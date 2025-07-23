import cv2
import numpy as np
import torch
import torchreid
from PIL import Image
import os
import glob
from collections import defaultdict
import time
import json
import faiss # For efficient similarity search

# --- Configuration and Global Variables ---

# Ensure OpenCV does not use multiple threads internally to avoid conflicts with other threading.
cv2.setNumThreads(1)

# Initialize Re-ID model (OSNet)
reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1041, # Number of classes in the pre-training dataset (e.g., MSMT17)
    loss='softmax',
    pretrained=False # Weights will be loaded manually
)

# Path to the pre-trained Re-ID model weights.
# IMPORTANT: You MUST update this path to the actual location of your .pth weight file.
# This should be the same path as used in detector.py.
weight_path = r'C:\Users\Omprakash\Desktop\pthon\osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'
try:
    torchreid.utils.load_pretrained_weights(reid_model, weight_path)
    print("Re-ID model weights loaded successfully for drone tracker.")
except FileNotFoundError:
    print(f" Error: Re-ID model weights not found at {weight_path}.")
    print("Please download the weights and update the 'weight_path' variable.")
    exit()
except Exception as e:
    print(f"Error loading Re-ID model weights for drone tracker: {e}")
    exit()

reid_model.eval() # Set model to evaluation mode
# Move model to GPU if available
if torch.cuda.is_available():
    reid_model.cuda()
    print("CUDA is available. Re-ID model moved to GPU for drone tracker.")
else:
    print("CUDA is not available. Re-ID model running on CPU for drone tracker (may be slower).")

# Define the image transformation pipeline for the Re-ID model.
reid_transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    norm_mean=[0.485, 0.456, 0.406], # ImageNet mean for normalization
    norm_std=[0.229, 0.224, 0.225]   # ImageNet std for normalization
)[0] # Take the first transform (training transform, suitable for inference here)

# Dictionaries to store loaded criminal data and tracking information
saved_criminal_features = {} # {person_id: feature_vector} for initial load
saved_criminal_images = {}   # {person_id: image_array} for initial load

# Comprehensive tracking data for each criminal, including multiple features/images
tracking_data = {} # {person_id: {'last_seen': frame, 'frames_tracked': int, 'stored_images': [], 'stored_features': [], ...}}

# Tracking parameters and thresholds
UPDATE_FREQUENCY = 10 # Frames to wait before considering adding a new image to a criminal's gallery
SIMILARITY_THRESHOLD = 0.60 # Minimum cosine similarity for a positive criminal identification
UPDATE_THRESHOLD = 0.45 # Lower threshold to check if a new image should update existing criminal data (currently unused, consider integrating)
MAX_TRACKING_FRAMES = 70 # Maximum number of frames to keep in tracking history for a criminal
MAX_IMAGES_PER_CRIMINAL = 7 # Maximum distinct images to store per criminal for multi-image matching
MIN_SIMILARITY_FOR_NEW_IMAGE = 0.24 # Minimum similarity to existing stored images for a new image to be considered 'distinct' enough to store.

# Decorator for Numba profiling (if Numba is used for specific functions, not directly here)
if 'profile' not in globals():
    def profile(func):
        return func

# --- Data Loading and Updating Functions ---

def load_saved_criminal_data():
    """
    Loads all saved criminal features (.npy) and images (.jpg) from the 'criminal_data/' directory.
    Initializes the `tracking_data` structure and updates the FAISS index with these features.
    """
    global saved_criminal_features, saved_criminal_images, tracking_data
    
    # Load feature files
    feature_files = glob.glob("criminal_data/features/*.npy")
    for feature_file in feature_files:
        filename = os.path.basename(feature_file)
        # Extract person_id from filename, assuming format like "criminal_ID1_ID2_..."
        parts = filename.replace('.npy', '').split('_')
        if len(parts) >= 4: # Expected format: criminal_X_Y_frame_... or criminal_X_Y_frame_updated_...
            try:
                # Assuming person_id is (X, Y) from filename parts[1] and parts[2]
                person_id = (int(parts[1]), int(parts[2]))
                feature = np.load(feature_file)
                feature = feature / np.linalg.norm(feature) # Ensure L2 normalization
                saved_criminal_features[person_id] = feature
            except ValueError:
                print(f"Warning: Could not parse person_id from feature file: {filename}")
                continue
    
    # Load image files
    image_files = glob.glob("criminal_data/images/*.jpg")
    for image_file in image_files:
        filename = os.path.basename(image_file)
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) >= 4:
            try:
                person_id = (int(parts[1]), int(parts[2]))
                image = cv2.imread(image_file)
                if image is not None:
                    saved_criminal_images[person_id] = image
            except ValueError:
                print(f"Warning: Could not parse person_id from image file: {filename}")
                continue
            
    # Initialize tracking_data for each loaded criminal
    for person_id in saved_criminal_features.keys():
        if person_id not in tracking_data: # Avoid re-initializing if already done
            tracking_data[person_id] = {
                'last_seen': 0, # Frame number when criminal was last seen
                'frames_tracked': 0, # Total frames this criminal has been tracked
                'last_update_frame': 0, # Last frame when stored_images/features were updated
                'stored_images': [img for img_id, img in saved_criminal_images.items() if img_id == person_id and img is not None], # List of stored distinct images
                'stored_features': [feat for feat_id, feat in saved_criminal_features.items() if feat_id == person_id], # List of stored distinct features
                'tracking_history': [], # List of (frame_number, similarity) tuples
                'confidence_scores': [] # List of similarity scores
            }
        # Ensure that if multiple images/features exist for the same person_id from `detector.py`, they are added.
        # This basic load just takes the last one loaded for a person_id if multiple exist,
        # A more robust loading would iterate all specific image/feature files for a given person_id.
        # For simplicity, we assume detector.py saves one primary image/feature per person_id.
        
    print(f"Loaded {len(saved_criminal_features)} unique criminals for tracking initialization.")
    
    # Update the FAISS index with all loaded criminal features for multi-image matching
    update_multi_feature_matrix(tracking_data)

def update_criminal_data(person_id, new_image, new_feature, frame_count, similarity):
    """
    Updates the tracking data for a criminal, including adding new, distinct images
    to their gallery (up to MAX_IMAGES_PER_CRIMINAL).

    Args:
        person_id (tuple): The ID of the criminal being tracked.
        new_image (np.array): The newly cropped image of the criminal.
        new_feature (np.array): The Re-ID feature extracted from the new_image.
        frame_count (int): The current frame number.
        similarity (float): The similarity score of the current detection to the matched criminal.
    """
    global saved_criminal_features, saved_criminal_images, tracking_data
    
    # Initialize tracking data for a new criminal ID if not already present
    if person_id not in tracking_data:
        tracking_data[person_id] = {
            'last_seen': frame_count,
            'frames_tracked': 1,
            'last_update_frame': frame_count,
            'stored_images': [new_image], # Start with the current new image
            'stored_features': [new_feature], # Start with the current new feature
            'tracking_history': [(frame_count, similarity)],
            'confidence_scores': [similarity]
        }
        # Save this new criminal data immediately
        save_updated_criminal_data(person_id, new_feature, new_image, frame_count, 0) # Index 0 for first image
        update_multi_feature_matrix(tracking_data) # Update FAISS index
        print(f"🆕 New criminal ID {person_id} added to tracking with first image.")
        return # Exit, as initial setup is done

    # Update existing criminal's tracking information
    tracking_data[person_id]['last_seen'] = frame_count
    tracking_data[person_id]['frames_tracked'] += 1
    tracking_data[person_id]['tracking_history'].append((frame_count, similarity))
    tracking_data[person_id]['confidence_scores'].append(similarity)
    
    # Keep tracking history within a defined length
    if len(tracking_data[person_id]['tracking_history']) > MAX_TRACKING_FRAMES:
        tracking_data[person_id]['tracking_history'] = tracking_data[person_id]['tracking_history'][-MAX_TRACKING_FRAMES:]
        tracking_data[person_id]['confidence_scores'] = tracking_data[person_id]['confidence_scores'][-MAX_TRACKING_FRAMES:]
    
    should_add_image = False
    
    # Decide whether to add a new image to the criminal's gallery
    if len(tracking_data[person_id]['stored_images']) < MAX_IMAGES_PER_CRIMINAL:
        # If there's capacity, initially allow adding images
        should_add_image = True
    else:
        # If gallery is full, check if the new image is sufficiently distinct
        if tracking_data[person_id]['stored_features']:
            existing_features = np.array(tracking_data[person_id]['stored_features'])
            # Calculate similarity of the new feature to all existing stored features
            similarities_to_existing = cosine_similarity_vectorized(new_feature, existing_features)
            max_similarity_to_existing = np.max(similarities_to_existing)
        else:
            max_similarity_to_existing = -1 # No existing features

        # Add image if it's distinct enough from currently stored ones
        if max_similarity_to_existing < MIN_SIMILARITY_FOR_NEW_IMAGE:
            should_add_image = True
            print(f"Criminal {person_id}: Distinct image detected (max similarity to existing: {max_similarity_to_existing:.3f}).")
    
    # If a new image should be added and the update frequency is met
    if should_add_image and \
       (frame_count - tracking_data[person_id]['last_update_frame'] >= UPDATE_FREQUENCY):
        
        tracking_data[person_id]['stored_images'].append(new_image)
        tracking_data[person_id]['stored_features'].append(new_feature)
        tracking_data[person_id]['last_update_frame'] = frame_count # Update last update frame
        
        # If adding a new image exceeds max capacity, remove the oldest one
        if len(tracking_data[person_id]['stored_images']) > MAX_IMAGES_PER_CRIMINAL:
            tracking_data[person_id]['stored_images'].pop(0) # Remove oldest image
            tracking_data[person_id]['stored_features'].pop(0) # Remove oldest feature
            print(f"🗑️ Criminal {person_id}: Max images reached. Removing oldest image.")
        
        # Save the updated criminal data to disk
        save_updated_criminal_data(
            person_id, 
            new_feature, 
            new_image, 
            frame_count, 
            len(tracking_data[person_id]['stored_images']) - 1 # Index of the newly added image
        )
        
        update_multi_feature_matrix(tracking_data) # Update FAISS index after adding new image/feature
        
        print(f"🔄 Criminal {person_id}: Added image {len(tracking_data[person_id]['stored_images'])}/{MAX_IMAGES_PER_CRIMINAL} (similarity to current detection: {similarity:.3f}).")

def save_updated_criminal_data(person_id, feature, image, frame_count, image_index):
    """
    Saves an updated criminal image and its feature to disk, along with metadata.

    Args:
        person_id (tuple): The ID of the criminal.
        feature (np.array): The Re-ID feature to save.
        image (np.array): The image to save.
        frame_count (int): The current frame number.
        image_index (int): The index of this image within the criminal's stored images.
    """
    # Create directories if they don't exist
    os.makedirs("criminal_data", exist_ok=True)
    os.makedirs("criminal_data/images", exist_ok=True)
    os.makedirs("criminal_data/features", exist_ok=True)
    
    # Define file paths with a unique name including frame_count and image_index
    image_path = f"criminal_data/images/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}_updated_{image_index}.jpg"
    cv2.imwrite(image_path, image) # Save image
    
    feature_path = f"criminal_data/features/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}_updated_{image_index}.npy"
    np.save(feature_path, feature) # Save feature
    
    # Save tracking metadata as a JSON file
    metadata_path = f"criminal_data/tracking_metadata_{person_id[0]}_{person_id[1]}.json"
    metadata = {
        'person_id': person_id,
        'last_update_frame': frame_count,
        'frames_tracked': tracking_data[person_id]['frames_tracked'],
        'images_stored': len(tracking_data[person_id]['stored_images']),
        'average_confidence': np.mean(tracking_data[person_id]['confidence_scores']) if tracking_data[person_id]['confidence_scores'] else 0,
        'max_confidence': np.max(tracking_data[person_id]['confidence_scores']) if tracking_data[person_id]['confidence_scores'] else 0,
        'tracking_history_length': len(tracking_data[person_id]['tracking_history'])
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2) # Save with pretty-print indentation

    # print(f"  Saved updated criminal data for {person_id}: {image_path}, {feature_path}, {metadata_path}")


# --- Re-ID Feature Extraction ---

def extract_reid_feature(image_bgr, model):
    """
    Extracts re-identification feature vector from a cropped person image.
    Optimized for speed by using `torch.no_grad()`.

    Args:
        image_bgr (np.array): Cropped person image in BGR format.
        model (torch.nn.Module): The Re-ID model.

    Returns:
        np.array: L2-normalized feature vector.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    pil_image = Image.fromarray(image_rgb) # Convert to PIL Image
    
    # Apply transformations, add batch dimension, and move to GPU if available
    image = reid_transform(pil_image).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    
    with torch.no_grad(): # Disable gradient calculation for inference
        features = model(image)
    
    features_np = features.cpu().numpy().flatten() # Move to CPU, convert to numpy, flatten
    features_np = features_np / np.linalg.norm(features_np) # L2 normalize features
    return features_np

# --- FAISS Multi-Image Similarity Search ---

all_features_matrix = None # Stores a concatenated matrix of all stored features from all criminals
all_person_ids = []        # Corresponding person IDs for `all_features_matrix`
all_image_indices = []     # Corresponding image indices within each criminal's gallery
faiss_index = None         # The FAISS index for multi-image search

def initialize_faiss_index_multi():
    """
    Initializes or re-initializes the FAISS index specifically for multi-image similarity search.
    Attempts to use GPU acceleration for FAISS.
    """
    global faiss_index
    dimension = 512 # Dimension of OSNet features
    # Use Inner Product (IP) index for cosine similarity with L2-normalized vectors
    faiss_index = faiss.IndexFlatIP(dimension)
    
    # Attempt to move the FAISS index to GPU
    if torch.cuda.is_available():
        try:
            if hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index) # Move to GPU 0
                
            else:
                print("FAISS StandardGpuResources not found, using CPU version for multi-image matching.")
        except Exception as e:
            print(f"FAISS GPU initialization failed: {e}, using CPU version for multi-image matching.")
    else:
        print("CUDA is not available. FAISS multi-image matching will run on CPU.")
    
    return faiss_index

def update_multi_feature_matrix(tracking_data):
    """
    Updates the global FAISS index with all stored features from all tracked criminals.
    This function rebuilds the index, which is necessary when new images/features are added.

    Args:
        tracking_data (dict): The comprehensive tracking data dictionary.
    """
    global all_features_matrix, all_person_ids, all_image_indices, faiss_index
    
    if not tracking_data:
        # If no tracking data, clear all related global variables and FAISS index
        all_features_matrix = None
        all_person_ids = []
        all_image_indices = []
        if faiss_index is not None:
            faiss_index.reset()
        return
    
    features_list = []
    person_ids_list = []
    image_indices_list = []
    
    # Aggregate all stored features and their corresponding IDs/indices
    for person_id, data in tracking_data.items():
        for i, feature in enumerate(data['stored_features']):
            features_list.append(feature)
            person_ids_list.append(person_id)
            image_indices_list.append(i) # Store index within that criminal's gallery
    
    if not features_list: # If no features were aggregated
        all_features_matrix = None
        all_person_ids = []
        all_image_indices = []
        if faiss_index is not None:
            faiss_index.reset()
        return
    
    all_features_matrix = np.array(features_list)
    all_person_ids = person_ids_list
    all_image_indices = image_indices_list
    
    # Ensure all features are L2-normalized (unit vectors) for IP search
    all_features_matrix = all_features_matrix / np.linalg.norm(all_features_matrix, axis=1, keepdims=True)
    
    # Initialize FAISS index if not done, or reset and add features
    if faiss_index is None:
        faiss_index = initialize_faiss_index_multi()
    else:
        faiss_index.reset() # Clear existing data
    
    # Add features to the FAISS index. FAISS expects float32.
    faiss_index.add(all_features_matrix.astype(np.float32))
    print(f"FAISS multi-image index updated with {faiss_index.ntotal} features.")

def cosine_similarity_vectorized(query_feature, features_matrix):
    """
    Calculates cosine similarity between a single query feature and a matrix of features.
    This is a fallback/helper function, usually FAISS is preferred for speed.

    Args:
        query_feature (np.array): The feature vector to compare.
        features_matrix (np.array): A matrix of feature vectors to compare against (rows are features).

    Returns:
        np.array: An array of similarity scores.
    """
    query_normalized = query_feature / np.linalg.norm(query_feature)
    # Dot product of normalized vectors gives cosine similarity
    similarities = features_matrix @ query_normalized
    return similarities

def find_best_match_multi_vectorized(query_feature, tracking_data, threshold=0.60):
    """
    Finds the best matching criminal for a query feature using the multi-image FAISS index.
    Compares the query feature against all stored features for all known criminals.

    Args:
        query_feature (np.array): The feature vector of the person detected in the current frame.
        tracking_data (dict): The comprehensive tracking data.
        threshold (float): The minimum similarity score for a match to be considered valid.

    Returns:
        tuple: (matched_person_id, similarity_score, image_index) if a match above threshold is found.
               Returns (None, best_similarity_found, -1) if no match or below threshold.
    """
    global all_features_matrix, all_person_ids, all_image_indices, faiss_index
    
    if not tracking_data:
        return None, -1, -1 # No criminals to track

    # Check if the FAISS index needs to be rebuilt or initialized.
    # This comparison (len(all_person_ids) != current_total_features) helps trigger
    # a rebuild when a new criminal or a new image for an existing criminal is added.
    current_total_features = sum(len(data['stored_features']) for data in tracking_data.values())
    if (all_features_matrix is None or 
        len(all_person_ids) != current_total_features or
        (faiss_index is not None and faiss_index.ntotal != current_total_features)): # Also check faiss_index size
        
        print("Rebuilding FAISS multi-image index due to data changes.")
        update_multi_feature_matrix(tracking_data) # Rebuild the index

    if all_features_matrix is None or len(all_features_matrix) == 0:
        return None, -1, -1 # No features to compare against

    query_normalized = query_feature / np.linalg.norm(query_feature) # L2 normalize query feature

    if faiss_index is not None and faiss_index.ntotal > 0:
        # Search the FAISS index for the top K (e.g., 10) most similar features
        k = min(faiss_index.ntotal, 10) # Search up to 10 nearest neighbors
        similarities, indices = faiss_index.search(
            query_normalized.reshape(1, -1).astype(np.float32), 
            k
        )
        similarities = similarities.flatten()
        indices = indices.flatten()
        
        best_idx = -1
        best_similarity = -1.0
        
        # Iterate through FAISS search results
        for sim, idx in zip(similarities, indices):
            if idx < len(all_person_ids) and sim >= threshold: # Check valid index and threshold
                if sim > best_similarity: # Find the best similarity above threshold
                    best_similarity = sim
                    best_idx = idx
        
        if best_idx >= 0:
            best_person_id = all_person_ids[best_idx]
            best_image_index = all_image_indices[best_idx]
            print(f"  FAISS match found: ID {best_person_id} (Image {best_image_index+1}), Similarity: {best_similarity:.3f}")
            return best_person_id, best_similarity, best_image_index
    
    # Fallback to pure NumPy calculation if FAISS fails or is not used
    # This block will execute if FAISS is not set up, or if no match was found via FAISS.
    print("  FAISS did not find a match above threshold or is not used. Falling back to NumPy comparison.")
    similarities_np = all_features_matrix @ query_normalized
    best_idx_np = np.argmax(similarities_np)
    best_similarity_np = similarities_np[best_idx_np]
    
    if best_similarity_np >= threshold:
        best_person_id_np = all_person_ids[best_idx_np]
        best_image_index_np = all_image_indices[best_idx_np]
        print(f"  NumPy fallback match: ID {best_person_id_np} (Image {best_image_index_np+1}), Similarity: {best_similarity_np:.3f}")
        return best_person_id_np, best_similarity_np, best_image_index_np
    
    return None, best_similarity_np, -1 # No match found or below threshold

# --- Drawing Functions for Visualization ---

def mark_criminal_found(frame, bbox, person_id, similarity, display_image=None, image_index=0):
    """
    Draws a bounding box and labels on the frame when a criminal is found.
    The bounding box is always RED for criminals.

    Args:
        frame (np.array): The image frame to draw on.
        bbox (list/np.array): Bounding box coordinates [x1, y1, x2, y2].
        person_id (tuple): The ID of the matched criminal.
        similarity (float): The similarity score of the match.
        display_image (np.array): An optional image (e.g., stored criminal image) to display.
        image_index (int): The index of the displayed stored image (for label).
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    color = (0, 0, 255) # Red color for criminals (BGR)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3) # Draw bounding box
    
    # Display criminal ID and image index
    label = f"🚨 CRIMINAL FOUND - ID: {person_id} (Image {image_index+1})"
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display similarity score
    similarity_text = f"Match: {similarity:.3f}"
    cv2.putText(frame, similarity_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display tracking information if available
    if person_id in tracking_data:
        tracking_info = tracking_data[person_id]
        frames_tracked = tracking_info['frames_tracked']
        images_stored = len(tracking_info['stored_images'])
        avg_confidence = np.mean(tracking_info['confidence_scores']) if tracking_info['confidence_scores'] else 0
        tracking_text = f"Tracked: {frames_tracked} frames, Images: {images_stored}/{MAX_IMAGES_PER_CRIMINAL}, Avg: {avg_confidence:.3f}"
        cv2.putText(frame, tracking_text, (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Display a small version of the stored criminal image for visual confirmation
    if display_image is not None and display_image.size != 0:
        h, w = display_image.shape[:2]
        display_h = 100 # Fixed height for display
        display_w = int(w * display_h / h) # Maintain aspect ratio
        display_img = cv2.resize(display_image, (display_w, display_h))
        
        # Position the image in the top-right corner
        img_y_offset = 10
        img_x_offset = frame.shape[1] - display_w - 10
        
        # Overlay the image, checking boundaries
        if img_y_offset + display_h <= frame.shape[0] and img_x_offset + display_w <= frame.shape[1] and \
           img_y_offset >= 0 and img_x_offset >= 0:
            frame[img_y_offset:img_y_offset+display_h, img_x_offset:img_x_offset+display_w] = display_img
            # Draw a green border around the displayed stored image
            cv2.rectangle(frame, (img_x_offset-2, img_y_offset-2), (img_x_offset+display_w+2, img_y_offset+display_h+2), (0, 255, 0), 2)
            cv2.putText(frame, f"Stored Image {image_index+1}", (img_x_offset, img_y_offset-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)


# --- Main Execution Block ---

if __name__ == "__main__":
    # Load YOLOv8 detection model (small version, pre-trained on COCO dataset)
    # IMPORTANT: Ensure 'yolov8s.pt' is accessible. Download from: https://github.com/ultralytics/ultralytics
    try:
        yolo_model = YOLO("yolov8s.pt")
        print(" model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv8s model: {e}")
        print("Please ensure 'yolov8s.pt' is in your current directory or provide full path.")
        exit()

    # Path to the drone video file
    # IMPORTANT: Update this path to your drone video file.
    drone_video_path = r".mp4"
        
    cap = cv2.VideoCapture(drone_video_path)
        
    if not cap.isOpened():
        print(f" Could not open video file: {drone_video_path}")
        exit()
        
    cv2.startWindowThread() # Required for some OpenCV installations to manage windows
    cv2.namedWindow("Drone Criminal Tracker", cv2.WINDOW_NORMAL) # Create a resizable window
        
    frame_count = 0 # Global frame counter for the video
    criminals_found = 0 # Maximum number of criminals found in a single frame
    tracking_history = defaultdict(list) # Detailed tracking history per criminal ID

    # Load previously saved criminal data before starting the video processing
    load_saved_criminal_data()
    # Main video processing loop
    while cap.isOpened():
        frame_start_time = time.time() # Record start time for frame processing
        ret, frame = cap.read() # Read a frame from the video
        if not ret:
            break # Exit loop if video ends
        
        frame_count += 1
        
        # Skip frames to improve performance (process every 2nd frame in this case)
        if frame_count % 2 != 0: # If frame_count is odd, skip this frame
            continue
        
        yolo_start = time.time()
        # Run YOLO detection. `conf=0.4` is the object detection confidence threshold.
        # `classes=[0]` specifically filters for 'person' class (COCO dataset class ID for person is 0).
        results = yolo_model(frame, conf=0.4, classes=[0])
        yolo_time = (time.time() - yolo_start) * 1000 # Time for YOLO inference in ms
        
        current_frame_criminals = 0 # Counter for criminals identified in the current frame
        people_detected = 0 # Total number of persons detected by YOLO in the current frame
        
        for r in results: # Iterate through YOLO results (each 'r' is a results object for a batch or single image)
            if r.boxes is not None:
                boxes = r.boxes.xyxy # Get bounding box coordinates [x1, y1, x2, y2]
                if hasattr(boxes, 'cpu'): # Ensure data is on CPU and is a NumPy array
                    boxes = boxes.cpu().numpy()
                
                people_detected = len(boxes) # Number of persons detected in this frame
                
                for j, box in enumerate(boxes): # Iterate through each detected person's bounding box
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Skip very small detections, as they are unlikely to be useful for Re-ID
                    if x2 - x1 < 30 or y2 - y1 < 50:
                        # print(f"Skipping small detection: w={x2-x1}, h={y2-y1}")
                        continue
                    
                    person_crop = frame[y1:y2, x1:x2] # Crop the person from the frame
                    
                    if person_crop.size == 0: # Skip if crop is empty (e.g., invalid bbox)
                        continue
                    
                    reid_start = time.time()
                    try:
                        # Extract Re-ID feature from the cropped person image
                        reid_feature = extract_reid_feature(person_crop, reid_model)
                        reid_time = (time.time() - reid_start) * 1000 # Time for Re-ID feature extraction in ms
                        
                        matching_start = time.time()
                        # Find the best match among all stored criminal features using multi-image FAISS
                        best_match, similarity, best_image_index = find_best_match_multi_vectorized(reid_feature, tracking_data, threshold=SIMILARITY_THRESHOLD)
                        matching_time = (time.time() - matching_start) * 1000 # Time for feature matching in ms
                        
                        if best_match: # If a criminal match is found above the similarity threshold
                            # Retrieve the image that provided the best match for display
                            display_image = tracking_data[best_match]['stored_images'][best_image_index]
                            # Mark the criminal on the frame
                            mark_criminal_found(frame, box, best_match, similarity, display_image, best_image_index)
                            
                            # Update the criminal's tracking data (including potentially adding a new image)
                            update_criminal_data(best_match, person_crop, reid_feature, frame_count, similarity)
                            
                            current_frame_criminals += 1 # Increment count of criminals in current frame
                            criminals_found = max(criminals_found, current_frame_criminals) # Update max simultaneous criminals
                            
                            # Add to specific criminal's tracking history
                            tracking_history[best_match].append((frame_count, similarity))
                            
                            print(f"🚨 CRIMINAL SPOTTED! ID: {best_match}, Similarity: {similarity:.3f}, Frame: {frame_count}")
                            
                    except Exception as e:
                        print(f"Error processing person {j} (bbox: {box}): {e}")
                        continue
            
        frame_time = (time.time() - frame_start_time) * 1000 # Total time to process the current frame in ms
        
        # --- Display real-time information on the frame ---
        cv2.putText(frame, f"People Detected: {people_detected}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2) # Green for general people
        cv2.putText(frame, f"Criminals Tracked: {current_frame_criminals}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red for criminals
        cv2.putText(frame, f"Similarity Threshold: {SIMILARITY_THRESHOLD:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2) # White
        cv2.putText(frame, f"Multi-Image Matching: ON ({MAX_IMAGES_PER_CRIMINAL} per criminal)", (10, 115), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2) # Yellow
        cv2.putText(frame, f"Frame: {frame_count}", (10, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1) # White
        
        current_fps = 1000 / frame_time if frame_time > 0 else 0 # Calculate FPS
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 165), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1) # Yellow
        cv2.putText(frame, f"Frame Time: {frame_time:.1f}ms", (10, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1) # Yellow
        
        # Information box in top-right corner
        info_box_x = frame.shape[1] - 200
        info_box_y = 10
        cv2.rectangle(frame, (info_box_x, info_box_y), (frame.shape[1]-10, info_box_y+90), (0, 0, 0), -1) # Black background
        cv2.putText(frame, "Red = Criminal (All Images)", (info_box_x + 10, info_box_y + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(frame, "Green = Normal Person", (info_box_x + 10, info_box_y + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "Press 'q' to quit", (info_box_x + 10, info_box_y + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.imshow("Drone Criminal Tracker", frame) # Display the annotated frame
        
        # --- Handle keyboard input ---
        key = cv2.waitKey(1) & 0xFF # Wait for 1 ms and get key press
        if key == ord("q"): # Press 'q' to quit
            break
        elif key == ord("t"): # Press 't' to adjust similarity threshold
            print(f"Current similarity threshold: {SIMILARITY_THRESHOLD:.2f}")
            try:
                new_threshold_str = input("Enter new threshold (0.1-0.9): ")
                new_threshold = float(new_threshold_str)
                if 0.1 <= new_threshold <= 0.9:
                    SIMILARITY_THRESHOLD = new_threshold
                    print(f"Threshold changed to: {SIMILARITY_THRESHOLD:.2f}")
                else:
                    print("Invalid threshold range (0.1-0.9), keeping current value.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        elif key == ord("p"):
            # Placeholder for performance summary (can be extended to show detailed timings)
            print("\nPerformance summary requested (detailed breakdown requires code modification):")
            print(f"  Last Frame Total Time: {frame_time:.1f} ms")
            # Example (requires accumulating times):
            # print(f"  Average YOLO Detection Time: {avg_yolo_time:.1f} ms")
            # print(f"  Average Re-ID Feature Extraction Time: {avg_reid_time:.1f} ms")
            # print(f"  Average FAISS Matching Time: {avg_matching_time:.1f} ms")
        
    # --- Cleanup after video processing loop ---
    cap.release() # Release the video capture object
    cv2.destroyAllWindows() # Close all OpenCV windows
    
    # --- Final Summary ---
    print(f"\n=== Drone Tracking Summary (MULTI-IMAGE MATCHING) ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Maximum criminals found in a single frame: {criminals_found}")
    print(f"Unique criminals tracked: {len(tracking_history)}")
    print(f"Final similarity threshold used: {SIMILARITY_THRESHOLD:.2f}")
    
    # Print detailed tracking stats for each criminal
    if tracking_history:
        print("\nIndividual Criminal Tracking Details:")
        for person_id, history in tracking_history.items():
            if history:
                max_similarity = max([sim for _, sim in history])
                frames_detected = len(history)
                if person_id in tracking_data and tracking_data[person_id]['confidence_scores']:
                    avg_confidence = np.mean(tracking_data[person_id]['confidence_scores'])
                    print(f"  Criminal {person_id}: Detected in {frames_detected} frames, Max similarity: {max_similarity:.3f}, Avg confidence: {avg_confidence:.3f}")
                else:
                    print(f"  Criminal {person_id}: Detected in {frames_detected} frames, Max similarity: {max_similarity:.3f} (No confidence history).")
            else:
                print(f"  Criminal {person_id}: No detection history.")
    else:
        print("No criminals were tracked during this session.")
