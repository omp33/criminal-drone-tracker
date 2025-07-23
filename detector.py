import cv2
import numpy as np
import math
from collections import defaultdict
import time
import torch
import torchreid
from PIL import Image
from threading import Thread
import os
import faiss
from ultralytics import YOLO
from numba import jit, njit, prange

# Ensure OpenCV does not use multiple threads internally, which can conflict with our threading
cv2.setNumThreads(1)

# Global configuration and state variables
threat_history = defaultdict(list)  # Stores recent threat detections for each person
persistence_frames = 7  # Number of frames to consider for confirming a threat
min_confidence = 0.45  # Minimum confidence for YOLO detections (currently not directly used in the loop, but good to have)
frame_count = 0  # Global frame counter
confirmed_threats = set()  # Set of person IDs confirmed as threats
reid_features = {}  # Stores re-identification features for detected individuals
detection_stopped = False  # Flag to stop detection after a criminal is identified and saved

def update_frame():
    """Increments the global frame counter."""
    global frame_count
    frame_count += 1

def _get_person_id(bbox):
    """
    Generates a unique ID for a person based on their bounding box center.
    This helps in tracking individuals coarsely across frames.
    """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    # Quantize coordinates to create a grid-based ID
    return (round(center_x / 50) * 50, round(center_y / 50) * 50)

def add_threat_detection(person_bbox, is_threat):
    """
    Updates the threat history for a person and determines if they are a confirmed threat.

    Args:
        person_bbox (list): Bounding box coordinates of the person.
        is_threat (bool): True if the current pose analysis indicates a threat, False otherwise.

    Returns:
        bool: True if the person is a confirmed threat, False otherwise.
    """
    global threat_history, confirmed_threats, frame_count
    person_id = _get_person_id(person_bbox)
    current_time = frame_count

    # Add current detection to history
    threat_history[person_id].append((current_time, is_threat))

    # Clean up old entries from history (only keep recent ones)
    threat_history[person_id] = [
        (t, threat) for t, threat in threat_history[person_id]
        if current_time - t <= persistence_frames * 2
    ]

    # Count recent threat detections
    recent_threats = [
        threat for t, threat in threat_history[person_id]
        if current_time - t <= persistence_frames and threat
    ]

    # Confirm threat if enough recent threat detections
    if len(recent_threats) >= persistence_frames // 2:
        confirmed_threats.add(person_id)
        return True
    else:
        confirmed_threats.discard(person_id)
        return False

# Numba-optimized functions for pose analysis (for performance)
if 'profile' not in globals():
    def profile(func):
        return func

@njit(fastmath=True, cache=True)
def calculate_angle_numba(x1, y1, x2, y2, x3, y3):
    """
    Calculates the angle between three points (p1-p2-p3) using Numba for optimization.

    Args:
        x1, y1 (float): Coordinates of point 1.
        x2, y2 (float): Coordinates of point 2 (vertex).
        x3, y3 (float): Coordinates of point 3.

    Returns:
        float: Angle in degrees.
    """
    v1x = x1 - x2
    v1y = y1 - y2
    
    v2x = x3 - x2
    v2y = y3 - y2
    
    dot_product = v1x * v2x + v1y * v2y
    
    mag1 = math.sqrt(v1x * v1x + v1y * v1y)
    mag2 = math.sqrt(v2x * v2x + v2y * v2y)
    
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    
    cos_angle = dot_product / (mag1 * mag2)
    
    # Clip cos_angle to avoid numerical issues that can lead to NaN from acos
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    
    angle = math.degrees(math.acos(cos_angle))
    return angle

@njit(fastmath=True, cache=True)
def is_arm_straight_numba(x1, y1, x2, y2, x3, y3, threshold=40.0):
    """
    Checks if an arm (shoulder-elbow-wrist) is straight using Numba.

    Args:
        x1, y1 (float): Shoulder coordinates.
        x2, y2 (float): Elbow coordinates.
        x3, y3 (float): Wrist coordinates.
        threshold (float): Maximum deviation from 180 degrees for a straight arm.

    Returns:
        bool: True if the arm is considered straight, False otherwise.
    """
    angle = calculate_angle_numba(x1, y1, x2, y2, x3, y3)
    return abs(180.0 - angle) <= threshold

@njit(fastmath=True, cache=True)
def is_forward_aiming_numba(shoulder_x, shoulder_y, wrist_x, wrist_y, body_center_x, body_center_y, max_y_diff=51.0):
    """
    Checks if an arm is aiming forward, relative to the body center, using Numba.

    Args:
        shoulder_x, shoulder_y (float): Shoulder coordinates.
        wrist_x, wrist_y (float): Wrist coordinates.
        body_center_x, body_center_y (float): Body center coordinates.
        max_y_diff (float): Maximum allowed vertical difference between wrist and shoulder for "forward aiming".

    Returns:
        bool: True if the arm is considered to be aiming forward, False otherwise.
    """
    dx = wrist_x - body_center_x
    dy = abs(wrist_y - shoulder_y)
    # Checks for significant horizontal displacement and small vertical displacement
    return abs(dx) > 50.0 and dy < max_y_diff

def calculate_angle(point1, point2, point3):
    """
    Calculates the angle between three points (p1-p2-p3).
    This is a non-Numba version, primarily for clarity or fallback.
    """
    try:
        p1 = np.array(point1)
        p2 = np.array(point2)
        p3 = np.array(point3)
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        return angle
    except:
        return None

def is_arm_straight(shoulder, elbow, wrist, threshold=40):
    """
    Checks if an arm (shoulder-elbow-wrist) is straight using the non-Numba angle calculation.
    """
    if None in [shoulder, elbow, wrist]:
        return False
    angle = calculate_angle(shoulder, elbow, wrist)
    if angle is None:
        return False
    return abs(180 - angle) <= threshold

def get_arm_angle_from_torso(shoulder, elbow, hip):
    """
    Calculates the angle of the arm relative to the torso (not directly used for threat detection, but useful).
    """
    if None in [shoulder, elbow, hip]:
        return None
    reference_point = (hip[0], hip[1] - 100) # A point directly above the hip to define a vertical line for reference
    angle = calculate_angle(reference_point, shoulder, elbow)
    return angle

def analyze_pose_threat(keypoints_batch, confidence_threshold=0.5):
    """
    Analyzes a batch of keypoints to determine if a threatening pose is detected.

    Args:
        keypoints_batch (np.array): Keypoints for one or more persons.
                                    Shape: (N, 17, 3) where N is number of persons,
                                    17 keypoints (x, y, confidence).
        confidence_threshold (float): Minimum confidence for a keypoint to be considered valid.

    Returns:
        list of dict: Analysis results for each person, including 'is_threat' and details.
                      If N=1, returns a single dictionary instead of a list.
    """
    keypoints_batch = np.asarray(keypoints_batch)
    if keypoints_batch.ndim == 2:
        keypoints_batch = keypoints_batch[None, ...] # Add batch dimension if only one person

    N = keypoints_batch.shape[0] # Number of persons in the batch

    # Mappings for keypoint indices
    idxs = {
        'left_shoulder': 5, 'right_shoulder': 6,
        'left_elbow': 7, 'right_elbow': 8,
        'left_wrist': 9, 'right_wrist': 10,
        'left_hip': 11, 'right_hip': 12
    }

    def get_xy(idx):
        """Extracts (x, y) coordinates for a given keypoint index, filtering by confidence."""
        conf = keypoints_batch[:, idx, 2] # Confidence score
        mask = conf > confidence_threshold # Mask for confident keypoints
        xy = keypoints_batch[:, idx, :2].astype(np.float32) # (x, y) coordinates
        xy[~mask] = np.nan # Set non-confident keypoints to NaN
        return xy

    # Get coordinates for relevant keypoints
    left_shoulder = get_xy(idxs['left_shoulder'])
    right_shoulder = get_xy(idxs['right_shoulder'])
    left_elbow = get_xy(idxs['left_elbow'])
    right_elbow = get_xy(idxs['right_elbow'])
    left_wrist = get_xy(idxs['left_wrist'])
    right_wrist = get_xy(idxs['right_wrist'])
    left_hip = get_xy(idxs['left_hip'])
    right_hip = get_xy(idxs['right_hip'])

    # Calculate torso center (midpoint of hips)
    torso_center = np.nanmean(np.stack([left_hip, right_hip], axis=1), axis=1)

    def arm_straight(shoulder, elbow, wrist, threshold=40):
        """Vectorized check for straight arm using numpy."""
        v1 = shoulder - elbow
        v2 = wrist - elbow
        dot = np.nansum(v1 * v2, axis=1) # Dot product
        mag1 = np.linalg.norm(v1, axis=1) # Magnitude
        mag2 = np.linalg.norm(v2, axis=1)
        
        # Handle division by zero for magnitudes
        with np.errstate(divide='ignore', invalid='ignore'):
            cos_angle = dot / (mag1 * mag2)
        
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        
        # Determine if straight, setting to False if angle is NaN (due to missing keypoints)
        straight = np.abs(180.0 - angles) <= threshold
        straight[np.isnan(angles)] = False
        return straight, angles

    # Check straightness of left and right arms
    left_arm_straight, left_arm_angle = arm_straight(left_shoulder, left_elbow, left_wrist)
    right_arm_straight, right_arm_angle = arm_straight(right_shoulder, right_elbow, right_wrist)

    def is_forward_aiming(shoulder, wrist, torso, max_y_diff=51.0):
        """Vectorized check if arm is aiming forward."""
        dx = wrist[:, 0] - torso[:, 0] # Horizontal displacement from torso center
        dy = np.abs(wrist[:, 1] - shoulder[:, 1]) # Vertical difference between wrist and shoulder
        
        # Aiming if horizontal displacement is significant and vertical is small
        aiming = (np.abs(dx) > 50.0) & (dy < max_y_diff)
        
        # Set to False if any component is NaN
        aiming[np.isnan(dx) | np.isnan(dy)] = False
        return aiming

    # Check if left and right arms are aiming forward
    left_forward = is_forward_aiming(left_shoulder, left_wrist, torso_center)
    right_forward = is_forward_aiming(right_shoulder, right_wrist, torso_center)

    # A person is a threat if either arm is straight AND aiming forward
    is_threat = (left_arm_straight & left_forward) | (right_arm_straight & right_forward)

    results = []
    for i in range(N):
        threat_details = []
        if left_arm_straight[i] and left_forward[i]:
            threat_details.append("Left arm aiming forward")
        if right_arm_straight[i] and right_forward[i]:
            threat_details.append("Right arm aiming forward")
        
        results.append({
            'is_threat': bool(is_threat[i]),
            'left_arm_straight': bool(left_arm_straight[i]),
            'right_arm_straight': bool(right_arm_straight[i]),
            'left_arm_angle': float(left_arm_angle[i]) if not np.isnan(left_arm_angle[i]) else None,
            'right_arm_angle': float(right_arm_angle[i]) if not np.isnan(right_arm_angle[i]) else None,
            'threat_details': threat_details
        })
    
    # If only one person, return a single dict for convenience
    if results and N == 1:
        return results[0]
    return results

def mark_confirmed_threat(frame, bbox, threat_details, confidence_level="HIGH"):
    """
    Draws a bounding box and threat labels on the frame for a confirmed threat.
    """
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4) # Red box for threat
    
    # Blinking effect for the outer rectangle
    if int(time.time() * 3) % 2:
        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 2)
    
    label_y = y1 - 15
    cv2.putText(frame, "🚨 CONFIRMED THREAT", (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display threat details
    for i, detail in enumerate(threat_details):
        detail_y = y1 - 35 - (i * 20)
        cv2.putText(frame, detail, (x1, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def crop_person(frame, bbox):
    """Crops the person bounding box from the frame."""
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0: # Handle empty crop if bbox is out of bounds or invalid
        return crop
    
    return crop

# Initialize Re-ID model
reid_model = torchreid.models.build_model(
    name='osnet_x1_0', # OSNet is a lightweight and effective Re-ID model
    num_classes=1041,  # Number of classes in the pre-training dataset (MSMT17)
    loss='softmax',
    pretrained=False   # We load weights manually
)
# Path to the pre-trained Re-ID model weights
# IMPORTANT: Update this path to where you've saved the .pth file
weight_path = r'C:\Users\Omprakash\Desktop\pthon\osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'
torchreid.utils.load_pretrained_weights(reid_model, weight_path)
reid_model.eval() # Set model to evaluation mode
reid_model.cuda() # Move model to GPU if available

# Define Re-ID image transformation pipeline
reid_transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    norm_mean=[0.485, 0.456, 0.406], # ImageNet mean for normalization
    norm_std=[0.229, 0.224, 0.225]   # ImageNet std for normalization
)[0] # Take the first transform (train transform is typically [0], val is [1])

def extract_reid_feature(image_bgr, model):
    """
    Extracts re-identification features from a cropped person image.

    Args:
        image_bgr (np.array): Cropped person image in BGR format.
        model (torch.nn.Module): The Re-ID model.

    Returns:
        np.array: Normalized feature vector.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # Convert BGR to RGB for PIL
    pil_image = Image.fromarray(image_rgb) # Convert to PIL Image
    image = reid_transform(pil_image).unsqueeze(0).cuda() # Apply transforms, add batch dim, move to GPU

    # Optional: Profile Re-ID inference (uncomment for detailed profiling)
    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as prof:
    with torch.no_grad(): # Disable gradient calculation for inference
        features = model(image)
    # prof.export_chrome_trace("reid_inference_trace.json") # Export trace if profiling
    
    features_np = features.cpu().numpy().flatten() # Move to CPU, convert to numpy, flatten
    features_np = features_np / np.linalg.norm(features_np) # L2 normalize features
    return features_np

# FAISS for efficient similarity search
feature_matrix = None # Stores all person features
person_ids = []       # Corresponding person IDs for the feature matrix
feature_index = None  # FAISS index for fast search

def initialize_faiss_index():
    """
    Initializes or re-initializes the FAISS index. Attempts to use GPU if available.
    """
    global feature_index
    dimension = 512 # Dimension of OSNet features
    feature_index = faiss.IndexFlatIP(dimension) # Inner Product for cosine similarity

    # Try to use FAISS GPU acceleration
    if torch.cuda.is_available():
        try:
            # Check for StandardGpuResources, which is the typical way to get GPU resources in newer FAISS versions
            if hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                feature_index = faiss.index_cpu_to_gpu(res, 0, feature_index) # Move index to GPU 0
                print("🚀 Using FAISS GPU acceleration")
            else:
                # Fallback for older FAISS versions or other configurations
                print("⚠️ FAISS GPU not available via StandardGpuResources, trying fallback or using CPU version")
                # You might need faiss.GpuResources or direct index_cpu_to_gpu with a GpuIndex
        except Exception as e:
            print(f"⚠️ FAISS GPU initialization failed: {e}, using CPU version")
    
    return feature_index

def update_feature_matrix(stored_features):
    """
    Updates the FAISS index with new or updated person features.

    Args:
        stored_features (dict): Dictionary of {person_id: reid_feature}.
    """
    global feature_matrix, person_ids, feature_index
    
    if not stored_features:
        feature_matrix = None
        person_ids = []
        if feature_index is not None:
            feature_index.reset() # Clear the index if no features
        return
    
    person_ids = list(stored_features.keys())
    feature_matrix = np.array([stored_features[pid] for pid in person_ids])
    
    # Re-normalize features if they are not already (important for IP index)
    feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    
    if feature_index is None:
        feature_index = initialize_faiss_index() # Initialize if not already
    else:
        feature_index.reset() # Clear existing data before adding new
    
    feature_index.add(feature_matrix.astype(np.float32)) # Add features to the index

def find_best_match_vectorized(query_feature, stored_features, threshold=0.38):
    """
    Finds the best matching person ID for a query feature using the FAISS index.

    Args:
        query_feature (np.array): Feature vector of the person to match.
        stored_features (dict): Dictionary of existing {person_id: feature_vector} pairs.
        threshold (float): Similarity threshold for a match to be considered valid.

    Returns:
        tuple: (matched_person_id, similarity_score) or (None, best_similarity_found) if no match.
    """
    global feature_matrix, person_ids, feature_index
    
    if not stored_features:
        return None, -1
    
    # Rebuild feature matrix and FAISS index if dimensions change or initially empty
    if feature_matrix is None or len(stored_features) != len(person_ids) or \
       (person_ids and sorted(list(stored_features.keys())) != sorted(person_ids)):
        update_feature_matrix(stored_features) # Call the update function to rebuild

    query_normalized = query_feature / np.linalg.norm(query_feature) # Normalize query feature
    
    if feature_index is not None and feature_index.ntotal > 0:
        # Search the FAISS index for the top 10 most similar features
        similarities, indices = feature_index.search(
            query_normalized.reshape(1, -1).astype(np.float32), 
            min(len(person_ids), 10) # Search up to 10 nearest neighbors
        )
        similarities = similarities.flatten()
        indices = indices.flatten()
        
        best_idx = -1
        best_similarity = -1.0
        
        # Iterate through search results and find the best match above threshold
        for i, (sim, idx) in enumerate(zip(similarities, indices)):
            if idx < len(person_ids) and sim > threshold:
                if sim > best_similarity:
                    best_similarity = sim
                    best_idx = idx
        
        if best_idx >= 0:
            return person_ids[best_idx], best_similarity
    
    # Fallback to pure numpy calculation if FAISS fails or for small sets (less efficient)
    # This part should ideally be rarely hit if FAISS is set up correctly
    if feature_matrix is not None:
        similarities = feature_matrix @ query_normalized
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        if best_similarity > threshold:
            return person_ids[best_idx], best_similarity
            
    return None, best_similarity # No match found or below threshold

def save_criminal_data(person_id, reid_feature, cropped_image, frame_count):
    """
    Saves the cropped image and re-identification feature of a detected criminal.
    """
    # Create directories if they don't exist
    os.makedirs("criminal_data", exist_ok=True)
    os.makedirs("criminal_data/images", exist_ok=True)
    os.makedirs("criminal_data/features", exist_ok=True)
    
    # Define file paths
    image_path = f"criminal_data/images/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}.jpg"
    feature_path = f"criminal_data/features/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}.npy"
    
    # Save image and feature
    cv2.imwrite(image_path, cropped_image)
    np.save(feature_path, reid_feature)
    
    print(f"💾 Saved criminal data: {image_path}, {feature_path}")
    return image_path, feature_path

# Main execution block
if __name__ == "__main__":
    # Load YOLOv8 pose model
    # IMPORTANT: Update this path to your yolov8n-pose.pt file
    model = YOLO(r"C:\Users\Omprakash\Desktop\pthon\yolov8n-pose.pt") 
    
    # Load video
    # IMPORTANT: Update this path to your video file
    video_path = r"C:\Users\Omprakash\Desktop\pthon\t1.mp4"
    cap = cv2.VideoCapture(video_path)

    # Initialize OpenCV window
    cv2.startWindowThread() # Required for some OpenCV installations to manage windows
    cv2.namedWindow("Persistent Threat Detection", cv2.WINDOW_NORMAL)

    total_threats = 0 # Keeps track of the maximum number of threats detected simultaneously
    active_threats = 0 # Number of threats in the current frame

    print("🚨 Starting criminal detection...")
    print("Press 'q' to quit, 'p' for performance summary (currently placeholder)")

    FRAME_SKIP = 2 # Process every Nth frame to improve performance

    while cap.isOpened():
        frame_start_time = time.time() # Start time for frame processing
        ret, frame = cap.read() # Read a frame
        if not ret:
            break # Exit if video ends

        update_frame() # Increment global frame counter
        
        process_this_frame = (frame_count % FRAME_SKIP == 0) # Check if we should process this frame

        if not detection_stopped and process_this_frame:
            yolo_start = time.time()
            results = model(frame, conf=0.3) # Run YOLO detection and pose estimation
            yolo_time = (time.time() - yolo_start) * 1000 # Time taken for YOLO
            
            annotated_frame = frame.copy() # Create a copy for drawing annotations
        else:
            annotated_frame = frame.copy()
            results = [] # No results if detection is stopped or frame is skipped
        
        current_frame_threats = 0 # Counter for threats in the current frame

        if not detection_stopped and process_this_frame:
            for r in results: # Iterate through YOLO results (each 'r' is a results object for a batch or single image)
                if r.keypoints is not None and r.boxes is not None:
                    # Iterate through each detected person's keypoints and bounding box
                    for j, (keypoints, box) in enumerate(zip(r.keypoints.data, r.boxes.xyxy)):
                        # Ensure data is on CPU and is a numpy array
                        if hasattr(keypoints, 'cpu'):
                            keypoints = keypoints.cpu().numpy()
                        if hasattr(box, 'cpu'):
                            box = box.cpu().numpy()

                        pose_start = time.time()
                        analysis = analyze_pose_threat(keypoints) # Analyze pose for threat
                        pose_time = (time.time() - pose_start) * 1000 # Time taken for pose analysis
                        
                        is_confirmed_threat = add_threat_detection(box, analysis['is_threat']) # Update threat history and confirm
                        person_id = _get_person_id(box) # Get ID for the current person

                        if person_id in confirmed_threats:
                            mark_confirmed_threat(annotated_frame, box, analysis['threat_details']) # Draw threat indicators
                            
                            crop_start = time.time()
                            cropped = crop_person(frame, box) # Crop the person's image
                            crop_time = (time.time() - crop_start) * 1000 # Time taken for cropping
                            
                            if cropped.size != 0: # Ensure the crop is not empty
                                reid_start = time.time()
                                reid_feature = extract_reid_feature(cropped, reid_model) # Extract Re-ID features
                                reid_time = (time.time() - reid_start) * 1000 # Time taken for Re-ID feature extraction
                                
                                reid_features[person_id] = reid_feature # Store the feature
                                
                                update_feature_matrix(reid_features) # Update FAISS index with new features
                                
                                save_start = time.time()
                                # Save criminal data in a separate thread to avoid blocking the main loop
                                Thread(target=save_criminal_data, args=(person_id, reid_feature, cropped, frame_count)).start()
                                # Save time is negligible due to threading here.
                                
                                detection_stopped = True # Stop detection after a criminal is identified and saved
                                print(f"🛑 Detection stopped after saving criminal {person_id}")
                                
                            current_frame_threats += 1 # Increment current frame threat count
                            
                            # Print threat details to console
                            left_angle = analysis['left_arm_angle'] if analysis['left_arm_angle'] is not None else "N/A"
                            right_angle = analysis['right_arm_angle'] if analysis['right_arm_angle'] is not None else "N/A"
                            print(f"🚨 PERSISTENT THREAT - Person {j+1}: "
                                  f"Left arm straight: {analysis['left_arm_straight']}, "
                                  f"Right arm straight: {analysis['right_arm_straight']}, "
                                  f"Left angle: {left_angle}, Right angle: {right_angle}")
        else:
            if detection_stopped:
                # Display message when detection is stopped
                cv2.putText(annotated_frame, "DETECTION STOPPED - Criminal Saved", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Criminals Saved: {len(reid_features)}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_time = (time.time() - frame_start_time) * 1000 # Total time to process the frame

        active_threats = current_frame_threats # Update active threats for display
        if current_frame_threats > 0:
            total_threats = max(total_threats, current_frame_threats) # Keep track of max threats

        # Display performance and status on the frame
        cv2.putText(annotated_frame, f"Active Threats: {active_threats}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(annotated_frame, f"Total Detected: {total_threats}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(annotated_frame, f"Frame: {frame_count}",
                    (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        current_fps = 1000 / frame_time if frame_time > 0 else 0
        cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(annotated_frame, f"Frame Time: {frame_time:.1f}ms", (10, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Persistent Threat Detection", annotated_frame) # Show the annotated frame
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"): # Quit
            break
        elif key == ord("p"):
            # Placeholder for performance summary (can be expanded to print detailed timings)
            print("\nPerformance summary requested:")
            # Example: print(f"YOLO Time: {yolo_time:.1f}ms, Pose Time: {pose_time:.1f}ms, Re-ID Time: {reid_time:.1f}ms")
            # Note: `yolo_time`, `pose_time`, `reid_time` are local to the loop and would need to be accumulated.

    # Release video capture and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

    # Final summary after the loop finishes
    print(f"\n=== Detection Summary ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Maximum simultaneous threats: {total_threats}")
    print(f"Confirmed threat IDs: {len(confirmed_threats)}")
    print(f"Criminal data saved: {len(reid_features)} persons")
    print(f"Data saved in: criminal_data/ directory")
