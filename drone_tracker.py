
from ultralytics import YOLO
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

cv2.setNumThreads(1)

reid_model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1041,
    loss='softmax',
    pretrained=False
)
weight_path = r'C:\Users\Omprakash\Desktop\pthon\osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth'
torchreid.utils.load_pretrained_weights(reid_model, weight_path)
reid_model.eval()
reid_model.cuda()

reid_transform = torchreid.data.transforms.build_transforms(
    height=256,
    width=128,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225]
)[0]

saved_criminal_features = {}
saved_criminal_images = {}
tracking_data = {}

UPDATE_FREQUENCY = 10
SIMILARITY_THRESHOLD = 0.60
UPDATE_THRESHOLD = 0.45
MAX_TRACKING_FRAMES = 70
MAX_IMAGES_PER_CRIMINAL = 7
MIN_SIMILARITY_FOR_NEW_IMAGE = 0.24

if 'profile' not in globals():
    def profile(func):
        return func

def load_saved_criminal_data():
    """Load all saved criminal features and images"""
    global saved_criminal_features, saved_criminal_images, tracking_data
    
    feature_files = glob.glob("criminal_data/features/*.npy")
    for feature_file in feature_files:
        filename = os.path.basename(feature_file)
        parts = filename.replace('.npy', '').split('_')
        if len(parts) >= 4:
            person_id = (int(parts[1]), int(parts[2]))
            feature = np.load(feature_file)
            feature = feature / np.linalg.norm(feature)
            saved_criminal_features[person_id] = feature
    
    image_files = glob.glob("criminal_data/images/*.jpg")
    for image_file in image_files:
        filename = os.path.basename(image_file)
        parts = filename.replace('.jpg', '').split('_')
        if len(parts) >= 4:
            person_id = (int(parts[1]), int(parts[2]))
            image = cv2.imread(image_file)
            saved_criminal_images[person_id] = image
    
    for person_id in saved_criminal_features.keys():
        tracking_data[person_id] = {
            'last_seen': 0,
            'frames_tracked': 0,
            'last_update_frame': 0,
            'stored_images': [saved_criminal_images.get(person_id)],
            'stored_features': [saved_criminal_features[person_id]],
            'tracking_history': [],
            'confidence_scores': []
        }
    
    print(f"Loaded {len(saved_criminal_features)} criminals for tracking")
    
    update_multi_feature_matrix(tracking_data)

def update_criminal_data(person_id, new_image, new_feature, frame_count, similarity):
    """Update stored criminal data with new tracking data - store up to 7 distinct images"""
    global saved_criminal_features, saved_criminal_images, tracking_data
    
    if person_id not in tracking_data:
        tracking_data[person_id] = {
            'last_seen': frame_count,
            'frames_tracked': 1,
            'last_update_frame': frame_count,
            'stored_images': [saved_criminal_images.get(person_id)],
            'stored_features': [saved_criminal_features[person_id]],
            'tracking_history': [(frame_count, similarity)],
            'confidence_scores': [similarity]
        }
    else:
        tracking_data[person_id]['last_seen'] = frame_count
        tracking_data[person_id]['frames_tracked'] += 1
        tracking_data[person_id]['tracking_history'].append((frame_count, similarity))
        tracking_data[person_id]['confidence_scores'].append(similarity)
        
        if len(tracking_data[person_id]['tracking_history']) > MAX_TRACKING_FRAMES:
            tracking_data[person_id]['tracking_history'] = tracking_data[person_id]['tracking_history'][-MAX_TRACKING_FRAMES:]
            tracking_data[person_id]['confidence_scores'] = tracking_data[person_id]['confidence_scores'][-MAX_TRACKING_FRAMES:]
        
        should_add_image = False
        
        if len(tracking_data[person_id]['stored_images']) < MAX_IMAGES_PER_CRIMINAL:
            should_add_image = True
        else:
            if len(tracking_data[person_id]['stored_features']) > 0:
                existing_features = np.array(tracking_data[person_id]['stored_features'])
                similarities = cosine_similarity_vectorized(new_feature, existing_features)
                max_similarity_to_existing = np.max(similarities)
            else:
                max_similarity_to_existing = -1
            
            if max_similarity_to_existing < MIN_SIMILARITY_FOR_NEW_IMAGE:
                should_add_image = True
                print(f"🆕 Distinct image detected (similarity: {max_similarity_to_existing:.3f})")
        
        if (should_add_image and 
            frame_count - tracking_data[person_id]['last_update_frame'] >= UPDATE_FREQUENCY):
            
            tracking_data[person_id]['stored_images'].append(new_image)
            tracking_data[person_id]['stored_features'].append(new_feature)
            tracking_data[person_id]['last_update_frame'] = frame_count
            
            if len(tracking_data[person_id]['stored_images']) > MAX_IMAGES_PER_CRIMINAL:
                tracking_data[person_id]['stored_images'] = tracking_data[person_id]['stored_images'][-MAX_IMAGES_PER_CRIMINAL:]
                tracking_data[person_id]['stored_features'] = tracking_data[person_id]['stored_features'][-MAX_IMAGES_PER_CRIMINAL:]
            
            save_updated_criminal_data(person_id, new_feature, new_image, frame_count, len(tracking_data[person_id]['stored_images']))
            
            update_multi_feature_matrix(tracking_data)
            
            print(f"Added image {len(tracking_data[person_id]['stored_images'])}/7 for criminal {person_id} (similarity: {similarity:.3f})")

def save_updated_criminal_data(person_id, feature, image, frame_count, image_index):
    """Save updated criminal data to disk"""
    os.makedirs("criminal_data", exist_ok=True)
    os.makedirs("criminal_data/images", exist_ok=True)
    os.makedirs("criminal_data/features", exist_ok=True)
    
    image_path = f"criminal_data/images/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}_updated_{image_index}.jpg"
    cv2.imwrite(image_path, image)
    
    feature_path = f"criminal_data/features/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}_updated_{image_index}.npy"
    np.save(feature_path, feature)
    
    metadata_path = f"criminal_data/tracking_metadata_{person_id[0]}_{person_id[1]}.json"
    metadata = {
        'person_id': person_id,
        'last_update_frame': frame_count,
        'frames_tracked': tracking_data[person_id]['frames_tracked'],
        'images_stored': len(tracking_data[person_id]['stored_images']),
        'average_confidence': np.mean(tracking_data[person_id]['confidence_scores']),
        'max_confidence': np.max(tracking_data[person_id]['confidence_scores']),
        'tracking_history_length': len(tracking_data[person_id]['tracking_history'])
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

import torch.profiler

def extract_reid_feature(image_bgr, model):
    """Extract ReID features from image - Optimized for speed"""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image_rgb)
    
    image = reid_transform(pil_image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        features = model(image)
    
    features_np = features.cpu().numpy().flatten()
    features_np = features_np / np.linalg.norm(features_np)
    return features_np

import faiss

all_features_matrix = None
all_person_ids = []
all_image_indices = []
faiss_index = None

def initialize_faiss_index_multi():
    """Initialize FAISS index for multi-image similarity search"""
    global faiss_index
    dimension = 512
    faiss_index = faiss.IndexFlatIP(dimension)
    
    if torch.cuda.is_available():
        try:
            if hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
                print("Using FAISS GPU acceleration for multi-image matching")
            else:
                print("FAISS GPU not available, using CPU version for multi-image matching")
        except Exception as e:
            print(f"FAISS GPU initialization failed: {e}, using CPU version for multi-image matching")
    
    return faiss_index

def update_multi_feature_matrix(tracking_data):
    """Update feature matrix for multi-image matching"""
    global all_features_matrix, all_person_ids, all_image_indices, faiss_index
    
    if not tracking_data:
        all_features_matrix = None
        all_person_ids = []
        all_image_indices = []
        return
    
    features_list = []
    person_ids_list = []
    image_indices_list = []
    
    for person_id, data in tracking_data.items():
        for i, feature in enumerate(data['stored_features']):
            features_list.append(feature)
            person_ids_list.append(person_id)
            image_indices_list.append(i)
    
    if not features_list:
        all_features_matrix = None
        all_person_ids = []
        all_image_indices = []
        return
    
    all_features_matrix = np.array(features_list)
    all_person_ids = person_ids_list
    all_image_indices = image_indices_list
    
    all_features_matrix = all_features_matrix / np.linalg.norm(all_features_matrix, axis=1, keepdims=True)
    
    if faiss_index is None:
        faiss_index = initialize_faiss_index_multi()
    else:
        faiss_index.reset()
    
    faiss_index.add(all_features_matrix.astype(np.float32))

def cosine_similarity_vectorized(query_feature, features_matrix):
    """Calculate cosine similarity using vectorized operations"""
    query_normalized = query_feature / np.linalg.norm(query_feature)
    similarities = features_matrix @ query_normalized
    return similarities

def find_best_match_multi_vectorized(query_feature, tracking_data, threshold=0.60):
    global all_features_matrix, all_person_ids, all_image_indices, faiss_index
    if not tracking_data:
        return None, -1, -1
    current_total_features = sum(len(data['stored_features']) for data in tracking_data.values())
    if (all_features_matrix is None or 
        len(all_person_ids) != current_total_features):
        features_list = []
        person_ids_list = []
        image_indices_list = []
        for person_id, data in tracking_data.items():
            for i, feature in enumerate(data['stored_features']):
                features_list.append(feature)
                person_ids_list.append(person_id)
                image_indices_list.append(i)
        if not features_list:
            all_features_matrix = None
            all_person_ids = []
            all_image_indices = []
            return None, -1, -1
        all_features_matrix = np.stack(features_list)
        all_person_ids = person_ids_list
        all_image_indices = image_indices_list
        all_features_matrix = all_features_matrix / np.linalg.norm(all_features_matrix, axis=1, keepdims=True)
        if faiss_index is not None:
            faiss_index.reset()
            faiss_index.add(all_features_matrix.astype(np.float32))
    if all_features_matrix is None or len(all_features_matrix) == 0:
        return None, -1, -1
    query_normalized = query_feature / np.linalg.norm(query_feature)
    if faiss_index is not None and faiss_index.ntotal > 0:
        k = min(len(all_features_matrix), 10)
        similarities, indices = faiss_index.search(
            query_normalized.reshape(1, -1).astype(np.float32), 
            k
        )
        similarities = similarities.flatten()
        indices = indices.flatten()
        best_idx = -1
        best_similarity = -1
        for sim, idx in zip(similarities, indices):
            if idx < len(all_person_ids) and sim >= threshold:
                if sim > best_similarity:
                    best_similarity = sim
                    best_idx = idx
        if best_idx >= 0:
            best_person_id = all_person_ids[best_idx]
            best_image_index = all_image_indices[best_idx]
            return best_person_id, best_similarity, best_image_index
    similarities = all_features_matrix @ query_normalized
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    if best_similarity >= threshold:
        best_person_id = all_person_ids[best_idx]
        best_image_index = all_image_indices[best_idx]
        print(f"  Best match: {best_person_id} (Image {best_image_index+1}), Best similarity: {best_similarity:.3f}")
        return best_person_id, best_similarity, best_image_index
    return None, best_similarity, -1

def mark_criminal_found(frame, bbox, person_id, similarity, display_image=None, image_index=0):
    """Mark when a criminal is found in drone footage - ALWAYS RED"""
    x1, y1, x2, y2 = map(int, bbox)
    
    color = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    
    label = f"CRIMINAL FOUND - ID: {person_id} (Image {image_index+1})"
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    similarity_text = f"Match: {similarity:.3f}"
    cv2.putText(frame, similarity_text, (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    if person_id in tracking_data:
        tracking_info = tracking_data[person_id]
        frames_tracked = tracking_info['frames_tracked']
        images_stored = len(tracking_info['stored_images'])
        avg_confidence = np.mean(tracking_info['confidence_scores']) if tracking_info['confidence_scores'] else 0
        tracking_text = f"Tracked: {frames_tracked} frames, Images: {images_stored}/7, Avg: {avg_confidence:.3f}"
        cv2.putText(frame, tracking_text, (x1, y1-50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if display_image is not None:
        h, w = display_image.shape[:2]
        display_h = 100
        display_w = int(w * display_h / h)
        display_img = cv2.resize(display_image, (display_w, display_h))
        
        frame[10:10+display_h, frame.shape[1]-display_w-10:frame.shape[1]-10] = display_img
        
        cv2.rectangle(frame, (frame.shape[1]-display_w-12, 8), (frame.shape[1]-8, 10+display_h+2), (0, 255, 0), 2)
        cv2.putText(frame, f"Stored Image {image_index+1}", (frame.shape[1]-display_w-10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

yolo_model = YOLO("yolov8s.pt")

drone_video_path = r".mp4"
    
cap = cv2.VideoCapture(drone_video_path)
    
if not cap.isOpened():
    print(f"Could not open video: {drone_video_path}")
    pass
    
cv2.startWindowThread()
cv2.namedWindow("Drone Criminal Tracker", cv2.WINDOW_NORMAL)
    
frame_count = 0
criminals_found = 0
tracking_history = defaultdict(list)
    
print("Starting real-time drone criminal tracking with MULTI-IMAGE MATCHING...")
print(f"Using similarity threshold: {SIMILARITY_THRESHOLD}")
print(f"Update threshold: {UPDATE_THRESHOLD}")
print(f"Update frequency: every {UPDATE_FREQUENCY} frames")
print(f"Max images per criminal: {MAX_IMAGES_PER_CRIMINAL}")
print(f" Min similarity for new image: {MIN_SIMILARITY_FOR_NEW_IMAGE}")
print(f"Video path: {drone_video_path}")
print(f"Processing every 3rd frame")
print("Press 'q' to quit, 't' to adjust threshold, 'p' for performance summary")
    
while cap.isOpened():
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    
    if frame_count % 2 != 0:
        continue
    
    yolo_start = time.time()
    results = yolo_model(frame, conf=0.4, classes=[0])
    yolo_time = (time.time() - yolo_start) * 1000
    
    current_frame_criminals = 0
    people_detected = 0
    
    for r in results:
        if r.boxes is not None:
            boxes = r.boxes.xyxy
            if hasattr(boxes, 'cpu'):
                boxes = boxes.cpu().numpy()
            
            people_detected = len(boxes)
            
            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                
                if x2 - x1 < 30 or y2 - y1 < 50:
                    continue
                
                person_crop = frame[y1:y2, x1:x2]
                
                if person_crop.size == 0:
                    continue
                
                reid_start = time.time()
                try:
                    reid_feature = extract_reid_feature(person_crop, reid_model)
                    reid_time = (time.time() - reid_start) * 1000
                    
                    matching_start = time.time()
                    best_match, similarity, best_image_index = find_best_match_multi_vectorized(reid_feature, tracking_data, threshold=SIMILARITY_THRESHOLD)
                    matching_time = (time.time() - matching_start) * 1000
                    
                    if best_match:
                        display_image = tracking_data[best_match]['stored_images'][best_image_index]
                        mark_criminal_found(frame, box, best_match, similarity, display_image, best_image_index)
                        
                        update_criminal_data(best_match, person_crop, reid_feature, frame_count, similarity)
                        
                        current_frame_criminals += 1
                        criminals_found = max(criminals_found, current_frame_criminals)
                        
                        tracking_history[best_match].append((frame_count, similarity))
                        
                        print(f"🚨 CRIMINAL SPOTTED! ID: {best_match}, Similarity: {similarity:.3f}, Frame: {frame_count}")
                        
                except Exception as e:
                    print(f"Error processing person {j}: {e}")
                    continue
        
    frame_time = (time.time() - frame_start_time) * 1000
    
    cv2.putText(frame, f"People: {people_detected}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Criminals: {current_frame_criminals}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Threshold: {SIMILARITY_THRESHOLD}", (10, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f"Multi-Image: 7 per criminal", (10, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"Frame: {frame_count}", (10, 140), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    current_fps = 1000 / frame_time if frame_time > 0 else 0
    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 165), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv2.putText(frame, f"Frame Time: {frame_time:.1f}ms", (10, 185), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    cv2.rectangle(frame, (frame.shape[1]-200, 10), (frame.shape[1]-10, 100), (0, 0, 0), -1)
    cv2.putText(frame, "Red = Criminal (All Images)", (frame.shape[1]-190, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(frame, "Green = Normal Person", (frame.shape[1]-190, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, "Multi-Image: 7 per criminal", (frame.shape[1]-190, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    cv2.putText(frame, "Press 'q' to quit", (frame.shape[1]-190, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    cv2.imshow("Drone Criminal Tracker", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("t"):
        print(f"Current threshold: {SIMILARITY_THRESHOLD}")
        new_threshold = input("Enter new threshold (0.1-0.9): ")
        try:
            SIMILARITY_THRESHOLD = float(new_threshold)
            print(f"Threshold changed to: {SIMILARITY_THRESHOLD}")
        except:
            print("Invalid threshold, keeping current value")
    elif key == ord("p"):
        pass
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== Drone Tracking Summary (MULTI-IMAGE MATCHING) ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Maximum criminals found in single frame: {criminals_found}")
    print(f"Unique criminals tracked: {len(tracking_history)}")
    print(f"Final threshold used: {SIMILARITY_THRESHOLD}")
    
    for person_id, history in tracking_history.items():
        max_similarity = max([sim for _, sim in history])
        frames_detected = len(history)
        if person_id in tracking_data:
            avg_confidence = np.mean(tracking_data[person_id]['confidence_scores'])
            print(f"Criminal {person_id}: Detected in {frames_detected} frames, Max similarity: {max_similarity:.3f}, Avg confidence: {avg_confidence:.3f}")

if __name__ == "__main__":
    pass
