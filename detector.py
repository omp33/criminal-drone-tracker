
from ultralytics import YOLO
import cv2
import numpy as np
import math
from collections import defaultdict
import time
import torch
import torchreid
from PIL import Image
from threading import Thread

cv2.setNumThreads(1)

threat_history = defaultdict(list)
persistence_frames = 7
min_confidence = 0.45
frame_count = 0
confirmed_threats = set()
reid_features = {}
detection_stopped = False

def update_frame():
    global frame_count
    frame_count += 1

def _get_person_id(bbox):
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return (round(center_x / 50) * 50, round(center_y / 50) * 50)

def add_threat_detection(person_bbox, is_threat):
    global threat_history, confirmed_threats, frame_count
    person_id = _get_person_id(person_bbox)
    current_time = frame_count
    threat_history[person_id].append((current_time, is_threat))
    threat_history[person_id] = [
        (t, threat) for t, threat in threat_history[person_id]
        if current_time - t <= persistence_frames * 2
    ]
    recent_threats = [
        threat for t, threat in threat_history[person_id]
        if current_time - t <= persistence_frames and threat
    ]
    if len(recent_threats) >= persistence_frames // 2:
        confirmed_threats.add(person_id)
        return True
    else:
        confirmed_threats.discard(person_id)
        return False

from numba import jit, njit, prange
import math

if 'profile' not in globals():
    def profile(func):
        return func

@njit(fastmath=True, cache=True)
def calculate_angle_numba(x1, y1, x2, y2, x3, y3):
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
    
    if cos_angle > 1.0:
        cos_angle = 1.0
    elif cos_angle < -1.0:
        cos_angle = -1.0
    
    angle = math.degrees(math.acos(cos_angle))
    return angle

@njit(fastmath=True, cache=True)
def is_arm_straight_numba(x1, y1, x2, y2, x3, y3, threshold=40.0):
    angle = calculate_angle_numba(x1, y1, x2, y2, x3, y3)
    return abs(180.0 - angle) <= threshold

@njit(fastmath=True, cache=True)
def is_forward_aiming_numba(shoulder_x, shoulder_y, wrist_x, wrist_y, body_center_x, body_center_y, max_y_diff=51.0):
    dx = wrist_x - body_center_x
    dy = abs(wrist_y - shoulder_y)
    return abs(dx) > 50.0 and dy < max_y_diff

def calculate_angle(point1, point2, point3):
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
    if None in [shoulder, elbow, wrist]:
        return False
    angle = calculate_angle(shoulder, elbow, wrist)
    if angle is None:
        return False
    return abs(180 - angle) <= threshold

def get_arm_angle_from_torso(shoulder, elbow, hip):
    if None in [shoulder, elbow, hip]:
        return None
    reference_point = (hip[0], hip[1] - 100)
    angle = calculate_angle(reference_point, shoulder, elbow)
    return angle

def analyze_pose_threat(keypoints_batch, confidence_threshold=0.5):
    keypoints_batch = np.asarray(keypoints_batch)
    if keypoints_batch.ndim == 2:
        keypoints_batch = keypoints_batch[None, ...]
    N = keypoints_batch.shape[0]
    idxs = {
        'left_shoulder': 5,
        'right_shoulder': 6,
        'left_elbow': 7,
        'right_elbow': 8,
        'left_wrist': 9,
        'right_wrist': 10,
        'left_hip': 11,
        'right_hip': 12
    }
    def get_xy(idx):
        conf = keypoints_batch[:, idx, 2]
        mask = conf > confidence_threshold
        xy = keypoints_batch[:, idx, :2]
        xy[~mask] = np.nan
        return xy
    left_shoulder = get_xy(idxs['left_shoulder'])
    right_shoulder = get_xy(idxs['right_shoulder'])
    left_elbow = get_xy(idxs['left_elbow'])
    right_elbow = get_xy(idxs['right_elbow'])
    left_wrist = get_xy(idxs['left_wrist'])
    right_wrist = get_xy(idxs['right_wrist'])
    left_hip = get_xy(idxs['left_hip'])
    right_hip = get_xy(idxs['right_hip'])
    torso_center = np.nanmean(np.stack([left_hip, right_hip], axis=1), axis=1)
    def arm_straight(shoulder, elbow, wrist, threshold=40):
        v1 = shoulder - elbow
        v2 = wrist - elbow
        dot = np.nansum(v1 * v2, axis=1)
        mag1 = np.linalg.norm(v1, axis=1)
        mag2 = np.linalg.norm(v2, axis=1)
        cos_angle = dot / (mag1 * mag2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_angle))
        straight = np.abs(180.0 - angles) <= threshold
        straight[np.isnan(angles)] = False
        return straight, angles
    left_arm_straight, left_arm_angle = arm_straight(left_shoulder, left_elbow, left_wrist)
    right_arm_straight, right_arm_angle = arm_straight(right_shoulder, right_elbow, right_wrist)
    def is_forward_aiming(shoulder, wrist, torso, max_y_diff=51.0):
        dx = wrist[:, 0] - torso[:, 0]
        dy = np.abs(wrist[:, 1] - shoulder[:, 1])
        aiming = (np.abs(dx) > 50.0) & (dy < max_y_diff)
        aiming[np.isnan(dx) | np.isnan(dy)] = False
        return aiming
    left_forward = is_forward_aiming(left_shoulder, left_wrist, torso_center)
    right_forward = is_forward_aiming(right_shoulder, right_wrist, torso_center)
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
    if results and N == 1:
        return results[0]
    return results

def mark_confirmed_threat(frame, bbox, threat_details, confidence_level="HIGH"):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    if int(time.time() * 3) % 2:
        cv2.rectangle(frame, (x1-5, y1-5), (x2+5, y2+5), (0, 0, 255), 2)
    label_y = y1 - 15
    cv2.putText(frame, "🚨 CONFIRMED THREAT", (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    for i, detail in enumerate(threat_details):
        detail_y = y1 - 35 - (i * 20)
        cv2.putText(frame, detail, (x1, detail_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

import torch.profiler

def crop_person(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    crop = frame[y1:y2, x1:x2]
    
    if crop.size == 0:
        return crop
    
    return crop

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

def extract_reid_feature(image_bgr, model):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    image = reid_transform(pil_image).unsqueeze(0).cuda()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            features = model(image)
    prof.export_chrome_trace("reid_inference_trace.json")
    features_np = features.cpu().numpy().flatten()
    features_np = features_np / np.linalg.norm(features_np)
    return features_np

import faiss
import numpy as np

feature_matrix = None
person_ids = []
feature_index = None

def initialize_faiss_index():
    global feature_index
    dimension = 512
    feature_index = faiss.IndexFlatIP(dimension)
    
    if torch.cuda.is_available():
        try:
            if hasattr(faiss, 'StandardGpuResources'):
                res = faiss.StandardGpuResources()
                feature_index = faiss.index_cpu_to_gpu(res, 0, feature_index)
                print("🚀 Using FAISS GPU acceleration")
            else:
                print("⚠️ FAISS GPU not available, using CPU version")
        except Exception as e:
            print(f"⚠️ FAISS GPU initialization failed: {e}, using CPU version")
    
    return feature_index

def update_feature_matrix(stored_features):
    global feature_matrix, person_ids, feature_index
    
    if not stored_features:
        feature_matrix = None
        person_ids = []
        return
    
    person_ids = list(stored_features.keys())
    feature_matrix = np.array([stored_features[pid] for pid in person_ids])
    
    feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)
    
    if feature_index is None:
        feature_index = initialize_faiss_index()
    else:
        feature_index.reset()
    
    feature_index.add(feature_matrix.astype(np.float32))

def cosine_similarity_vectorized(query_feature, stored_features_matrix):
    query_normalized = query_feature / np.linalg.norm(query_feature)
    similarities = stored_features_matrix @ query_normalized
    return similarities


def find_best_match_vectorized(query_feature, stored_features, threshold=0.38):
    global feature_matrix, person_ids, feature_index
    
    if not stored_features:
        return None, -1
    
    if feature_matrix is None or len(stored_features) != len(person_ids):
        person_ids = list(stored_features.keys())
        feature_matrix = np.stack(list(stored_features.values()))
        feature_matrix = feature_matrix / np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        if feature_index is not None:
            feature_index.reset()
            feature_index.add(feature_matrix.astype(np.float32))
    
    query_normalized = query_feature / np.linalg.norm(query_feature)
    
    if feature_index is not None and feature_index.ntotal > 0:
        similarities, indices = feature_index.search(
            query_normalized.reshape(1, -1).astype(np.float32), 
            min(len(person_ids), 10)
        )
        similarities = similarities.flatten()
        indices = indices.flatten()
        best_idx = -1
        best_similarity = -1
        for i, (sim, idx) in enumerate(zip(similarities, indices)):
            if idx < len(person_ids) and sim > threshold:
                if sim > best_similarity:
                    best_similarity = sim
                    best_idx = idx
        if best_idx >= 0:
            return person_ids[best_idx], best_similarity
    
    similarities = feature_matrix @ query_normalized
    best_idx = np.argmax(similarities)
    best_similarity = similarities[best_idx]
    if best_similarity > threshold:
        return person_ids[best_idx], best_similarity
    return None, best_similarity

def save_criminal_data(person_id, reid_feature, cropped_image, frame_count):
    import os
    
    os.makedirs("criminal_data", exist_ok=True)
    os.makedirs("criminal_data/images", exist_ok=True)
    os.makedirs("criminal_data/features", exist_ok=True)
    
    image_path = f"criminal_data/images/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}.jpg"
    cv2.imwrite(image_path, cropped_image)
    
    feature_path = f"criminal_data/features/criminal_{person_id[0]}_{person_id[1]}_frame_{frame_count}.npy"
    np.save(feature_path, reid_feature)
    
    print(f"💾 Saved criminal data: {image_path}, {feature_path}")
    return image_path, feature_path

model = YOLO(r"C:\Users\Omprakash\Desktop\pthon\yolov8n-pose.pt")
video_path = r"C:\Users\Omprakash\Desktop\pthon\t1.mp4"
cap = cv2.VideoCapture(video_path)

cv2.startWindowThread()
cv2.namedWindow("Persistent Threat Detection", cv2.WINDOW_NORMAL)

total_threats = 0
active_threats = 0

print("🚨 Starting criminal detection...")
print("Press 'q' to quit, 'p' for performance summary")

FRAME_SKIP = 2

while cap.isOpened():
    frame_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    update_frame()
    
    process_this_frame = (frame_count % FRAME_SKIP == 0)

    if not detection_stopped and process_this_frame:
        yolo_start = time.time()
        results = model(frame, conf=0.3)
        yolo_time = (time.time() - yolo_start) * 1000
        
        annotated_frame = frame.copy()
    else:
        annotated_frame = frame.copy()
        results = []
    
    current_frame_threats = 0

    if not detection_stopped and process_this_frame:
        for r in results:
            if r.keypoints is not None and r.boxes is not None:
                for j, (keypoints, box) in enumerate(zip(r.keypoints.data, r.boxes.xyxy)):
                    if hasattr(keypoints, 'cpu'):
                        keypoints = keypoints.cpu().numpy()
                    if hasattr(box, 'cpu'):
                        box = box.cpu().numpy()

                    pose_start = time.time()
                    analysis = analyze_pose_threat(keypoints)
                    pose_time = (time.time() - pose_start) * 1000
                    
                    is_confirmed_threat = add_threat_detection(box, analysis['is_threat'])
                    person_id = _get_person_id(box)

                    if person_id in confirmed_threats:
                        mark_confirmed_threat(annotated_frame, box, analysis['threat_details'])
                        
                        crop_start = time.time()
                        cropped = crop_person(frame, box)
                        crop_time = (time.time() - crop_start) * 1000
                        
                        if cropped.size != 0:
                            reid_start = time.time()
                            reid_feature = extract_reid_feature(cropped, reid_model)
                            reid_time = (time.time() - reid_start) * 1000
                            
                            reid_features[person_id] = reid_feature
                            
                            update_feature_matrix(reid_features)
                            
                            save_start = time.time()
                            Thread(target=save_criminal_data, args=(person_id, reid_feature, cropped, frame_count)).start()
                            
                            detection_stopped = True
                            print(f"🛑 Detection stopped after saving criminal {person_id}")
                            
                        current_frame_threats += 1
                        left_angle = analysis['left_arm_angle'] if analysis['left_arm_angle'] is not None else "N/A"
                        right_angle = analysis['right_arm_angle'] if analysis['right_arm_angle'] is not None else "N/A"
                        print(f"🚨 PERSISTENT THREAT - Person {j+1}: "
                              f"Left arm straight: {analysis['left_arm_straight']}, "
                              f"Right arm straight: {analysis['right_arm_straight']}, "
                              f"Left angle: {left_angle}, Right angle: {right_angle}")
    else:
        if detection_stopped:
            cv2.putText(annotated_frame, "DETECTION STOPPED - Criminal Saved", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Criminals Saved: {len(reid_features)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    frame_time = (time.time() - frame_start_time) * 1000

    active_threats = current_frame_threats
    if current_frame_threats > 0:
        total_threats = max(total_threats, current_frame_threats)

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

    cv2.imshow("Persistent Threat Detection", annotated_frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("p"):
        pass

cap.release()
cv2.destroyAllWindows()

print(f"\n=== Detection Summary ===")
print(f"Total frames processed: {frame_count}")
print(f"Maximum simultaneous threats: {total_threats}")
print(f"Confirmed threat IDs: {len(confirmed_threats)}")
print(f"Criminal data saved: {len(reid_features)} persons")
print(f"Data saved in: criminal_data/ directory")

if __name__ == "__main__":
    pass