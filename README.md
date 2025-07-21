# Real-Time Criminal Detection & Drone Tracking System 🚨🛩️

## Why
In some gun-related crimes that occur in public spaces, the attacker commits the act and disappears from the scene within seconds — and sometimes, they’re never caught.

I started thinking — what if we had a system that could react just as fast?

A system that could detect the crime in real time and immediately launch a drone within seconds — tracking the suspect live and continuously streaming their location to law enforcement, giving them the upper hand when every second counts.

My goal is a system where fixed surveillance cameras detect the threat, and within seconds, a drone launches autonomously to track the criminal while continuously updating their live location to officers.

---

## How 🛠️

- **Ground Camera Detection:**
  - Uses YOLOv8 pose estimation to detect threatening actions (e.g., gun aiming) in real time.
  - Confirms threats over multiple frames for reliability.
  - Crops and saves high-quality images of the suspect when a threat is confirmed.

- **Criminal Profiling & Re-Identification:**
  - Extracts robust feature vectors from cropped images using OSNet (torchreid).
  - Stores these as a digital fingerprint for the suspect.

- **Dynamic, Multi-Image Learning:**
  - Continuously updates the criminal’s profile with up to 7 distinct images as the drone tracks the suspect.
  - Adapts to changes in angle, lighting, and appearance for robust re-identification.

- **Aerial Drone Tracking:**
  - The drone-side script loads the suspect’s features and matches people in the drone’s video feed using fast, vectorized similarity search (NumPy and FAISS).
  - Continuously updates the profile with new images for improved tracking.

- **Performance & Optimization:**
  - Frame skipping for higher FPS and smoother video.
  - All heavy computations (feature matching, pose analysis) are fully vectorized.
  - Blocking disk I/O (saving images/features) is offloaded to background threads.
  - GPU-accelerated where possible, with CPU fallback.
  - Minimal GUI overhead for maximum speed.
  - OpenCV thread control (`cv2.setNumThreads(1)`).

---

## What 🚀

- **Real-Time Threat Detection:**
  - Detects and confirms threatening behavior (like gun aiming) in live video.

- **Seamless Ground-to-Drone Handover:**
  - As soon as a threat is confirmed, the drone can immediately begin tracking using the latest, most accurate profile.

- **Dynamic, Continually-Learning Suspect Profile:**
  - Stores and updates up to 7 distinct images/features per criminal for robust re-identification.

- **Efficient, Scalable Matching:**
  - Uses FAISS and vectorized NumPy for fast, scalable matching—capable of handling large crowds and multiple suspects in real time.

- **Threaded I/O and Frame Skipping:**
  - Ensures high FPS and smooth operation by offloading slow operations and only processing a subset of frames.

- **Modular, Extensible Design:**
  - Ready for further expansion—such as integrating drone launch logic, live location streaming, or more advanced drone mobility control.

- **Actively Developed:**
  - Still learning and building, especially on the drone-side: next steps include automating drone launch, improving aerial suspect re-identification, and adding real-time mobility control and live location streaming for law enforcement.

---

## Quick Start ⚡

### Requirements
- Python 3.8+
- torch, torchvision, torchreid
- ultralytics (YOLOv8)
- opencv-python
- faiss (CPU or GPU)
- numpy, pillow

Install dependencies (example):
```bash
pip install torch torchvision opencv-python ultralytics faiss-cpu numpy pillow
# For torchreid, follow: https://github.com/KaiyangZhou/deep-person-reid
```

### Setup
1. Place your YOLOv8 and OSNet weights in the project directory.
2. Prepare your input videos (ground and drone footage).
3. Run the ground detector:
   ```bash
   python detector.py
   ```
   - This will save cropped images and features of confirmed threats to `criminal_data/`.
4. Run the drone tracker:
   ```bash
   python drone_tracker.py
   ```
   - This will load the saved criminal data and perform real-time tracking in the drone video.

---

## Example Output 🖼️
- Red bounding boxes highlight confirmed threats in both ground and drone views.
- The system updates the criminal’s profile with new images as tracking continues.
- All data is saved in the `criminal_data/` directory.

---

## Acknowledgements 🙏
- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics)
- [torchreid (OSNet)](https://github.com/KaiyangZhou/deep-person-reid)
- [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)

---

## Contact & Contributing
- Open to feedback, suggestions, and collaboration!
- Feel free to open issues or pull requests.
- Connect with me on [LinkedIn](https://www.linkedin.com/)

---

> **Note:** This project is for research and demonstration purposes. Always comply with local laws and regulations regarding surveillance and privacy. 