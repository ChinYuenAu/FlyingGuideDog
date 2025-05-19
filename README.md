# 👨‍💻Indoor Navigation Drone with Real-Time Person Tracking and Obstacle Avoidance  🤖

**Author:** Au, Chin Yuen (Isaac)

## 📌 Project Overview
Project Goal: This project focuses on the core real-time navigation system for a future **“Flying Guide Dog”**—an “over‑the‑shoulder” assistive companion drone designed to help visually impaired individuals navigate complex indoor environments safely and autonomously, thus enhancing indoor mobility. While the broader “Flying Guide Dog” initiative envisions features like human-drone interaction, voice feedback, and semantic scene understanding, this project provides the **foundational autonomous navigation engine** that makes those higher-level features possible. <br/> Conventional drone systems encounter significant challenges in indoor environments, primarily due to the **absence of GPS signals**, unreliable obstacle detection under **varying lighting conditions**, and unstable target tracking in dynamic scenes. Notably, this system is designed to run entirely on a **DJI Tello EDU** — a **lightweight**, **low-cost** drone with **limited onboard sensing and only one forward-facing camera**. Overcoming these constraints required rethinking robotics pipelines and optimizing real-time computer vision to work under tight hardware limitations.  <br/>


This work delivers:
- **Reliable person-following** via ArUco marker tracking  
- **Monocular obstacle avoidance** powered by MiDaS depth estimation  
- **Smooth PD control** for stable and responsive flight behavior

| DJI Tello EDU                                 | ArUco marker                                      |
|---------------------------------------------- |---------------------------------------------------|
|<img src="assets/DJI_Tello_EDU.JPG" width="200"> |<img src="assets/ArUco_6x6_1000-50.svg" width="200"> |

## 🚀 Features

| Component                        | Description |
|----------------------------------|-------------|
| 🎯 **ArUco Marker Tracking**     | Primary tracking system using marker pose (position + yaw angle) |
| 🛑 **Obstacle Avoidance**        | MiDaS depth estimation + spatial logic for halting or rerouting |
| 📡 **PD Control**                | Smooth real-time drone control across yaw, up/down, forward/backward, and lateral axes |
| 🔄 **Dynamic Turn Handling**     | Adjusts drone behavior based on yaw angle changes (turns) to minimize corner collisions |
| 🧠 **State-Aware Logic**         | Transitions between ArUco tracking mode and obstacle avoidance override |
| 🖥️ **Live Video + Debug UI**     | Real-time annotations: marker box, yaw angle, fallback IDs, obstacle warnings, depth map |

## 📂 Repository Structure

```
.
├── assets/                                   # Example outputs, evaluation images, logs
├── src/
|   ├──droneMain.py                           # Unified control script
├── README.md
├── requirements.txt
```

## 🧠 Methodology

### 1. **ArUco Marker Detection**
- Marker detector: `cv2.aruco.ArucoDetector.detectMarkers`
- Pose estimation: `cv2.solvePnP`
- Extracts marker center, distance, and yaw angle from `rvec` + `tvec`
- Used to derive control commands:
  - `yaw` → align heading with marker angle
  - `lr`  → lateral correction
  - `fb`  → forward-backward distance control
  - `ud`  → altitude stabilization
- Maintains smooth control using visual tracking and position memory

### 2. **Obstacle Avoidance**
- Uses MiDaS to generate depth maps
- Assume obstacle apperance on sides as this project focuses on target tracking
- If mean depth in Region of Interest(ROI) < threshold, halts or redirects
- Obstacle override has highest priority

### 3. **PD Controller with Smoothing**
- Stabilizes drone motion via proportional-derivative control
- Smooths jitter using:
  - Low-pass filtering
  - Deadbands and clipping
  - Turn-based dynamic acceleration logic

## 🧪 Evaluation

| Metric                             | Value / Notes |
|-----------------------------------|----------------|
| ArUco Tracking Latency            | ~7 ± 1 ms                              |
| End-to-End Avoidance Success      | 90% of 30 indoor corner runs           |
| Turn Responsiveness               | Drone reduces radius and accelerates during sharp marker yaw changes |
| Stability                         | Controlled oscillation with PID smoothing and min-speed gating       |

## 🧪 Example Output

| Drone Camera View <br/> BGR Frame(Left), Depth Map (Right)  | Third person point of view |
|-------------------------------------------------------- |--------------------------- |
| <img src="assets/TrackingScreenshot.png" width="500">     | <img src="assets/RouteView_RoomExitAndNavigation.png" width="500"> | 
| <img src="assets/TrackingScreenshot2.png" width="500">    | <img src="assets/RouteView_CorridorNavigation.png" width="500"> | 

## 🕹️ Requirements
This project is tested on Apple M1 Pro chip. The following libraries and frameworks are required:
- Python 3.10+
- OpenCV
- NumPy
- PyTorch
- djitellopy (DJI Tello SDK)

Install with:
```bash
pip install -r requirements.txt
```

## 🎮 Controls

| Key | Action             |
|-----|--------------------|
| `s` | Takeoff            |
| `l` | Land               |
| `q` | Emergency Quit     |

## 🧩Discussion
- Focuses
  - Tracking
    - PID tuning alone is insufficient - must be integrated with deadbands, dynamic clipping, boosting and smoothing to handle complex drone behavior such as - 1) Approach a stationary user from behind and stablize within target range
      without overshooting, and 2) Quickly react to a moving user accelerating forward from a hover state
    - Maintain tracking and alignment with user making turns by adopting optimal deceleration, yaw and lateral movements
      
  - Turning
    - Close the trailing distance so the drone can follow turns without veering into the opposite wall
    - Favor lateral shifts over rotational adjustments to replicate how humans turn in navigating confined or structured indoor environments
    - Control forward velocity with turn-aware deceleration to prevent overshooting
    - Define accurate yaw angle is critical for understanding user turns
      
  - Yaw and LR relations
    - Balance the tradeoff between stability in straight path and responsiveness in turns with weights tuning
    - Prioritize marker orientation in turns, and pixel error when moving straight
    - Achieve tight and responsive navigation around corners and maintain lateral alignment with back of target user by producing oppositely signed yaw and lateral commands – such as peeking left while rotating right
    - Avoid excessive lateral oscillation
 
  - Obstacle avoidance Region of Interest(ROI) definition
    - Monitor only the left and right sides for obstacles based on experimental findings and project nature. Obstacles are not likely to intrude between target user and drone in the middle part of frame
    - Full image height and width are not considered to avoid false positive classifications
    - ROI thresholds are empirically calibrated to balance sensitivity—avoiding values that are too little (missing obstacles) or too large (capturing irrelevant outliers)
    - ROI shrinks dynamically at closer distances to account for reduced false positives

- Limitations
  - Performace
    - Market tracking might be lost if user turns too fast

  - Blind spots
    - The drone lacks cameras on its left, right, and rear sides, resulting in limited spatial perception in those directions. Incorporating additional side- and rear-facing cameras or proximity sensors can significantly improve
      environmental awareness and reduce blind zones. 

  - YOLO Usage and limitations
    - Initially used YOLOv8n.pt to detect obstacles along the drone’s path. Experiments showed minimal obstruction between user and drone, so YOLO was repurposed as a fallback tracker when the ArUco marker is lost
    - The closest person box to the last known marker position is used to guide reacquisition
    - Performance Constraints on macOS (M1/MPS):
      - YOLOv8 is not MPS-compatible and runs on CPU with ~300 ms latency (~3 FPS with MiDaS and other modules). ONNXRuntime offers minor improvements but remains unsuitable for real-time use
      - YOLOv8n on GPU is only feasible via CoreML + Swift, which doesn't align with the current Python-based pipeline
      - Alternative direction includes considering YOLOv5n with TorchScript, which offers 60–120 ms latency per online benchmarks. YOLO is temporarily removed from the script for future exploration directions


## 💡 Conclusion 
By **empowering entry-level drones** with advanced real time autonomous behavior, this project enhances **accessibility, scalability, and affordability**, paving the way for assistive drone technologies to reach a **wider audience** beyond high-end, research-grade hardware.

Future work includes
- semantic scene understanding with more complex models
- integrating LLM voice assistant for human interaction
- hardware enhancement such as extra cameras or proximity sensors for improved spatial awareness
- eliminate the usage of visual marker while maintaining performance under dynamic environment
- build a model to learn and tune weights for drone movement

## 🛠️ Appendix1: Replacing drone motor without soldering

Replacing a damaged or worn-out motor on the DJI Tello EDU can be done without soldering, making it more accessible for beginners and faster for prototyping.
The visual guide below outlines each step, enabling quick motor swaps using only basic tools.

|    Identifying the correct motor orientation     | Clockwise (Blue/Black) or <br/> Counter-clockwise(White/Black)| 
| ----------------------------------------------   | ------------------------------------------------           |
|<img src="assets/replace_motor1.jpg" width="200"> | <img src="assets/replace_motor2.jpg" width="200">          |
| Reconnecting wires with <br/> staggered lengths to prevent short circuits | Securing wires with non-conductive tape | 
|<img src="assets/replace_motor3.jpg" width="200"> | <img src="assets/replace_motor4.jpg" width="200">          |


## 🛠️ Appendix2: Git workflow

Setting apart a working feature branch from the main branch. Might be useful for other contributors who are new to git control.

|    Action                         | Command
| --------------------------------- | ------------------------------------------------           |
|Pull latest main                   | > git checkout main && git pull origin main    |
|Create feature branch              | > git checkout -b feature/tracking-bugfix | 
|Work and commit                    | > git add . && git commit -m "..."        |
|(Optional) Update with latest main | > git checkout main && git pull origin main && git checkout feature/tracking-bugfix && git rebase main    |
|Merge back to main                 | > git checkout main && git merge feature/tracking-bugfix      |
|Push to remote                     | > git push origin main       |

## Contribution
Au, Chin Yuen (Isaac): Designed and implemented the complete system architecture, integrated ArUco tracking and MiDaS depth-based avoidance, tuned PD controllers, performed tests and evaluation, wrote documentation and deployment instructions.

## References
MiDaS: Ranftl et al., CVPR 2021 <br/>
ArUco: Garrido-Jurado et al., Pattern Recognition 2014 <br/>
DJI Tello SDK docs <br/>
Tello EDU Motor replacement: https://youtu.be/jU3gfmurPMk?si=GEEZ-b87soSrQNEm

## 👨‍🏫 Acknowledgements

This project is part of a research initiative at Northeastern University under the guidance of Prof. Ilmi Yoon and Prof Jeongkyu Lee.  
Special thanks to DJI Tello EDU SDK and open-source contributions from MiDaS, Deep SORT, and Ultralytics.
