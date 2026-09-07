# 🤖 Real-Time AI Avatar

A real-time computer vision-based avatar system that tracks a user's facial and upper-body movements through a webcam and synchronizes them with a responsive 2D digital avatar.

The project combines **MediaPipe Face Mesh, MediaPipe Pose, OpenCV, NumPy, and Python** to create a lightweight real-time avatar animation pipeline that works on a **CPU-based system without requiring a dedicated GPU**.

---

## 📌 Project Overview

The **Real-Time AI Avatar** project transforms live webcam input into synchronized 2D avatar movements.

The system captures the user's webcam feed, detects facial landmarks and body keypoints, analyzes movement, and maps the detected information to different avatar states.

The avatar responds to:

* Head movement
* Head rotation
* Eye blinking
* Mouth/talking movement
* Facial expression states
* Left arm movement
* Right arm movement

The goal was to build a practical real-time avatar pipeline using lightweight computer vision techniques while maintaining smooth and responsive performance on CPU hardware.

---

## ✨ Key Features

### 👤 Facial Tracking

The system uses **MediaPipe Face Mesh** to detect facial landmarks in real time.

It tracks important facial points such as:

* Nose
* Eyes
* Mouth
* Facial regions required for expression analysis

### 🧠 Head Movement Tracking

The detected facial landmarks are used to estimate:

* Head movement along the X-axis
* Head movement along the Y-axis
* Head rotation

The avatar's head position and rotation are then adjusted according to the user's movement.

### 👁️ Eye Blink Detection

The system analyzes eye landmarks to detect blinking.

When a blink is detected, the avatar switches to a dedicated blink state.

This creates a more natural avatar animation instead of keeping the avatar's eyes permanently open.

### 👄 Mouth / Talking Animation

The distance between relevant mouth landmarks is analyzed to detect mouth opening.

Depending on the detected state, the avatar can switch between:

* Neutral mouth
* Talking/open mouth

This allows the avatar to visually respond when the user is speaking.

### 😊 Facial Expression States

Facial landmark information is also used to determine basic facial states and synchronize the corresponding avatar appearance.

### 💪 Arm Movement Tracking

The project also integrates **MediaPipe Pose** to track upper-body landmarks.

The system tracks:

* Left shoulder
* Right shoulder
* Left wrist
* Right wrist

Based on the detected wrist positions, the avatar's left and right arms can switch between different positions.

### 🎨 Layer-Based 2D Avatar Rendering

Instead of using a complex 3D model, the project uses separate PNG layers for the avatar.

Example layers include:

```text
body
head
arms
eyes
mouth
```

These components are dynamically combined to create the final avatar frame.

### ⚡ Real-Time Performance Optimization

Several techniques were implemented to improve responsiveness:

* Frame smoothing
* Movement smoothing
* Lightweight landmark processing
* Efficient image composition
* FPS monitoring
* Light image sharpening
* CPU-friendly processing

The project was specifically developed and tested without relying on a dedicated GPU.

---

# 🏗️ System Architecture

```text
                Webcam Input
                     │
                     ▼
              Frame Capture
                     │
                     ▼
            OpenCV Processing
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
   MediaPipe Face Mesh    MediaPipe Pose
          │                     │
          ▼                     ▼
   Facial Landmarks       Body Landmarks
          │                     │
    ┌─────┼─────┐             │
    ▼     ▼     ▼             ▼
   Head  Blink  Mouth       Arm Tracking
    │     │      │             │
    └─────┴──────┴─────────────┘
                     │
                     ▼
              Movement Mapping
                     │
                     ▼
             Avatar Rendering
                     │
                     ▼
          Real-Time Avatar Output
```

---

# 🔄 How It Works

The complete pipeline works in several stages.

### 1. Webcam Capture

OpenCV captures frames from the user's webcam.

```text
Webcam → Video Frames
```

Each frame is processed continuously.

### 2. Face Detection & Landmark Tracking

MediaPipe Face Mesh detects facial landmarks from the webcam frame.

These landmarks provide information about the position of different facial regions.

### 3. Head Pose & Movement Estimation

Important facial landmarks are analyzed to estimate head movement and rotation.

The resulting values are smoothed before being applied to the avatar.

```text
User Head Movement
        ↓
Landmark Detection
        ↓
Movement Estimation
        ↓
Smoothing
        ↓
Avatar Head Movement
```

### 4. Eye Blink Detection

Eye landmarks are analyzed to determine whether the user's eyes are open or closed.

```text
Eyes Open  → Normal Avatar
Eyes Closed → Blink Avatar
```

### 5. Mouth Tracking

Mouth landmarks are used to determine the mouth opening state.

```text
Mouth Closed → Neutral
Mouth Open   → Talking
```

### 6. Arm Tracking

MediaPipe Pose detects the user's shoulders and wrists.

The system compares wrist positions with predefined thresholds and maps them to avatar arm states.

```text
User Arm Movement
        ↓
Pose Landmarks
        ↓
Wrist Position
        ↓
Arm State
        ↓
Avatar Arm Movement
```

### 7. Avatar Composition

The body, head, and arm assets are loaded and dynamically composed into a single avatar frame.

The final avatar is rendered alongside the original webcam feed for real-time comparison.

---

# 🛠️ Technology Stack

| Technology                    | Purpose                              |
| ----------------------------- | ------------------------------------ |
| **Python**                    | Core programming language            |
| **OpenCV**                    | Webcam capture and image processing  |
| **MediaPipe Face Mesh**       | Facial landmark detection            |
| **MediaPipe Pose**            | Body and arm tracking                |
| **NumPy**                     | Numerical processing                 |
| **Live2D / 2D Avatar Assets** | Avatar design and animation workflow |
| **Git & GitHub**              | Version control and project hosting  |

---

# 📁 Project Structure

```text
Real Time AI Video/
│
├── avatar/
│   └── assets/
│       └── mie/
│           ├── body/
│           │   └── body_mie.png
│           │
│           ├── head/
│           │   ├── head_neutral_mie.png
│           │   ├── head_blink_mie.png
│           │   └── head_talk_mie.png
│           │
│           └── arms/
│               ├── left_arm_down.png
│               ├── left_arm_up.png
│               ├── right_arm_down.png
│               └── right_arm_up.png
│
├── main.py
├── facial_landmarks.py
├── face_tracking.py
├── head_pose_tracking.py
├── optimized_tracking.py
├── requirements.txt
├── .gitignore
└── README.md
```

> Project structure may evolve as additional modules and improvements are added.

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/real-time-ai-avatar.git
cd real-time-ai-avatar
```

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📦 Requirements

The core project dependencies are:

```text
opencv-python
mediapipe
numpy
```

---

# ▶️ Running the Project

After installing the dependencies, run:

```bash
python main.py
```

The application will access the webcam and start the real-time tracking and avatar rendering pipeline.

The output displays the user's webcam feed together with the corresponding animated avatar.

---

# 🎯 Real-Time Tracking Pipeline

The complete processing flow can be summarized as:

```text
Webcam
   ↓
OpenCV Frame Capture
   ↓
Face Mesh + Pose Detection
   ↓
Facial & Body Landmarks
   ↓
Head / Blink / Mouth / Arm Analysis
   ↓
Movement Smoothing
   ↓
Avatar State Mapping
   ↓
2D Avatar Composition
   ↓
Real-Time Display
```

---

# 🚧 Challenges & Solutions

## Challenge 1 — CPU-Only Real-Time Processing

One of the main challenges was maintaining real-time performance without a dedicated GPU.

### Solution

The pipeline was kept lightweight by using MediaPipe's efficient tracking models and applying optimized frame processing and smoothing techniques.

---

## Challenge 2 — Smooth Head Movement

Directly mapping landmark positions to avatar movement can result in jittery animation.

### Solution

Movement smoothing was introduced to reduce sudden changes and create more natural avatar motion.

---

## Challenge 3 — Blink Detection

The avatar needed to respond naturally to the user's eye movements.

### Solution

Eye landmark distances were analyzed to detect when the eyes were closed and switch to the blink avatar state.

---

## Challenge 4 — Mouth Animation

A static mouth makes a talking avatar look unnatural.

### Solution

Mouth landmark distances were used to determine whether the mouth was open or closed and switch between neutral and talking states.

---

## Challenge 5 — Arm Synchronization

Tracking arms and converting real-world wrist positions into simple 2D avatar states required additional logic.

### Solution

MediaPipe Pose landmarks were used to track shoulders and wrists, while threshold-based state mapping was used for the avatar arms.

---

# 📊 Performance

The project was designed with CPU-based execution in mind.

Performance considerations included:

* Lightweight computer vision models
* Efficient frame processing
* Movement smoothing
* FPS monitoring
* Reduced unnecessary processing
* Simple 2D image composition

This makes the project suitable for experimentation on systems without dedicated GPU hardware.

---

# 🔮 Future Enhancements

The current version focuses on the core real-time tracking and avatar animation pipeline.

Possible future improvements include:

### 🎥 Virtual Camera Integration

Integrate the rendered avatar output with applications such as:

* Zoom
* Microsoft Teams
* Discord
* Other video communication platforms

### 🧍 Full-Body Tracking

Extend the current upper-body tracking to support:

* Full-body movement
* Leg tracking
* More detailed pose estimation

### 🎭 Advanced Facial Expressions

Add more detailed expression tracking for:

* Happiness
* Sadness
* Surprise
* Anger
* Other facial states

### 🧠 AI-Based Avatar Generation

Future versions could explore generative AI or neural rendering approaches for more realistic avatar transformation.

### ⚡ GPU Acceleration

ONNX Runtime, TensorRT, CUDA, or other optimized inference technologies could be explored for higher-resolution and more complex real-time models.

---

# 🎓 Learning Outcomes

Through this project, I gained practical experience in:

* Real-time computer vision
* OpenCV
* MediaPipe Face Mesh
* MediaPipe Pose
* Facial landmark detection
* Head pose estimation
* Eye blink detection
* Mouth movement detection
* Pose estimation
* Real-time avatar animation
* Image processing
* Performance optimization
* CPU-based AI processing
* Git and GitHub project management

---

# 📌 Project Status

**Status: Core Project Completed ✅**

The current implementation successfully demonstrates:

* ✅ Real-time webcam processing
* ✅ Facial landmark tracking
* ✅ Head movement tracking
* ✅ Head rotation
* ✅ Eye blink detection
* ✅ Mouth/talking animation
* ✅ Facial state mapping
* ✅ Arm movement tracking
* ✅ 2D avatar rendering
* ✅ Real-time side-by-side visualization
* ✅ Performance optimization
* ✅ CPU-based execution

**Virtual Camera integration is planned as a future enhancement and is not part of the current completed implementation.**

---

# 👩‍💻 Developer

**Sania Aijaz**

BS Computer Science Student
AI / Machine Learning / Computer Vision Enthusiast

---

# ⭐ Acknowledgment

This project was developed as a hands-on exploration of real-time computer vision and avatar animation, combining facial tracking, pose estimation, image processing, and real-time rendering into a single pipeline.

---

## 📜 License

This project is intended for educational and portfolio purposes.

If you reuse or extend this project, please provide appropriate credit to the original work and any third-party assets used.
