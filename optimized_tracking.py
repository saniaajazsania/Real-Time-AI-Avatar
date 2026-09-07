import cv2
import mediapipe as mp
import numpy as np
import time


# =========================================================
# 1. MediaPipe Face Mesh
# =========================================================

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================================================
# 2. Webcam Configuration
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam open nahi ho raha.")
    exit()


# Set processing resolution
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# Reduce camera buffer to lower latency
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# =========================================================
# 3. Important Face Landmark IDs
# =========================================================

NOSE = 1
CHIN = 152

LEFT_EYE = 33
RIGHT_EYE = 263

LEFT_MOUTH = 61
RIGHT_MOUTH = 291


# =========================================================
# 4. 3D Face Model
# =========================================================

model_points = np.array([
    (0.0, 0.0, 0.0),            # Nose
    (0.0, -330.0, -65.0),       # Chin
    (-225.0, 170.0, -135.0),     # Left Eye
    (225.0, 170.0, -135.0),      # Right Eye
    (-150.0, -150.0, -125.0),    # Left Mouth
    (150.0, -150.0, -125.0)      # Right Mouth
], dtype=np.float64)


# =========================================================
# 5. Camera Matrix
# =========================================================

focal_length = FRAME_WIDTH

camera_matrix = np.array([
    [focal_length, 0, FRAME_WIDTH / 2],
    [0, focal_length, FRAME_HEIGHT / 2],
    [0, 0, 1]
], dtype=np.float64)


distortion_coefficients = np.zeros((4, 1))


# =========================================================
# 6. FPS Variables
# =========================================================

previous_time = time.perf_counter()

fps = 0.0


# =========================================================
# 7. Smoothing Variables
# =========================================================

previous_yaw = 0.0
previous_pitch = 0.0
previous_roll = 0.0


# Smoothing factor
SMOOTHING = 0.7


def smooth_value(previous, current):
    """
    Smooths head-pose values to reduce jitter.
    """

    return (
        SMOOTHING * previous
        + (1 - SMOOTHING) * current
    )


# =========================================================
# 8. Main Loop
# =========================================================

while True:

    # -----------------------------------------------------
    # Capture frame
    # -----------------------------------------------------

    success, frame = cap.read()

    if not success:
        print("❌ Frame capture failed.")
        break


    # -----------------------------------------------------
    # Mirror webcam
    # -----------------------------------------------------

    frame = cv2.flip(frame, 1)


    # -----------------------------------------------------
    # Resize frame
    # -----------------------------------------------------

    frame = cv2.resize(
        frame,
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_LINEAR
    )


    height, width, _ = frame.shape


    # =====================================================
    # Convert BGR → RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # MediaPipe Processing
    # =====================================================

    results = face_mesh.process(rgb_frame)


    # =====================================================
    # Face Detection
    # =====================================================

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]


        # =================================================
        # Get Important Landmarks
        # =================================================

        image_points = np.array([
            (
                face_landmarks.landmark[NOSE].x * width,
                face_landmarks.landmark[NOSE].y * height
            ),

            (
                face_landmarks.landmark[CHIN].x * width,
                face_landmarks.landmark[CHIN].y * height
            ),

            (
                face_landmarks.landmark[LEFT_EYE].x * width,
                face_landmarks.landmark[LEFT_EYE].y * height
            ),

            (
                face_landmarks.landmark[RIGHT_EYE].x * width,
                face_landmarks.landmark[RIGHT_EYE].y * height
            ),

            (
                face_landmarks.landmark[LEFT_MOUTH].x * width,
                face_landmarks.landmark[LEFT_MOUTH].y * height
            ),

            (
                face_landmarks.landmark[RIGHT_MOUTH].x * width,
                face_landmarks.landmark[RIGHT_MOUTH].y * height
            )

        ], dtype=np.float64)


        # =================================================
        # Nose Position
        # =================================================

        nose_x = int(image_points[0][0])
        nose_y = int(image_points[0][1])


        cv2.circle(
            frame,
            (nose_x, nose_y),
            6,
            (0, 0, 255),
            -1
        )


        # =================================================
        # Position-Based Tracking
        # =================================================

        center_x = width // 2
        center_y = height // 2

        horizontal_threshold = 80
        vertical_threshold = 60


        # X Position

        if nose_x < center_x - horizontal_threshold:

            position_x = "LEFT"

        elif nose_x > center_x + horizontal_threshold:

            position_x = "RIGHT"

        else:

            position_x = "CENTER"


        # Y Position

        if nose_y < center_y - vertical_threshold:

            position_y = "UP"

        elif nose_y > center_y + vertical_threshold:

            position_y = "DOWN"

        else:

            position_y = "CENTER"


        # =================================================
        # Head Pose Estimation
        # =================================================

        success_pose, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )


        if success_pose:

            # Rotation vector → rotation matrix

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )


            # Rotation matrix → Euler angles

            rq_result = cv2.RQDecomp3x3(
                rotation_matrix
            )

            angles = rq_result[0]


            pitch = float(angles[0])
            yaw = float(angles[1])
            roll = float(angles[2])


            # =================================================
            # Smooth Head Pose
            # =================================================

            yaw = smooth_value(
                previous_yaw,
                yaw
            )

            pitch = smooth_value(
                previous_pitch,
                pitch
            )

            roll = smooth_value(
                previous_roll,
                roll
            )


            # Save values

            previous_yaw = yaw
            previous_pitch = pitch
            previous_roll = roll


            # =================================================
            # YAW
            # =================================================

            if yaw < -10:

                head_direction = "LEFT"

            elif yaw > 10:

                head_direction = "RIGHT"

            else:

                head_direction = "CENTER"


            # =================================================
            # PITCH
            # =================================================

            if pitch > 10:

                head_vertical = "UP"

            elif pitch < -10:

                head_vertical = "DOWN"

            else:

                head_vertical = "CENTER"


            # =================================================
            # ROLL
            # =================================================

            if roll < -10:

                head_tilt = "LEFT TILT"

            elif roll > 10:

                head_tilt = "RIGHT TILT"

            else:

                head_tilt = "STRAIGHT"


            # =================================================
            # Display Position
            # =================================================

            cv2.putText(
                frame,
                f"Position X: {position_x}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            cv2.putText(
                frame,
                f"Position Y: {position_y}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


            # =================================================
            # Display Yaw
            # =================================================

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f} ({head_direction})",
                (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


            # =================================================
            # Display Pitch
            # =================================================

            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f} ({head_vertical})",
                (15, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


            # =================================================
            # Display Roll
            # =================================================

            cv2.putText(
                frame,
                f"Roll: {roll:.1f} ({head_tilt})",
                (15, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )


        # =================================================
        # Draw Face Mesh
        # =================================================

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(180, 180, 180),
                thickness=1
            )
        )


        # =================================================
        # Draw Face Contours
        # =================================================

        mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(0, 255, 0),
                thickness=2
            )
        )


        # =================================================
        # Face Status
        # =================================================

        cv2.putText(
            frame,
            "Face: TRACKING",
            (15, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )


    else:

        cv2.putText(
            frame,
            "Face: NOT DETECTED",
            (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    # =====================================================
    # FPS Calculation
    # =====================================================

    current_time = time.perf_counter()

    elapsed_time = current_time - previous_time

    if elapsed_time > 0:

        current_fps = 1.0 / elapsed_time

        # Smooth FPS
        fps = (
            0.9 * fps
            + 0.1 * current_fps
        )


    previous_time = current_time


    # =====================================================
    # Display FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (500, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2
    )


    # =====================================================
    # Display Window
    # =====================================================

    cv2.imshow(
        "Phase 1 - Optimized Face Tracking",
        frame
    )


    # =====================================================
    # Exit
    # =====================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================================================
# Cleanup
# =========================================================

cap.release()

cv2.destroyAllWindows()

face_mesh.close()

print("✅ Optimized Face Tracking stopped.")