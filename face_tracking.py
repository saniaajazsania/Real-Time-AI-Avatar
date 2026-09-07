import cv2
import mediapipe as mp
import time
import math


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
# 2. Webcam
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam open nahi ho raha.")
    exit()


# =========================================================
# 3. FPS
# =========================================================

previous_time = 0


# =========================================================
# 4. Previous Face Position
# =========================================================

previous_x = None
previous_y = None


# Movement threshold
MOVEMENT_THRESHOLD = 8


# =========================================================
# 5. Main Loop
# =========================================================

while True:

    success, frame = cap.read()

    if not success:
        print("❌ Frame capture failed.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # Get frame dimensions
    height, width, _ = frame.shape


    # =====================================================
    # Convert BGR → RGB
    # =====================================================

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # Detect Facial Landmarks
    # =====================================================

    results = face_mesh.process(rgb_frame)


    # Default movement
    movement = "CENTER"


    # =====================================================
    # If Face Detected
    # =====================================================

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]


        # -------------------------------------------------
        # Nose Landmark
        # -------------------------------------------------
        # Landmark 1 = approximate nose area

        nose = face_landmarks.landmark[1]


        # Convert normalized coordinates to pixels

        nose_x = int(nose.x * width)
        nose_y = int(nose.y * height)


        # -------------------------------------------------
        # Draw Nose Point
        # -------------------------------------------------

        cv2.circle(
            frame,
            (nose_x, nose_y),
            8,
            (0, 0, 255),
            -1
        )


        # -------------------------------------------------
        # Calculate Face Center
        # -------------------------------------------------

        face_x = nose_x
        face_y = nose_y


        # -------------------------------------------------
        # Compare with Previous Frame
        # -------------------------------------------------

        if previous_x is not None:

            dx = face_x - previous_x
            dy = face_y - previous_y


            # Horizontal movement

            if abs(dx) > MOVEMENT_THRESHOLD:

                if dx > 0:
                    movement = "RIGHT"
                else:
                    movement = "LEFT"


            # Vertical movement

            elif abs(dy) > MOVEMENT_THRESHOLD:

                if dy > 0:
                    movement = "DOWN"
                else:
                    movement = "UP"


            else:

                movement = "CENTER"


        # Save current position

        previous_x = face_x
        previous_y = face_y


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
        # Display Face Position
        # =================================================

        cv2.putText(
            frame,
            f"Face X: {face_x}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Face Y: {face_y}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =================================================
        # Display Movement
        # =================================================

        cv2.putText(
            frame,
            f"Movement: {movement}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # =================================================
        # Face Status
        # =================================================

        cv2.putText(
            frame,
            "Face: TRACKING",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )


    else:

        # Face not detected

        cv2.putText(
            frame,
            "Face: NOT DETECTED",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )


        # Reset position

        previous_x = None
        previous_y = None


    # =====================================================
    # FPS Calculation
    # =====================================================

    current_time = time.time()

    if previous_time != 0:

        fps = 1 / (current_time - previous_time)

    else:

        fps = 0


    previous_time = current_time


    # =====================================================
    # Display FPS
    # =====================================================

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )


    # =====================================================
    # Display Window
    # =====================================================

    cv2.imshow(
        "Phase 1 - Basic Face Tracking",
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

print("✅ Face tracking stopped.")