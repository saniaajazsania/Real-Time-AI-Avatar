import cv2
import numpy as np

from camera.webcam import start_webcam
from facial_landmarks import FacialLandmarks
from avatar.expression_tracking import ExpressionTracker
from avatar.avatar_loader import AvatarLoader


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def blend_with_background(
    image_bgra,
    background_color=(230, 230, 230)
):
    """
    Converts transparent BGRA avatar into
    a normal BGR image for display.
    """

    if image_bgra.shape[2] != 4:
        return image_bgra

    bgr = image_bgra[:, :, :3].astype(float)

    alpha = (
        image_bgra[:, :, 3].astype(float)
        / 255.0
    )

    alpha = alpha[:, :, np.newaxis]

    background = np.full(
        bgr.shape,
        background_color,
        dtype=float
    )

    blended = (
        alpha * bgr
        + (1 - alpha) * background
    )

    return blended.astype(np.uint8)


def main():

    print("=" * 60)
    print("PHASE 2 - STEP 6 TEST")
    print("EXPRESSION -> AVATAR STATE")
    print("=" * 60)

    # =========================================================
    # 1. Webcam
    # =========================================================

    cap = start_webcam()

    if cap is None:

        print("❌ Could not start webcam.")
        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        FRAME_WIDTH
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        FRAME_HEIGHT
    )

    cap.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    print("✅ Webcam initialized.")

    # =========================================================
    # 2. Facial Landmarks
    # =========================================================

    landmark_detector = FacialLandmarks(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("✅ Facial landmarks initialized.")

    # =========================================================
    # 3. Expression Tracker
    # =========================================================

    expression_tracker = ExpressionTracker()

    print("✅ Expression tracker initialized.")

    # =========================================================
    # 4. Avatar Loader
    # =========================================================

    avatar_loader = AvatarLoader(
        avatar_folder="avatar/assets/clara"
    )

    # Load all avatar states

    avatar_loader.load_state(
        "neutral",
        "head_neutral.png"
    )

    avatar_loader.load_state(
        "blink",
        "head_blink.png"
    )

    avatar_loader.load_state(
        "talk",
        "head_talk.png"
    )

    print("✅ Avatar states loaded.")

    print("-" * 60)
    print("Expression -> Avatar mapping is running.")
    print("Blink = closed eyes")
    print("Talk = open mouth")
    print("Normal = neutral")
    print("Press Q to quit.")
    print("-" * 60)

    # =========================================================
    # 5. Main Loop
    # =========================================================

    while True:

        ret, frame = cap.read()

        if not ret:

            print("❌ Frame capture failed.")
            break

        # -----------------------------------------------------
        # Mirror
        # -----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )

        # -----------------------------------------------------
        # Resize
        # -----------------------------------------------------

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # =====================================================
        # Facial Landmarks
        # =====================================================

        face_landmarks = landmark_detector.process(
            frame
        )

        # =====================================================
        # Default Avatar
        # =====================================================

        current_state = "neutral"

        avatar_image = avatar_loader.get_state(
            "neutral"
        )

        # =====================================================
        # Expression Detection
        # =====================================================

        if face_landmarks is not None:

            expression = expression_tracker.detect(
                face_landmarks
            )

            if expression is not None:

                # ---------------------------------------------
                # Get detected expression
                # ---------------------------------------------

                detected_expression = (
                    expression["expression"]
                )

                # ---------------------------------------------
                # Expression -> Avatar mapping
                # ---------------------------------------------

                if detected_expression == "BLINK":

                    current_state = "blink"

                elif detected_expression == "TALK":

                    current_state = "talk"

                else:

                    current_state = "neutral"

                # ---------------------------------------------
                # Get avatar image
                # ---------------------------------------------

                avatar_image = avatar_loader.get_state(
                    current_state
                )

                # ---------------------------------------------
                # Show expression information
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Expression: {detected_expression}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Avatar State: {current_state}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                # ---------------------------------------------
                # Eye State
                # ---------------------------------------------

                eye_state = (
                    "CLOSED"
                    if expression["eyes_closed"]
                    else "OPEN"
                )

                cv2.putText(
                    frame,
                    f"Eyes: {eye_state}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Mouth State
                # ---------------------------------------------

                mouth_state = (
                    "OPEN"
                    if expression["mouth_open"]
                    else "CLOSED"
                )

                cv2.putText(
                    frame,
                    f"Mouth: {mouth_state}",
                    (15, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        else:

            cv2.putText(
                frame,
                "Face: NOT DETECTED",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # =====================================================
        # Prepare Avatar for Display
        # =====================================================

        avatar_display = blend_with_background(
            avatar_image
        )

        avatar_display = cv2.resize(
            avatar_display,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # =====================================================
        # Combine Webcam + Avatar
        # =====================================================

        combined = np.hstack(
            (
                frame,
                avatar_display
            )
        )

        # =====================================================
        # Display
        # =====================================================

        cv2.imshow(
            "Phase 2 Step 6 - Expression -> Avatar",
            combined
        )

        # =====================================================
        # Quit
        # =====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # =========================================================
    # Cleanup
    # =========================================================

    cap.release()

    landmark_detector.close()

    expression_tracker.close()

    cv2.destroyAllWindows()

    print("✅ Expression -> Avatar test complete.")


if __name__ == "__main__":

    main()