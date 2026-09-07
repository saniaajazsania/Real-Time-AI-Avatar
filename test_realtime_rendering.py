import cv2
import numpy as np
import time

from camera.webcam import start_webcam
from facial_landmarks import FacialLandmarks
from head_pose_tracking import HeadPoseTracker

from avatar.avatar_loader import AvatarLoader
from avatar.avatar_renderer import AvatarRenderer
from avatar.expression_tracking import ExpressionTracker


def main():

    # ==========================================
    # 1. Start Webcam
    # ==========================================

    cap = start_webcam()

    if cap is None:
        print("❌ Could not start webcam.")
        return

    # ==========================================
    # 2. Initialize Face/Landmark Tracking
    # ==========================================

    facial_landmarks = FacialLandmarks()
    head_tracker = HeadPoseTracker()

    # ==========================================
    # 3. Initialize Avatar System
    # ==========================================

    avatar_loader = AvatarLoader(
        "avatar/assets/clara"
    )

    avatar_renderer = AvatarRenderer()

    expression_tracker = ExpressionTracker()

    # ==========================================
    # 4. Load Avatar States
    # ==========================================

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

    neutral_avatar = avatar_loader.get_state("neutral")
    blink_avatar = avatar_loader.get_state("blink")
    talk_avatar = avatar_loader.get_state("talk")

    if neutral_avatar is None:
        print("❌ Neutral avatar not found.")
        return

    print("✅ Avatar states loaded successfully.")

    # ==========================================
    # 5. FPS Variables
    # ==========================================

    previous_time = time.time()
    fps = 0

    print("\n========================================")
    print(" REAL-TIME AVATAR RENDERING TEST")
    print("========================================")
    print("Press Q to quit.")
    print("========================================\n")

    # ==========================================
    # 6. Main Real-Time Loop
    # ==========================================

    while True:

        # --------------------------------------
        # Capture webcam frame
        # --------------------------------------

        ret, frame = cap.read()

        if not ret:
            print("❌ Failed to read webcam frame.")
            break

        # Mirror webcam
        frame = cv2.flip(frame, 1)

        # --------------------------------------
        # Facial Landmarks
        # --------------------------------------

        landmarks = facial_landmarks.process(frame)

        # Default values
        expression_name = "NO FACE"

        pose = {
            "yaw": 0,
            "pitch": 0,
            "roll": 0
        }

        base_avatar = neutral_avatar

        # --------------------------------------
        # If face detected
        # --------------------------------------

        if landmarks is not None:

            # ==================================
            # Head Pose Tracking
            # ==================================

            pose = head_tracker.calculate_pose(
                landmarks,
                frame.shape
            )

            # ==================================
            # Expression Tracking
            # ==================================

            expression = expression_tracker.detect(
                landmarks
            )

            if expression is not None:

                expression_name = expression["expression"]

                # Select avatar state
                if expression_name == "BLINK":

                    base_avatar = blink_avatar

                elif expression_name == "TALK":

                    base_avatar = talk_avatar

                else:

                    base_avatar = neutral_avatar

        # ==========================================
        # Real-Time Avatar Rendering
        # ==========================================

        rendered_avatar = avatar_renderer.render(
            base_avatar,
            yaw=pose["yaw"],
            pitch=pose["pitch"],
            roll=pose["roll"]
        )

        # ==========================================
        # Prepare Avatar Display
        # ==========================================

        avatar_display = rendered_avatar.copy()

        # --------------------------------------
        # If avatar has Alpha channel
        # --------------------------------------

        if len(avatar_display.shape) == 3 and avatar_display.shape[2] == 4:

            avatar_bgr = avatar_display[:, :, :3]

            alpha = (
                avatar_display[:, :, 3].astype(float)
                / 255.0
            )

            alpha = alpha[:, :, None]

            # White background
            white_background = (
                255
                * np.ones(
                    avatar_bgr.shape,
                    dtype="uint8"
                )
            )

            avatar_display = (
                avatar_bgr.astype(float) * alpha
                +
                white_background.astype(float) * (1 - alpha)
            )

            avatar_display = avatar_display.astype("uint8")

        # ==========================================
        # Resize Avatar
        # ==========================================

        target_height = frame.shape[0]

        avatar_height = avatar_display.shape[0]
        avatar_width = avatar_display.shape[1]

        scale = target_height / avatar_height

        target_width = int(
            avatar_width * scale
        )

        avatar_display = cv2.resize(
            avatar_display,
            (target_width, target_height)
        )

        # ==========================================
        # FPS Calculation
        # ==========================================

        current_time = time.time()

        elapsed = current_time - previous_time

        if elapsed > 0:

            fps = 1.0 / elapsed

        previous_time = current_time

        # ==========================================
        # Webcam Information
        # ==========================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Expression: {expression_name}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        if landmarks is not None:

            cv2.putText(
                frame,
                f"Yaw: {pose['yaw']:.1f}",
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Pitch: {pose['pitch']:.1f}",
                (20, 135),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Roll: {pose['roll']:.1f}",
                (20, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

        # ==========================================
        # Labels
        # ==========================================

        cv2.putText(
            frame,
            "WEBCAM",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            avatar_display,
            "AI AVATAR",
            (20, avatar_display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        # ==========================================
        # Combine Webcam + Avatar
        # ==========================================

        combined = cv2.hconcat(
            [
                frame,
                avatar_display
            ]
        )

        # ==========================================
        # Display
        # ==========================================

        cv2.imshow(
            "Step 9 - Real-Time Avatar Rendering",
            combined
        )

        # ==========================================
        # Press Q to Quit
        # ==========================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # ==========================================
    # Cleanup
    # ==========================================

    cap.release()

    cv2.destroyAllWindows()

    facial_landmarks.close()
    head_tracker.close()
    expression_tracker.close()

    print("\n✅ Real-time rendering test completed.")


# ==============================================
# Run Program
# ==============================================

if __name__ == "__main__":
    main()