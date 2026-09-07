import cv2
import numpy as np

from camera.webcam import start_webcam
from facial_landmarks import FacialLandmarks
from head_pose_tracking import HeadPoseTracker

from avatar.avatar_loader import AvatarLoader
from avatar.avatar_renderer import AvatarRenderer
from avatar.expression_tracking import ExpressionTracker


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def blend_with_background(
    image_bgra,
    background_color=(230, 230, 230)
):
    """
    Converts BGRA avatar into BGR
    for display.
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
    print("PHASE 2 - STEP 8")
    print("FULL AVATAR ANIMATION TEST")
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
    # 3. Head Pose
    # =========================================================

    head_tracker = HeadPoseTracker(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        smoothing=0.7
    )

    print("✅ Head pose tracker initialized.")

    # =========================================================
    # 4. Expression Tracker
    # =========================================================

    expression_tracker = ExpressionTracker()

    print("✅ Expression tracker initialized.")

    # =========================================================
    # 5. Avatar Loader
    # =========================================================

    avatar_loader = AvatarLoader(
        avatar_folder="avatar/assets/clara"
    )

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

    # =========================================================
    # 6. Avatar Renderer
    # =========================================================

    avatar_renderer = AvatarRenderer()

    print("✅ Avatar renderer initialized.")

    print("-" * 60)
    print("FULL AVATAR ANIMATION IS RUNNING")
    print()
    print("Move head LEFT / RIGHT")
    print("Tilt head")
    print("Blink")
    print("Open mouth")
    print("Press Q to quit.")
    print("-" * 60)

    # =========================================================
    # MAIN LOOP
    # =========================================================

    while True:

        # =====================================================
        # Frame Capture
        # =====================================================

        ret, frame = cap.read()

        if not ret:

            print("❌ Frame capture failed.")
            break

        # =====================================================
        # Mirror
        # =====================================================

        frame = cv2.flip(
            frame,
            1
        )

        # =====================================================
        # Resize
        # =====================================================

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # =====================================================
        # Default Avatar
        # =====================================================

        avatar_state = "neutral"

        base_avatar = avatar_loader.get_state(
            "neutral"
        )

        rendered_avatar = base_avatar

        # =====================================================
        # Facial Landmarks
        # =====================================================

        face_landmarks = landmark_detector.process(
            frame
        )

        # =====================================================
        # Tracking
        # =====================================================

        if face_landmarks is not None:

            # -------------------------------------------------
            # HEAD POSE
            # -------------------------------------------------

            pose = head_tracker.calculate_pose(
                face_landmarks,
                frame.shape
            )

            # -------------------------------------------------
            # EXPRESSION
            # -------------------------------------------------

            expression = expression_tracker.detect(
                face_landmarks
            )

            # =================================================
            # Expression → Avatar State
            # =================================================

            if expression is not None:

                detected_expression = (
                    expression["expression"]
                )

                if detected_expression == "BLINK":

                    avatar_state = "blink"

                elif detected_expression == "TALK":

                    avatar_state = "talk"

                else:

                    avatar_state = "neutral"

            # =================================================
            # Get Avatar Image
            # =================================================

            base_avatar = avatar_loader.get_state(
                avatar_state
            )

            # =================================================
            # HEAD POSE → AVATAR
            # =================================================

            if pose is not None:

                rendered_avatar = avatar_renderer.render(
                    base_avatar,
                    yaw=pose["yaw"],
                    pitch=pose["pitch"],
                    roll=pose["roll"]
                )

                # -------------------------------------------------
                # Show tracking information
                # -------------------------------------------------

                cv2.putText(
                    frame,
                    f"Yaw: {pose['yaw']:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Pitch: {pose['pitch']:.1f}",
                    (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Roll: {pose['roll']:.1f}",
                    (15, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # =================================================
            # Expression Information
            # =================================================

            if expression is not None:

                cv2.putText(
                    frame,
                    f"Expression: {expression['expression']}",
                    (15, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Avatar State: {avatar_state}",
                    (15, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
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
        # Avatar Display
        # =====================================================

        avatar_display = blend_with_background(
            rendered_avatar
        )

        avatar_display = cv2.resize(
            avatar_display,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # =====================================================
        # Side-by-Side Display
        # =====================================================

        combined = np.hstack(
            (
                frame,
                avatar_display
            )
        )

        cv2.imshow(
            "Phase 2 Step 8 - Full Avatar Animation",
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

    head_tracker.close()

    expression_tracker.close()

    cv2.destroyAllWindows()

    print("✅ Full avatar animation test complete.")


if __name__ == "__main__":

    main()