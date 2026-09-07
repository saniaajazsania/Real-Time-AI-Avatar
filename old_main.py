import cv2
import numpy as np

from camera.webcam import start_webcam
from performance.fps import FPSCounter
from face_detection.detector import FaceDetector

from facial_landmarks import FacialLandmarks
from head_pose_tracking import HeadPoseTracker

from avatar.avatar_loader import AvatarLoader
from avatar.avatar_renderer import AvatarRenderer
from avatar.expression_tracking import ExpressionTracker

from pose_tracking.detector import ArmTracker


# =========================================================
# Configuration
# =========================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Shoulder pivot point in body_no_arm.png / only_arm.png pixel
# coordinates (estimated — adjust these two numbers if the arm
# rotates around the wrong point).
SHOULDER_PIVOT_X = 320
SHOULDER_PIVOT_Y = 365

# The angle (in the same units get_arm_angle returns) that
# corresponds to the arm's REST position as drawn in only_arm.png.
# Start at 15 and adjust after testing.
NEUTRAL_ARM_ANGLE = 15

# Maximum degrees the arm is allowed to rotate away from neutral,
# in either direction (safety clamp).
MAX_ARM_ROTATION = 70


def alpha_composite(base_bgra, overlay_bgra):
    """
    Composites overlay_bgra on top of base_bgra (both same size,
    BGRA). Returns a new BGRA image.
    """

    base_bgr = base_bgra[:, :, :3].astype(float)
    base_a = base_bgra[:, :, 3:4].astype(float) / 255.0

    ov_bgr = overlay_bgra[:, :, :3].astype(float)
    ov_a = overlay_bgra[:, :, 3:4].astype(float) / 255.0

    out_bgr = ov_bgr * ov_a + base_bgr * (1 - ov_a)
    out_a = (ov_a + base_a * (1 - ov_a)) * 255.0

    out = np.dstack([out_bgr, out_a]).astype("uint8")

    return out


def main():

    print("=" * 60)
    print("REAL-TIME AI VIDEO")
    print("PHASE 1 + PHASE 2 - COMPLETE PIPELINE (+ ARM MOVEMENT)")
    print("=" * 60)


    # =====================================================
    # 1. Webcam
    # =====================================================

    cap = start_webcam()

    if cap is None:
        print("❌ Could not start webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("✅ Webcam initialized.")


    # =====================================================
    # 2. FPS Counter
    # =====================================================

    fps_counter = FPSCounter()

    print("✅ FPS counter initialized.")


    # =====================================================
    # 3. Face Detector
    # =====================================================

    face_detector = FaceDetector(
        min_detection_confidence=0.5
    )

    print("✅ Face detector initialized.")


    # =====================================================
    # 4. Facial Landmarks
    # =====================================================

    landmark_detector = FacialLandmarks(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("✅ Facial landmarks initialized.")


    # =====================================================
    # 5. Head Pose Tracker
    # =====================================================

    head_tracker = HeadPoseTracker(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        smoothing=0.7
    )

    print("✅ Head pose tracker initialized.")


    # =====================================================
    # 6. Avatar System
    # =====================================================

    avatar_loader = AvatarLoader("avatar/assets/clara")

    avatar_loader.load_state("neutral", "head_neutral_fulls.png")
    avatar_loader.load_state("blink", "head_blink_fulls.png")
    avatar_loader.load_state("talk", "head_talk_fulls.png")
    avatar_loader.load_state("smile", "head_smile.png")
    avatar_loader.load_state("body_no_arm", "body_no_arm.png")
    avatar_loader.load_state("arm", "only_arm.png")

    neutral_avatar = avatar_loader.get_state("neutral")
    blink_avatar = avatar_loader.get_state("blink")
    talk_avatar = avatar_loader.get_state("talk")
    body_no_arm = avatar_loader.get_state("body_no_arm")
    arm_image = avatar_loader.get_state("arm")

    if neutral_avatar is None:
        print("❌ Neutral avatar not found.")
        return

    if body_no_arm is None or arm_image is None:
        print("⚠️ Arm assets not found — arm movement will be disabled.")

    avatar_renderer = AvatarRenderer()

    expression_tracker = ExpressionTracker()

    print("✅ Avatar system initialized.")


    # =====================================================
    # 7. Arm Tracker
    # =====================================================

    arm_tracker = ArmTracker()

    print("✅ Arm tracker initialized.")


    print("-" * 60)
    print("Phase 1 + Phase 2 pipeline is running.")
    print("Press Q to quit.")
    print("-" * 60)


    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:


        # =================================================
        # Frame Capture
        # =================================================

        ret, frame = cap.read()

        if not ret:
            print("❌ Frame capture failed.")
            break


        # =================================================
        # Mirror + Resize
        # =================================================

        frame = cv2.flip(frame, 1)

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT),
            interpolation=cv2.INTER_LINEAR
        )


        # =================================================
        # FACE DETECTION (visual overlay on webcam side)
        # =================================================

        faces = face_detector.detect(frame)

        frame = face_detector.draw_detections(frame, faces)


        # =================================================
        # FACIAL LANDMARKS
        # =================================================

        face_landmarks = landmark_detector.process(frame)


        # =================================================
        # Defaults (used when no face is detected)
        # =================================================

        expression_name = "NO FACE"

        pose = {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

        base_avatar = neutral_avatar
        use_dynamic_arm = True   # default (neutral) allows arm movement


        # =================================================
        # If Face Detected
        # =================================================

        if face_landmarks is not None:


            # ---------------------------------------------
            # Head Pose
            # ---------------------------------------------

            calculated_pose = head_tracker.calculate_pose(
                face_landmarks,
                frame.shape
            )

            if calculated_pose is not None:
                pose = calculated_pose


            # ---------------------------------------------
            # Expression Tracking
            # ---------------------------------------------

            expression = expression_tracker.detect(
                face_landmarks
            )

            if expression is not None:

                expression_name = expression["expression"]

                if expression_name == "BLINK":
                    base_avatar = blink_avatar
                    use_dynamic_arm = False

                elif expression_name == "TALK":
                    base_avatar = talk_avatar
                    use_dynamic_arm = False

                else:
                    base_avatar = neutral_avatar
                    use_dynamic_arm = True

            else:

                expression_name = "NEUTRAL"


        # =================================================
        # ARM TRACKING
        # =================================================

        arm_angle_offset = 0.0

        if use_dynamic_arm and body_no_arm is not None and arm_image is not None:

            pose_landmarks = arm_tracker.process(frame)

            if pose_landmarks is not None:

                raw_angle = arm_tracker.get_arm_angle(pose_landmarks, frame.shape)

                if raw_angle is not None:

                    arm_angle_offset = raw_angle - NEUTRAL_ARM_ANGLE
                    arm_angle_offset = max(
                        -MAX_ARM_ROTATION,
                        min(MAX_ARM_ROTATION, arm_angle_offset)
                    )


        # =================================================
        # AVATAR RENDERING
        # =================================================

        if use_dynamic_arm and body_no_arm is not None and arm_image is not None:

            # Rotate the arm image around the shoulder pivot
            h, w = arm_image.shape[:2]

            rot_matrix = cv2.getRotationMatrix2D(
                (SHOULDER_PIVOT_X, SHOULDER_PIVOT_Y),
                -arm_angle_offset,
                1.0
            )

            rotated_arm = cv2.warpAffine(
                arm_image,
                rot_matrix,
                (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            # Composite rotated arm onto the body (before head-pose rotation)
            composed_base = alpha_composite(body_no_arm, rotated_arm)

            rendered_avatar = avatar_renderer.render(
                composed_base,
                yaw=pose["yaw"],
                pitch=pose["pitch"],
                roll=pose["roll"]
            )

        else:

            rendered_avatar = avatar_renderer.render(
                base_avatar,
                yaw=pose["yaw"],
                pitch=pose["pitch"],
                roll=pose["roll"]
            )


        # =================================================
        # Alpha Blend Avatar onto White Background
        # =================================================

        avatar_display = rendered_avatar.copy()

        if len(avatar_display.shape) == 3 and avatar_display.shape[2] == 4:

            avatar_bgr = avatar_display[:, :, :3]

            alpha = avatar_display[:, :, 3].astype(float) / 255.0
            alpha = alpha[:, :, None]

            white_background = 255 * np.ones(
                avatar_bgr.shape,
                dtype="uint8"
            )

            avatar_display = (
                avatar_bgr.astype(float) * alpha
                + white_background.astype(float) * (1 - alpha)
            )

            avatar_display = avatar_display.astype("uint8")


        # =================================================
        # Resize Avatar to Match Webcam Frame Height
        # =================================================

        target_height = frame.shape[0]

        avatar_height = avatar_display.shape[0]
        avatar_width = avatar_display.shape[1]

        scale = target_height / avatar_height
        target_width = int(avatar_width * scale)

        avatar_display = cv2.resize(
            avatar_display,
            (target_width, target_height)
        )


        # =================================================
        # FPS
        # =================================================

        fps_counter.update()
        fps = fps_counter.get_fps()


        # =================================================
        # Webcam Side — Overlay Text
        # =================================================

        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Expression: {expression_name}",
            (15, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {pose['yaw']:.1f}  Pitch: {pose['pitch']:.1f}  Roll: {pose['roll']:.1f}",
            (15, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Arm angle offset: {arm_angle_offset:.1f}",
            (15, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 200, 0),
            2
        )

        cv2.putText(
            frame,
            "WEBCAM",
            (15, frame.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # =================================================
        # Avatar Side — Label
        # =================================================

        cv2.putText(
            avatar_display,
            "AI AVATAR",
            (15, avatar_display.shape[0] - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )


        # =================================================
        # Combine Webcam + Avatar Side-by-Side
        # =================================================

        combined = cv2.hconcat([frame, avatar_display])


        # =================================================
        # Display
        # =================================================

        cv2.imshow(
            "Real Time AI Video - Phase 1 + Phase 2",
            combined
        )


        # =================================================
        # Quit
        # =================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


    # =====================================================
    # CLEANUP
    # =====================================================

    print("\nStopping pipeline...")

    cap.release()

    face_detector.close()
    landmark_detector.close()
    head_tracker.close()
    expression_tracker.close()
    arm_tracker.close()

    cv2.destroyAllWindows()

    print("✅ Phase 1 + Phase 2 pipeline completed successfully.")


# =========================================================
# Entry Point
# =========================================================

if __name__ == "__main__":
    main()