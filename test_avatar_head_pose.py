import cv2
import numpy as np

from camera.webcam import start_webcam
from facial_landmarks import FacialLandmarks
from head_pose_tracking import HeadPoseTracker
from avatar.avatar_loader import AvatarLoader
from avatar.avatar_renderer import AvatarRenderer


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def blend_with_background(image_bgra, background_color=(230, 230, 230)):
    """Blends a BGRA image onto a solid background for display."""

    if image_bgra.shape[2] != 4:
        return image_bgra

    bgr = image_bgra[:, :, :3].astype(float)
    alpha = image_bgra[:, :, 3].astype(float) / 255.0
    alpha = alpha[:, :, np.newaxis]

    background = np.full(bgr.shape, background_color, dtype=float)
    blended = (alpha * bgr) + ((1 - alpha) * background)

    return blended.astype(np.uint8)


def main():

    print("=" * 60)
    print("PHASE 2 - STEP 4 TEST: Landmark -> Avatar Mapping")
    print("=" * 60)

    # -----------------------------------------------------
    # Setup — reusing the exact same working Phase 1 pieces
    # -----------------------------------------------------

    cap = start_webcam()

    if cap is None:
        print("❌ Could not start webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    landmark_detector = FacialLandmarks(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    head_tracker = HeadPoseTracker(
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        smoothing=0.7
    )

    avatar_loader = AvatarLoader(avatar_folder="avatar/assets/clara")
    avatar_loader.load_state("neutral", "head_neutral.png")

    avatar_renderer = AvatarRenderer()

    print("✅ All systems initialized.")
    print("Move your head to see the avatar follow. Press Q to quit.")
    print("-" * 60)

    # -----------------------------------------------------
    # Main loop
    # -----------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            print("❌ Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        face_landmarks = landmark_detector.process(frame)

        # Default avatar frame (no rotation) in case no face is found
        base_avatar = avatar_loader.get_state("neutral")
        rendered_avatar = base_avatar

        if face_landmarks is not None:

            pose = head_tracker.calculate_pose(face_landmarks, frame.shape)

            if pose is not None:

                rendered_avatar = avatar_renderer.render(
                    base_avatar,
                    yaw=pose["yaw"],
                    pitch=pose["pitch"],
                    roll=pose["roll"]
                )

                cv2.putText(
                    frame,
                    f"Yaw: {pose['yaw']:.1f}  Pitch: {pose['pitch']:.1f}  Roll: {pose['roll']:.1f}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

        # -----------------------------------------------------
        # Combine webcam feed (left) + avatar (right) side by side
        # -----------------------------------------------------

        avatar_display = blend_with_background(rendered_avatar)

        # Resize avatar to match webcam frame height for side-by-side display
        avatar_display = cv2.resize(
            avatar_display,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        combined = np.hstack((frame, avatar_display))

        cv2.imshow("Phase 2 Step 4 - Head Pose -> Avatar", combined)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # -----------------------------------------------------
    # Cleanup
    # -----------------------------------------------------

    cap.release()
    landmark_detector.close()
    head_tracker.close()
    cv2.destroyAllWindows()

    print("✅ Test complete.")


if __name__ == "__main__":
    main()