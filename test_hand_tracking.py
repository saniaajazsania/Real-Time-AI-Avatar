import cv2

from camera.webcam import start_webcam
from hand_tracking.detector import HandTracker


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():

    print("=" * 60)
    print("HAND TRACKING TEST")
    print("=" * 60)

    cap = start_webcam()

    if cap is None:
        print("❌ Could not start webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    hand_tracker = HandTracker()

    print("✅ Hand tracker initialized. Show your hand. Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        hand_landmarks = hand_tracker.process(frame)

        if hand_landmarks is not None:

            wrist_x, wrist_y = hand_tracker.get_wrist_position(
                hand_landmarks,
                frame.shape
            )

            cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"Wrist: ({wrist_x}, {wrist_y})",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "No hand detected",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.imshow("Hand Tracking Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hand_tracker.close()
    cv2.destroyAllWindows()

    print("✅ Test complete.")


if __name__ == "__main__":
    main()