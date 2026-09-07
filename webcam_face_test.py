import cv2
import mediapipe as mp


# MediaPipe Face Detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils


def main():

    # Webcam open
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam open nahi ho raha.")
        return

    print("Webcam started...")
    print("Press Q to exit.")

    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    ) as face_detection:

        while True:

            # Webcam frame
            ret, frame = cap.read()

            if not ret:
                print("ERROR: Frame read nahi ho raha.")
                break

            # Mirror effect
            frame = cv2.flip(frame, 1)

            # BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Face detection
            results = face_detection.process(rgb_frame)

            # Draw detected face
            if results.detections:

                for detection in results.detections:

                    mp_draw.draw_detection(
                        frame,
                        detection
                    )

                cv2.putText(
                    frame,
                    "FACE DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "NO FACE",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            # Show webcam
            cv2.imshow(
                "Real Webcam Face Tracking",
                frame
            )

            # Q = exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()