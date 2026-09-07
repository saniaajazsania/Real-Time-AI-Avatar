import cv2
import mediapipe as mp


# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh


def main():

    # Webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Webcam open nahi ho raha.")
        return

    print("Head tracking started...")
    print("Move your head LEFT / RIGHT")
    print("Press Q to exit.")

    # Face Mesh
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:

            ret, frame = cap.read()

            if not ret:
                print("ERROR: Frame read nahi ho raha.")
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            # Frame dimensions
            h, w, _ = frame.shape

            # BGR → RGB
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # Face landmarks
            results = face_mesh.process(rgb_frame)

            direction = "NO FACE"

            if results.multi_face_landmarks:

                face_landmarks = results.multi_face_landmarks[0]

                # Nose landmark
                nose = face_landmarks.landmark[1]

                # Nose pixel position
                nose_x = int(nose.x * w)
                nose_y = int(nose.y * h)

                # Draw nose point
                cv2.circle(
                    frame,
                    (nose_x, nose_y),
                    8,
                    (0, 255, 0),
                    -1
                )

                # Center of webcam
                center_x = w // 2

                # Difference from center
                difference = nose_x - center_x

                # LEFT / CENTER / RIGHT
                if difference < -60:
                    direction = "HEAD LEFT"

                elif difference > 60:
                    direction = "HEAD RIGHT"

                else:
                    direction = "HEAD CENTER"

                # Show nose position
                cv2.putText(
                    frame,
                    f"Nose X: {nose_x}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            # Show direction
            cv2.putText(
                frame,
                direction,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Show webcam
            cv2.imshow(
                "Mie Head Tracking Test",
                frame
            )

            # Q = quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    