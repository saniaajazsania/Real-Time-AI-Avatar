import cv2

from camera.webcam import start_webcam
from facial_landmarks import FacialLandmarks
from avatar.expression_tracking import ExpressionTracker


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():

    print("=" * 60)
    print("PHASE 2 - STEP 5 TEST")
    print("EXPRESSION TRACKING")
    print("=" * 60)

    # ---------------------------------------------------------
    # Webcam
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Facial Landmarks
    # ---------------------------------------------------------

    landmark_detector = FacialLandmarks(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # ---------------------------------------------------------
    # Expression Tracker
    # ---------------------------------------------------------

    expression_tracker = ExpressionTracker()

    print("✅ Systems initialized.")
    print("Try blinking and opening your mouth.")
    print("Press Q to quit.")
    print("-" * 60)

    # ---------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:

            print("❌ Frame capture failed.")
            break

        # Mirror

        frame = cv2.flip(
            frame,
            1
        )

        # Resize

        frame = cv2.resize(
            frame,
            (FRAME_WIDTH, FRAME_HEIGHT)
        )

        # -----------------------------------------------------
        # Facial Landmarks
        # -----------------------------------------------------

        face_landmarks = landmark_detector.process(
            frame
        )

        # -----------------------------------------------------
        # Expression Tracking
        # -----------------------------------------------------

        if face_landmarks is not None:

            expression = expression_tracker.detect(
                face_landmarks
            )

            if expression is not None:

                # ---------------------------------------------
                # Expression
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Expression: {expression['expression']}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Eye Ratio
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Eye Ratio: {expression['eye_ratio']:.3f}",
                    (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Mouth Ratio
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Mouth Ratio: {expression['mouth_ratio']:.3f}",
                    (15, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
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
                    (15, 140),
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
                    (15, 175),
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

        # -----------------------------------------------------
        # Display
        # -----------------------------------------------------

        cv2.imshow(
            "Phase 2 Step 5 - Expression Tracking",
            frame
        )

        # -----------------------------------------------------
        # Quit
        # -----------------------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    cap.release()

    landmark_detector.close()

    expression_tracker.close()

    cv2.destroyAllWindows()

    print("✅ Expression tracking test complete.")


if __name__ == "__main__":

    main()