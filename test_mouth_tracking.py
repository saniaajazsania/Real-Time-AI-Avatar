import cv2

from camera.webcam import start_webcam
from facial_landmarks import FacialLandmarks
from avatar.expression_tracking import ExpressionTracker


FRAME_WIDTH = 640
FRAME_HEIGHT = 480


def main():

    print("=" * 60)
    print("PHASE 2 - STEP 7 TEST")
    print("MOUTH MOVEMENT TRACKING")
    print("=" * 60)

    # =========================================================
    # Webcam
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
    # Facial Landmarks
    # =========================================================

    landmark_detector = FacialLandmarks(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    print("✅ Facial landmarks initialized.")

    # =========================================================
    # Expression Tracker
    # =========================================================

    expression_tracker = ExpressionTracker()

    print("✅ Expression tracker initialized.")

    print("-" * 60)
    print("Open and close your mouth slowly.")
    print("Watch Mouth Ratio and Mouth Level.")
    print("Press Q to quit.")
    print("-" * 60)

    # =========================================================
    # Main Loop
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

        if face_landmarks is not None:

            # =================================================
            # Expression Detection
            # =================================================

            result = expression_tracker.detect(
                face_landmarks
            )

            if result is not None:

                mouth_ratio = result[
                    "mouth_ratio"
                ]

                mouth_level = result[
                    "mouth_level"
                ]

                mouth_open = result[
                    "mouth_open"
                ]

                # ---------------------------------------------
                # Expression
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Expression: {result['expression']}",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Mouth Ratio
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Mouth Ratio: {mouth_ratio:.3f}",
                    (15, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                # ---------------------------------------------
                # Mouth Level
                # ---------------------------------------------

                cv2.putText(
                    frame,
                    f"Mouth Level: {mouth_level:.1f}%",
                    (15, 115),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                # ---------------------------------------------
                # Mouth State
                # ---------------------------------------------

                state = (
                    "OPEN"
                    if mouth_open
                    else "CLOSED"
                )

                cv2.putText(
                    frame,
                    f"Mouth: {state}",
                    (15, 155),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                # =================================================
                # Visual Mouth Level Bar
                # =================================================

                bar_x = 15
                bar_y = 200

                bar_width = 300
                bar_height = 25

                # Background

                cv2.rectangle(
                    frame,
                    (
                        bar_x,
                        bar_y
                    ),
                    (
                        bar_x + bar_width,
                        bar_y + bar_height
                    ),
                    (80, 80, 80),
                    -1
                )

                # Filled level

                fill_width = int(
                    bar_width *
                    mouth_level /
                    100.0
                )

                cv2.rectangle(
                    frame,
                    (
                        bar_x,
                        bar_y
                    ),
                    (
                        bar_x + fill_width,
                        bar_y + bar_height
                    ),
                    (0, 255, 0),
                    -1
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
        # Display
        # =====================================================

        cv2.imshow(
            "Phase 2 Step 7 - Mouth Tracking",
            frame
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

    print("✅ Mouth movement tracking test complete.")


if __name__ == "__main__":

    main()