import cv2
import mediapipe as mp

from face_region.extractor import FaceRegionExtractor


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

extractor = FaceRegionExtractor(
    padding=30
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Webcam open nahi ho raha.")
    exit()


while True:

    success, frame = cap.read()

    if not success:
        print("❌ Frame capture failed.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:

        face_landmarks = results.multi_face_landmarks[0]

        # Extract face region
        face_region, box = extractor.extract(
            frame,
            face_landmarks
        )

        x_min, y_min, x_max, y_max = box

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # Display status
        cv2.putText(
            frame,
            "Face Region Extracted",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # Show cropped face
        if face_region.size > 0:

            cv2.imshow(
                "Face Region",
                face_region
            )

    else:

        cv2.putText(
            frame,
            "Face Not Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Phase 2 - Face Region Extraction",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
face_mesh.close()
cv2.destroyAllWindows()