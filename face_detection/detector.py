import cv2
import mediapipe as mp


class FaceDetector:

    def __init__(self, min_detection_confidence=0.5):

        self.mp_face_detection = mp.solutions.face_detection

        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def detect(self, frame):

        # OpenCV BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = self.detector.process(rgb_frame)

        faces = []

        if results.detections:

            height, width, _ = frame.shape

            for detection in results.detections:

                bbox = detection.location_data.relative_bounding_box

                x = int(bbox.xmin * width)
                y = int(bbox.ymin * height)

                w = int(bbox.width * width)
                h = int(bbox.height * height)

                # Keep coordinates inside frame
                x = max(0, x)
                y = max(0, y)

                w = min(w, width - x)
                h = min(h, height - y)

                confidence = detection.score[0]

                faces.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "confidence": confidence
                })

        return faces

    def draw_detections(self, frame, faces):

        for face in faces:

            x = face["x"]
            y = face["y"]
            w = face["width"]
            h = face["height"]

            confidence = face["confidence"]

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display confidence
            text = f"Face: {confidence:.2f}"

            cv2.putText(
                frame,
                text,
                (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        return frame

    def close(self):

        self.detector.close()