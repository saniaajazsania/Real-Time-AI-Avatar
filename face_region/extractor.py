import cv2


class FaceRegionExtractor:

    def __init__(self, padding=30):
        self.padding = padding

    def extract(self, frame, face_landmarks):

        height, width, _ = frame.shape

        # Get all landmark coordinates
        x_coordinates = [
            int(landmark.x * width)
            for landmark in face_landmarks.landmark
        ]

        y_coordinates = [
            int(landmark.y * height)
            for landmark in face_landmarks.landmark
        ]

        # Find bounding box
        x_min = min(x_coordinates)
        x_max = max(x_coordinates)

        y_min = min(y_coordinates)
        y_max = max(y_coordinates)

        # Add padding
        x_min = max(0, x_min - self.padding)
        y_min = max(0, y_min - self.padding)

        x_max = min(width, x_max + self.padding)
        y_max = min(height, y_max + self.padding)

        # Crop face region
        face_region = frame[
            y_min:y_max,
            x_min:x_max
        ]

        bounding_box = (
            x_min,
            y_min,
            x_max,
            y_max
        )

        return face_region, bounding_box