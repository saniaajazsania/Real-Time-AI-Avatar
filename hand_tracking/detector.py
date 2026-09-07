import cv2
import mediapipe as mp


class HandTracker:
    """
    Detects a hand in the frame using MediaPipe Hands and
    returns the wrist position, which we use as a simple
    reference point to move a hand-icon overlay.
    """

    def __init__(
        self,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    ):

        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame):
        """Returns hand landmarks for the first detected hand, or None."""

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            return results.multi_hand_landmarks[0]

        return None

    def get_wrist_position(self, hand_landmarks, frame_shape):
        """Landmark 0 is the wrist point in MediaPipe's hand model."""

        height, width = frame_shape[:2]

        wrist = hand_landmarks.landmark[0]

        x = int(wrist.x * width)
        y = int(wrist.y * height)

        return x, y

    def close(self):
        self.hands.close()