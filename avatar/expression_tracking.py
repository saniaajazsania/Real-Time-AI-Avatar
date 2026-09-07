import math


class ExpressionTracker:
    """
    Tracks basic facial expressions and continuous
    mouth movement using MediaPipe facial landmarks.

    Detects:
        - BLINK
        - TALK
        - NEUTRAL

    Also provides:
        - Eye openness
        - Mouth openness
        - Mouth movement level
    """

    def __init__(
        self,
        eye_closed_threshold=0.20,
        mouth_open_threshold=0.035
    ):

        self.eye_closed_threshold = (
            eye_closed_threshold
        )

        self.mouth_open_threshold = (
            mouth_open_threshold
        )

    # =========================================================
    # Distance Between Two Landmarks
    # =========================================================

    def distance(self, p1, p2):

        return math.sqrt(
            (p1.x - p2.x) ** 2 +
            (p1.y - p2.y) ** 2
        )

    # =========================================================
    # Eye Aspect Ratio
    # =========================================================

    def eye_aspect_ratio(
        self,
        landmarks,
        top_id,
        bottom_id,
        left_id,
        right_id
    ):

        vertical_distance = self.distance(
            landmarks.landmark[top_id],
            landmarks.landmark[bottom_id]
        )

        horizontal_distance = self.distance(
            landmarks.landmark[left_id],
            landmarks.landmark[right_id]
        )

        if horizontal_distance == 0:

            return 0.0

        return (
            vertical_distance /
            horizontal_distance
        )

    # =========================================================
    # Mouth Opening Ratio
    # =========================================================

    def mouth_open_ratio(
        self,
        landmarks
    ):

        mouth_top = landmarks.landmark[13]

        mouth_bottom = landmarks.landmark[14]

        mouth_left = landmarks.landmark[61]

        mouth_right = landmarks.landmark[291]

        vertical_distance = self.distance(
            mouth_top,
            mouth_bottom
        )

        horizontal_distance = self.distance(
            mouth_left,
            mouth_right
        )

        if horizontal_distance == 0:

            return 0.0

        return (
            vertical_distance /
            horizontal_distance
        )

    # =========================================================
    # Convert Mouth Ratio into Animation Level
    # =========================================================

    def mouth_movement_level(
        self,
        mouth_ratio
    ):
        """
        Converts mouth opening ratio into
        a 0-100 animation value.
        """

        minimum = 0.015
        maximum = 0.100

        # Clamp

        mouth_ratio = max(
            minimum,
            min(maximum, mouth_ratio)
        )

        # Normalize

        level = (
            (mouth_ratio - minimum)
            /
            (maximum - minimum)
        ) * 100.0

        return level

    # =========================================================
    # Main Detection
    # =========================================================

    def detect(
        self,
        face_landmarks
    ):

        if face_landmarks is None:

            return None

        landmarks = face_landmarks

        # =====================================================
        # LEFT EYE
        # =====================================================

        left_ear = self.eye_aspect_ratio(
            landmarks,
            top_id=159,
            bottom_id=145,
            left_id=33,
            right_id=133
        )

        # =====================================================
        # RIGHT EYE
        # =====================================================

        right_ear = self.eye_aspect_ratio(
            landmarks,
            top_id=386,
            bottom_id=374,
            left_id=362,
            right_id=263
        )

        # =====================================================
        # Average Eye Ratio
        # =====================================================

        eye_ratio = (
            left_ear +
            right_ear
        ) / 2.0

        # =====================================================
        # Mouth
        # =====================================================

        mouth_ratio = self.mouth_open_ratio(
            landmarks
        )

        # =====================================================
        # Mouth Movement
        # =====================================================

        mouth_level = self.mouth_movement_level(
            mouth_ratio
        )

        # =====================================================
        # Blink
        # =====================================================

        eyes_closed = (
            eye_ratio <
            self.eye_closed_threshold
        )

        # =====================================================
        # Mouth Open
        # =====================================================

        mouth_open = (
            mouth_ratio >
            self.mouth_open_threshold
        )

        # =====================================================
        # Expression
        # =====================================================

        if eyes_closed:

            expression = "BLINK"

        elif mouth_open:

            expression = "TALK"

        else:

            expression = "NEUTRAL"

        # =====================================================
        # Return
        # =====================================================

        return {

            "expression": expression,

            "left_eye_ratio": left_ear,

            "right_eye_ratio": right_ear,

            "eye_ratio": eye_ratio,

            "mouth_ratio": mouth_ratio,

            "mouth_level": mouth_level,

            "eyes_closed": eyes_closed,

            "mouth_open": mouth_open
        }

    # =========================================================
    # Close
    # =========================================================

    def close(self):

        pass