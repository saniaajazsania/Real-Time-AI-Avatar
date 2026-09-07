import cv2
import numpy as np


class AvatarRenderer:
    """
    Renders the 2D avatar according to head pose.

    Yaw   -> left/right movement
    Pitch -> up/down movement
    Roll  -> head tilt/rotation
    """

    def __init__(self, max_yaw_shift=25, max_pitch_shift=30):
        # Maximum avatar movement in pixels
        self.max_yaw_shift = max_yaw_shift
        self.max_pitch_shift = max_pitch_shift

    def render(self, avatar_image_bgra, yaw, pitch, roll):
        """
        avatar_image_bgra: BGRA avatar image
        yaw:   left/right head rotation
        pitch: up/down head movement
        roll:  head tilt

        Returns transformed BGRA avatar.
        """

        height, width = avatar_image_bgra.shape[:2]

        # Avatar center
        center = (width // 2, height // 2)

        # --------------------------------------------------
        # 1. SAFETY CLAMP
        # --------------------------------------------------

        yaw = max(-45, min(45, yaw))
        pitch = max(-45, min(45, pitch))
        roll = max(-30, min(30, roll))

        # --------------------------------------------------
        # 2. HEAD TILT / ROLL
        # --------------------------------------------------

        rotation_matrix = cv2.getRotationMatrix2D(
            center,
            -roll,
            1.0
        )

        # --------------------------------------------------
        # 3. LEFT / RIGHT MOVEMENT
        # --------------------------------------------------

        shift_x = -(yaw / 45.0) * self.max_yaw_shift

        # --------------------------------------------------
        # 4. UP / DOWN MOVEMENT
        # --------------------------------------------------

        shift_y = (pitch / 45.0) * self.max_pitch_shift

        # Apply movement
        rotation_matrix[0, 2] += shift_x
        rotation_matrix[1, 2] += shift_y

        # --------------------------------------------------
        # 5. APPLY TRANSFORMATION
        # --------------------------------------------------

        transformed = cv2.warpAffine(
            avatar_image_bgra,
            rotation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )

        return transformed