
import cv2
import numpy as np


class HeadPoseTracker:
    """
    Estimates head position and head rotation
    using facial landmarks.

    Outputs:
        Yaw
        Pitch
        Roll
        LEFT / RIGHT
        UP / DOWN
        HEAD TILT
    """

    def __init__(
        self,
        frame_width=640,
        frame_height=480,
        smoothing=0.7
    ):

        self.frame_width = frame_width
        self.frame_height = frame_height

        self.smoothing = smoothing

        # =================================================
        # Important Landmark IDs
        # =================================================

        self.NOSE = 1
        self.CHIN = 152

        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263

        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291


        # =================================================
        # 3D Face Model
        # =================================================

        self.model_points = np.array([
            (0.0, 0.0, 0.0),
            (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0),
            (150.0, -150.0, -125.0)
        ], dtype=np.float64)


        # =================================================
        # Camera Matrix
        # =================================================

        focal_length = frame_width

        self.camera_matrix = np.array([
            [
                focal_length,
                0,
                frame_width / 2
            ],

            [
                0,
                focal_length,
                frame_height / 2
            ],

            [
                0,
                0,
                1
            ]

        ], dtype=np.float64)


        self.distortion_coefficients = np.zeros(
            (4, 1)
        )


        # =================================================
        # Previous Pose Values (for smoothing)
        # =================================================

        self.previous_yaw = 0.0
        self.previous_pitch = 0.0
        self.previous_roll = 0.0

        # Raw (unsmoothed) previous values, used for
        # outlier-jump detection.
        self.previous_raw_yaw = 0.0
        self.previous_raw_pitch = 0.0
        self.previous_raw_roll = 0.0

        # =================================================
        # Previous solvePnP solution (for warm-starting)
        # =================================================

        self.previous_rotation_vector = None
        self.previous_translation_vector = None

        self.has_valid_pose = False


    # =====================================================
    # Smoothing
    # =====================================================

    def smooth_value(
        self,
        previous,
        current
    ):

        return (
            self.smoothing * previous
            + (1 - self.smoothing) * current
        )


    # =====================================================
    # Stable Rotation Matrix -> Euler Angles (atan2-based)
    # =====================================================

    def rotation_matrix_to_angles(self, rotation_matrix):

        sy = np.sqrt(
            rotation_matrix[0, 0] ** 2
            + rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:

            x_angle = np.arctan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2]
            )

            y_angle = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            z_angle = np.arctan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0]
            )

        else:

            x_angle = np.arctan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1]
            )

            y_angle = np.arctan2(
                -rotation_matrix[2, 0],
                sy
            )

            z_angle = 0.0

        pitch = np.degrees(x_angle)
        yaw = np.degrees(y_angle)
        roll = np.degrees(z_angle)

        return pitch, yaw, roll


    # =====================================================
    # Calculate Head Pose
    # =====================================================

    def calculate_pose(
        self,
        face_landmarks,
        frame_shape
    ):

        height, width = frame_shape[:2]


        # =================================================
        # Get 2D Image Points
        # =================================================

        image_points = np.array([

            (
                face_landmarks.landmark[
                    self.NOSE
                ].x * width,

                face_landmarks.landmark[
                    self.NOSE
                ].y * height
            ),

            (
                face_landmarks.landmark[
                    self.CHIN
                ].x * width,

                face_landmarks.landmark[
                    self.CHIN
                ].y * height
            ),

            (
                face_landmarks.landmark[
                    self.LEFT_EYE
                ].x * width,

                face_landmarks.landmark[
                    self.LEFT_EYE
                ].y * height
            ),

            (
                face_landmarks.landmark[
                    self.RIGHT_EYE
                ].x * width,

                face_landmarks.landmark[
                    self.RIGHT_EYE
                ].y * height
            ),

            (
                face_landmarks.landmark[
                    self.LEFT_MOUTH
                ].x * width,

                face_landmarks.landmark[
                    self.LEFT_MOUTH
                ].y * height
            ),

            (
                face_landmarks.landmark[
                    self.RIGHT_MOUTH
                ].x * width,

                face_landmarks.landmark[
                    self.RIGHT_MOUTH
                ].y * height
            )

        ], dtype=np.float64)


        # =================================================
        # Nose Position
        # =================================================

        nose_x = int(image_points[0][0])
        nose_y = int(image_points[0][1])


        # =================================================
        # Position Tracking
        # =================================================

        center_x = width // 2
        center_y = height // 2

        horizontal_threshold = 80
        vertical_threshold = 60


        # Horizontal

        if nose_x < center_x - horizontal_threshold:

            position_x = "LEFT"

        elif nose_x > center_x + horizontal_threshold:

            position_x = "RIGHT"

        else:

            position_x = "CENTER"


        # Vertical

        if nose_y < center_y - vertical_threshold:

            position_y = "UP"

        elif nose_y > center_y + vertical_threshold:

            position_y = "DOWN"

        else:

            position_y = "CENTER"


        # =================================================
        # Head Pose (warm-started with previous solution
        # to avoid the ambiguous "flipped" solution)
        # =================================================

        if self.has_valid_pose:

            success_pose, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    self.model_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    rvec=self.previous_rotation_vector.copy(),
                    tvec=self.previous_translation_vector.copy(),
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            )

        else:

            success_pose, rotation_vector, translation_vector = (
                cv2.solvePnP(
                    self.model_points,
                    image_points,
                    self.camera_matrix,
                    self.distortion_coefficients,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            )


        if not success_pose:

            return None


        # =================================================
        # Rotation Vector → Matrix
        # =================================================

        rotation_matrix, _ = cv2.Rodrigues(
            rotation_vector
        )


        # =================================================
        # Rotation Matrix → Euler Angles (stable version)
        # =================================================

        raw_pitch, raw_yaw, raw_roll = self.rotation_matrix_to_angles(
            rotation_matrix
        )


        # =================================================
        # OUTLIER REJECTION
        # =================================================
        # If this frame's raw pose jumped too far from the
        # last raw pose (a sign that solvePnP locked onto
        # the wrong / flipped ambiguous solution), reject
        # this frame's solve and reuse the previous good
        # solution instead of updating with bad data.

        JUMP_THRESHOLD = 60  # degrees

        pose_is_valid_jump = True

        if self.has_valid_pose:

            yaw_jump = abs(raw_yaw - self.previous_raw_yaw)
            pitch_jump = abs(raw_pitch - self.previous_raw_pitch)
            roll_jump = abs(raw_roll - self.previous_raw_roll)

            if (
                yaw_jump > JUMP_THRESHOLD
                or pitch_jump > JUMP_THRESHOLD
                or roll_jump > JUMP_THRESHOLD
            ):

                pose_is_valid_jump = False


        if pose_is_valid_jump:

            # Good solution — accept it and remember it
            # for next frame's warm-start.

            self.previous_rotation_vector = rotation_vector
            self.previous_translation_vector = translation_vector
            self.has_valid_pose = True

            self.previous_raw_yaw = raw_yaw
            self.previous_raw_pitch = raw_pitch
            self.previous_raw_roll = raw_roll

            pitch = raw_pitch
            yaw = raw_yaw
            roll = raw_roll

        else:

            # Bad/outlier solution — ignore it, keep using
            # the last known good smoothed values so the
            # avatar doesn't jump/flip for this frame.

            pitch = self.previous_pitch
            yaw = self.previous_yaw
            roll = self.previous_roll


        # =================================================
        # Smooth Values
        # =================================================

        yaw = self.smooth_value(
            self.previous_yaw,
            yaw
        )

        pitch = self.smooth_value(
            self.previous_pitch,
            pitch
        )

        roll = self.smooth_value(
            self.previous_roll,
            roll
        )


        self.previous_yaw = yaw
        self.previous_pitch = pitch
        self.previous_roll = roll


        # =================================================
        # YAW
        # =================================================

        if yaw < -10:

            head_direction = "LEFT"

        elif yaw > 10:

            head_direction = "RIGHT"

        else:

            head_direction = "CENTER"


        # =================================================
        # PITCH
        # =================================================

        if pitch > 10:

            head_vertical = "UP"

        elif pitch < -10:

            head_vertical = "DOWN"

        else:

            head_vertical = "CENTER"


        # =================================================
        # ROLL
        # =================================================

        if roll < -10:

            head_tilt = "LEFT TILT"

        elif roll > 10:

            head_tilt = "RIGHT TILT"

        else:

            head_tilt = "STRAIGHT"


        # =================================================
        # Return All Data
        # =================================================

        return {

            "pitch": pitch,

            "yaw": yaw,

            "roll": roll,

            "direction": head_direction,

            "vertical": head_vertical,

            "tilt": head_tilt,

            "position_x": position_x,

            "position_y": position_y,

            "nose_x": nose_x,

            "nose_y": nose_y
        }


    # =====================================================
    # Close
    # =====================================================

    def close(self):
        """
        Currently no external resource to release.
        """
        pass