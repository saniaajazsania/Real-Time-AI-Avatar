import cv2
import mediapipe as mp
import numpy as np


class ArmTracker:
    """
    Uses MediaPipe Pose to track the shoulder and elbow (as seen on
    screen, i.e. the person's own right arm since the webcam feed
    is mirrored) and calculates the arm's angle relative to vertical,
    so it can drive a rotating arm overlay on the avatar.
    """

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.5):

        self.mp_pose = mp.solutions.pose

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process(self, frame):

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if results.pose_landmarks:
            return results.pose_landmarks

        return None

    def get_arm_angle(self, pose_landmarks, frame_shape):
        """
        Landmark 11 = left shoulder, 13 = left elbow, as seen on
        screen (mirrored view = the person's actual right arm).

        Returns the angle in degrees of the shoulder->elbow line
        relative to straight down (0 = arm hanging down,
        positive = raised outward).
        """

        height, width = frame_shape[:2]

        shoulder = pose_landmarks.landmark[11]
        elbow = pose_landmarks.landmark[13]

        # Only trust this if both points are reasonably visible
        if shoulder.visibility < 0.5 or elbow.visibility < 0.5:
            return None

        shoulder_x = shoulder.x * width
        shoulder_y = shoulder.y * height

        elbow_x = elbow.x * width
        elbow_y = elbow.y * height

        dx = elbow_x - shoulder_x
        dy = elbow_y - shoulder_y

        angle = np.degrees(np.arctan2(dx, dy))

        return angle

    def close(self):
        self.pose.close()