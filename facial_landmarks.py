import cv2
import mediapipe as mp


class FacialLandmarks:
    """
    Detects and draws 468/478 facial landmarks
    using MediaPipe Face Mesh.
    """

    def __init__(
        self,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):

        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )


    def process(self, frame):
        """
        Detect facial landmarks from a frame.

        Returns:
            face_landmarks if face detected
            None otherwise
        """

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = self.face_mesh.process(
            rgb_frame
        )

        if results.multi_face_landmarks:

            # Single-user system
            return results.multi_face_landmarks[0]

        return None


    def draw_landmarks(
        self,
        frame,
        face_landmarks
    ):
        """
        Draw complete face mesh and face contours.
        """

        # -------------------------------------------------
        # Face Mesh / Tessellation
        # -------------------------------------------------

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
            )
        )


        # -------------------------------------------------
        # Face Contours
        # -------------------------------------------------

        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=(
                self.mp_drawing_styles
                .get_default_face_mesh_contours_style()
            )
        )

        return frame


    def close(self):
        """
        Release MediaPipe resources.
        """

        self.face_mesh.close()