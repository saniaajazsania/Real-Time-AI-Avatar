
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import time

# =========================================================
# PHASE 1 IMPORTS
# =========================================================

from camera.webcam import start_webcam
from performance.fps import FPSCounter

# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

AVATAR_DIR = PROJECT_DIR / "avatar" / "assets" / "mie"

BODY_PATH = AVATAR_DIR / "body" / "body_mie.png"

HEAD_PATH = AVATAR_DIR / "head" / "head_neutral_mie.png"

HEAD_BLINK_PATH = AVATAR_DIR / "head" / "head_blink_mie.png"

HEAD_TALK_PATH = AVATAR_DIR / "head" / "head_talk_mie.png"

LEFT_ARM_DOWN_PATH = AVATAR_DIR / "arms" / "left_arm_down.png"

LEFT_ARM_UP_PATH = AVATAR_DIR / "arms" / "left_arm_up.png"

RIGHT_ARM_DOWN_PATH = AVATAR_DIR / "arms" / "right_arm_down.png"

RIGHT_ARM_UP_PATH = AVATAR_DIR / "arms" / "right_arm_up.png"

# =========================================================
# MIE CANVAS / AVATAR SETTINGS
# SAME VALUES AS avatar_complete_tracking.py
# =========================================================

CANVAS_WIDTH = 620
CANVAS_HEIGHT = 700

AVATAR_HEIGHT = 500

BOTTOM_MARGIN = 25

# =========================================================
# DISPLAY SIZE
# ONLY DISPLAY SIZE CHANGED
# =========================================================

AVATAR_DISPLAY_SCALE = 1.35

# =========================================================
# MIE HEAD MOVEMENT SETTINGS
# SAME VALUES
# =========================================================

MAX_HEAD_X = 15
MAX_HEAD_Y = 12
MAX_HEAD_ROTATION = 15

SMOOTHING = 0.25

# =========================================================
# MIE NECK PIVOT
# SAME VALUE
# =========================================================

NECK_PIVOT_RATIO = 0.88

# =========================================================
# BLINK SETTINGS
# SAME VALUES
# =========================================================

BLINK_THRESHOLD = 0.20

BLINK_MIN_TIME = 0.08
BLINK_MAX_TIME = 0.25

# =========================================================
# MOUTH / TALK SETTINGS
# SAME VALUES
# =========================================================

MOUTH_OPEN_THRESHOLD = 0.035

MOUTH_HOLD_TIME = 0.08

# =========================================================
# ARM SETTINGS
# SAME VALUES
# =========================================================

ARM_HEIGHT_RATIO = 0.28

LEFT_SHOULDER_X = 0.27
RIGHT_SHOULDER_X = 0.64

SHOULDER_Y = 0.15

# =========================================================
# ARM MOVEMENT
# SAME VALUE
# =========================================================

UP_ARM_MOVE = 35

# =========================================================
# ARM SMOOTHING
# SAME VALUE
# =========================================================

ARM_SMOOTHING = 0.20

# =========================================================
# ARM NATURAL SHOULDER MOVEMENT
# =========================================================

# Pivot arm ke upper area / shoulder ke qareeb rahega
# Is se arm body ke saath connected lagega

ELBOW_PIVOT_RATIO_Y = 0.10

# Natural rotation

ARM_SWING_ANGLE = 12

# =========================================================
# LOAD PNG
# =========================================================

def load_png(path):

    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    if len(image.shape) == 2:

        image = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGRA
        )

    elif image.shape[2] == 3:

        alpha = np.ones(
            (
                image.shape[0],
                image.shape[1],
                1
            ),
            dtype=np.uint8
        ) * 255

        image = np.concatenate(
            [image, alpha],
            axis=2
        )

    return image

# =========================================================
# REMOVE TRANSPARENT BORDER
# =========================================================

def crop_transparent(image):

    alpha = image[:, :, 3]

    points = cv2.findNonZero(alpha)

    if points is None:
        return image

    x, y, w, h = cv2.boundingRect(points)

    return image[
        y:y + h,
        x:x + w
    ]

# =========================================================
# RESIZE KEEP RATIO
# =========================================================

def resize_keep_ratio(image, target_height):

    h, w = image.shape[:2]

    scale = target_height / h

    new_width = max(1, int(w * scale))
    new_height = max(1, int(h * scale))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

# =========================================================
# ALPHA COMPOSITING
# =========================================================

def overlay_rgba(background, foreground, x, y):

    fg_height, fg_width = foreground.shape[:2]

    bg_height, bg_width = background.shape[:2]

    x1 = max(x, 0)
    y1 = max(y, 0)

    x2 = min(x + fg_width, bg_width)
    y2 = min(y + fg_height, bg_height)

    if x1 >= x2 or y1 >= y2:
        return

    fx1 = x1 - x
    fy1 = y1 - y

    fx2 = fx1 + (x2 - x1)
    fy2 = fy1 + (y2 - y1)

    fg = foreground[
        fy1:fy2,
        fx1:fx2
    ]

    alpha = (
        fg[:, :, 3:4].astype(np.float32)
        / 255.0
    )

    fg_rgb = fg[:, :, :3].astype(np.float32)

    bg = background[
        y1:y2,
        x1:x2
    ].astype(np.float32)

    result = (
        fg_rgb * alpha
        +
        bg * (1.0 - alpha)
    )

    background[
        y1:y2,
        x1:x2
    ] = result.astype(np.uint8)

# =========================================================
# ROTATE HEAD
# SAME LOGIC
# =========================================================

def rotate_image(image, angle):

    h, w = image.shape[:2]

    center = (
        w // 2,
        int(h * NECK_PIVOT_RATIO)
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated

# =========================================================
# ROTATE ARM FROM SHOULDER AREA
# =========================================================

def rotate_arm(image, angle):

    if angle == 0:
        return image

    h, w = image.shape[:2]

    # Pivot arm ke upper / shoulder area ke qareeb
    center = (
        w // 2,
        int(h * ELBOW_PIVOT_RATIO_Y)
    )

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

    return rotated

# =========================================================
# POST PROCESSING
# =========================================================

def post_process_frame(current_frame, previous_frame=None):

    processed = current_frame.copy()

    # -----------------------------------------------------
    # FRAME SMOOTHING
    # -----------------------------------------------------

    if previous_frame is not None:

        if previous_frame.shape == processed.shape:

            processed = cv2.addWeighted(
                processed,
                0.85,
                previous_frame,
                0.15,
                0
            )

    # -----------------------------------------------------
    # LIGHT SHARPENING
    # -----------------------------------------------------

    blur = cv2.GaussianBlur(
        processed,
        (0, 0),
        1.0
    )

    processed = cv2.addWeighted(
        processed,
        1.10,
        blur,
        -0.10,
        0
    )

    return processed

# =========================================================
# MIE AVATAR SETUP
# =========================================================

def create_mie_avatar():

    body = load_png(BODY_PATH)

    head = load_png(HEAD_PATH)

    head_blink = load_png(HEAD_BLINK_PATH)

    head_talk = load_png(HEAD_TALK_PATH)

    left_arm_down = load_png(LEFT_ARM_DOWN_PATH)

    left_arm_up = load_png(LEFT_ARM_UP_PATH)

    right_arm_down = load_png(RIGHT_ARM_DOWN_PATH)

    right_arm_up = load_png(RIGHT_ARM_UP_PATH)

    # =====================================================
    # CROP
    # =====================================================

    body = crop_transparent(body)

    head = crop_transparent(head)

    head_blink = crop_transparent(head_blink)

    head_talk = crop_transparent(head_talk)

    left_arm_down = crop_transparent(left_arm_down)

    left_arm_up = crop_transparent(left_arm_up)

    right_arm_down = crop_transparent(right_arm_down)

    right_arm_up = crop_transparent(right_arm_up)

    # =====================================================
    # BODY SIZE
    # =====================================================

    body = resize_keep_ratio(
        body,
        AVATAR_HEIGHT
    )

    body_height, body_width = body.shape[:2]

    # =====================================================
    # HEAD SIZE
    # =====================================================

    head_target_height = int(
        body.shape[0] * 0.47
    )

    head = resize_keep_ratio(
        head,
        head_target_height
    )

    head_blink = resize_keep_ratio(
        head_blink,
        head_target_height
    )

    head_talk = resize_keep_ratio(
        head_talk,
        head_target_height
    )

    # =====================================================
    # ARM SIZE
    # =====================================================

    arm_target_height = int(
        body_height * ARM_HEIGHT_RATIO
    )

    left_arm_down = resize_keep_ratio(
        left_arm_down,
        arm_target_height
    )

    left_arm_up = resize_keep_ratio(
        left_arm_up,
        arm_target_height
    )

    right_arm_down = resize_keep_ratio(
        right_arm_down,
        arm_target_height
    )

    right_arm_up = resize_keep_ratio(
        right_arm_up,
        arm_target_height
    )

    # =====================================================
    # BODY POSITION
    # =====================================================

    body_x = (
        CANVAS_WIDTH - body_width
    ) // 2

    body_y = (
        CANVAS_HEIGHT
        - body_height
        - BOTTOM_MARGIN
    )

    # =====================================================
    # NECK ANCHOR
    # =====================================================

    neck_x = (
        body_x
        + int(body_width * 0.50)
    )

    neck_y = (
        body_y
        + int(body_height * 0.18)
    )

    # =====================================================
    # HEAD BASE POSITION
    # =====================================================

    head_height, head_width = head.shape[:2]

    head_base_x = (
        neck_x
        - head_width // 2
    )

    head_base_y = (
        neck_y
        - int(head_height * 0.70)
    )

    # =====================================================
    # SAFETY CLAMP — HEAD
    # =====================================================

    TOP_MARGIN = 20

    if head_base_y < TOP_MARGIN:

        shift_down = TOP_MARGIN - head_base_y

        body_y += shift_down
        neck_y += shift_down
        head_base_y += shift_down

    # =====================================================
    # SAFETY CLAMP — FEET
    # =====================================================

    bottom_of_body = (
        body_y + body_height
    )

    if bottom_of_body > CANVAS_HEIGHT - 5:

        overflow = (
            bottom_of_body
            - (CANVAS_HEIGHT - 5)
        )

        body_y -= overflow
        neck_y -= overflow
        head_base_y -= overflow

    # =====================================================
    # ARM SHOULDER ANCHORS
    # =====================================================

    left_shoulder_x = int(
        body_x
        + body_width * LEFT_SHOULDER_X
    )

    right_shoulder_x = int(
        body_x
        + body_width * RIGHT_SHOULDER_X
    )

    shoulder_y = int(
        body_y
        + body_height * SHOULDER_Y
    )

    # =====================================================
    # ARM DOWN POSITIONS
    # =====================================================

    left_arm_x = int(
        left_shoulder_x
        - left_arm_down.shape[1] * 0.50
    )

    left_arm_y = int(
        shoulder_y
        - left_arm_down.shape[0] * 0.02
    )

    right_arm_x = int(
        right_shoulder_x
        - right_arm_down.shape[1] * 0.50
    )

    right_arm_y = int(
        shoulder_y
        - right_arm_down.shape[0] * 0.02
    )

    return {

        "body": body,

        "head": head,

        "head_blink": head_blink,

        "head_talk": head_talk,

        "left_arm_down": left_arm_down,

        "left_arm_up": left_arm_up,

        "right_arm_down": right_arm_down,

        "right_arm_up": right_arm_up,

        "body_x": body_x,

        "body_y": body_y,

        "head_base_x": head_base_x,

        "head_base_y": head_base_y,

        "left_arm_x": left_arm_x,

        "left_arm_y": left_arm_y,

        "right_arm_x": right_arm_x,

        "right_arm_y": right_arm_y
    }

# =========================================================
# MEDIAPIPE TRACKING MODELS
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

mp_pose = mp.solutions.pose

# =========================================================
# MAIN
# =========================================================

def main():

    # =====================================================
    # WEBCAM + FPS
    # =====================================================

    cap = start_webcam()

    fps_counter = FPSCounter()

    # =====================================================
    # LOAD AVATAR
    # =====================================================

    print("Loading Mie complete avatar...")

    mie = create_mie_avatar()

    # =====================================================
    # FACE MESH
    # =====================================================

    face_mesh = mp_face_mesh.FaceMesh(

        static_image_mode=False,

        max_num_faces=1,

        refine_landmarks=True,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5
    )

    # =====================================================
    # POSE
    # =====================================================

    pose = mp_pose.Pose(

        static_image_mode=False,

        model_complexity=1,

        smooth_landmarks=True,

        min_detection_confidence=0.5,

        min_tracking_confidence=0.5
    )

    # =====================================================
    # AVATAR STATES
    # =====================================================

    smooth_x = 0.0
    smooth_y = 0.0
    smooth_angle = 0.0

    left_state = 0.0
    right_state = 0.0

    last_blink_time = 0.0
    is_blinking = False
    blink_start_time = 0.0

    last_mouth_change = 0.0
    is_talking = False

    # =====================================================
    # POST PROCESSING STATE
    # =====================================================

    previous_avatar_frame = None

    # =====================================================
    # FPS DISPLAY
    # =====================================================

    fps_start_time = time.time()

    fps_frame_count = 0

    display_fps = 0.0

    # =====================================================
    # MAIN LOOP
    # =====================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # =================================================
        # MIRROR WEBCAM
        # =================================================

        frame = cv2.flip(
            frame,
            1
        )

        phase1_frame = frame.copy()

        # =================================================
        # FPS
        # =================================================

        try:
            fps_counter.update()
        except Exception:
            pass

        fps_frame_count += 1

        fps_elapsed = (
            time.time()
            - fps_start_time
        )

        if fps_elapsed >= 1.0:

            display_fps = (
                fps_frame_count
                / fps_elapsed
            )

            fps_frame_count = 0

            fps_start_time = time.time()

        # =================================================
        # CONVERT TO RGB
        # =================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # =================================================
        # FACE + POSE PROCESSING
        # =================================================

        face_results = face_mesh.process(rgb)

        pose_results = pose.process(rgb)

        # =================================================
        # DEFAULT HEAD VALUES
        # =================================================

        target_x = 0.0
        target_y = 0.0
        target_angle = 0.0

        current_time = time.time()

        # =================================================
        # FACE DETECTED
        # =================================================

        if face_results.multi_face_landmarks:

            landmarks = (
                face_results
                .multi_face_landmarks[0]
                .landmark
            )

            # =================================================
            # HEAD TRACKING
            # =================================================

            nose = landmarks[1]

            left_eye = landmarks[33]

            right_eye = landmarks[263]

            face_x = nose.x

            normalized_x = (
                face_x - 0.5
            ) * 2.0

            normalized_x = np.clip(
                normalized_x,
                -1.0,
                1.0
            )

            target_x = (
                normalized_x
                * MAX_HEAD_X
            )

            face_y = nose.y

            normalized_y = (
                face_y - 0.5
            ) * 2.0

            normalized_y = np.clip(
                normalized_y,
                -1.0,
                1.0
            )

            target_y = (
                normalized_y
                * MAX_HEAD_Y
            )

            # =================================================
            # HEAD ROTATION
            # =================================================

            dx = (
                right_eye.x
                - left_eye.x
            )

            dy = (
                right_eye.y
                - left_eye.y
            )

            angle = np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )

            target_angle = np.clip(
                -angle,
                -MAX_HEAD_ROTATION,
                MAX_HEAD_ROTATION
            )

            # =================================================
            # BLINK
            # =================================================

            blink_left_outer = landmarks[33]
            blink_left_inner = landmarks[133]

            blink_left_top = landmarks[159]
            blink_left_bottom = landmarks[145]

            blink_right_outer = landmarks[263]
            blink_right_inner = landmarks[362]

            blink_right_top = landmarks[386]
            blink_right_bottom = landmarks[374]

            left_eye_width = abs(
                blink_left_inner.x
                - blink_left_outer.x
            )

            left_eye_height = abs(
                blink_left_bottom.y
                - blink_left_top.y
            )

            right_eye_width = abs(
                blink_right_inner.x
                - blink_right_outer.x
            )

            right_eye_height = abs(
                blink_right_bottom.y
                - blink_right_top.y
            )

            if (
                left_eye_width > 0
                and right_eye_width > 0
            ):

                left_ratio = (
                    left_eye_height
                    / left_eye_width
                )

                right_ratio = (
                    right_eye_height
                    / right_eye_width
                )

                eye_ratio = (
                    left_ratio
                    + right_ratio
                ) / 2.0

            else:

                eye_ratio = 1.0

            if eye_ratio < BLINK_THRESHOLD:

                if (
                    not is_blinking
                    and current_time
                    - last_blink_time
                    > 0.30
                ):

                    is_blinking = True

                    blink_start_time = current_time

                    last_blink_time = current_time

            else:

                if is_blinking:

                    blink_duration = (
                        current_time
                        - blink_start_time
                    )

                    if blink_duration >= BLINK_MIN_TIME:

                        is_blinking = False

            # =================================================
            # MOUTH / TALK
            # =================================================

            upper_lip = landmarks[13]
            lower_lip = landmarks[14]

            mouth_left = landmarks[61]
            mouth_right = landmarks[291]

            mouth_height = abs(
                lower_lip.y
                - upper_lip.y
            )

            mouth_width = abs(
                mouth_right.x
                - mouth_left.x
            )

            if mouth_width > 0:

                mouth_ratio = (
                    mouth_height
                    / mouth_width
                )

            else:

                mouth_ratio = 0.0

            if mouth_ratio > MOUTH_OPEN_THRESHOLD:

                is_talking = True

                last_mouth_change = current_time

            else:

                if (
                    current_time
                    - last_mouth_change
                    > MOUTH_HOLD_TIME
                ):

                    is_talking = False

        # =================================================
        # SMOOTH HEAD
        # SAME VALUES
        # =================================================

        smooth_x += (
            target_x
            - smooth_x
        ) * SMOOTHING

        smooth_y += (
            target_y
            - smooth_y
        ) * SMOOTHING

        smooth_angle += (
            target_angle
            - smooth_angle
        ) * SMOOTHING

        # =================================================
        # ARM DEFAULT
        # =================================================

        target_left = 0.0
        target_right = 0.0

        # =================================================
        # POSE DETECTED
        # =================================================

        if pose_results.pose_landmarks:

            pose_landmarks = (
                pose_results
                .pose_landmarks
                .landmark
            )

            # =================================================
            # YOUR LEFT ARM
            # MEDIAPIPE RIGHT
            # =================================================

            your_left_shoulder = pose_landmarks[
                mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            ]

            your_left_wrist = pose_landmarks[
                mp_pose.PoseLandmark.RIGHT_WRIST.value
            ]

            # =================================================
            # YOUR RIGHT ARM
            # MEDIAPIPE LEFT
            # =================================================

            your_right_shoulder = pose_landmarks[
                mp_pose.PoseLandmark.LEFT_SHOULDER.value
            ]

            your_right_wrist = pose_landmarks[
                mp_pose.PoseLandmark.LEFT_WRIST.value
            ]

            # =================================================
            # LEFT ARM
            # =================================================

            if (
                your_left_wrist.y
                < your_left_shoulder.y - 0.08
            ):

                target_left = 1.0

            else:

                target_left = 0.0

            # =================================================
            # RIGHT ARM
            # =================================================

            if (
                your_right_wrist.y
                < your_right_shoulder.y - 0.08
            ):

                target_right = 1.0

            else:

                target_right = 0.0

        # =================================================
        # SMOOTH ARM STATES
        # =================================================

        left_state += (
            target_left
            - left_state
        ) * ARM_SMOOTHING

        right_state += (
            target_right
            - right_state
        ) * ARM_SMOOTHING

        # =================================================
        # ARM STATUS
        # =================================================

        left_arm_up_status = (
            left_state > 0.5
        )

        right_arm_up_status = (
            right_state > 0.5
        )

        # =================================================
        # SELECT LEFT ARM
        # NATURAL SHOULDER ROTATION
        # =================================================

        if left_arm_up_status:

            left_angle = (
                ARM_SWING_ANGLE
                * left_state
            )

            current_left_arm = rotate_arm(
                mie["left_arm_up"],
                left_angle
            )

            current_left_x = mie[
                "left_arm_x"
            ]

            current_left_y = (
                mie["left_arm_y"]
                - int(
                    UP_ARM_MOVE
                    * left_state
                )
            )

        else:

            current_left_arm = mie[
                "left_arm_down"
            ]

            current_left_x = mie[
                "left_arm_x"
            ]

            current_left_y = mie[
                "left_arm_y"
            ]

        # =================================================
        # SELECT RIGHT ARM
        # NATURAL SHOULDER ROTATION
        # =================================================

        if right_arm_up_status:

            right_angle = (
                -ARM_SWING_ANGLE
                * right_state
            )

            current_right_arm = rotate_arm(
                mie["right_arm_up"],
                right_angle
            )

            current_right_x = mie[
                "right_arm_x"
            ]

            current_right_y = (
                mie["right_arm_y"]
                - int(
                    UP_ARM_MOVE
                    * right_state
                )
            )

        else:

            current_right_arm = mie[
                "right_arm_down"
            ]

            current_right_x = mie[
                "right_arm_x"
            ]

            current_right_y = mie[
                "right_arm_y"
            ]

        # =================================================
        # CREATE MIE CANVAS
        # =================================================

        canvas = np.ones(
            (
                CANVAS_HEIGHT,
                CANVAS_WIDTH,
                3
            ),
            dtype=np.uint8
        ) * 255

        # =================================================
        # BODY
        # =================================================

        overlay_rgba(
            canvas,
            mie["body"],
            mie["body_x"],
            mie["body_y"]
        )

        # =================================================
        # SELECT HEAD
        # BLINK > TALK > NORMAL
        # =================================================

        if is_blinking:

            selected_head = mie[
                "head_blink"
            ]

        elif is_talking:

            selected_head = mie[
                "head_talk"
            ]

        else:

            selected_head = mie[
                "head"
            ]

        # =================================================
        # ROTATE HEAD
        # =================================================

        transformed_head = rotate_image(
            selected_head,
            smooth_angle
        )

        # =================================================
        # HEAD POSITION
        # =================================================

        current_head_x = int(
            mie["head_base_x"]
            + smooth_x
        )

        current_head_y = int(
            mie["head_base_y"]
            + smooth_y
        )

        # =================================================
        # DRAW HEAD
        # =================================================

        overlay_rgba(
            canvas,
            transformed_head,
            current_head_x,
            current_head_y
        )

        # =================================================
        # LEFT ARM
        # BODY KE FRONT PAR
        # =================================================

        overlay_rgba(
            canvas,
            current_left_arm,
            current_left_x,
            current_left_y
        )

        # =================================================
        # RIGHT ARM
        # BODY KE FRONT PAR
        # =================================================

        overlay_rgba(
            canvas,
            current_right_arm,
            current_right_x,
            current_right_y
        )

        # =================================================
        # POST PROCESSING
        # =================================================

        processed_avatar = post_process_frame(
            canvas,
            previous_avatar_frame
        )

        previous_avatar_frame = processed_avatar.copy()

        # =================================================
        # MIE STATUS
        # =================================================

        cv2.putText(
            processed_avatar,
            "MIE AVATAR",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        # =================================================
        # RESIZE MIE
        # =================================================

        webcam_h, webcam_w = frame.shape[:2]

        avatar_h, avatar_w = processed_avatar.shape[:2]

        scale = min(
            webcam_w / avatar_w,
            webcam_h / avatar_h
        )

        scale = (
            scale
            * AVATAR_DISPLAY_SCALE
        )

        new_avatar_w = max(
            1,
            int(avatar_w * scale)
        )

        new_avatar_h = max(
            1,
            int(avatar_h * scale)
        )

        avatar_display = cv2.resize(
            processed_avatar,
            (
                new_avatar_w,
                new_avatar_h
            ),
            interpolation=cv2.INTER_AREA
        )

        # =================================================
        # WHITE AVATAR PANEL
        # =================================================

        avatar_panel = np.ones(
            (
                webcam_h,
                webcam_w,
                3
            ),
            dtype=np.uint8
        ) * 255

        # =================================================
        # CENTER AVATAR
        # =================================================

        x_offset = (
            webcam_w
            - new_avatar_w
        ) // 2

        y_offset = (
            webcam_h
            - new_avatar_h
        ) // 2

        src_x1 = max(
            0,
            -x_offset
        )

        src_y1 = max(
            0,
            -y_offset
        )

        src_x2 = min(
            new_avatar_w,
            webcam_w - x_offset
        )

        src_y2 = min(
            new_avatar_h,
            webcam_h - y_offset
        )

        dst_x1 = max(
            0,
            x_offset
        )

        dst_y1 = max(
            0,
            y_offset
        )

        dst_x2 = dst_x1 + (
            src_x2 - src_x1
        )

        dst_y2 = dst_y1 + (
            src_y2 - src_y1
        )

        if (
            src_x2 > src_x1
            and src_y2 > src_y1
        ):

            avatar_panel[
                dst_y1:dst_y2,
                dst_x1:dst_x2
            ] = avatar_display[
                src_y1:src_y2,
                src_x1:src_x2
            ]

        avatar_display = avatar_panel

        # =================================================
        # REAL PERSON TITLE
        # =================================================

        cv2.putText(
            phase1_frame,
            "REAL PERSON",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # =================================================
        # STATUS PANEL
        # =================================================

        status_x = 20
        status_y = 65
        line_gap = 27

        # FPS

        cv2.putText(
            phase1_frame,
            f"FPS: {display_fps:.1f}",
            (
                status_x,
                status_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # EXPRESSION

        if is_blinking:

            expression_status = "Blink"

        elif is_talking:

            expression_status = "Talk"

        else:

            expression_status = "Neutral"

        cv2.putText(
            phase1_frame,
            f"Expression: {expression_status}",
            (
                status_x,
                status_y + line_gap
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # EYES

        eyes_status = (
            "Closed"
            if is_blinking
            else "Open"
        )

        cv2.putText(
            phase1_frame,
            f"Eyes: {eyes_status}",
            (
                status_x,
                status_y + line_gap * 2
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # MOUTH

        mouth_status = (
            "Open"
            if is_talking
            else "Closed"
        )

        cv2.putText(
            phase1_frame,
            f"Mouth: {mouth_status}",
            (
                status_x,
                status_y + line_gap * 3
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # LEFT ARM

        cv2.putText(
            phase1_frame,
            "Left Arm: "
            + (
                "UP"
                if left_arm_up_status
                else "DOWN"
            ),
            (
                status_x,
                status_y + line_gap * 4
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # RIGHT ARM

        cv2.putText(
            phase1_frame,
            "Right Arm: "
            + (
                "UP"
                if right_arm_up_status
                else "DOWN"
            ),
            (
                status_x,
                status_y + line_gap * 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # =================================================
        # SIDE BY SIDE
        # =================================================

        combined = np.hstack(
            (
                phase1_frame,
                avatar_display
            )
        )

        # =================================================
        # DISPLAY
        # =================================================

        cv2.imshow(
            "Real Person + Mie Avatar",
            combined
        )

        # =================================================
        # QUIT
        # =================================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    # =====================================================
    # CLEANUP
    # =====================================================

    cap.release()

    face_mesh.close()

    pose.close()

    cv2.destroyAllWindows()

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
