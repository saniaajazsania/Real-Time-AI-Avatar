
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import time


# =========================================================
# PATHS
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
# CANVAS / AVATAR SETTINGS
# =========================================================

CANVAS_WIDTH = 620

CANVAS_HEIGHT = 700

AVATAR_HEIGHT = 500

BOTTOM_MARGIN = 25


# =========================================================
# HEAD MOVEMENT SETTINGS
# =========================================================

MAX_HEAD_X = 15

MAX_HEAD_Y = 12

MAX_HEAD_ROTATION = 15

SMOOTHING = 0.25


# =========================================================
# NECK PIVOT SETTING
# =========================================================

NECK_PIVOT_RATIO = 0.88


# =========================================================
# BLINK SETTINGS
# =========================================================

BLINK_THRESHOLD = 0.20

BLINK_MIN_TIME = 0.08

BLINK_MAX_TIME = 0.25

last_blink_time = 0

is_blinking = False

blink_start_time = 0


# =========================================================
# MOUTH / TALK SETTINGS
# =========================================================

MOUTH_OPEN_THRESHOLD = 0.035

MOUTH_HOLD_TIME = 0.08

last_mouth_change = 0

is_talking = False


# =========================================================
# ARM SETTINGS
# =========================================================

ARM_HEIGHT_RATIO = 0.28

LEFT_SHOULDER_X = 0.27

RIGHT_SHOULDER_X = 0.64

SHOULDER_Y = 0.15


# =========================================================
# UP ARM MOVEMENT
# =========================================================

UP_ARM_MOVE = 18


# =========================================================
# ARM SMOOTHING
# =========================================================

ARM_SMOOTHING = 0.20


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

def resize_keep_ratio(
    image,
    target_height
):

    h, w = image.shape[:2]

    scale = target_height / h

    new_width = int(w * scale)

    new_height = int(h * scale)

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


# =========================================================
# ALPHA COMPOSITING
# =========================================================

def overlay_rgba(
    background,
    foreground,
    x,
    y
):

    fg_height, fg_width = foreground.shape[:2]

    bg_height, bg_width = background.shape[:2]

    x1 = max(x, 0)

    y1 = max(y, 0)

    x2 = min(
        x + fg_width,
        bg_width
    )

    y2 = min(
        y + fg_height,
        bg_height
    )

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
        fg[:, :, 3:4]
        .astype(np.float32)
        / 255.0
    )

    fg_rgb = fg[:, :, :3].astype(
        np.float32
    )

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
# =========================================================

def rotate_image(
    image,
    angle
):

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
# LOAD AVATAR
# =========================================================

print("Loading Mie complete avatar...")


body = load_png(BODY_PATH)

head = load_png(HEAD_PATH)

head_blink = load_png(HEAD_BLINK_PATH)

head_talk = load_png(HEAD_TALK_PATH)


left_arm_down = load_png(
    LEFT_ARM_DOWN_PATH
)

left_arm_up = load_png(
    LEFT_ARM_UP_PATH
)

right_arm_down = load_png(
    RIGHT_ARM_DOWN_PATH
)

right_arm_up = load_png(
    RIGHT_ARM_UP_PATH
)


# =========================================================
# CROP
# =========================================================

body = crop_transparent(body)

head = crop_transparent(head)

head_blink = crop_transparent(head_blink)

head_talk = crop_transparent(head_talk)


left_arm_down = crop_transparent(
    left_arm_down
)

left_arm_up = crop_transparent(
    left_arm_up
)

right_arm_down = crop_transparent(
    right_arm_down
)

right_arm_up = crop_transparent(
    right_arm_up
)


# =========================================================
# BODY SIZE
# =========================================================

body = resize_keep_ratio(
    body,
    AVATAR_HEIGHT
)

body_height, body_width = body.shape[:2]


# =========================================================
# HEAD SIZE
# =========================================================

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


# =========================================================
# ARM SIZE
# =========================================================

arm_target_height = int(
    body_height
    *
    ARM_HEIGHT_RATIO
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


# =========================================================
# BODY POSITION
# =========================================================

body_x = (
    CANVAS_WIDTH
    -
    body_width
) // 2


body_y = (
    CANVAS_HEIGHT
    -
    body_height
    -
    BOTTOM_MARGIN
)


# =========================================================
# NECK ANCHOR
# =========================================================

neck_x = (
    body_x
    +
    int(body_width * 0.50)
)

neck_y = (
    body_y
    +
    int(body_height * 0.18)
)


# =========================================================
# HEAD BASE POSITION
# =========================================================

head_height, head_width = head.shape[:2]


head_base_x = (
    neck_x
    -
    head_width // 2
)


head_base_y = (
    neck_y
    -
    int(head_height * 0.70)
)


# =========================================================
# SAFETY CLAMP — HEAD
# =========================================================

TOP_MARGIN = 20


if head_base_y < TOP_MARGIN:

    shift_down = (
        TOP_MARGIN
        -
        head_base_y
    )

    body_y += shift_down

    neck_y += shift_down

    head_base_y += shift_down


# =========================================================
# SAFETY CLAMP — FEET
# =========================================================

bottom_of_body = (
    body_y
    +
    body_height
)


if bottom_of_body > CANVAS_HEIGHT - 5:

    overflow = (
        bottom_of_body
        -
        (CANVAS_HEIGHT - 5)
    )

    body_y -= overflow

    neck_y -= overflow

    head_base_y -= overflow


# =========================================================
# ARM SHOULDER ANCHORS
# =========================================================

left_shoulder_x = int(
    body_x
    +
    body_width
    *
    LEFT_SHOULDER_X
)


right_shoulder_x = int(
    body_x
    +
    body_width
    *
    RIGHT_SHOULDER_X
)


shoulder_y = int(
    body_y
    +
    body_height
    *
    SHOULDER_Y
)


# =========================================================
# ARM DOWN POSITIONS
# =========================================================

left_arm_x = int(
    left_shoulder_x
    -
    left_arm_down.shape[1]
    *
    0.50
)


left_arm_y = int(
    shoulder_y
    -
    left_arm_down.shape[0]
    *
    0.02
)


right_arm_x = int(
    right_shoulder_x
    -
    right_arm_down.shape[1]
    *
    0.50
)


right_arm_y = int(
    shoulder_y
    -
    right_arm_down.shape[0]
    *
    0.02
)


# =========================================================
# MEDIAPIPE FACE MESH
# =========================================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(

    static_image_mode=False,

    max_num_faces=1,

    refine_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# =========================================================
# MEDIAPIPE POSE
# =========================================================

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(

    static_image_mode=False,

    model_complexity=1,

    smooth_landmarks=True,

    min_detection_confidence=0.5,

    min_tracking_confidence=0.5
)


# =========================================================
# SMOOTHING VALUES
# =========================================================

smooth_x = 0.0

smooth_y = 0.0

smooth_angle = 0.0


left_state = 0.0

right_state = 0.0


# =========================================================
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("====================================")
print("       MIE COMPLETE TRACKING")
print("====================================")
print()
print("HEAD  : LEFT / RIGHT / TILT")
print("EYES  : BLINK")
print("MOUTH : TALK")
print("ARMS  : LEFT / RIGHT UP-DOWN")
print()
print("Press Q to quit")
print()
print("====================================")


# =========================================================
# MAIN LOOP
# =========================================================

while True:

    ret, frame = cap.read()

    if not ret:

        break


    # =====================================================
    # MIRROR WEBCAM
    # =====================================================

    frame = cv2.flip(
        frame,
        1
    )


    # =====================================================
    # RGB FRAME
    # =====================================================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # FACE + POSE
    # =====================================================

    face_results = face_mesh.process(
        rgb
    )

    pose_results = pose.process(
        rgb
    )


    # =====================================================
    # DEFAULT HEAD VALUES
    # =====================================================

    target_x = 0.0

    target_y = 0.0

    target_angle = 0.0


    # =====================================================
    # FACE DETECTED
    # =====================================================

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
            *
            MAX_HEAD_X
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
            *
            MAX_HEAD_Y
        )


        # =================================================
        # HEAD ROTATION
        # =================================================

        dx = (
            right_eye.x
            -
            left_eye.x
        )


        dy = (
            right_eye.y
            -
            left_eye.y
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
        # EYE BLINK DETECTION
        # =================================================

        blink_left_outer = landmarks[33]

        blink_left_inner = landmarks[133]

        blink_left_top = landmarks[159]

        blink_left_bottom = landmarks[145]


        blink_right_outer = landmarks[263]

        blink_right_inner = landmarks[362]

        blink_right_top = landmarks[386]

        blink_right_bottom = landmarks[374]


        # =================================================
        # LEFT EYE RATIO
        # =================================================

        left_eye_width = abs(
            blink_left_inner.x
            -
            blink_left_outer.x
        )


        left_eye_height = abs(
            blink_left_bottom.y
            -
            blink_left_top.y
        )


        # =================================================
        # RIGHT EYE RATIO
        # =================================================

        right_eye_width = abs(
            blink_right_inner.x
            -
            blink_right_outer.x
        )


        right_eye_height = abs(
            blink_right_bottom.y
            -
            blink_right_top.y
        )


        # =================================================
        # AVERAGE EYE RATIO
        # =================================================

        if (
            left_eye_width > 0
            and
            right_eye_width > 0
        ):

            left_ratio = (
                left_eye_height
                /
                left_eye_width
            )


            right_ratio = (
                right_eye_height
                /
                right_eye_width
            )


            eye_ratio = (
                left_ratio
                +
                right_ratio
            ) / 2.0

        else:

            eye_ratio = 1.0


        # =================================================
        # CURRENT TIME
        # =================================================

        current_time = time.time()


        # =================================================
        # BLINK STATE
        # =================================================

        if eye_ratio < BLINK_THRESHOLD:

            if (
                not is_blinking
                and
                current_time
                -
                last_blink_time
                >
                0.30
            ):

                is_blinking = True

                blink_start_time = (
                    current_time
                )

                last_blink_time = (
                    current_time
                )


        else:

            if is_blinking:

                blink_duration = (
                    current_time
                    -
                    blink_start_time
                )


                if (
                    blink_duration
                    >=
                    BLINK_MIN_TIME
                ):

                    is_blinking = False


        # =================================================
        # MOUTH / TALK DETECTION
        # =================================================

        upper_lip = landmarks[13]

        lower_lip = landmarks[14]

        mouth_left = landmarks[61]

        mouth_right = landmarks[291]


        mouth_height = abs(
            lower_lip.y
            -
            upper_lip.y
        )


        mouth_width = abs(
            mouth_right.x
            -
            mouth_left.x
        )


        if mouth_width > 0:

            mouth_ratio = (
                mouth_height
                /
                mouth_width
            )

        else:

            mouth_ratio = 0.0


        # =================================================
        # TALKING STATE
        # =================================================

        if mouth_ratio > MOUTH_OPEN_THRESHOLD:

            is_talking = True

            last_mouth_change = (
                current_time
            )

        else:

            if (
                current_time
                -
                last_mouth_change
                >
                MOUTH_HOLD_TIME
            ):

                is_talking = False


    # =====================================================
    # SMOOTH HEAD
    # =====================================================

    smooth_x += (
        target_x
        -
        smooth_x
    ) * SMOOTHING


    smooth_y += (
        target_y
        -
        smooth_y
    ) * SMOOTHING


    smooth_angle += (
        target_angle
        -
        smooth_angle
    ) * SMOOTHING


    # =====================================================
    # ARM DEFAULT
    # =====================================================

    target_left = 0.0

    target_right = 0.0


    # =====================================================
    # POSE DETECTED
    # =====================================================

    if pose_results.pose_landmarks:

        pose_landmarks = (
            pose_results
            .pose_landmarks
            .landmark
        )


        # =================================================
        # YOUR LEFT ARM
        # MediaPipe RIGHT
        # =================================================

        your_left_shoulder = pose_landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]


        your_left_wrist = pose_landmarks[
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]


        # =================================================
        # YOUR RIGHT ARM
        # MediaPipe LEFT
        # =================================================

        your_right_shoulder = pose_landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]


        your_right_wrist = pose_landmarks[
            mp_pose.PoseLandmark.LEFT_WRIST
        ]


        # =================================================
        # YOUR LEFT ARM
        # =================================================

        if (
            your_left_wrist.y
            <
            your_left_shoulder.y
            -
            0.08
        ):

            target_left = 1.0

        else:

            target_left = 0.0


        # =================================================
        # YOUR RIGHT ARM
        # =================================================

        if (
            your_right_wrist.y
            <
            your_right_shoulder.y
            -
            0.08
        ):

            target_right = 1.0

        else:

            target_right = 0.0


    # =====================================================
    # SMOOTH ARM STATES
    # =====================================================

    left_state += (
        target_left
        -
        left_state
    ) * ARM_SMOOTHING


    right_state += (
        target_right
        -
        right_state
    ) * ARM_SMOOTHING


    # =====================================================
    # SELECT LEFT ARM
    # =====================================================

    if left_state > 0.5:

        current_left_arm = left_arm_up

        current_left_x = left_arm_x

        current_left_y = (
            left_arm_y
            -
            UP_ARM_MOVE
        )

    else:

        current_left_arm = left_arm_down

        current_left_x = left_arm_x

        current_left_y = left_arm_y


    # =====================================================
    # SELECT RIGHT ARM
    # =====================================================

    if right_state > 0.5:

        current_right_arm = right_arm_up

        current_right_x = right_arm_x

        current_right_y = (
            right_arm_y
            -
            UP_ARM_MOVE
        )

    else:

        current_right_arm = right_arm_down

        current_right_x = right_arm_x

        current_right_y = right_arm_y


    # =====================================================
    # CREATE CANVAS
    # =====================================================

    canvas = np.ones(
        (
            CANVAS_HEIGHT,
            CANVAS_WIDTH,
            3
        ),
        dtype=np.uint8
    ) * 255


    # =====================================================
    # DRAW BODY
    # =====================================================

    overlay_rgba(
        canvas,
        body,
        body_x,
        body_y
    )


    # =====================================================
    # SELECT HEAD
    #
    # BLINK > TALK > NORMAL
    # =====================================================

    if is_blinking:

        selected_head = head_blink

    elif is_talking:

        selected_head = head_talk

    else:

        selected_head = head


    # =====================================================
    # ROTATE HEAD
    # =====================================================

    transformed_head = rotate_image(
        selected_head,
        smooth_angle
    )


    # =====================================================
    # HEAD POSITION
    # =====================================================

    current_head_x = int(
        head_base_x
        +
        smooth_x
    )


    current_head_y = int(
        head_base_y
        +
        smooth_y
    )


    # =====================================================
    # DRAW HEAD
    # =====================================================

    overlay_rgba(
        canvas,
        transformed_head,
        current_head_x,
        current_head_y
    )


    # =====================================================
    # DRAW LEFT ARM
    # =====================================================

    overlay_rgba(
        canvas,
        current_left_arm,
        current_left_x,
        current_left_y
    )


    # =====================================================
    # DRAW RIGHT ARM
    # =====================================================

    overlay_rgba(
        canvas,
        current_right_arm,
        current_right_x,
        current_right_y
    )


    # =====================================================
    # STATUS
    # =====================================================

    cv2.putText(
        canvas,
        "MIE AVATAR",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )


    # =====================================================
    # DISPLAY
    # =====================================================

    display = np.ascontiguousarray(
        canvas
    )


    cv2.imshow(
        "Mie Avatar - Complete Tracking",
        display
    )


    # =====================================================
    # QUIT
    # =====================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# =========================================================
# CLEANUP
# =========================================================

cap.release()

face_mesh.close()

pose.close()

cv2.destroyAllWindows()