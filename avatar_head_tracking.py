
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

# ONLY ADDED
HEAD_BLINK_PATH = AVATAR_DIR / "head" / "head_blink_mie.png"
HEAD_TALK_PATH = AVATAR_DIR / "head" / "head_talk_mie.png"


# =========================================================
# CANVAS / AVATAR SETTINGS
# =========================================================

CANVAS_WIDTH = 620

CANVAS_HEIGHT = 700

AVATAR_HEIGHT = 500


# =========================================================
# BOTTOM MARGIN
# =========================================================

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

# Mouth opening threshold
MOUTH_OPEN_THRESHOLD = 0.035

# How long mouth stays in talking state
MOUTH_HOLD_TIME = 0.08

last_mouth_change = 0

is_talking = False


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

    if image.shape[2] == 3:

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
# RESIZE WHILE KEEPING PROPORTION
# =========================================================

def resize_keep_ratio(image, target_height):

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

print("Loading Mie avatar...")

body = load_png(BODY_PATH)

head = load_png(HEAD_PATH)

head_blink = load_png(HEAD_BLINK_PATH)

head_talk = load_png(HEAD_TALK_PATH)


body = crop_transparent(body)

head = crop_transparent(head)

head_blink = crop_transparent(head_blink)

head_talk = crop_transparent(head_talk)


print("Original body:", body.shape)

print("Original head:", head.shape)


# =========================================================
# BODY SIZE
# =========================================================

body = resize_keep_ratio(
    body,
    AVATAR_HEIGHT
)


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


# IMPORTANT:
# Blink and talk heads are resized to EXACT
# same dimensions as neutral head.

head_blink = resize_keep_ratio(
    head_blink,
    head_target_height
)

head_talk = resize_keep_ratio(
    head_talk,
    head_target_height
)


print("Final body:", body.shape)

print("Final head:", head.shape)


# =========================================================
# BODY POSITION
# =========================================================

body_height, body_width = body.shape[:2]

body_x = (
    CANVAS_WIDTH - body_width
) // 2

body_y = (
    CANVAS_HEIGHT
    - body_height
    - BOTTOM_MARGIN
)


# =========================================================
# NECK ANCHOR
# =========================================================

neck_x = (
    body_x
    + int(body_width * 0.50)
)

neck_y = (
    body_y
    + int(body_height * 0.18)
)


# =========================================================
# HEAD BASE POSITION
# =========================================================

head_height, head_width = head.shape[:2]

head_base_x = (
    neck_x
    - head_width // 2
)

head_base_y = (
    neck_y
    - int(head_height * 0.70)
)


# =========================================================
# SAFETY CLAMP — HEAD
# =========================================================

TOP_MARGIN = 20

if head_base_y < TOP_MARGIN:

    shift_down = TOP_MARGIN - head_base_y

    body_y += shift_down

    neck_y += shift_down

    head_base_y += shift_down

    print(
        f"Adjusted vertical position by "
        f"{shift_down}px to avoid clipping."
    )


# =========================================================
# SAFETY CLAMP — FEET
# =========================================================

bottom_of_body = body_y + body_height

if bottom_of_body > CANVAS_HEIGHT - 5:

    overflow = (
        bottom_of_body
        -
        (CANVAS_HEIGHT - 5)
    )

    body_y -= overflow

    neck_y -= overflow

    head_base_y -= overflow

    print(
        f"Adjusted vertical position by "
        f"-{overflow}px to keep feet visible."
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
# SMOOTHING VALUES
# =========================================================

smooth_x = 0.0

smooth_y = 0.0

smooth_angle = 0.0


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

print("        MIE HEAD TRACKING")

print("====================================")

print()

print("Move your head LEFT / RIGHT")

print("Tilt your head slightly")

print("Blink your eyes")

print("Open/close your mouth")

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


    frame = cv2.flip(
        frame,
        1
    )


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = face_mesh.process(rgb)


    target_x = 0.0

    target_y = 0.0

    target_angle = 0.0


    # =====================================================
    # FACE DETECTED
    # =====================================================

    if results.multi_face_landmarks:

        landmarks = (
            results
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
        # EYE BLINK DETECTION
        # =================================================

        # LEFT EYE
        blink_left_outer = landmarks[33]
        blink_left_inner = landmarks[133]

        blink_left_top = landmarks[159]
        blink_left_bottom = landmarks[145]


        # RIGHT EYE
        blink_right_outer = landmarks[263]
        blink_right_inner = landmarks[362]

        blink_right_top = landmarks[386]
        blink_right_bottom = landmarks[374]


        # -------------------------------------------------
        # LEFT EYE RATIO
        # -------------------------------------------------

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


        # -------------------------------------------------
        # RIGHT EYE RATIO
        # -------------------------------------------------

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


        # -------------------------------------------------
        # AVERAGE EYE RATIO
        # -------------------------------------------------

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
        # BLINK STATE
        # =================================================

        current_time = time.time()


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

        # Upper and lower lip
        upper_lip = landmarks[13]

        lower_lip = landmarks[14]

        # Left/right mouth corners
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
    # SMOOTH HEAD MOVEMENT
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
    # =====================================================

    # Priority:
    #
    # BLINK > TALK > NORMAL
    #
    # So if blinking while talking,
    # blink image is shown.

    if is_blinking:

        selected_head = head_blink

    elif is_talking:

        selected_head = head_talk

    else:

        selected_head = head


    # =====================================================
    # ROTATE SELECTED HEAD
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
    # DISPLAY
    # =====================================================

    display = np.ascontiguousarray(
        canvas
    )


    cv2.putText(
        display,
        "MIE AVATAR",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )


    cv2.imshow(
        "Mie Avatar - Head Tracking",
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

cv2.destroyAllWindows()

