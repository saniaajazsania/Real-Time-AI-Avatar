
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp


# =========================================================
# PATHS
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

AVATAR_DIR = PROJECT_DIR / "avatar" / "assets" / "mie"

BODY_PATH = AVATAR_DIR / "body" / "body_mie.png"

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
# ARM SIZE
# =========================================================

ARM_HEIGHT_RATIO = 0.28


# =========================================================
# ARM POSITION
# =========================================================

LEFT_SHOULDER_X = 0.27
RIGHT_SHOULDER_X = 0.64

SHOULDER_Y = 0.15


# =========================================================
# UP ARM MOVEMENT
# =========================================================

# Arm will move only a little upward.
# X position will NOT change.

UP_ARM_MOVE = 18


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
# LOAD AVATAR
# =========================================================

print("Loading Mie body and arms...")


body = load_png(
    BODY_PATH
)

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
# SHOULDER ANCHORS
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
# ARM POSITIONS
# =========================================================

# DOWN POSITION
# Keep exactly as current working version.

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
# WEBCAM
# =========================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


print()
print("====================================")
print("        MIE ARM TRACKING")
print("====================================")
print()
print("Raise YOUR LEFT arm")
print("Raise YOUR RIGHT arm")
print()
print("Avatar sides are corrected.")
print()
print("Press Q to quit")
print()
print("====================================")


# =========================================================
# SMOOTHING
# =========================================================

left_state = 0.0

right_state = 0.0

ARM_SMOOTHING = 0.20


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


    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    results = pose.process(
        rgb
    )


    # =====================================================
    # DEFAULT
    # =====================================================

    target_left = 0.0

    target_right = 0.0


    # =====================================================
    # POSE DETECTED
    # =====================================================

    if results.pose_landmarks:

        landmarks = (
            results
            .pose_landmarks
            .landmark
        )


        # =================================================
        # YOUR LEFT ARM
        # MediaPipe RIGHT
        # =================================================

        your_left_shoulder = landmarks[
            mp_pose.PoseLandmark.RIGHT_SHOULDER
        ]

        your_left_wrist = landmarks[
            mp_pose.PoseLandmark.RIGHT_WRIST
        ]


        # =================================================
        # YOUR RIGHT ARM
        # MediaPipe LEFT
        # =================================================

        your_right_shoulder = landmarks[
            mp_pose.PoseLandmark.LEFT_SHOULDER
        ]

        your_right_wrist = landmarks[
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
    # SMOOTH STATES
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
    # CANVAS
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
    # BODY
    # =====================================================

    overlay_rgba(
        canvas,
        body,
        body_x,
        body_y
    )


    # =====================================================
    # LEFT ARM
    # =====================================================

    overlay_rgba(
        canvas,
        current_left_arm,
        current_left_x,
        current_left_y
    )


    # =====================================================
    # RIGHT ARM
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

    left_status = (
        "YOUR LEFT: UP"
        if left_state > 0.5
        else
        "YOUR LEFT: DOWN"
    )

    right_status = (
        "YOUR RIGHT: UP"
        if right_state > 0.5
        else
        "YOUR RIGHT: DOWN"
    )


    cv2.putText(
        canvas,
        left_status,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )


    cv2.putText(
        canvas,
        right_status,
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
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
        "Mie Avatar - Arm Tracking",
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

pose.close()

cv2.destroyAllWindows()