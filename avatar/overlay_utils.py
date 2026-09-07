import numpy as np


def overlay_image_alpha(background_bgr, overlay_bgra, center_x, center_y, scale=1.0):
    """
    Alpha-blends a BGRA overlay image (e.g. the hand) onto a BGR
    background image (e.g. the rendered avatar), centered at
    (center_x, center_y). Handles the overlay going partly off
    the edge of the background without crashing.

    background_bgr: the avatar canvas (H, W, 3), modified in place
                     and also returned
    overlay_bgra:    the hand image (h, w, 4) with alpha channel
    center_x, center_y: where the overlay's center should land on
                     the background
    scale: optional resize factor for the overlay (1.0 = original size)
    """

    if scale != 1.0:
        new_w = max(1, int(overlay_bgra.shape[1] * scale))
        new_h = max(1, int(overlay_bgra.shape[0] * scale))
        overlay_bgra = _resize(overlay_bgra, new_w, new_h)

    overlay_h, overlay_w = overlay_bgra.shape[:2]
    bg_h, bg_w = background_bgr.shape[:2]

    # Top-left corner where the overlay would start
    x1 = int(center_x - overlay_w / 2)
    y1 = int(center_y - overlay_h / 2)
    x2 = x1 + overlay_w
    y2 = y1 + overlay_h

    # Clip to background boundaries
    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1)
    bg_x2 = min(bg_w, x2)
    bg_y2 = min(bg_h, y2)

    # If completely off-screen, do nothing
    if bg_x1 >= bg_x2 or bg_y1 >= bg_y2:
        return background_bgr

    # Corresponding region within the overlay image
    ov_x1 = bg_x1 - x1
    ov_y1 = bg_y1 - y1
    ov_x2 = ov_x1 + (bg_x2 - bg_x1)
    ov_y2 = ov_y1 + (bg_y2 - bg_y1)

    overlay_crop = overlay_bgra[ov_y1:ov_y2, ov_x1:ov_x2]
    background_crop = background_bgr[bg_y1:bg_y2, bg_x1:bg_x2]

    if overlay_crop.shape[2] == 4:
        overlay_rgb = overlay_crop[:, :, :3].astype(float)
        alpha = overlay_crop[:, :, 3].astype(float) / 255.0
        alpha = alpha[:, :, None]
    else:
        overlay_rgb = overlay_crop.astype(float)
        alpha = np.ones(overlay_rgb.shape[:2] + (1,), dtype=float)

    blended = (
        overlay_rgb * alpha
        + background_crop.astype(float) * (1 - alpha)
    )

    background_bgr[bg_y1:bg_y2, bg_x1:bg_x2] = blended.astype("uint8")

    return background_bgr


def _resize(image, width, height):
    import cv2
    return cv2.resize(image, (width, height))


def map_position(
    source_x, source_y,
    source_width, source_height,
    target_width, target_height
):
    """
    Converts an (x, y) position from one coordinate space (e.g. the
    webcam frame) into the equivalent position in another coordinate
    space (e.g. the avatar canvas), preserving relative position.
    """

    ratio_x = source_x / source_width
    ratio_y = source_y / source_height

    target_x = ratio_x * target_width
    target_y = ratio_y * target_height

    return int(target_x), int(target_y)