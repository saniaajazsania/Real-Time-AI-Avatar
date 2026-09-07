import cv2
import numpy as np

from avatar.avatar_loader import AvatarLoader


def blend_with_background(image_bgra, background_color=(200, 200, 200)):
    """
    Blends a BGRA (with alpha channel) image onto a solid-color
    background, so transparent areas display correctly in a plain
    OpenCV window instead of appearing black.
    """

    if image_bgra.shape[2] != 4:
        # No alpha channel — nothing to blend, just return as-is
        return image_bgra

    bgr = image_bgra[:, :, :3].astype(float)
    alpha = image_bgra[:, :, 3].astype(float) / 255.0
    alpha = alpha[:, :, np.newaxis]  # shape (H, W, 1) for broadcasting

    background = np.full(bgr.shape, background_color, dtype=float)

    blended = (alpha * bgr) + ((1 - alpha) * background)

    return blended.astype(np.uint8)


def main():

    print("=" * 60)
    print("PHASE 2 - STEP 3 TEST: Avatar Preparation (neutral state)")
    print("=" * 60)

    # -----------------------------------------------------
    # Load the avatar
    # -----------------------------------------------------

    avatar = AvatarLoader(avatar_folder="avatar/assets/clara")

    neutral_image = avatar.load_state("neutral", "head_neutral.png")

    # -----------------------------------------------------
    # Blend onto a plain background for correct display
    # -----------------------------------------------------

    display_image = blend_with_background(
        neutral_image,
        background_color=(230, 230, 230)  # light gray
    )

    # -----------------------------------------------------
    # Show it
    # -----------------------------------------------------

    cv2.imshow("Phase 2 - Avatar Neutral Test", display_image)

    print("✅ Image displayed. Press any key on the image window to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("✅ Test complete.")


if __name__ == "__main__":
    main()