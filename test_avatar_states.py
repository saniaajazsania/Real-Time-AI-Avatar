import cv2
import numpy as np

from avatar.avatar_loader import AvatarLoader


def blend_with_background(image_bgra, background_color=(230, 230, 230)):
    """Blends a BGRA image onto a solid background for display."""

    if image_bgra.shape[2] != 4:
        return image_bgra

    bgr = image_bgra[:, :, :3].astype(float)
    alpha = image_bgra[:, :, 3].astype(float) / 255.0
    alpha = alpha[:, :, np.newaxis]

    background = np.full(bgr.shape, background_color, dtype=float)
    blended = (alpha * bgr) + ((1 - alpha) * background)

    return blended.astype(np.uint8)


def main():

    print("=" * 60)
    print("PHASE 2 - STEP 3 TEST: All Avatar States")
    print("=" * 60)

    avatar = AvatarLoader(avatar_folder="avatar/assets/clara")

    avatar.load_state("neutral", "head_neutral.png")
    avatar.load_state("blink", "head_blink.png")
    avatar.load_state("talk", "head_talk.png")

    print("-" * 60)
    print("Controls:")
    print("  1 = Neutral")
    print("  2 = Blink")
    print("  3 = Talk")
    print("  Q = Quit")
    print("-" * 60)

    current_state = "neutral"

    while True:

        image = avatar.get_state(current_state)
        display_image = blend_with_background(image)

        # Show which state is active
        cv2.putText(
            display_image,
            f"State: {current_state.upper()}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        cv2.imshow("Phase 2 - Avatar State Switcher", display_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            current_state = "neutral"
        elif key == ord("2"):
            current_state = "blink"
        elif key == ord("3"):
            current_state = "talk"
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("✅ Test complete.")


if __name__ == "__main__":
    main()