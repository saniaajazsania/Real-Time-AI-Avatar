import cv2
import os


class AvatarLoader:
    """
    Loads avatar image states (neutral, blink, talk, etc.) for a given
    avatar folder. Images are expected to be PNGs with transparency
    (alpha channel), since they were exported from Krita with a
    transparent background.
    """

    def __init__(self, avatar_folder):
        self.avatar_folder = avatar_folder
        self.states = {}

    def load_state(self, state_name, filename):
        """
        Loads a single avatar state image (e.g. 'neutral' -> 'head_neutral.png').
        Uses cv2.IMREAD_UNCHANGED so the alpha (transparency) channel
        is preserved. Without this flag, OpenCV drops transparency
        and the image would load as a plain 3-channel BGR image.
        """

        path = os.path.join(self.avatar_folder, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Avatar image not found: {path}")

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            raise ValueError(f"❌ Could not read avatar image: {path}")

        # Warn if the image has no alpha channel (shape would be
        # (H, W, 3) instead of (H, W, 4))
        if len(image.shape) < 3 or image.shape[2] != 4:
            print(
                f"⚠️  Warning: {filename} has no alpha channel "
                f"(shape: {image.shape}). Transparency will not work."
            )

        self.states[state_name] = image

        print(f"✅ Loaded '{state_name}' state from {filename} — shape: {image.shape}")

        return image

    def get_state(self, state_name):
        """Returns a previously loaded state image, or None if not loaded."""
        return self.states.get(state_name)