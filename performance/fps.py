import time


class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frame_count = 0

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed_time = time.time() - self.start_time

        if elapsed_time <= 0:
            return 0.0

        return self.frame_count / elapsed_time