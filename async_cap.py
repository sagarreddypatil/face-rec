import cv2
import time
from threading import Thread


class AsyncCap:
    def __init__(self, cap_name, fps=30):
        self.cap = cv2.VideoCapture(cap_name)
        self.frame = self.cap.read()
        self.fps = fps
        self.thread = Thread(target=self.runner)
        self.thread.start()

    def read(self):
        return self.frame

    def runner(self):
        while True:
            start = time.time()
            self.frame = self.cap.read()
            time.sleep(max(0, (1 / self.fps) - (time.time() - start)))
