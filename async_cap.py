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
        self.stopped = False

    def stop(self):
        self.stopped = True

    def read(self):
        return self.frame

    def runner(self):
        while True:
            self.frame = self.cap.read()
            if(self.stopped):
                break
