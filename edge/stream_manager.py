import cv2
import time
import logging

class StreamManager:
    def __init__(self, source, zoom_factor=1.0):
        self.source = source
        self.zoom_factor = float(zoom_factor)
        self.cap = cv2.VideoCapture(self.source)

    def set_zoom(self, zoom_factor: float):
        self.zoom_factor = max(1.0, float(zoom_factor))

    def read_frame(self):
        if not self.cap.isOpened():
            self.cap.open(self.source)
            time.sleep(0.5)
            return False, None

        ret, frame = self.cap.read()
        if not ret or frame is None:
            return False, None

        # digital zoom crop aur resize
        if self.zoom_factor > 1.0:
            h, w = frame.shape[:2]
            crop_h = int(h / self.zoom_factor)
            crop_w = int(w / self.zoom_factor)
            
            y1 = (h - crop_h) // 2
            x1 = (w - crop_w) // 2
            cropped = frame[y1:y1 + crop_h, x1:x1 + crop_w]
            frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

        return True, frame

    def release(self):
        if self.cap.isOpened():
            self.cap.release()