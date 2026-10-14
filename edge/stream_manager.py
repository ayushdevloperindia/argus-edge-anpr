import cv2
import time
import logging

class StreamManager:
    def __init__(self, source):
        self.source = source
        self.cap = None
        self._connect()

    def _connect(self):
        logging.info(f"[STREAM] Connecting to source: {self.source}")
        self.cap = cv2.VideoCapture(self.source)

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            self._reconnect()

        ret, frame = self.cap.read()
        if not ret:
            # Agar static image ya video file hai toh loop rewind karega
            if isinstance(self.source, str) and not self.source.startswith("rtsp://") and not self.source.startswith("http://"):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
            else:
                self._reconnect()
                return False, None
        return ret, frame

    def _reconnect(self):
        logging.warning("[STREAM] Connection lost. Reconnecting in 2 seconds...")
        if self.cap:
            self.cap.release()
        time.sleep(2.0)
        self._connect()

    def release(self):
        if self.cap:
            self.cap.release()