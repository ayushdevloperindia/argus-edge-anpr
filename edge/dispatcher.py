import os
import time
import requests
import logging
import threading
from queue import Queue

class Dispatcher:
    def __init__(self, target_url="http://127.0.0.1:8000/api/detections"):
        self.target_url = target_url
        self.queue = Queue()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()

    def dispatch(self, payload: dict):
        self.queue.put(payload)

    def _worker(self):
        while self.running:
            try:
                payload = self.queue.get(timeout=1)
            except Exception:
                continue

            sent = False
            while not sent and self.running:
                try:
                    res = requests.post(self.target_url, json=payload, timeout=2)
                    if res.status_code in [200, 201]:
                        sent = True
                        logging.info(f"dispatched successfully: {payload.get('plate_number', 'EVENT')}")
                    else:
                        time.sleep(3)
                except Exception:
                    # server unreachable hone par wait karega aur retry karega
                    time.sleep(3)
            self.queue.task_done()