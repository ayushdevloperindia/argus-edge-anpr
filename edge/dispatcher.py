import queue
import threading
import requests
import logging

class Dispatcher:
    def __init__(self, target_url: str, timeout: float = 3.0):
        self.target_url = target_url
        self.timeout = timeout
        self.queue = queue.Queue(maxsize=100)
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def dispatch(self, payload: dict):
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            logging.warning("[DISPATCH] Queue full, dropping payload.")

    def _run(self):
        while self.running:
            try:
                payload = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                resp = requests.post(self.target_url, json=payload, timeout=self.timeout)
                logging.info(f"[DISPATCH] Sent {payload.get('plate')} -> HTTP {resp.status_code}")
            except Exception as e:
                # Laptop 2 offline ho tab bhi camera loop freeze nahi hoga
                logging.warning(f"[DISPATCH] Server unreachable: {e}")
            finally:
                self.queue.task_done()

    def stop(self):
        self.running = False