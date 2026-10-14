from ultralytics import YOLO
import numpy as np

class PlateDetector:
    def __init__(self, weights_path: str, conf_threshold: float = 0.35):
        self.model = YOLO(weights_path)
        self.conf = conf_threshold

    def detect_and_track(self, frame: np.ndarray):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            verbose=False
        )
        detections = []
        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                track_id = int(boxes.id[i].item()) if boxes.id is not None else None
                conf = float(boxes.conf[i].item())
                detections.append({
                    "bbox": xyxy.tolist(),
                    "track_id": track_id,
                    "confidence": conf
                })
        return detections