import numpy as np
import time

class AccidentDetector:
    def __init__(self, iou_threshold=0.35, stationary_cooldown=3.0):
        self.iou_threshold = iou_threshold
        self.stationary_cooldown = stationary_cooldown
        self.track_history = {}
        self.last_alert_time = 0

    def compute_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0:
            return 0.0

        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def check_collision(self, detections):
        current_time = time.time()
        active_boxes = []

        for det in detections:
            track_id = det.get("track_id")
            bbox = det.get("bbox")
            if track_id is not None and bbox:
                active_boxes.append((track_id, bbox))
                # Center point record karo movement check karne ke liye
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                self.track_history[track_id] = (cx, cy, current_time)

        # Pairwise collision check
        num_boxes = len(active_boxes)
        for i in range(num_boxes):
            for j in range(i + 1, num_boxes):
                id1, box1 = active_boxes[i]
                id2, box2 = active_boxes[j]

                iou = self.compute_iou(box1, box2)
                if iou >= self.iou_threshold:
                    if current_time - self.last_alert_time > self.stationary_cooldown:
                        self.last_alert_time = current_time
                        return True, [box1, box2], (id1, id2)

        return False, [], ()