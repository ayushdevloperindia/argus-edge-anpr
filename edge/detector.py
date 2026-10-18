import cv2
import re
import numpy as np
from ultralytics import YOLO
import easyocr

class PlateDetector:
    def __init__(self, weights_path="edge/models/plate_yolo11.pt", conf_threshold=0.25):
        self.model = YOLO(weights_path)
        self.conf = conf_threshold
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.vehicle_classes = [1, 2, 3, 5, 7]
        self.last_plate = {}
        self.frame_num = 0

    def correct_indian_plate(self, s: str) -> str:
        s = re.sub(r'[^A-Z0-9]', '', s.upper())
        if len(s) < 8:
            return s

        char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'A': '4', 'T': '1', 'D': '0'}
        num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '4': 'A', '7': 'V'}

        res = list(s)

        for i in [0, 1]:
            if i < len(res) and res[i] in num_to_char:
                res[i] = num_to_char[res[i]]

        for i in [2, 3]:
            if i < len(res) and res[i] in char_to_num:
                res[i] = char_to_num[res[i]]

        for i in range(max(4, len(res) - 4), len(res)):
            if res[i] in char_to_num:
                res[i] = char_to_num[res[i]]

        if len(res) >= 10:
            for i in range(4, len(res) - 4):
                if res[i] in num_to_char:
                    res[i] = num_to_char[res[i]]

        cleaned = "".join(res)
        m = re.search(r'[A-Z]{2}[0-9]{2}[A-Z]{1,2}[0-9]{4}', cleaned)
        if m:
            return m.group(0)
        return cleaned

    def read_plate(self, crop):
        if crop is None or crop.size == 0 or crop.shape[0] < 15 or crop.shape[1] < 30:
            return ""

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (0, 0), fx=3.5, fy=3.5, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(resized)
        
        thresh = cv2.adaptiveThreshold(
            contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 17, 7
        )
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        clean_img = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        res = self.reader.readtext(
            clean_img, 
            detail=0, 
            paragraph=False,
            allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        )
        raw = "".join(res)

        if len(raw) < 8:
            res_contrast = self.reader.readtext(
                contrast, 
                detail=0, 
                paragraph=False,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            raw = "".join(res_contrast)

        return self.correct_indian_plate(raw)

    def detect_and_track(self, frame):
        detections = []
        self.frame_num += 1

        results = self.model(
            frame, 
            conf=self.conf, 
            classes=self.vehicle_classes, 
            verbose=False
        )

        if not results or len(results) == 0 or results[0].boxes is None:
            return detections

        h, w = frame.shape[:2]

        for i, box in enumerate(results[0].boxes):
            xyxy = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, xyxy)
            cls_id = int(box.cls[0].item())
            v_type = self.model.names.get(cls_id, "VEHICLE").upper()

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            box_w = x2 - x1
            box_h = y2 - y1

            py1 = int(y1 + box_h * 0.50)
            py2 = int(y1 + box_h * 0.70)
            px1 = int(x1 + box_w * 0.25)
            px2 = int(x2 - box_w * 0.25)

            plate_text = self.last_plate.get(i, "")

            if self.frame_num % 2 == 0:
                crop = frame[py1:py2, px1:px2]
                found = self.read_plate(crop)
                if len(found) >= 8:
                    self.last_plate[i] = found
                    plate_text = found
                    print(f"[EXACT PLATE DETECTED] -> {v_type} : {plate_text}")
                elif not plate_text and found:
                    plate_text = found

            detections.append({
                "track_id": i + 1,
                "bbox": [x1, y1, x2, y2],
                "vehicle_type": v_type,
                "plate_text": plate_text
            })

        return detections