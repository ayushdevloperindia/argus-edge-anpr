import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import Tuple

class OCREngine:
    def __init__(self):
        # Stable v2.8 initialization with angle classifier
        self.ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    def extract_text(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        result = self.ocr.ocr(processed, cls=True)
        if not result or not result[0]:
            return "", 0.0

        texts = []
        confs = []
        for line in result[0]:
            text, score = line[1]
            texts.append(text)
            confs.append(score)

        full_text = "".join(texts)
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return full_text, float(avg_conf)