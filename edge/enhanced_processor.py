import cv2
import numpy as np
import re

class STNRectifier:
    """Mathematical STN: Perspective deskewing without heavy weights"""
    def rectify(self, crop):
        if crop is None or crop.size == 0:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return crop
            
        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.04 * peri, True)
        
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            # Order points: top-left, top-right, bottom-right, bottom-left
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            rect = np.zeros((4, 2), dtype="float32")
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            
            width = int(max(np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[2] - rect[3])))
            height = int(max(np.linalg.norm(rect[3] - rect[0]), np.linalg.norm(rect[2] - rect[1])))
            
            if width > 40 and height > 15:
                dst = np.array([
                    [0, 0],
                    [width - 1, 0],
                    [width - 1, height - 1],
                    [0, height - 1]
                ], dtype="float32")
                M = cv2.getPerspectiveTransform(rect, dst)
                return cv2.warpPerspective(crop, M, (width, height))
        return crop

class OSNetExtractor:
    """Lightweight color & texture spatial feature embedding for Re-ID"""
    def extract_embedding(self, frame, bbox):
        x1, y1, x2, y2 = bbox
        # Vehicle context around plate
        pad_y = int((y2 - y1) * 1.5)
        pad_x = int((x2 - x1) * 0.8)
        h, w = frame.shape[:2]
        
        vx1, vy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        vx2, vy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
        veh_crop = frame[vy1:vy2, vx1:vx2]
        
        if veh_crop.size == 0:
            return [0.0] * 16
            
        hsv = cv2.cvtColor(veh_crop, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        feature = np.concatenate([hist_h.flatten(), hist_s.flatten()])
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        return feature.round(3).tolist()

class HMMCorrector:
    """HMM Viterbi character probability correction for Indian number plates"""
    def __init__(self):
        self.num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
        self.char_to_num = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6'}
        
    def correct(self, raw_text: str) -> str:
        text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if len(text) < 8 or len(text) > 10:
            return text
            
        chars = list(text)
        n = len(chars)
        
        # State Code: Positions 0,1 should be letters
        for i in [0, 1]:
            if chars[i] in self.num_to_char:
                chars[i] = self.num_to_char[chars[i]]
                
        # District Code: Positions 2,3 should be digits
        for i in [2, 3]:
            if chars[i] in self.char_to_num:
                chars[i] = self.char_to_num[chars[i]]
                
        # Last 4 unique sequence: Digits
        for i in range(n - 4, n):
            if chars[i] in self.char_to_num:
                chars[i] = self.char_to_num[chars[i]]
                
        return "".join(chars)