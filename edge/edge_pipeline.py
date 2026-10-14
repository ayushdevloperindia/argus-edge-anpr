import os
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

import cv2
import yaml
import time
import logging
from datetime import datetime, timezone, timedelta
from edge.stream_manager import StreamManager
from edge.detector import PlateDetector
from edge.ocr_engine import OCREngine
from edge.ocr_cleaner import clean_and_validate_plate
from edge.debouncer import Debouncer
from edge.dispatcher import Dispatcher
from edge.enhanced_processor import STNRectifier, OSNetExtractor, HMMCorrector
from edge.accident_detector import AccidentDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SNAPSHOT_DIR = os.path.join("edge", "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

def get_ist_timestamp() -> str:
    tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(tz).isoformat()

def save_snapshot(image, prefix: str) -> str:
    if image is None or image.size == 0:
        return None
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{prefix}_{timestamp_str}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(filepath, image)
    return filepath

def run_pipeline():
    with open("edge/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    detector = PlateDetector(
        weights_path=config["model"]["weights"],
        conf_threshold=config["model"]["conf_threshold"]
    )
    ocr = OCREngine()
    debouncer = Debouncer(cooldown_seconds=config["inference"]["debounce_cooldown_seconds"])
    dispatcher = Dispatcher(target_url=config["server"]["target_url"])
    
    stn = STNRectifier()
    hmm = HMMCorrector()
    osnet = OSNetExtractor()
    accident_engine = AccidentDetector(iou_threshold=0.30, stationary_cooldown=5.0)

    cam = config["cameras"][0]
    stream = StreamManager(cam["source"])
    logging.info(f"[BRAIN] Ingesting stream from {cam['id']}: {cam['name']}")

    processed_tracks = set()

    try:
        while True:
            ret, frame = stream.read_frame()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            detections = detector.detect_and_track(frame)

            # Accident detection logic
            is_accident, crash_boxes, crash_ids = accident_engine.check_collision(detections)
            if is_accident:
                logging.warning(f"[ACCIDENT ALERT] Collision detected between ID {crash_ids[0]} and ID {crash_ids[1]}")
                cv2.putText(frame, "CRITICAL: ACCIDENT DETECTED", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Save incident frame proof
                crash_snapshot = save_snapshot(frame, f"accident_{cam['id']}")

                accident_payload = {
                    "event_type": "ACCIDENT",
                    "camera_id": cam["id"],
                    "timestamp": get_ist_timestamp(),
                    "vehicle_ids": list(crash_ids),
                    "snapshot_path": crash_snapshot
                }
                dispatcher.dispatch(accident_payload)

            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                track_id = det["track_id"]

                color = (0, 0, 255) if is_accident else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if track_id is not None and track_id not in processed_tracks:
                    raw_crop = frame[y1:y2, x1:x2]
                    rectified_crop = stn.rectify(raw_crop)
                    raw_text, ocr_conf = ocr.extract_text(rectified_crop)

                    if ocr_conf >= config["inference"]["ocr_min_confidence"]:
                        corrected_text = hmm.correct(raw_text)
                        clean_plate = clean_and_validate_plate(corrected_text)
                        
                        if clean_plate and debouncer.should_emit(cam["id"], clean_plate):
                            processed_tracks.add(track_id)
                            vehicle_embedding = osnet.extract_embedding(frame, [x1, y1, x2, y2])
                            
                            # Save verified plate crop proof
                            plate_snapshot = save_snapshot(rectified_crop, f"plate_{clean_plate}")
                            
                            payload = {
                                "plate": clean_plate,
                                "raw_ocr": raw_text,
                                "confidence": round(ocr_conf, 2),
                                "camera_id": cam["id"],
                                "timestamp": get_ist_timestamp(),
                                "track_id": track_id,
                                "bbox": [x1, y1, x2, y2],
                                "reid_embedding": vehicle_embedding,
                                "snapshot_path": plate_snapshot
                            }
                            logging.info(f"[BRAIN] EMIT SIGHTING: {clean_plate} (snapshot saved: {plate_snapshot})")
                            dispatcher.dispatch(payload)

            cv2.imshow("ARGUS Laptop 1 - CV Monitor", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()
        dispatcher.stop()

if __name__ == "__main__":
    run_pipeline()