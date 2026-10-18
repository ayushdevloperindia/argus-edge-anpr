import os
import time
import logging
import cv2
import yaml
from datetime import datetime

from edge.detector import PlateDetector
from edge.dispatcher import Dispatcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def clean_old_crops(crops_dir, retention_days=7):
    if not os.path.exists(crops_dir):
        return
    now = time.time()
    cutoff_seconds = retention_days * 86400
    for filename in os.listdir(crops_dir):
        file_path = os.path.join(crops_dir, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > cutoff_seconds:
                try:
                    os.remove(file_path)
                    logging.info(f"deleted expired crop: {filename}")
                except Exception as e:
                    logging.warning(f"failed to delete {filename}: {e}")

def load_config(config_path="edge/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_pipeline():
    config = load_config()

    camera_id = str(config.get("camera", {}).get("id", "CAM-01"))
    crops_dir = config.get("storage", {}).get("crops_dir", "edge/crops")
    retention_days = config.get("storage", {}).get("retention_days", 7)
    cooldown_seconds = config.get("edge", {}).get("cooldown_seconds", 10)
    target_url = config.get("server", {}).get("target_url", "http://172.30.139.101:8000/api/detections")

    os.makedirs(crops_dir, exist_ok=True)
    clean_old_crops(crops_dir, retention_days)

    detector = PlateDetector()
    dispatcher = Dispatcher(target_url=target_url)

    video_stream_url = "http://172.30.139.198:8080/video"
    cap = cv2.VideoCapture(video_stream_url)
    if not cap.isOpened():
        logging.error("mobile camera stream open nahi ho paya, ip aur wifi check karo")
        return

    logging.info(f"edge pipeline started for camera {camera_id}")
    logging.info(f"target server url: {target_url}")

    last_seen_tracker = {}
    frame_count = 0
    detections = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("frame capture fail ho gaya")
                break

            frame_count += 1
            current_time = time.time()

            if frame_count % 3 == 0:
                detections = detector.detect_and_track(frame)

                for det in detections:
                    plate_text = det.get("plate_text")
                    if not plate_text:
                        continue

                    plate_text = str(plate_text).strip().upper()
                    if len(plate_text) < 6:
                        continue

                    last_time = last_seen_tracker.get(plate_text, 0)
                    if (current_time - last_time) < cooldown_seconds:
                        continue

                    last_seen_tracker[plate_text] = current_time

                    crop_img = det.get("crop")
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                    crop_filename = f"{plate_text}_{timestamp_str}.jpg"
                    crop_filepath = os.path.join(crops_dir, crop_filename)

                    if crop_img is not None and crop_img.size > 0:
                        cv2.imwrite(crop_filepath, crop_img)

                    raw_conf = det.get("confidence", 0.95)
                    conf_pct = int(raw_conf * 100) if raw_conf <= 1.0 else int(raw_conf)
                    conf_str = f"{conf_pct}%"

                    payload = {
                        "plate": plate_text,
                        "camera_id": camera_id,
                        "speed": "0 KM/H",
                        "confidence": conf_str
                    }

                    dispatcher.dispatch(payload)
                    logging.info(f"queued sighting for dashboard: {plate_text} (cam: {camera_id})")

            for det in detections:
                box = det.get("bbox")
                plate_text = det.get("plate_text", "")
                v_type = det.get("vehicle_type", "vehicle")

                if box:
                    x1, y1, x2, y2 = map(int, box)
                    label = f"{v_type}: {plate_text}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(y1 - 10, 25)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Edge Pipeline Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("pipeline stopped cleanly")

if __name__ == "__main__":
    run_pipeline()