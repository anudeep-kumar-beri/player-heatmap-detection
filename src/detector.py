# src/detector.py
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Runs YOLO detection on a single frame.
        Returns bounding boxes in format: [x1, y1, x2, y2, confidence, class_id]
        """
        results = self.model(frame, verbose=False)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            detections.append([x1, y1, x2, y2, conf, cls])

        return detections
