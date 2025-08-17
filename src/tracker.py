# src/tracker.py
import numpy as np
from src.sort.sort import Sort

class PlayerTracker:
    def __init__(self):
        self.tracker = Sort()

    def update(self, detections):
        """
        Updates tracker with YOLO detections.
        Detections: [[x1, y1, x2, y2, conf, class_id], ...]
        Returns: [[x1, y1, x2, y2, id], ...]
        """
        dets_for_sort = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            if cls == 0:  # class 0 = person
                dets_for_sort.append([x1, y1, x2, y2, conf])

        # 🔴 Ensure numpy array (important!)
        if len(dets_for_sort) == 0:
            dets_for_sort = np.empty((0, 5))
        else:
            dets_for_sort = np.array(dets_for_sort)

        tracked = self.tracker.update(dets_for_sort)
        return tracked
