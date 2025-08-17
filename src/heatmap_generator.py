import numpy as np
import cv2

class HeatmapGenerator:
    def __init__(self, width, height, downscale=2):
        # store original resolution
        self.width = width
        self.height = height

        # apply downscaling factor for faster accumulation
        self.downscale = downscale
        self.h = height // downscale
        self.w = width // downscale

        # initialize empty heatmap
        self.heatmap = np.zeros((self.h, self.w), dtype=np.float32)

    def add_points(self, points):
        """
        Add detected player positions to heatmap.
        Points: list of (x, y) in original resolution.
        """
        for (x, y) in points:
            xs = int(x // self.downscale)
            ys = int(y // self.downscale)
            if 0 <= xs < self.w and 0 <= ys < self.h:
                self.heatmap[ys, xs] += 1

    def get_heatmap(self):
        # Normalize to [0, 255]
        heatmap_norm = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        return heatmap_norm.astype(np.uint8)

    def overlay_on_frame(self, frame, alpha=0.5):
        # Resize heatmap to match frame
        heatmap_resized = cv2.resize(self.get_heatmap(), (frame.shape[1], frame.shape[0]))
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 1 - alpha, heatmap_colored, alpha, 0)
