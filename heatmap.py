# heatmap.py (root)
import os
import cv2
import numpy as np

class HeatmapGenerator:
    """
    Accumulates per-frame (x,y) hits into a 2D heatmap and can render overlays.
    Works at full frame size or an optional downscaled internal grid for speed.
    """
    def __init__(self, frame_width: int, frame_height: int, downscale: int = 1):
        """
        downscale: accumulate on a smaller grid (e.g., 2 or 4) to save memory/CPU.
                   Final overlay is always returned at original frame size.
        """
        assert downscale >= 1, "downscale must be >= 1"
        self.w = frame_width
        self.h = frame_height
        self.ds = int(downscale)
        self.grid_w = self.w // self.ds
        self.grid_h = self.h // self.ds
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

    def add_points(self, tracks_xy):
        """
        tracks_xy: Iterable of (x, y) pixel centers in original frame coords.
        """
        if not tracks_xy:
            return
        xs = []
        ys = []
        for (x, y) in tracks_xy:
            gx = int(x) // self.ds
            gy = int(y) // self.ds
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                xs.append(gx)
                ys.append(gy)
        if xs:
            self.grid[ys, xs] += 1.0  # vectorized increment

    def render_overlay(self, frame, alpha: float = 0.35):
        """
        Returns frame blended with colored heatmap.
        alpha: heatmap opacity (0..1)
        """
        # Normalize grid to 0..255
        if self.grid.max() > 0:
            norm = cv2.normalize(self.grid, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm = self.grid.copy()
        heat_small = norm.astype(np.uint8)
        heat_color_small = cv2.applyColorMap(heat_small, cv2.COLORMAP_JET)

        # Upscale to frame size if needed
        if self.ds != 1:
            heat_color = cv2.resize(heat_color_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        else:
            heat_color = heat_color_small

        overlay = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)
        return overlay

    def save_heatmap_image(self, out_path: str):
        """
        Saves a standalone heatmap image (no background frame), upscaled to frame size.
        """
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        if self.grid.max() > 0:
            norm = cv2.normalize(self.grid, None, 0, 255, cv2.NORM_MINMAX)
        else:
            norm = self.grid.copy()
        heat_small = norm.astype(np.uint8)
        heat_color_small = cv2.applyColorMap(heat_small, cv2.COLORMAP_JET)
        if self.ds != 1:
            heat_color = cv2.resize(heat_color_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        else:
            heat_color = heat_color_small
        cv2.imwrite(out_path, heat_color)
