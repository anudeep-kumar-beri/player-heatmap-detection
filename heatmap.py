import os
import cv2
import numpy as np


class HeatmapGenerator:
    """
    Accumulates per-frame (x,y) hits into a 2D heatmap and can render overlays.
    Works at full frame size or an optional downscaled internal grid for speed.
    """

    def __init__(self, frame_width: int, frame_height: int, downscale: int = 1, fixed_norm_max: float | None = None):
        """
        downscale: accumulate on a smaller grid (e.g., 2 or 4) to save memory/CPU.
                  Final overlay is always returned at frame size.
        fixed_norm_max: if set (>0), use this fixed max count for normalization so
                        overlay intensity is stable and doesn't "fade" over time.
                        If None, use a persistent global max observed so far.
        """
        assert downscale >= 1, "downscale must be >= 1"
        self.w = frame_width
        self.h = frame_height
        self.ds = int(downscale)
        self.grid_w = int(max(1, self.w // self.ds))
        self.grid_h = int(max(1, self.h // self.ds))
        self.grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        self._global_max = 0.0
        self.fixed_norm_max = float(fixed_norm_max) if fixed_norm_max and fixed_norm_max > 0 else None

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
            # Update running maximum for persistent normalization (if not using fixed)
            if self.fixed_norm_max is None:
                current_max = float(self.grid.max())
                if current_max > self._global_max:
                    self._global_max = current_max

    def _to_colormap(self):
        if self.fixed_norm_max is not None and self.fixed_norm_max > 0:
            denom = self.fixed_norm_max
        else:
            denom = self._global_max if self._global_max > 0 else None

        if denom is None:
            heat_small = self.grid.astype(np.uint8)
        else:
            heat_small = np.clip((self.grid / denom) * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(heat_small, cv2.COLORMAP_JET)

    def render_overlay(self, frame, alpha: float = 0.35):
        """
        Returns frame blended with colored heatmap.
        alpha: heatmap opacity (0..1)
        """
        heat_color_small = self._to_colormap()
        # Upscale to frame size if needed
        if self.ds != 1:
            heat_color = cv2.resize(heat_color_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        else:
            heat_color = heat_color_small
        # Ensure sizes match exactly
        if (heat_color.shape[1], heat_color.shape[0]) != (frame.shape[1], frame.shape[0]):
            heat_color = cv2.resize(heat_color, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)

    def save_heatmap_image(self, out_path: str):
        # Save a standalone heatmap image (colored) at full resolution
        heat_color_small = self._to_colormap()
        if self.ds != 1:
            heat_color = cv2.resize(heat_color_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        else:
            heat_color = heat_color_small
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, heat_color)

    # --- New: Save heatmap over a football pitch background ---
    def _generate_pitch_background(self) -> np.ndarray:
        """Generate a simple football pitch drawing at (w,h)."""
        bg = np.full((self.h, self.w, 3), (40, 130, 40), dtype=np.uint8)  # green grass
        line_color = (255, 255, 255)
        thickness = max(2, int(round(min(self.w, self.h) * 0.004)))

        # Metric-based scaling (UEFA: 105m x 68m)
        pitch_w_m = 105.0
        pitch_h_m = 68.0
        sx = self.w / pitch_w_m
        sy = self.h / pitch_h_m
        s = min(sx, sy)
        # Use center as reference, fit to min scale
        field_w_px = int(round(pitch_w_m * s))
        field_h_px = int(round(pitch_h_m * s))
        x0 = (self.w - field_w_px) // 2
        y0 = (self.h - field_h_px) // 2
        x1 = x0 + field_w_px
        y1 = y0 + field_h_px

        # Outer lines
        cv2.rectangle(bg, (x0, y0), (x1, y1), line_color, thickness)

        # Halfway line
        cv2.line(bg, (x0 + field_w_px // 2, y0), (x0 + field_w_px // 2, y1), line_color, thickness)

        # Center circle and spot
        center = (x0 + field_w_px // 2, y0 + field_h_px // 2)
        r_cc = int(round(9.15 * s))  # center circle radius 9.15m
        cv2.circle(bg, center, r_cc, line_color, thickness)
        cv2.circle(bg, center, max(1, thickness // 2), line_color, -1)

        # Penalty areas and spots (depth 16.5m, width 40.32m)
        pa_depth = int(round(16.5 * s))
        pa_half_w = int(round((40.32 / 2.0) * s))
        # Left PA
        cx_left = x0 + int(round(11.0 * s))
        cv2.rectangle(bg, (x0, center[1] - pa_half_w), (x0 + pa_depth, center[1] + pa_half_w), line_color, thickness)
        cv2.circle(bg, (x0 + int(round(11.0 * s)), center[1]), max(1, thickness // 2), line_color, -1)
        # Right PA
        cv2.rectangle(bg, (x1 - pa_depth, center[1] - pa_half_w), (x1, center[1] + pa_half_w), line_color, thickness)
        cv2.circle(bg, (x1 - int(round(11.0 * s)), center[1]), max(1, thickness // 2), line_color, -1)

        # Goal areas (depth 5.5m, width 18.32m)
        ga_depth = int(round(5.5 * s))
        ga_half_w = int(round((18.32 / 2.0) * s))
        cv2.rectangle(bg, (x0, center[1] - ga_half_w), (x0 + ga_depth, center[1] + ga_half_w), line_color, thickness)
        cv2.rectangle(bg, (x1 - ga_depth, center[1] - ga_half_w), (x1, center[1] + ga_half_w), line_color, thickness)

        # Corner arcs (optional minimal representation)
        r_corner = int(round(1.0 * s))
        cv2.ellipse(bg, (x0, y0), (r_corner, r_corner), 0, 0, 90, line_color, thickness)
        cv2.ellipse(bg, (x1, y0), (r_corner, r_corner), 0, 90, 180, line_color, thickness)
        cv2.ellipse(bg, (x0, y1), (r_corner, r_corner), 0, 270, 360, line_color, thickness)
        cv2.ellipse(bg, (x1, y1), (r_corner, r_corner), 0, 180, 270, line_color, thickness)

        return bg

    def save_heatmap_over_pitch(self, out_path: str, pitch_image: str | None = None, alpha: float = 0.5):
        """Save heatmap blended over a football pitch background image or a generated pitch."""
        # Prepare colored heatmap at full resolution
        heat_color_small = self._to_colormap()
        if self.ds != 1:
            heat_color = cv2.resize(heat_color_small, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        else:
            heat_color = heat_color_small

        # Load background or generate one
        bg = None
        if pitch_image and os.path.isfile(pitch_image):
            bg = cv2.imread(pitch_image)
            if bg is not None:
                bg = cv2.resize(bg, (self.w, self.h), interpolation=cv2.INTER_AREA)
        if bg is None:
            bg = self._generate_pitch_background()

        blended = cv2.addWeighted(bg, 1.0 - alpha, heat_color, alpha, 0)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, blended)
