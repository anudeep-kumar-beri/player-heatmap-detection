"""
Heatmap utilities for player tracking.
- HeatmapGenerator accumulates (x,y) points into a grid with optional homography
  mapping, Gaussian smoothing, and colormap rendering.
- Convenience functions generate_heatmap/save_heatmap_image are used by run.py
"""

from __future__ import annotations
import typing as T
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


class HeatmapGenerator:
    """Accumulate player positions and generate heatmaps.

    Parameters
    ----------
    field_size : tuple[int, int]
        (width, height) of the target canvas for the heatmap, in pixels.
    bin_size : int
        Size of spatial bins in pixels.
    homography : np.ndarray | None
        Optional 3x3 homography matrix to map image coordinates to field coords.
    dtype : numpy dtype used for accumulator.
    """

    def __init__(self, field_size: T.Tuple[int, int] = (1280, 720), bin_size: int = 5,
                 homography: T.Optional[np.ndarray] = None, dtype=np.float32):
        self.field_w, self.field_h = int(field_size[0]), int(field_size[1])
        self.bin_size = int(bin_size)
        self.homography = None if homography is None else np.asarray(homography, dtype=np.float64)
        self.dtype = dtype
        self.cols = int(np.ceil(self.field_w / self.bin_size))
        self.rows = int(np.ceil(self.field_h / self.bin_size))
        self.acc = np.zeros((self.rows, self.cols), dtype=self.dtype)

    def reset(self) -> None:
        self.acc.fill(0)

    def _project_points(self, pts: np.ndarray) -> np.ndarray:
        if self.homography is None:
            return pts.astype(np.float32)
        ones = np.ones((pts.shape[0], 1), dtype=np.float64)
        homo = np.hstack([pts.astype(np.float64), ones])
        mapped = (self.homography @ homo.T).T
        mapped[:, 0] /= mapped[:, 2]
        mapped[:, 1] /= mapped[:, 2]
        return mapped[:, :2].astype(np.float32)

    def add_positions(self, positions: T.Iterable[T.Tuple[float, float]],
                      weights: T.Optional[T.Iterable[float]] = None) -> None:
        pts = np.asarray(list(positions), dtype=np.float32)
        if pts.size == 0:
            return
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("positions must be an iterable of (x,y) pairs")
        pts_field = self._project_points(pts)
        if weights is None:
            w = np.ones((pts_field.shape[0],), dtype=self.dtype)
        else:
            w = np.asarray(list(weights), dtype=self.dtype)
            if w.shape[0] != pts_field.shape[0]:
                raise ValueError("weights length must match positions length")
        xs = np.clip((pts_field[:, 0] / self.bin_size).astype(int), 0, self.cols - 1)
        ys = np.clip((pts_field[:, 1] / self.bin_size).astype(int), 0, self.rows - 1)
        for xi, yi, wi in zip(xs, ys, w):
            self.acc[yi, xi] += wi

    def add_tracks(self, tracks: T.Iterable[dict]) -> None:
        pts: list[tuple[float, float]] = []
        for t in tracks:
            if 'center' in t:
                cx, cy = t['center']
            elif 'bbox' in t:
                x, y, w, h = t['bbox']
                cx = x + w / 2.0
                cy = y + h / 2.0
            elif 'x' in t and 'y' in t:
                cx, cy = t['x'], t['y']
            else:
                continue
            pts.append((float(cx), float(cy)))
        self.add_positions(pts)

    def get_accumulator(self) -> np.ndarray:
        return self.acc.copy()

    def generate_heatmap(self, blur_sigma: float = 2.0, normalize: bool = True,
                         cmap: str = 'jet', overlay_on: T.Optional[np.ndarray] = None,
                         alpha: float = 0.6) -> np.ndarray:
        acc = self.acc.astype(np.float32)
        base = acc if acc.max() > 0 else np.zeros_like(acc)
        sm = gaussian_filter(base, sigma=blur_sigma, mode='constant')
        if normalize:
            maxv = sm.max()
            if maxv > 0:
                sm = sm / maxv
        heat_resized = cv2.resize(sm, (self.field_w, self.field_h), interpolation=cv2.INTER_LINEAR)
        cmap_fn = plt.get_cmap(cmap)
        colored = (cmap_fn(np.clip(heat_resized, 0.0, 1.0))[:, :, :3] * 255).astype(np.uint8)
        if overlay_on is None:
            return colored
        bg = overlay_on.copy()
        if bg.dtype != np.uint8:
            bg = (bg * 255).astype(np.uint8) if bg.max() <= 1.0 else bg.astype(np.uint8)
        if bg.shape[0] != self.field_h or bg.shape[1] != self.field_w:
            bg = cv2.resize(bg, (self.field_w, self.field_h), interpolation=cv2.INTER_LINEAR)
        overlay = cv2.addWeighted(bg, 1.0 - alpha, colored, alpha, 0)
        return overlay


def compute_homography_from_points(src_pts: T.Iterable[T.Tuple[float, float]],
                                   dst_pts: T.Iterable[T.Tuple[float, float]]) -> np.ndarray:
    src = np.asarray(list(src_pts), dtype=np.float32)
    dst = np.asarray(list(dst_pts), dtype=np.float32)
    if src.shape[0] < 4 or dst.shape[0] < 4:
        raise ValueError('At least 4 point correspondences are required to compute homography')
    H, status = cv2.findHomography(src, dst, method=cv2.RANSAC)
    if H is None:
        raise RuntimeError('Homography estimation failed')
    return H


# Convenience wrappers used by src/run.py

def generate_heatmap(
    points: np.ndarray,
    canvas_size: T.Tuple[int, int],
    title: str | None = None,
    background: T.Optional[np.ndarray] = None,
    *,
    bin_size: int = 5,
    blur_sigma: float = 2.0,
    cmap: str = "jet",
    alpha: float = 0.6,
) -> np.ndarray:
    """Convenience function to create a heatmap image from (x,y) points.

    Parameters
    - points: array-like of shape (N,2) with integer or float coordinates in the target canvas.
    - canvas_size: (width, height) of the output canvas.
    - title: optional text to annotate on the heatmap.
    - background: optional BGR image to overlay heatmap on (e.g., a video frame).
    - bin_size, blur_sigma, cmap, alpha: rendering parameters.

    Returns RGB uint8 image.
    """
    if points is None:
        points = np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)

    W, H = int(canvas_size[0]), int(canvas_size[1])
    hm = HeatmapGenerator(field_size=(W, H), bin_size=bin_size, homography=None)
    if pts.size > 0:
        hm.add_positions([(float(x), float(y)) for x, y in pts])

    # Convert background from BGR->RGB if provided
    overlay_rgb = None
    if background is not None:
        bg = background
        if bg.ndim == 2:
            bg = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
        if bg.shape[0] != H or bg.shape[1] != W:
            bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)
        overlay_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    img = hm.generate_heatmap(blur_sigma=blur_sigma, normalize=True, cmap=cmap, overlay_on=overlay_rgb, alpha=alpha)

    # Optional title: draw using OpenCV for simplicity
    if title:
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(
            bgr,
            str(title),
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return img


def save_heatmap_image(img_rgb: np.ndarray, out_path: str) -> None:
    """Save an RGB heatmap image to disk as PNG/JPEG."""
    if img_rgb is None:
        raise ValueError("img_rgb is None")
    if img_rgb.ndim == 2:
        # grayscale -> BGR
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    else:
        bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)