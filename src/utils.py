import argparse
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

try:
    # ultralytics + supervision provide YOLOv8 inference and ByteTrack
    from ultralytics import YOLO
    import supervision as sv
except Exception as e:  # pragma: no cover
    YOLO = None  # type: ignore
    sv = None  # type: ignore


@dataclass
class HomographyConfig:
    src_points: List[List[float]]  # 4 points in image/frame space
    dst_points: List[List[float]]  # 4 points in target/canvas space
    canvas_size: Tuple[int, int]  # (width, height) of the target canvas


class HomographyMapper:
    """Maps 2D points from image coordinates to a target top-down canvas using homography."""

    def __init__(self, config: HomographyConfig):
        self.config = config
        src = np.array(config.src_points, dtype=np.float32)
        dst = np.array(config.dst_points, dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst)

    def map_point(self, x: float, y: float) -> Tuple[int, int]:
        pts = np.array([[[x, y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(pts, self.M)[0, 0]
        return int(round(mapped[0])), int(round(mapped[1]))

    def map_points(self, xy: np.ndarray) -> np.ndarray:
        # xy shape: (N, 2)
        pts = xy.reshape(-1, 1, 2).astype(np.float32)
        mapped = cv2.perspectiveTransform(pts, self.M).reshape(-1, 2)
        return mapped


def load_homography_from_yaml(path: str) -> HomographyConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    src = data["src_points"]
    dst = data["dst_points"]
    canvas_size = tuple(data.get("canvas_size", [1280, 720]))
    return HomographyConfig(src_points=src, dst_points=dst, canvas_size=(int(canvas_size[0]), int(canvas_size[1])))


class PlayerTrackerPipeline:
    def __init__(
        self,
        video_path: str,
        model_path: str = "yolov8n.pt",
        conf: float = 0.4,
        iou: float = 0.5,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,  # COCO class ids, 0 for person
        frame_stride: int = 1,
        resize: Optional[Tuple[int, int]] = None,  # (width, height)
        homography: Optional[HomographyMapper] = None,
    ) -> None:
        self.video_path = video_path
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = device
        self.classes = classes if classes is not None else [0]  # default: person
        self.frame_stride = max(1, frame_stride)
        self.resize = resize
        self.homography = homography

        if YOLO is None or sv is None:
            raise RuntimeError(
                "Required packages not found. Please install 'ultralytics' and 'supervision' per requirements.txt"
            )
        self.model = YOLO(self.model_path)
        self.tracker = sv.ByteTrack()

    def run(self, progress: bool = True) -> Tuple[pd.DataFrame, Tuple[int, int]]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.resize is not None:
            out_w, out_h = self.resize
            canvas_size = (out_w, out_h)
        else:
            canvas_size = (width, height)

        # Storage for tracking logs
        rows = []

        pbar_iter: Iterable = range(total_frames) if total_frames > 0 else iter(int, 1)  # infinite if unknown
        if progress and total_frames > 0:
            pbar_iter = tqdm(range(total_frames), desc="Processing frames", unit="f")

        frame_idx = 0
        iter_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if self.resize is not None:
                frame = cv2.resize(frame, self.resize)

            if iter_count % self.frame_stride != 0:
                iter_count += 1
                if progress and total_frames > 0:
                    try:
                        pbar_iter.update(1)
                    except Exception:  # tqdm range fallback
                        pass
                continue

            # Inference
            result = self.model(frame, conf=self.conf, iou=self.iou, device=self.device, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)

            # Filter classes
            if detections.class_id is not None and len(detections) > 0:
                mask = np.isin(detections.class_id, np.array(self.classes))
                detections = detections[mask]

            # Update tracker
            tracked = self.tracker.update_with_detections(detections)

            # Log centers
            for i in range(len(tracked)):
                xyxy = tracked.xyxy[i]
                track_id = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else -1
                conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0

                cx = float((xyxy[0] + xyxy[2]) / 2.0)
                cy = float((xyxy[1] + xyxy[3]) / 2.0)

                if self.homography is not None:
                    cx_m, cy_m = self.homography.map_point(cx, cy)
                else:
                    cx_m, cy_m = int(round(cx)), int(round(cy))

                rows.append(
                    {
                        "frame": frame_idx,
                        "track_id": track_id,
                        "x": cx_m,
                        "y": cy_m,
                        "w": float(xyxy[2] - xyxy[0]),
                        "h": float(xyxy[3] - xyxy[1]),
                        "conf": conf,
                    }
                )

            frame_idx += 1
            iter_count += 1

            if progress and total_frames > 0:
                try:
                    pbar_iter.update(1)
                except Exception:
                    pass

        cap.release()

        df = pd.DataFrame(rows)
        return df, canvas_size


def make_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_tracks_csv(df: pd.DataFrame, out_path: str) -> None:
    df.to_csv(out_path, index=False)


def df_points(df: pd.DataFrame, track_id: Optional[int] = None) -> np.ndarray:
    d = df if track_id is None else df[df["track_id"] == track_id]
    return d[["x", "y"]].to_numpy().astype(np.int32)


def parse_resize(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if not s:
        return None
    parts = s.lower().split("x")
    if len(parts) != 2:
        raise ValueError("resize must be in the form WIDTHxHEIGHT, e.g., 1280x720")
    return int(parts[0]), int(parts[1])


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Player tracking and heatmap generation")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model weights path or name")
    ap.add_argument("--conf", type=float, default=0.4, help="Detection confidence threshold")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    ap.add_argument("--resize", type=str, default=None, help="Resize frames: WIDTHxHEIGHT")
    ap.add_argument("--homography", type=str, default=None, help="Path to YAML with src/dst points and canvas_size")
    ap.add_argument("--per_player", action="store_true", help="Generate per-player heatmaps")
    return ap


def compute_per_track_metrics(df: pd.DataFrame, fps: float, frame_stride: int = 1, meters_per_pixel: float | None = None) -> pd.DataFrame:
    """
    Compute simple metrics per track:
    - duration_s
    - distance_px, distance_m (if meters_per_pixel provided)
    - avg_speed_px_s, max_speed_px_s
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "track_id", "duration_s", "distance_px", "distance_m",
            "avg_speed_px_s", "max_speed_px_s"
        ])

    results = []
    dt = (frame_stride / fps) if fps > 0 else 0.0
    for tid, g in df.sort_values(["track_id", "frame"]).groupby("track_id"):
        xs = g["x"].to_numpy(dtype=np.float32)
        ys = g["y"].to_numpy(dtype=np.float32)
        frames = g["frame"].to_numpy()
        if len(xs) < 2:
            duration_s = 0.0
            dist_px = 0.0
            avg_speed = 0.0
            max_speed = 0.0
        else:
            # account for potential skipped frames (non-consecutive ids)
            dxy = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
            # time deltas can vary if frames not consecutive; approximate by dt * frame gaps
            frame_gaps = np.maximum(1, np.diff(frames))
            time_deltas = frame_gaps * dt if dt > 0 else np.zeros_like(frame_gaps, dtype=np.float32)
            duration_s = float(time_deltas.sum())
            dist_px = float(dxy.sum())
            # instantaneous speed per step
            inst_speed = np.divide(dxy, time_deltas, out=np.zeros_like(dxy), where=time_deltas > 0)
            avg_speed = float(inst_speed.mean()) if inst_speed.size else 0.0
            max_speed = float(inst_speed.max()) if inst_speed.size else 0.0
        dist_m = (dist_px * meters_per_pixel) if meters_per_pixel else None
        results.append({
            "track_id": int(tid),
            "duration_s": duration_s,
            "distance_px": dist_px,
            "distance_m": dist_m,
            "avg_speed_px_s": avg_speed,
            "max_speed_px_s": max_speed,
        })
    return pd.DataFrame(results)


# This file exposes the core pipeline and CLI args; the actual CLI is implemented in run.py
