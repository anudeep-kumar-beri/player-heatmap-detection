# main.py
import os
import cv2
import csv
import numpy as np
from tqdm import tqdm

from heatmap import HeatmapGenerator
from src.detector import PlayerDetector
from src.tracker import PlayerTracker


# ---------------- CONFIG ----------------
VIDEO_PATH = os.getenv("VIDEO_PATH", "test.mp4")  # put your video here or set env VIDEO_PATH
OUTPUT_DIR = "outputs"
TRACKS_CSV = os.path.join(OUTPUT_DIR, "tracks", "player_positions.csv")
FINAL_HEATMAP = os.path.join(OUTPUT_DIR, "heatmaps", "final_heatmap.png")
OVERLAY_VIDEO_OUT = os.getenv("OVERLAY_VIDEO_OUT", os.path.join(OUTPUT_DIR, "heatmaps", "overlay.mp4"))  # set to empty string to disable
FINAL_HEATMAP_PITCH = os.getenv("FINAL_HEATMAP_PITCH", os.path.join(OUTPUT_DIR, "heatmaps", "final_heatmap_on_pitch.png"))
PITCH_IMAGE = os.getenv("PITCH_IMAGE", "").strip()  # optional custom pitch image path

TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", "640"))        # resize frames to this width (keep aspect ratio)
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))               # process every Nth frame (3 => ~10 fps if 30 fps video)
HEATMAP_DOWNSCALE = int(os.getenv("HEATMAP_DOWNSCALE", "2")) # internal heat grid downscale (1=full res; 2 or 4 saves CPU/RAM)
ALPHA = float(os.getenv("HEATMAP_ALPHA", "0.35"))            # heatmap overlay opacity
# Optional fixed normalization max for stable overlay intensity over long videos (set empty to auto)
HEATMAP_FIXED_MAX_ENV = os.getenv("HEATMAP_FIXED_MAX", "").strip()
HEATMAP_FIXED_MAX = float(HEATMAP_FIXED_MAX_ENV) if HEATMAP_FIXED_MAX_ENV else None
SHOW_PREVIEW = False  # Force silent mode, no live preview
# ----------------------------------------


def ensure_dirs():
    os.makedirs(os.path.join(OUTPUT_DIR, "tracks"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "heatmaps"), exist_ok=True)


def resize_keep_aspect(frame, target_width):
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    scale = target_width / float(w)
    new_w = target_width
    new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def main():
    ensure_dirs()

    # Initialize detector/tracker
    detector = PlayerDetector("yolov8n.pt")
    tracker = PlayerTracker()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    # Probe first frame for size → create heatmap + compute output sizes
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Empty or unreadable video.")

    frame0 = resize_keep_aspect(frame0, TARGET_WIDTH)
    h, w = frame0.shape[:2]

    # Create heatmap at resized resolution (optionally with fixed normalization)
    heatmap = HeatmapGenerator(w, h, downscale=HEATMAP_DOWNSCALE, fixed_norm_max=HEATMAP_FIXED_MAX)

    # Rewind to start for processing
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0
    print(f"Video: {VIDEO_PATH} | {total_frames} frames @ {fps:.1f} fps")

    # Prepare outputs
    writer = None
    if OVERLAY_VIDEO_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OVERLAY_VIDEO_OUT, fourcc, fps, (w, h))
    csv_file = open(TRACKS_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "id", "x", "y"])

    frame_idx = 0
    with tqdm(total=total_frames, desc="Processing frames", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % FRAME_SKIP != 0:
                frame_idx += 1
                pbar.update(1)
                continue
            frame = resize_keep_aspect(frame, TARGET_WIDTH)
            # Detect players
            detections = detector.detect(frame)
            # Track players
            tracks = tracker.update(detections)
            centers = [(int((x1 + x2) / 2), int((y1 + y2) / 2)) for x1, y1, x2, y2, tid in tracks]
            for x1, y1, x2, y2, tid in tracks:
                csv_writer.writerow([frame_idx, int(tid), int((x1 + x2) / 2), int((y1 + y2) / 2)])
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (40, 220, 40), 2)
                cv2.putText(frame, f"ID {int(tid)}", (int(x1), max(0, int(y1)-8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 2)

            # Update heatmap
            heatmap.add_points(centers)

            # Render overlay + write

            overlay = heatmap.render_overlay(frame, alpha=ALPHA)

            if writer is not None:
                writer.write(overlay)

            # No live preview, just progress bar
            frame_idx += 1
            pbar.update(1)

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # Save final standalone heatmap image
    heatmap.save_heatmap_image(FINAL_HEATMAP)
    # Save heatmap over pitch (generated or from PITCH_IMAGE)
    heatmap.save_heatmap_over_pitch(FINAL_HEATMAP_PITCH, pitch_image=PITCH_IMAGE, alpha=0.5)

    print(f"✅ Done.\n- Tracks CSV: {TRACKS_CSV}\n- Final heatmap: {FINAL_HEATMAP}")
    if OVERLAY_VIDEO_OUT:
        print(f"- Overlay video: {OVERLAY_VIDEO_OUT}")
    if FINAL_HEATMAP_PITCH:
        print(f"- Heatmap on pitch: {FINAL_HEATMAP_PITCH}")


if __name__ == "__main__":
    main()
    
# main.py (end of file)
