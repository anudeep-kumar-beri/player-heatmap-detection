# main.py
import os
import cv2
import csv
import numpy as np

from heatmap import HeatmapGenerator
from src.detector import PlayerDetector
from src.tracker import PlayerTracker


# ---------------- CONFIG ----------------
VIDEO_PATH = os.getenv("VIDEO_PATH", "test.mp4")  # put your video here or set env VIDEO_PATH
OUTPUT_DIR = "outputs"
TRACKS_CSV = os.path.join(OUTPUT_DIR, "tracks", "player_positions.csv")
FINAL_HEATMAP = os.path.join(OUTPUT_DIR, "heatmaps", "final_heatmap.png")
OVERLAY_VIDEO_OUT = os.getenv("OVERLAY_VIDEO_OUT", os.path.join(OUTPUT_DIR, "heatmaps", "overlay.mp4"))  # set to empty string to disable

TARGET_WIDTH = int(os.getenv("TARGET_WIDTH", "640"))        # resize frames to this width (keep aspect ratio)
FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))               # process every Nth frame (3 => ~10 fps if 30 fps video)
HEATMAP_DOWNSCALE = int(os.getenv("HEATMAP_DOWNSCALE", "2")) # internal heat grid downscale (1=full res; 2 or 4 saves CPU/RAM)
ALPHA = float(os.getenv("HEATMAP_ALPHA", "0.35"))            # heatmap overlay opacity
SHOW_PREVIEW = os.getenv("SHOW_PREVIEW", "1") == "1"         # show live window (set SHOW_PREVIEW=0 to disable)
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

    # Probe first frame for size → create heatmap + writer
    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Empty or unreadable video.")

    frame0 = resize_keep_aspect(frame0, TARGET_WIDTH)
    h, w = frame0.shape[:2]

    heatmap = HeatmapGenerator(w, h, downscale=HEATMAP_DOWNSCALE)

    writer = None
    if OVERLAY_VIDEO_OUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Try to keep approx fps, else default to 10
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1 or np.isnan(fps):
            fps = 10
        writer = cv2.VideoWriter(OVERLAY_VIDEO_OUT, fourcc, fps / max(1, FRAME_SKIP), (w, h))

    # CSV logging
    csv_file = open(TRACKS_CSV, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["frame", "player_id", "x", "y"])

    # Rewind to process from first (already-read) frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        # Frame skipping
        if frame_idx % FRAME_SKIP != 0:
            continue

        frame = resize_keep_aspect(frame, TARGET_WIDTH)

        # Detect
        detections = detector.detect(frame)
        # Expected format per detection: [x1, y1, x2, y2, conf, cls]

        # Track
        tracked = tracker.update(detections)
        # Expected shape: Nx5 -> [x1, y1, x2, y2, id]

        # Collect centers for heatmap + log CSV
        centers = []
        for x1, y1, x2, y2, tid in tracked:
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            centers.append((cx, cy))
            csv_writer.writerow([frame_idx, int(tid), cx, cy])

            # Optional: draw tracks (debug)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (40, 220, 40), 2)
            cv2.putText(frame, f"ID {int(tid)}", (int(x1), max(0, int(y1)-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 40), 2)

        # Update heatmap
        heatmap.add_points(centers)

        # Render overlay + write/preview
        overlay = heatmap.render_overlay(frame, alpha=ALPHA)

        if writer is not None:
            writer.write(overlay)

        if SHOW_PREVIEW:
            cv2.imshow("Player Tracking + Heatmap", overlay)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    # Cleanup
    cap.release()
    if writer is not None:
        writer.release()
    csv_file.close()
    cv2.destroyAllWindows()

    # Save final standalone heatmap image
    heatmap.save_heatmap_image(FINAL_HEATMAP)

    print(f"✅ Done.\n- Tracks CSV: {TRACKS_CSV}\n- Final heatmap: {FINAL_HEATMAP}")
    if OVERLAY_VIDEO_OUT:
        print(f"- Overlay video: {OVERLAY_VIDEO_OUT}")


if __name__ == "__main__":
    main()
    
# main.py (end of file)
