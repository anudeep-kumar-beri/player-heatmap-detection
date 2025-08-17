# Player Tracking & Heatmap Generation

A minimal pipeline to detect and track players from a sports video and generate movement heatmaps.

Features
- YOLOv8 object detection (person class)
- ByteTrack multi-object tracking (via supervision)
- Optional homography mapping to a top-down canvas
- CSV export of tracked centers (frame, track_id, x, y, w, h, conf)
- Global and per-player heatmap images

Quickstart
1) Create a Python environment and install dependencies
   - pip install -r requirements.txt

2) Prepare an input video
   - Place your video under data/, e.g., data/match.mp4

3) (Optional) Prepare homography YAML
   - File example (paths are in image coordinates, clockwise order):
```yaml
src_points: [[100,700],[1800,700],[1900,100],[50,100]]
dst_points: [[0,720],[1280,720],[1280,0],[0,0]]
canvas_size: [1280,720]
```

4) Run
   - python -m src.run --video data/match.mp4 --out outputs --model yolov8n.pt --per_player
   - With homography: add --homography data/homography.yaml
   - Resize processing: --resize 1280x720
   - Process every Nth frame: --stride 2

Outputs
- outputs/tracks.csv
- outputs/heatmap_all.png
- outputs/heatmap_player_<id>.png (when --per_player)

Notes
- For GPU acceleration, install the PyTorch version matching your CUDA, then reinstall ultralytics if necessary.
- The default model yolov8n.pt is downloaded on first run.
- Adjust --conf and --iou for detection quality.
