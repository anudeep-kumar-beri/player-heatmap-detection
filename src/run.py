import os
import sys
from pathlib import Path

import cv2

from .utils import (
    HomographyMapper,
    HomographyConfig,
    PlayerTrackerPipeline,
    build_arg_parser,
    df_points,
    load_homography_from_yaml,
    make_dirs,
    save_tracks_csv,
    parse_resize,
)
from .heatmap import generate_heatmap, save_heatmap_image


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    make_dirs(str(out_dir))

    resize = parse_resize(args.resize)

    homography = None
    canvas_size = None
    if args.homography:
        cfg = load_homography_from_yaml(args.homography)
        homography = HomographyMapper(cfg)
        canvas_size = cfg.canvas_size

    pipeline = PlayerTrackerPipeline(
        video_path=args.video,
        model_path=args.model,
        conf=args.conf,
        iou=args.iou,
        frame_stride=args.stride,
        resize=resize,
        homography=homography,
    )

    df, default_canvas = pipeline.run(progress=True)
    if canvas_size is None:
        canvas_size = default_canvas

    tracks_csv = out_dir / "tracks.csv"
    save_tracks_csv(df, str(tracks_csv))

    # Optional background: first frame resized to canvas
    cap = cv2.VideoCapture(args.video)
    ret, bg = cap.read()
    cap.release()
    if ret:
        if resize is not None:
            bg = cv2.resize(bg, resize)
        # If homography exists and dst is a rectangle, no easy inverse background. Use resized first frame.
        background = bg
    else:
        background = None

    # Global heatmap
    pts_all = df_points(df)
    heat_all = generate_heatmap(pts_all, canvas_size=canvas_size, title="All Players", background=background)
    save_heatmap_image(heat_all, str(out_dir / "heatmap_all.png"))

    # Per-player heatmaps
    if args.per_player and not df.empty:
        for tid, sub in df.groupby("track_id"):
            pts = df_points(sub)
            heat = generate_heatmap(pts, canvas_size=canvas_size, title=f"Player {tid}", background=background)
            save_heatmap_image(heat, str(out_dir / f"heatmap_player_{int(tid)}.png"))

    print(f"Saved: {tracks_csv}")
    print(f"Saved: {out_dir / 'heatmap_all.png'}")
    if args.per_player:
        print("Saved per-player heatmaps in output directory")


if __name__ == "__main__":
    main()
