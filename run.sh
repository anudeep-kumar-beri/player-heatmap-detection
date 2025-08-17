#!/usr/bin/env bash
set -euo pipefail

# Simple runner for the pipeline
# Usage:
#   ./run.sh [video_path]
# Env vars:
#   SHOW_PREVIEW=0 to disable imshow (useful on headless servers)
#   FRAME_SKIP=3 etc.

VIDEO_PATH_ARG=${1:-}
if [[ -n "${VIDEO_PATH_ARG}" ]]; then
  export VIDEO_PATH="${VIDEO_PATH_ARG}"
fi

python3 -m pip install -r requirements.txt
python3 main.py
