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

# Activate venv, install, and run
VENV_PATH=".venv"

if [ ! -d "$VENV_PATH" ]; then
  echo "Virtual environment not found. Creating one..."
  python3 -m venv "$VENV_PATH"
fi

# Use python from venv
PYTHON_EXEC="$VENV_PATH/bin/python3"

"$PYTHON_EXEC" -m pip install -r requirements.txt
"$PYTHON_EXEC" main.py
