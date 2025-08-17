#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

# Install dependencies
pip install -r requirements.txt

# Run the player heatmap pipeline
python -m src.run --video data/match.mp4 --out outputs --per_player
