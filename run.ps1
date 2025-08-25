# PowerShell runner for the pipeline
# Usage:
#   .\run.ps1 [video_path]
# Env vars:
#   $env:SHOW_PREVIEW=0 to disable imshow (useful on headless servers)
#   $env:FRAME_SKIP=3 etc.

param(
    [string]$video_path = ""
)

if ($video_path -ne "") {
    $env:VIDEO_PATH = $video_path
}

$VENV_PATH = ".venv"
$PYTHON_EXEC = Join-Path $VENV_PATH "Scripts/python.exe"

if (-not (Test-Path $VENV_PATH)) {
    Write-Host "Virtual environment not found. Creating one..."
    python -m venv $VENV_PATH
}

& $PYTHON_EXEC -m pip install -r requirements.txt
& $PYTHON_EXEC main.py
