#!/usr/bin/env bash
# Run the annotation UI locally: instant saves to data/annotations/, pushed to the HF dataset in the
# background with the local HF token. Same gradio version as the Space (space/README.md sdk_version).
# Open http://127.0.0.1:7860. Leave it running a few seconds after the last save so the push finishes;
# anything unpushed is re-queued at the next start.
set -euo pipefail
cd "$(dirname "$0")/.."
ANNOTATOR_LOCAL=1 exec uv run --no-project --with gradio==5.49.1 --with huggingface_hub python space/app.py "$@"
