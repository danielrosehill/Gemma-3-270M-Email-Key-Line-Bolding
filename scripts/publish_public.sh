#!/usr/bin/env bash
# Refresh the public GitHub copy (danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding) from this repo's
# tracked files. The public repo has its own history, started 2026-10-21, so nothing from this private
# repo's history is ever pushed there. Commit here first; only committed files are copied.
set -euo pipefail
cd "$(dirname "$0")/.."
PUB="${PUB:-$HOME/repos/github/audio-dictation/Gemma-3-270M-Email-Key-Line-Bolding}"
[ -d "$PUB/.git" ] || gh repo clone danielrosehill/Gemma-3-270M-Email-Key-Line-Bolding "$PUB"
git -C "$PUB" pull -q --ff-only || true
find "$PUB" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git archive HEAD | tar -x -C "$PUB"
git -C "$PUB" add -A
git -C "$PUB" diff --cached --quiet && { echo "public copy already current"; exit 0; }
git -C "$PUB" commit -q -m "Sync from working repo $(git rev-parse --short HEAD)"
git -C "$PUB" push -q
echo "pushed $(git -C "$PUB" rev-parse --short HEAD)"
