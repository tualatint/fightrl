#!/usr/bin/env bash
# End-to-end demo on the built-in mock game: train -> evaluate -> video.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}"

python -m fightrl.train --game mock \
  --opponent league:runs/demo/league \
  --steps "${1:-120000}" --n-envs 8 --out runs/demo \
  --save-every 20000 --device auto

python -m fightrl.evaluate --game mock --model runs/demo/best.zip \
  --opponent aggressive --episodes 10 --video runs/demo/replay.mp4
