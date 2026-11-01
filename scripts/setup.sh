#!/usr/bin/env bash
# One-shot setup for a real game: cores, ROM placement, start state, smoke test.
#
#   bash scripts/setup.sh /path/to/samsho2.zip
#
# The ROM stays yours: nothing here downloads, fetches or redistributes game
# content.  Cores are open source and come from buildbot.libretro.com.
set -euo pipefail
cd "$(dirname "$0")/.."

GAME="${GAME:-samsho2_neogeo}"
CORE="${CORE:-fbneo}"
ROM_SRC="${1:-}"

if [ -z "$ROM_SRC" ]; then
  echo "usage: bash scripts/setup.sh /path/to/samsho2.zip   (or set GAME=...)"
  exit 1
fi
if [ ! -f "$ROM_SRC" ]; then
  echo "error: $ROM_SRC not found"
  exit 1
fi

echo "==> python"
python3 -c "import gymnasium, stable_baselines3, torch" 2>/dev/null || {
  echo "missing dependencies; run: pip install -e '.[arena,dev]'"
  exit 1
}

echo "==> emulator core"
if [ ! -f "cores/${CORE}_libretro.so" ]; then
  bash scripts/fetch_cores.sh cores "$CORE"
else
  echo "already present: cores/${CORE}_libretro.so"
fi

echo "==> ROM"
mkdir -p roms
case "$GAME" in
  samsho2_neogeo) ROM_DST="roms/samsho2.zip" ;;
  *) echo "error: unknown GAME=$GAME"; exit 1 ;;
esac
if [ -f "$ROM_DST" ] && [ ! -L "$ROM_DST" ]; then
  echo "already present: $ROM_DST"
else
  rm -f "$ROM_DST"   # never follow an existing symlink into the user's own copy
  cp "$ROM_SRC" "$ROM_DST"
  echo "installed: $ROM_DST ($(du -h "$ROM_DST" | cut -f1))"
fi

echo "==> start state"
STATE="states/${GAME}.state"
if [ -f "$STATE" ]; then
  echo "already present: $STATE"
else
  echo "missing: $STATE"
  echo "record one by playing until the first match frame, then saving:"
  echo "    python -m fightrl.capture --game $GAME"
  echo "the config's start_state must point at it; episodes then start there."
  exit 1
fi

echo "==> smoke test (1 episode against the aggressive bot)"
python3 -m fightrl.evaluate --game "$GAME" --model weights/samsho2_ppo_best.zip \
  --opponent aggressive --episodes 1 --seed 0 | tail -8

echo
echo "ready.  train:  python -m fightrl.train --game $GAME --opponent aggressive \\"
echo "                     --steps 2000000 --n-envs 8 --out runs/${GAME}"
echo "        watch:  python -m fightrl.arena --game $GAME --port 8767"
