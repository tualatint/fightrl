#!/usr/bin/env bash
# Download the libretro cores used by the bundled game configs.
# Cores are open source; the games' ROMs are NOT included -- use your own dumps.
set -euo pipefail

DIR="${1:-cores}"
BASE="https://buildbot.libretro.com/nightly/linux/x86_64/latest"
mkdir -p "$DIR"

for core in snes9x genesis_plus_gx fbneo; do
  if [ -f "$DIR/${core}_libretro.so" ]; then
    echo "already present: ${core}_libretro.so"
    continue
  fi
  echo "downloading ${core} ..."
  curl -L --fail --retry 3 -o "$DIR/${core}.zip" "$BASE/${core}_libretro.so.zip"
  unzip -o -q "$DIR/${core}.zip" -d "$DIR"
  rm -f "$DIR/${core}.zip"
done

echo
echo "done. Place your ROMs in ./roms and the Neo Geo BIOS (neogeo.zip) in ./$DIR"
