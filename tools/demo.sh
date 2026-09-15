#!/bin/bash
# One command from a fresh checkout to a picture: fetch the published MaleCNS v1.0
# tables, compile them into a pack, and draw the activity film from a run on it.
#
#   ./tools/demo.sh                     # writes docs/figures/activity-film.png
#   ./tools/demo.sh /tmp/film.png       # writes somewhere else
#
# Every step is cached, so only the first run pays for it: the fetch keeps files
# whose size already matches, and the compile is skipped when the pack is there.
# What each step costs is printed as it finishes, because the first run is not
# quick -- the connectivity table alone is 1.0 GB, and how long that takes is your
# bandwidth, not this machine.
#
# MaleCNS v1.0 is CC BY 4.0. Credit FlyEM at HHMI Janelia, the University of
# Cambridge, the MRC LMB, Google Research and the MaleCNS collaboration.
# Release terms and citation: https://male-cns.janelia.org/download/
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="$ROOT/data/pack/male_cns_v1"
OUT="${1:-$ROOT/docs/figures/activity-film.png}"
# A bare `[ ... ] && ...` chain would end the script under `set -e` whenever the
# test fails, which is the fresh-checkout case this demo exists for.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x "$ROOT/.venv/bin/python" ]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python"
fi

step() {  # label, seconds
  printf '\n--- %s: %d s\n' "$1" "$2"
}

echo "drosophila-brain-mlx demo"
echo "  1. fetch   MaleCNS v1.0 tables (~1.06 GB, cached by size)"
echo "  2. compile data/pack/male_cns_v1 (skipped when it exists)"
echo "  3. draw    $OUT"

t0=$SECONDS
"$ROOT/tools/fetch_male_cns.sh"
step "fetch" $((SECONDS - t0))

t1=$SECONDS
if [ -f "$PACK/manifest.json" ]; then
  echo "pack already compiled: $PACK"
else
  "$PY" -m lif.compile_pack_malecns --out "$PACK"
fi
step "compile" $((SECONDS - t1))

t2=$SECONDS
"$PY" -m lif.activity_film --out "$OUT"
step "draw" $((SECONDS - t2))

step "total" $SECONDS
echo
echo "picture: $OUT"
echo "MaleCNS v1.0 is CC BY 4.0 -- credit FlyEM at HHMI Janelia, the University of"
echo "Cambridge, the MRC LMB, Google Research and the MaleCNS collaboration."
