#!/bin/bash
# Downloads the published MaleCNS v1.0 flat-connectome tables (~1.06 GB, CC BY 4.0)
# from the FlyEM release bucket. Nothing here is redistributed with this
# repository. Resumable: re-run it after an interruption.
#
#   ./tools/fetch_male_cns.sh
#   python -m lif.compile_pack_malecns
#
# Release terms and citation: https://male-cns.janelia.org/download/
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/data/raw/male_cns"
BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
mkdir -p "$DEST"

fetch() {  # filename, expected bytes
  local out="$DEST/$1" sz=0
  [ -f "$out" ] && sz=$(stat -f%z "$out" 2>/dev/null || stat -c%s "$out" 2>/dev/null || echo 0)
  if [ "$sz" = "$2" ]; then echo "  ok    $1"; return; fi
  echo "  fetch $1 ($2 bytes)"
  # -C - resumes a partial file; the bucket is slow enough that this matters.
  curl -fL --progress-bar --retry 10 --retry-all-errors --retry-delay 5 -C - -o "$out" "$BASE/$1"
}

echo "MaleCNS v1.0 -> data/raw/male_cns/"
fetch body-annotations-male-cns-v1.0-minconf-0.5.feather        14483314
fetch body-neurotransmitters-male-cns-v1.0.feather              43282834
fetch connectome-weights-male-cns-v1.0-minconf-0.5.feather    1051241946

echo
echo "sha256 (recorded in the pack manifest when you compile):"
shasum -a 256 "$DEST"/*.feather
echo
echo "MaleCNS v1.0 is CC BY 4.0. Credit FlyEM at HHMI Janelia, the University of"
echo "Cambridge, the MRC LMB, Google Research and the MaleCNS collaboration."
