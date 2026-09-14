#!/bin/bash
# Downloads the published FlyWire v630 model files (~90 MB) and the Brian2
# reference implementation from philshiu/Drosophila_brain_model (MIT).
# Nothing here is redistributed with this repository.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"; REF="$ROOT/data/ref"
BASE="https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
mkdir -p "$RAW" "$REF"
fetch() {  # url, dest, expected bytes (0 = unknown)
  local sz=0; [ -f "$2" ] && sz=$(stat -f%z "$2" 2>/dev/null || echo 0)
  if [ "$3" != "0" ] && [ "$sz" = "$3" ]; then echo "  ok    $(basename "$2")"; return; fi
  echo "  fetch $(basename "$2")"
  curl -fL --progress-bar --retry 5 --retry-all-errors -C - -o "$2" "$1"
}
echo "model files -> data/raw/"
fetch "$BASE/2023_03_23_completeness_630_final.csv"     "$RAW/completeness_630.csv"     3057611
fetch "$BASE/2023_03_23_connectivity_630_final.parquet" "$RAW/connectivity_630.parquet" 86630944
echo "brian2 reference -> data/ref/"
for f in model.py utils.py Readme.md; do fetch "$BASE/$f" "$REF/$f" 0; done
echo
echo "expected sha256:"
echo "  completeness  e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb"
echo "  connectivity  94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3"
shasum -a 256 "$RAW/completeness_630.csv" "$RAW/connectivity_630.parquet"
