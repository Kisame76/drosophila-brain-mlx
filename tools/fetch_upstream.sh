#!/bin/bash
# Downloads the published FlyWire v630 model files (~90 MB), the Brian2 reference
# implementation, its example notebook and the five result files upstream
# published for that notebook (~3 MB), which python -m lif.validate_notebook
# compares against, from philshiu/Drosophila_brain_model (MIT).
# Nothing here is redistributed with this repository.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/raw"; REF="$ROOT/data/ref"; RES="$ROOT/data/ref/results/example"
BASE="https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"
mkdir -p "$RAW" "$REF" "$RES"
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
fetch "$BASE/example.ipynb" "$REF/example.ipynb" 10282
echo "published example results -> data/ref/results/example/"
fetch "$BASE/results/example/sugarR-720575940617937543.parquet" "$RES/sugarR-720575940617937543.parquet" 537629
fetch "$BASE/results/example/sugarR-720575940621754367.parquet" "$RES/sugarR-720575940621754367.parquet" 553968
fetch "$BASE/results/example/sugarR-720575940622695448.parquet" "$RES/sugarR-720575940622695448.parquet" 578825
fetch "$BASE/results/example/sugarR.parquet"                    "$RES/sugarR.parquet"                    948526
fetch "$BASE/results/example/sugarR_100Hz.parquet"              "$RES/sugarR_100Hz.parquet"              557849
echo
echo "expected sha256:"
echo "  completeness_630.csv               e6b71e17671a9bdb05f55e4bc6774640a1418cb7a05125e0fc994ad40f9bfdfb"
echo "  connectivity_630.parquet           94db8c650533bc36ffa3223f2e62325d5648b8d6bd31c3a4e1c804628c7557b3"
echo "  example.ipynb                      1737c3043af700504c4791c4bfc02d1856bbeb0d36c469832e4b9399dfddda3a"
echo "  sugarR-720575940617937543.parquet  e0d39699bd979a5761138dbab964dbfe1c55feb945349cd24bc1c1f9a925753b"
echo "  sugarR-720575940621754367.parquet  f7d7cd7386663cbce864d6ca132ce5992d0cdb65f281b30768c3afe0804514a6"
echo "  sugarR-720575940622695448.parquet  c6b5e102249e23c30989ad66ad897d81a6453681ba3756dd30c83e956aee803e"
echo "  sugarR.parquet                     83921d50089d966785e402d6c0e3be5ab3eba5beb680410a09c557b0f22dcc28"
echo "  sugarR_100Hz.parquet               93722ec03c3faa5a85790d77ed16bdd7f1e38a7c847ed0bc3af3dc2e4d0c3d34"
shasum -a 256 "$RAW/completeness_630.csv" "$RAW/connectivity_630.parquet" "$REF/example.ipynb" "$RES"/*.parquet
