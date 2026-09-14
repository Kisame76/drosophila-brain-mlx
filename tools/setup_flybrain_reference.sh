#!/bin/bash
# Baut die flyBrain-Referenzengine fuer den Direktvergleich.
# Schneidet die MuJoCo-abhaengigen Module heraus: der Neural-Benchmark braucht
# sie nicht, und metal_engine.rs (der Rechenkern) bleibt dabei unangetastet.
set -e
export PATH="$HOME/.cargo/bin:$PATH"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK=$ROOT/work
mkdir -p "$WORK" && cd "$WORK"
[ -d flyBrain ] || git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 \
  clone --depth 1 --filter=blob:none --sparse https://github.com/mehrantsi/flyBrain.git
cd flyBrain
git sparse-checkout set rust vendor .cargo tools src
[ -f rust/src/lib.rs.orig ] || cp rust/src/lib.rs rust/src/lib.rs.orig
python3 - <<'PY'
from pathlib import Path
p=Path("Cargo.toml"); s=p.read_text()
for b in ("flybrain-world","cns-numerics","flybrain-browser"):
    i=s.find(f'name = "{b}"')
    if i<0: continue
    start=s.rfind("[[bin]]",0,i); end=s.find("[[bin]]",i)
    if end<0: end=s.find("[dependencies]",i)
    s=s[:start]+s[end:]
for blk in ('[target.\'cfg(target_os = "emscripten")\'.dependencies.mujoco-rs]\npath = "vendor/mujoco-rs"\n\n',
            '[target.\'cfg(target_os = "macos")\'.dependencies.mujoco-rs]\npath = "vendor/mujoco-rs"\nfeatures = ["viewer-ui", "renderer-winit-fallback"]\n\n',
            '[patch.crates-io]\nglutin = { git = "https://github.com/davidhozic/glutin", rev = "e95fe97f1857c0498ed2236c0e8d60e5a30a2854" }\n'):
    s=s.replace(blk,'')
p.write_text(s)
KEEP={"cns_pathway","fixture","npy","pack","parameters","protocol","reference","stimulus","output","metal_engine"}
out=[];pend=[]
for line in Path("rust/src/lib.rs.orig").read_text().splitlines():
    st=line.strip()
    if st.startswith("#["): pend.append(line); continue
    if st.startswith(("pub mod ","mod ")):
        if st.split("mod ",1)[1].rstrip(";").strip() in KEEP: out.extend(pend); out.append(line)
        pend=[]; continue
    if st=="" and pend: continue
    out.extend(pend); pend=[]; out.append(line)
out.extend(pend)
Path("rust/src/lib.rs").write_text("\n".join(out)+"\n")
PY
cargo build --release --bin flybrain-rs 2>&1 | tail -3
[ -d "$WORK/pack_v630" ] || PYTHONPATH="$WORK/flyBrain/src" "$ROOT/.venv/bin/python" -m flybrain.cli pack \
  --completeness "$ROOT/data/raw/completeness_630.csv" \
  --connectivity "$ROOT/data/raw/connectivity_630.parquet" \
  --output "$WORK/pack_v630" --materialization 630 >/dev/null
echo "READY: $WORK/flyBrain/target/release/flybrain-rs + $WORK/pack_v630"
