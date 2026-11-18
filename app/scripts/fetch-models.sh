#!/usr/bin/env bash
# Download the trained ONNX models from a GitHub release asset and verify their checksums.
# The models carry neuron-level connectivity derived from hemibrain (CC BY 4.0), so they are
# published as release assets and on the site, never committed to Git (decision D-040).
#
# Usage: app/scripts/fetch-models.sh [release-tag]   (needs `gh`, authenticated, or GH_TOKEN)
set -euo pipefail

TAG="${1:-models-v1}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/public/models"
MANIFEST="$DIR/manifest.json"

mkdir -p "$DIR"
gh release download "$TAG" --repo kat1478/NeuroFlyNav --pattern "*.onnx" --dir "$DIR" --clobber
python3 - "$MANIFEST" "$DIR" <<'PY'
import hashlib, json, pathlib, sys
manifest, folder = json.loads(pathlib.Path(sys.argv[1]).read_text()), pathlib.Path(sys.argv[2])
for name, model in manifest["models"].items():
    data = (folder / model["file"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != model["sha256"]:
        sys.exit(f"checksum mismatch for {model['file']}: {digest}")
    print(f"{model['file']}: {len(data)} bytes, sha256 ok")
PY
