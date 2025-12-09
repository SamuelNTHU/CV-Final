#!/usr/bin/env bash
set -euo pipefail

# Minimal helper to run Gaussian Grouping after masks are ready.
# Usage: ./run_gaussian_grouping.sh <scene_name> [scale]
# Expects:
#   - Frames under Gaussian-Grouping/data/<scene_name>/images or images_<scale>
#   - Masks under Gaussian-Grouping/data/<scene_name>/object_mask
#   - COLMAP available; Gaussian-Grouping dependencies installed. 

SCENE="${1:-}"
SCALE="${2:-1}"

if [[ -z "$SCENE" ]]; then
  echo "Usage: $0 <scene_name> [scale]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GG_ROOT_DEFAULT="$(cd "$SCRIPT_DIR/../../Gaussian-Grouping" 2>/dev/null && pwd || true)"
GG_ROOT="${GG_ROOT:-$GG_ROOT_DEFAULT}"
if [[ -z "$GG_ROOT" || ! -d "$GG_ROOT" ]]; then
  echo "Set GG_ROOT to your Gaussian-Grouping path (current: '$GG_ROOT')." >&2
  exit 1
fi

DATA_DIR="${DATA_DIR:-$GG_ROOT/data/$SCENE}"
IMG_DIR="$DATA_DIR/images"
if [[ "$SCALE" != "1" ]]; then
  IMG_DIR="${DATA_DIR}/images_${SCALE}"
fi

if [[ ! -d "$IMG_DIR" ]]; then
  echo "Image dir not found: $IMG_DIR" >&2
  exit 1
fi
if [[ ! -d "$DATA_DIR/object_mask" ]]; then
  echo "Mask dir not found: $DATA_DIR/object_mask" >&2
  exit 1
fi

# Basic dependency check so we fail fast with guidance instead of Python tracebacks.
MISSING_PKGS="$(python - <<'PY'
import importlib.util
mods = ["diff_gaussian_rasterization", "plyfile", "simple_knn"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
print(" ".join(missing))
PY
)"
if [[ -n "$MISSING_PKGS" ]]; then
  cat <<EOF >&2
Missing Python packages: $MISSING_PKGS
Install them in your Gaussian Grouping environment:
  pip install plyfile==0.8.1
  pip install "$GG_ROOT/submodules/diff-gaussian-rasterization"
  pip install "$GG_ROOT/submodules/simple-knn"
EOF
  exit 1
fi

cd "$GG_ROOT"
python convert.py -s "$DATA_DIR"
bash script/train.sh "$SCENE" "$SCALE"
