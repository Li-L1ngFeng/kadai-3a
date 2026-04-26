#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
IMG_DIR="${1:-/export/space0/li-l/imgdata/kadai3a}"
OUT_DIR="${2:-$ROOT_DIR/features}"
PY_PREFIX="/usr/local/anaconda3/bin/conda run -p /home/yanai-lab/li-l/.conda/envs/env1 --no-capture-output python /host/home/yanai-lab/Sotsuken25/li-l/.vscode-server/extensions/ms-python.python-2026.4.0-linux-x64/python_files/get_output_via_markers.py"

mkdir -p "$OUT_DIR"

echo "[1/4] Extract histogram features..."
$PY_PREFIX "$ROOT_DIR/scripts/extract_features.py" \
  --img_dir "$IMG_DIR" \
  --out "$OUT_DIR/mvp_hist_features.npz" \
  --bins 8

echo "[2/4] Extract gabor features..."
$PY_PREFIX "$ROOT_DIR/scripts/extract_gabor_features.py" \
  --img_dir "$IMG_DIR" \
  --out "$OUT_DIR/feat_gabor.npz"

echo "[3/4] Merge gabor into index..."
$PY_PREFIX "$ROOT_DIR/scripts/merge_feature_npz.py" \
  --base "$OUT_DIR/mvp_hist_features.npz" \
  --extra "$OUT_DIR/feat_gabor.npz" \
  --out "$OUT_DIR/index_plus_gabor.npz"

echo "[4/4] Try DCNN (optional)..."
set +e
$PY_PREFIX "$ROOT_DIR/scripts/extract_dcnn_features.py" \
  --img_dir "$IMG_DIR" \
  --out "$OUT_DIR/feat_dcnn.npz" \
  --batch_size 8 \
  --backend torch
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
  $PY_PREFIX "$ROOT_DIR/scripts/merge_feature_npz.py" \
    --base "$OUT_DIR/index_plus_gabor.npz" \
    --extra "$OUT_DIR/feat_dcnn.npz" \
    --out "$OUT_DIR/index_full.npz"
  echo "[OK] Full index generated: $OUT_DIR/index_full.npz"
else
  echo "[WARN] DCNN step skipped (backend missing)."
  echo "[OK] Current best index: $OUT_DIR/index_plus_gabor.npz"
fi
