#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REG_ROOT="$REPO_ROOT/feature_registration"

# ---- User: change this to your scene dir ----
#SCENE_DIR=

# 1) Remove any previously installed rasterizer
python -m pip uninstall -y diff-gaussian-rasterization diff_gaussian_rasterization || true

# 2) Install the langsplat rasterizer (IMPORTANT: point to the folder that contains setup.py / pyproject.toml)
python -m pip install -v "$REG_ROOT/submodules/langsplat-rasterization"

# 3) Verify which one is being imported
python - <<'PY'
import diff_gaussian_rasterization as d
print("Using diff_gaussian_rasterization from:", d.__file__)
PY

# ---------------------------------------------
CKPT_PATH="$SCENE_DIR/out_pre_registration/chkpnt30000.pth"

cd "$REG_ROOT"

python train.py \
    -s "/content/ramen/" \
    -m "/content/ramen/" \
    --start_checkpoint "/content/ramen/out_pre_registration/chkpnt30000.pth" \
    --feature_level 1 \
    --topk 10 \
    --name_extra pq_openclip \
    --use_pq \
    --pq_index ckpts/pq_index.faiss \
    --port 55560 \
    --eval \
    --use_clusters \
    --pairs_per_chunk 400000 \
    --pixel_stride 1 \
    --spmm_cluster_block 0 \
    --eps_contrib 0.0000
