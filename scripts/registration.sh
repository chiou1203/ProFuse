#!/usr/bin/env bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRSPLAT_ROOT="$REPO_ROOT/registration/DrSplat"

# ---- User: change this to your scene dir ----
SCENE_DIR="/ABSOLUTE/PATH/TO/teatime"
# ---------------------------------------------
CKPT_PATH="$SCENE_DIR/teatime/chkpnt30000.pth"

cd "$DRSPLAT_ROOT"

python registration.py \
    -s "$SCENE_DIR" \
    -m "$SCENE_DIR" \
    --start_checkpoint "$CKPT_PATH" \
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
