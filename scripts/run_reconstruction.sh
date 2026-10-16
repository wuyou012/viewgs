#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

SOURCE_PATH=""
DIFF_PATH=""
MODEL_PATH=""
GPU=0
N_VIEWS=2
ITERATIONS=10000
EXTRA_ARGS=()

usage() {
  echo "Usage: $0 --source-path PATH --diff-path PATH --model-path PATH [--gpu N] [--n-views N] [--iterations N] [-- EXTRA_TRAIN_ARGS...]"
}

while (($#)); do
  case "$1" in
    --source-path) SOURCE_PATH="$2"; shift 2 ;;
    --diff-path) DIFF_PATH="$2"; shift 2 ;;
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --n-views) N_VIEWS="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$SOURCE_PATH" && -n "$DIFF_PATH" && -n "$MODEL_PATH" ]] || { usage; exit 2; }
validate_runtime
require_file "$SOURCE_PATH/0/points3D.ply"
require_file "$SOURCE_PATH/0/images.txt"
require_dir "$DIFF_PATH/diffusion"
mkdir -p "$MODEL_PATH"

export CUDA_VISIBLE_DEVICES="$GPU"
cd "$VIEWGS_ROOT"
"$PYTHON_BIN" train_mip_diff_1.py \
  --source_path "$SOURCE_PATH" \
  --model_path "$MODEL_PATH" \
  --eval \
  --diff_path "$DIFF_PATH" \
  -i images_full \
  --use_diff --no_pseudo \
  --n_views "$N_VIEWS" \
  --sub_ratio 20 \
  --loss_downsample --l_down_ratio 0.5 \
  --dif_l1_loss_weight 0 \
  --dif_pear_loss_weight 1 \
  --gray_corr_weight 0 \
  --lap_weight 1 \
  --sample_pseudo_interval 1 \
  --training_iters "$ITERATIONS" \
  --save_iterations "$ITERATIONS" \
  "${EXTRA_ARGS[@]}"

