#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

SCENE_ROOT=""
SCENE_NAME=""
OUTPUT_ROOT=""
GPU=0
N_VIEWS=2
ITERATIONS=10000
DDIM_STEPS=50
FORCE=0

usage() {
  echo "Usage: $0 --scene-root PATH --scene-name NAME --output-root PATH [--gpu N] [--iterations N] [--ddim-steps N] [--force]"
}

while (($#)); do
  case "$1" in
    --scene-root) SCENE_ROOT="$2"; shift 2 ;;
    --scene-name) SCENE_NAME="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --n-views) N_VIEWS="$2"; shift 2 ;;
    --iterations) ITERATIONS="$2"; shift 2 ;;
    --ddim-steps) DDIM_STEPS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ -n "$SCENE_ROOT" && -n "$SCENE_NAME" && -n "$OUTPUT_ROOT" ]] || { usage; exit 2; }
validate_runtime
require_dir "$SCENE_ROOT/images"
require_dir "$SCENE_ROOT/images_full"

SPARSE_PATH="$SCENE_ROOT/sparse_$N_VIEWS"
DIFF_ROOT="$OUTPUT_ROOT/diffusion"
DIFF_PATH="$DIFF_ROOT/$SCENE_NAME"
MODEL_PATH="$OUTPUT_ROOT/reconstruction/$SCENE_NAME"
PAIR_DIR="$OUTPUT_ROOT/inputs/${SCENE_NAME}_${N_VIEWS}view_pair"
LOG_DIR="$OUTPUT_ROOT/logs/$SCENE_NAME"
mkdir -p "$DIFF_ROOT" "$PAIR_DIR" "$LOG_DIR"
export CUDA_VISIBLE_DEVICES="$GPU"

if [[ $FORCE -eq 1 || ! -f "$SPARSE_PATH/0/points3D.ply" ]]; then
  (cd "$INSTANTSPLAT_ROOT" && "$PYTHON_BIN" "$SCRIPT_DIR/run_init_geo_compat.py" "$INSTANTSPLAT_ROOT/init_geo.py" \
    --source_path "$SCENE_ROOT" \
    --model_path "$LOG_DIR/mast3r" \
    --ckpt_path "$MAST3R_CKPT" \
    --device cuda --n_views "$N_VIEWS" --co_vis_dsp)
fi

mapfile -t IMAGE_NAMES < <(awk 'NF >= 10 && $1 ~ /^[0-9]+$/ && $10 ~ /\.(png|PNG|jpg|JPG|jpeg|JPEG)$/ {print $10}' "$SPARSE_PATH/0/images.txt")
[[ ${#IMAGE_NAMES[@]} -eq $N_VIEWS ]] || die "expected $N_VIEWS selected images, found ${#IMAGE_NAMES[@]}"
for name in "${IMAGE_NAMES[@]}"; do
  src="$SCENE_ROOT/images_full/$name"
  [[ -f "$src" ]] || src="$SCENE_ROOT/images/$name"
  require_file "$src"
  ln -sfn "$src" "$PAIR_DIR/$name"
done

# The current ViewCrafter loader always reserves index 0 as an LLFF test view.
# Add a disposable duplicate before the two real endpoints so both endpoints
# remain in its training pair without changing the ViewCrafter source tree.
first_src="$SCENE_ROOT/images_full/${IMAGE_NAMES[0]}"
[[ -f "$first_src" ]] || first_src="$SCENE_ROOT/images/${IMAGE_NAMES[0]}"
first_ext=".${IMAGE_NAMES[0]##*.}"
ln -sfn "$first_src" "$PAIR_DIR/000000_holdout$first_ext"

diff_count=0
if [[ -d "$DIFF_PATH/diffusion" ]]; then
  diff_count="$(find "$DIFF_PATH/diffusion" -maxdepth 1 -type f -name '*.png' | wc -l)"
fi
if [[ $FORCE -eq 1 || $diff_count -lt 25 ]]; then
  (cd "$VIEWCRAFTER_ROOT" && "$PYTHON_BIN" inference_diff.py \
    --image_dir "$PAIR_DIR" \
    --exp_name "$SCENE_NAME" \
    --out_dir "$DIFF_ROOT" \
    --cam_dir ./dataset \
    --mode sparse_view_interp \
    --n_views "$N_VIEWS" \
    --bg_trd 0.2 --seed 123 \
    --ckpt_path "$VIEWCRAFTER_CKPT" \
    --config configs/inference_pvd_512.yaml \
    --ddim_steps "$DDIM_STEPS" --video_length 25 \
    --device cuda:0 --height 320 --width 512 \
    --model_path "$DUST3R_CKPT")
fi

"$SCRIPT_DIR/run_reconstruction.sh" \
  --source-path "$SPARSE_PATH" \
  --diff-path "$DIFF_PATH" \
  --model-path "$MODEL_PATH" \
  --gpu "$GPU" --n-views "$N_VIEWS" --iterations "$ITERATIONS"

