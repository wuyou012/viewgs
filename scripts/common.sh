#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -f "$REPO_ROOT/.env" ]]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
fi

CONDA_ROOT="${CONDA_ROOT:-/mnt/ssd2/zongqi/miniconda3}"
ENV_NAME="${ENV_NAME:-viewgs-unified}"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_ROOT/envs/$ENV_NAME/bin/python}"
INSTANTSPLAT_ROOT="${INSTANTSPLAT_ROOT:-/mnt/ssd2/zongqi/ViewGS/work1/InstantSplat}"
VIEWCRAFTER_ROOT="${VIEWCRAFTER_ROOT:-/mnt/ssd2/zongqi/ViewGS/work1/ViewCrafter}"
VIEWGS_ROOT="${VIEWGS_ROOT:-/mnt/ssd2/zongqi/ViewGS/work1/work5}"

MAST3R_CKPT="${MAST3R_CKPT:-$INSTANTSPLAT_ROOT/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth}"
VIEWCRAFTER_CKPT="${VIEWCRAFTER_CKPT:-$VIEWCRAFTER_ROOT/checkpoints/model_sparse.ckpt}"
DUST3R_CKPT="${DUST3R_CKPT:-$VIEWCRAFTER_ROOT/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth}"

die() { echo "ERROR: $*" >&2; exit 1; }
require_file() { [[ -f "$1" ]] || die "missing file: $1"; }
require_dir() { [[ -d "$1" ]] || die "missing directory: $1"; }

validate_runtime() {
  require_file "$PYTHON_BIN"
  require_dir "$INSTANTSPLAT_ROOT"
  require_dir "$VIEWCRAFTER_ROOT"
  require_dir "$VIEWGS_ROOT"
  require_file "$MAST3R_CKPT"
  require_file "$VIEWCRAFTER_CKPT"
  require_file "$DUST3R_CKPT"
}

