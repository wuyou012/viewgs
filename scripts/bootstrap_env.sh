#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

CONDA_BIN="$CONDA_ROOT/bin/conda"
require_file "$CONDA_BIN"

if [[ ! -x "$PYTHON_BIN" ]]; then
  "$CONDA_BIN" env create -f "$REPO_ROOT/environment/environment.yml"
fi

"$PYTHON_BIN" -m pip install -r "$REPO_ROOT/environment/requirements-unified.txt"
# A cloned Conda environment can retain Pillow 11 files alongside stale 9.4
# metadata. Force a clean install because ViewCrafter uses Image.ANTIALIAS.
"$PYTHON_BIN" -m pip install --force-reinstall --no-deps Pillow==9.4.0

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-11.8}"
export CUDA_HOME
"$PYTHON_BIN" -m pip install --no-build-isolation \
  "$VIEWGS_ROOT/submodules/diff-gaussian-rasterization-confidence" \
  "$VIEWGS_ROOT/submodules/simple-knn"

"$PYTHON_BIN" -c "import torch, pytorch3d, open3d, diff_gaussian_rasterization, simple_knn; print('unified environment OK:', torch.__version__, torch.version.cuda)"
(cd "$INSTANTSPLAT_ROOT" && "$PYTHON_BIN" init_geo.py --help >/dev/null)
(cd "$VIEWCRAFTER_ROOT" && "$PYTHON_BIN" inference_diff.py --help >/dev/null)
(cd "$VIEWGS_ROOT" && "$PYTHON_BIN" train_mip_diff_1.py --help >/dev/null)
echo "Environment ready: $PYTHON_BIN"

