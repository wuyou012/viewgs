#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"

validate_runtime

"$PYTHON_BIN" -c "import torch, pytorch3d, open3d, diff_gaussian_rasterization, simple_knn; print(torch.__version__, torch.version.cuda)"
(cd "$INSTANTSPLAT_ROOT" && "$PYTHON_BIN" "$SCRIPT_DIR/run_init_geo_compat.py" "$INSTANTSPLAT_ROOT/init_geo.py" --help >/dev/null)
(cd "$VIEWCRAFTER_ROOT" && "$PYTHON_BIN" inference_diff.py --help >/dev/null)
(cd "$VIEWGS_ROOT" && "$PYTHON_BIN" train_mip_diff_1.py --help >/dev/null)

echo "Install validation passed."
