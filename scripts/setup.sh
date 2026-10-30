#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/prepare_sources.sh"
"$SCRIPT_DIR/bootstrap_env.sh"
"$SCRIPT_DIR/validate_install.sh"
