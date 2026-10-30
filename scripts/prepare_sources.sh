#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/sources.lock.env"

RUNTIME_ROOT="${RUNTIME_ROOT:-$REPO_ROOT/runtime}"
mkdir -p "$RUNTIME_ROOT"

prepare_repo() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="$RUNTIME_ROOT/$name"
  local overlay_stamp="$destination/.viewgs-repro-overlay"

  if [[ -f "$overlay_stamp" ]]; then
    [[ "$(git -C "$destination" rev-parse HEAD)" == "$commit" ]] || {
      echo "ERROR: pinned commit changed under prepared source: $destination" >&2
      exit 1
    }
    echo "Reusing prepared source: $destination"
    return
  fi

  if [[ ! -d "$destination/.git" ]]; then
    [[ ! -e "$destination" ]] || {
      echo "ERROR: $destination exists but is not a Git checkout" >&2
      exit 1
    }
    git clone --no-tags "$url" "$destination"
  fi

  local configured_url
  configured_url="$(git -C "$destination" remote get-url origin)"
  [[ "$configured_url" == "$url" ]] || {
    echo "ERROR: unexpected origin for $destination: $configured_url" >&2
    exit 1
  }

  if ! git -C "$destination" diff --quiet || \
     ! git -C "$destination" diff --cached --quiet; then
    echo "ERROR: source checkout has tracked modifications: $destination" >&2
    echo "Remove runtime/ and rerun, or preserve the changes elsewhere." >&2
    exit 1
  fi

  git -C "$destination" fetch --no-tags origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

prepare_repo InstantSplat "$INSTANTSPLAT_URL" "$INSTANTSPLAT_COMMIT"
prepare_repo ViewCrafter "$VIEWCRAFTER_URL" "$VIEWCRAFTER_COMMIT"
prepare_repo ViewGS "$VIEWGS_URL" "$VIEWGS_COMMIT"

git -C "$RUNTIME_ROOT/InstantSplat" submodule update --init croco
git -C "$RUNTIME_ROOT/ViewGS" submodule update --init submodules/simple-knn

cp -a "$REPO_ROOT/overlays/viewcrafter/." "$RUNTIME_ROOT/ViewCrafter/"
cp -a "$REPO_ROOT/overlays/viewgs/." "$RUNTIME_ROOT/ViewGS/"
printf '%s\n' "$VIEWCRAFTER_COMMIT" > "$RUNTIME_ROOT/ViewCrafter/.viewgs-repro-overlay"
printf '%s\n' "$VIEWGS_COMMIT" > "$RUNTIME_ROOT/ViewGS/.viewgs-repro-overlay"

echo "Prepared pinned sources:"
echo "  InstantSplat $INSTANTSPLAT_COMMIT"
echo "  ViewCrafter  $VIEWCRAFTER_COMMIT + overlay"
echo "  ViewGS       $VIEWGS_COMMIT + overlay"
