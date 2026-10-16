#!/usr/bin/env python3
"""Run InstantSplat init_geo.py on PyTorch 1.13 without editing its source tree."""

import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: run_init_geo_compat.py /path/to/init_geo.py [args...]")

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        raise SystemExit(f"missing init_geo.py: {target}")

    # InstantSplat sets expandable_segments unconditionally. PyTorch 1.13 does
    # not recognize it, while the allocator works normally with the variable
    # absent. Keep this compatibility behavior scoped to the child process.
    os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    source = target.read_text(encoding="utf-8")
    assignment = "os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'"
    if assignment not in source:
        raise SystemExit("InstantSplat allocator assignment changed; review compatibility wrapper")
    source = source.replace(assignment, "os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)", 1)

    sys.argv = [str(target), *sys.argv[2:]]
    sys.path.insert(0, str(target.parent))
    os.chdir(target.parent)
    globals_dict = {"__name__": "__main__", "__file__": str(target)}
    exec(compile(source, str(target), "exec"), globals_dict)


if __name__ == "__main__":
    main()

