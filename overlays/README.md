# Runtime overlays

These files are the minimal behavior-preserving delta required by the active two-view pipeline.

- `viewcrafter/` is copied over ViewCrafter commit `c2451dc3e33afaa7382b1c4f1cdb0ea2474d6c52`.
- `viewgs/` is copied over FSGS commit `a536a64c5b366b1088be64eeadf9e791ca26897c`.
- InstantSplat needs no source overlay; its PyTorch 1.13 allocator compatibility is handled by `scripts/run_init_geo_compat.py`.

Backups, experiment launchers, output directories, checkpoints, datasets, bytecode, and logs are intentionally excluded. The upstream repositories retain their original licenses and attribution.

`runtime/` is generated and must not be edited. Rerun `prepare_sources.sh` to reapply updated overlays; remove `runtime/` only when a completely fresh source assembly is desired.
