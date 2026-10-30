# Server validation

Validated initially on 2026-10-16 and repeated from pinned sources on 2026-10-30 on `user-M12SWA-TF` with an NVIDIA RTX 4090.

## Unified environment

- Clean-test environment: `/mnt/ssd2/zongqi/miniconda3/envs/viewgs-repro-clean`
- Python 3.9.16
- PyTorch 1.13.1 + CUDA 11.7 runtime
- PyTorch3D 0.7.5
- ViewGS `diff_gaussian_rasterization` and `simple_knn` compiled successfully with CUDA 11.8
- All three project entry points passed import and `--help` checks

`scripts/prepare_sources.sh` cloned the three upstream repositories at the exact commits in `sources.lock.env`, initialized the required submodules, applied the tracked overlays, and passed a second idempotent run. The test did not import code from the original dirty source trees.

## End-to-end smoke test

Command used the Barn scene, GPU 1, two DDIM steps, and five ViewGS training iterations. The reduced counts validate wiring only, not final quality.

Expected artifacts were produced:

- MASt3R `points3D.ply`: present, 293,901 downsampled points
- ViewCrafter diffusion PNGs: 25
- ViewGS supervision views after endpoint removal: 23
- ViewGS `point_cloud/iteration_5/point_cloud.ply`: present
- Training completed with test PSNR 13.05519 at iteration 5

The clean test used GPU 1 and wrote only to `/mnt/ssd2/zongqi/ViewGS/repro_smoke_clean_20261030_2`. It confirms installation and data flow; five iterations and two DDIM steps are not intended as a quality benchmark.
