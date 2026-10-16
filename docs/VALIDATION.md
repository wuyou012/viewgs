# Server validation

Validated on 2026-10-16 on `user-M12SWA-TF` with an NVIDIA RTX 4090.

## Unified environment

- Environment: `/mnt/ssd2/zongqi/miniconda3/envs/viewgs-unified`
- Python 3.9.16
- PyTorch 1.13.1 + CUDA 11.7 runtime
- PyTorch3D 0.7.5
- ViewGS `diff_gaussian_rasterization` and `simple_knn` compiled successfully with CUDA 11.8
- All three project entry points passed import and `--help` checks

## End-to-end smoke test

Command used the Barn scene, GPU 1, two DDIM steps, and five ViewGS training iterations. The reduced counts validate wiring only, not final quality.

Expected artifacts were produced:

- MASt3R `points3D.ply`: present, 293,895 downsampled points
- ViewCrafter diffusion PNGs: 25
- ViewGS supervision views after endpoint removal: 23
- ViewGS `point_cloud/iteration_5/point_cloud.ply`: present
- Training completed with test PSNR 13.03205 at iteration 5

