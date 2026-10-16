# ViewGS reproducible two-view pipeline

This repository wraps the currently validated workflow without modifying the three research repositories:

1. InstantSplat/MASt3R builds the sparse point cloud and endpoint cameras.
2. ViewCrafter runs on the exact same two source images and produces 25 interpolation frames.
3. ViewGS trains from the MASt3R point cloud, two real images, and 23 interior diffusion frames.

The priority is behavior-preserving code organization and reproducibility, not reconstruction-quality improvement. See [`docs/CODEBASE_MAP.md`](docs/CODEBASE_MAP.md) for the active dependency closure and cleanup boundary.

Known technical discrepancies are recorded in [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). See [`docs/VALIDATION.md`](docs/VALIDATION.md) for the server smoke test and [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md) for weight hashes.

## Server setup

```bash
cp config.example.env .env
bash scripts/bootstrap_env.sh
```

The tested server environment is `viewgs-unified`: Python 3.9.16, PyTorch 1.13.1, CUDA 11.7 runtime, PyTorch3D 0.7.5. CUDA extensions are compiled with the server CUDA 11.8 toolkit.
The pipeline uses a scoped compatibility launcher because InstantSplat enables a PyTorch 2.x-only allocator option; the original source tree is not edited.
It also stages one disposable holdout copy because the current ViewCrafter loader reserves the first sorted image as an LLFF test frame.

## Full pipeline

```bash
bash scripts/run_pipeline.sh \
  --scene-root /mnt/ssd2/zongqi/ViewGS/data/Tanks_v2/Barn/24_views \
  --scene-name Barn \
  --output-root /mnt/ssd2/zongqi/ViewGS/repro_runs/Barn \
  --gpu 1 \
  --iterations 10000
```

The script reuses completed MASt3R and ViewCrafter artifacts by default. Pass `--force` to regenerate them.

## Reconstruction only

```bash
bash scripts/run_reconstruction.sh \
  --source-path /mnt/ssd2/zongqi/ViewGS/data/Tanks_v2/Barn/24_views/sparse_2 \
  --diff-path /mnt/ssd2/zongqi/ViewGS/data/output_TT_2view/Barn \
  --model-path /mnt/ssd2/zongqi/ViewGS/repro_runs/Barn/reconstruction/Barn \
  --gpu 1 \
  --iterations 10000
```

For a quick validation, use a new model path and `--iterations 5`. Full research results still require the original 10,000 iterations.

## External source versions tested

- InstantSplat: `b951567`
- ViewCrafter: `c2451dc` plus local experimental changes
- ViewGS/FSGS: `a536a64` plus local experimental changes

The local experimental source trees are intentionally not committed by this orchestration repository. They must be preserved separately until their active modifications are reduced into reviewable patches.
