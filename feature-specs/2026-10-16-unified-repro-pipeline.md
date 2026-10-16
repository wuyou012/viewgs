# Unified reproducible pipeline

## Goal

Use one environment and two stable entry points for the current two-view workflow: MASt3R geometry, ViewCrafter interpolation, and ViewGS reconstruction.

## Acceptance criteria

- One isolated Conda environment imports all three projects and their CUDA extensions.
- `run_pipeline.sh` selects the exact MASt3R image pair and executes all three stages.
- `run_reconstruction.sh` independently reproduces the active ViewGS training command.
- Paths, GPU, view count, diffusion steps, and training iterations are configurable.
- Existing dirty repositories are not modified or committed.
- A real server smoke test reaches every stage and produces expected artifacts.

## Architecture

`scene/images` -> InstantSplat/MASt3R -> `scene/sparse_2` -> selected original pair -> ViewCrafter -> `diffusion/<scene>` -> ViewGS -> `reconstruction/<scene>`.

