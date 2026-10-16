# Project Instructions

## Purpose

This repository makes the existing MASt3R -> ViewCrafter -> ViewGS workflow reproducible. The primary goal is code organization and simplification, not model or metric improvement.

## Runtime

- Run and test only on SSH host `WeLight_4090_dentist`.
- Use `/mnt/ssd2/zongqi/miniconda3/envs/viewgs-unified/bin/python`.
- Never commit datasets, checkpoints, generated frames, logs, or reconstruction outputs.

## Behavior Preservation

- Keep DUSt3R as the ViewCrafter backend unless a separate experiment explicitly changes it.
- Do not change trajectory interpolation, losses, frame selection, or training defaults during cleanup.
- Treat the Barn smoke-test artifacts in `docs/VALIDATION.md` as the current behavioral baseline.

## Structure

- `scripts/`: stable user-facing entry points.
- `environment/`: unified environment specification.
- `docs/`: architecture, known issues, validation, and checkpoint hashes.
- `feature-specs/`: implementation decisions and acceptance criteria.

## Verification

- Run `bash -n scripts/*.sh` after shell changes.
- Run Python compilation/import checks after Python changes.
- Run the reduced Barn end-to-end smoke test after pipeline changes.
- Use a new output directory for tests; do not overwrite research results.

## Git

- Keep changes small and reviewable.
- Separate behavior-preserving cleanup from experimental algorithm changes.
- Preserve upstream attribution and licenses when source files are extracted.

