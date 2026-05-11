# LIBERO Smoke Experiment

This is the first reproducible slice for New-IL. The goal is not to claim SOTA;
it is to test whether PA-TCS improves the failure modes that fixed-time
supervised fine-tuning tends to hide behind one success-rate number.

## Scope

- Benchmark: `libero_spatial` first, then `libero_object` if the pipeline is stable.
- Data: official LIBERO demonstrations, capped by `configs/libero_smoke.yaml`.
- Models: `lerobot/smolvla_base` is the preferred public VLA candidate because
  its main checkpoint is about 907 MB. If dependency or data-format friction is
  too high for the first pass, use a small ACT-style action chunk policy under
  1 GB as the controlled baseline.
- Baselines: fixed-time BC versus PA-TCS on the same architecture and data split.
- Metrics: success rate, tube violation rate, event timing error, event pose error,
  wrong crossing rate, and progress backward rate.

## Reproducible Commands

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --extra dev
UV_CACHE_DIR=/tmp/uv-cache uv run pytest
```

Download data through an existing LIBERO checkout:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run new-il-download-libero \
  --libero-root /path/to/LIBERO \
  --datasets libero_spatial \
  --use-huggingface
```

Or download the first single-task HDF5 directly from the official Hugging Face
dataset mirror:

```bash
mkdir -p data/libero_small_hdf5/libero_object
curl -L --fail --continue-at - \
  --output data/libero_small_hdf5/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5 \
  https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5
```

Create a tiny manifest:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run new-il-prepare-libero-subset \
  --dataset-root /path/to/LIBERO/datasets \
  --suite libero_spatial \
  --max-files 2 \
  --max-demos-per-file 10 \
  --output data/libero_smoke_manifest.json
```

## Data And Model Candidates

- First official HDF5 target: one `libero_object` task such as
  `pick_up_the_orange_juice_and_place_it_in_the_basket_demo.hdf5`, then expand
  to 1-2 `libero_spatial` tasks.
- First public model target: `lerobot/smolvla_base` under the 1 GB constraint.
- Reference-only larger models: OpenVLA LIBERO checkpoints and OpenPI/pi0.5 are
  useful for understanding strong LIBERO evaluation paths, but they violate the
  requested small-model constraint.

## Decision Log

- We keep LIBERO, model checkpoints, and generated manifests out of git.
- We start with official HDF5 demonstrations because event extraction from gripper
  transitions is enough for the first PA-TCS smoke test.
- SmolVLA base is the most suitable first public VLA candidate found so far;
  LIBERO-finetuned SmolVLA variants may be more directly useful but appear to sit
  slightly above 1 GB.
