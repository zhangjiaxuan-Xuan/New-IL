# OpenVLA LIBERO RLDS And AE Training Plan

## Data

Use the public OpenVLA modified LIBERO RLDS dataset:

- `libero_spatial_no_noops`
- `libero_object_no_noops`
- `libero_goal_no_noops`
- `libero_10_no_noops`, also called LIBERO-Long

Pinned repository revision:

```text
openvla/modified_libero_rlds@6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551
```

Download:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --extra data new-il-download-openvla-rlds \
  --output data/openvla_modified_libero_rlds \
  --suites all
```

The downloader writes:

```text
data/openvla_modified_libero_rlds/new_il_openvla_rlds_manifest.json
```

Current local download status:

- `libero_spatial_no_noops`: 1.8G
- `libero_object_no_noops`: 2.7G
- `libero_goal_no_noops`: 1.8G
- `libero_10_no_noops`: 3.5G

## Training Reference

OpenVLA's official LIBERO recipe fine-tunes four LIBERO suites independently.
Their public LoRA setup uses rank 32, learning rate `5e-4`, batch size 128 on
8 A100 80GB GPUs, 80K gradient steps, no quantization, no gradient accumulation,
and a shuffle buffer of 100K. That is a strong reference point, but too large for
the first New-IL action-expert ablation.

For New-IL, follow the safer action-expert pattern:

- Freeze the VLM language backbone and vision encoder by default.
- Train the action expert and action projection/progress heads.
- Add PA-TCS only after fixed-time BC reproduces sane loss curves.
- Keep all run metadata under `runs/new_il/`.

## Run Guard

Always launch long training through `new-il-run` so terminal output and parameters
survive interruption:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run new-il-run \
  --name ae_full_libero_spatial \
  --config configs/openvla_libero_ae_full.yaml \
  --min-free-gb 20 \
  --memory-fraction 0.9 \
  -- \
  python -m new_il.training.train_ae
```

The guard writes:

- `run_config.json`
- `logs/terminal.log`
- `status.json`

It also sets conservative runtime environment defaults:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- `XLA_PYTHON_CLIENT_PREALLOCATE=false`
- `XLA_PYTHON_CLIENT_MEM_FRACTION=<memory_fraction>`
- `TOKENIZERS_PARALLELISM=false`
- `NEW_IL_PER_DEVICE_BATCH_SIZE=<planned_batch>`
- `NEW_IL_GRAD_ACCUMULATION_STEPS=<planned_accumulation>`
- `NEW_IL_EFFECTIVE_BATCH_SIZE=<planned_effective_batch>`

You can inspect the planner directly:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run new-il-plan-memory \
  --min-free-gb 20 \
  --max-gpus 2 \
  --batch-multiple 4 \
  --target-global-batch 128
```

Manual per-device batches are checked before training:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run new-il-run \
  --name ae_full_libero_spatial \
  --config configs/openvla_libero_ae_full.yaml \
  --max-gpus 2 \
  --per-device-batch-size 16 \
  -- \
  python -m new_il.training.train_ae
```

If the manual batch exceeds the safe estimate, the launcher rejects the run.
Use `--allow-oom-risk` only when you intentionally want to override the guard.
