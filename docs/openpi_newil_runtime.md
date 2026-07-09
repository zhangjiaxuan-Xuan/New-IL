# OpenPI And New-IL Runtime Split

New-IL is the main environment. It owns data conversion, SmolVLA interfaces,
LIBERO simulation, video writing, action health checks, and parallel rollout
scheduling.

OpenPI is a separate stable Python 3.11 environment. It owns pi0/pi0.5 model
loading and inference only. Do not run OpenPI's `examples/libero/main.py` for
New-IL experiments; that would bypass New-IL's LIBERO links, queue ledger, video
checks, and collection format.

## Process Layout

For a full unattended 8-GPU collection job, use the one-line preset:

```bash
cd /data/L202500340/New-IL
mkdir -p runs/openpi_libero_spatial_8gpu_6s6w && nohup bash scripts/run-openpi-libero-8gpu.sh > runs/openpi_libero_spatial_8gpu_6s6w/nohup.log 2>&1 &
```

By default this starts 8 GPUs x 6 OpenPI servers x 6 New-IL workers. Tasks are
claimed dynamically from the shared deficit queue; each worker is bound to one
server port on its GPU. The lower-level orchestrator is
`scripts/run-openpi-libero-collect-8gpu.sh`.

1. Start the OpenPI pi0.5 policy server in the OpenPI environment:

```bash
OPENPI_CONDA_ENV=openpi PORT=8000 scripts/start-openpi-pi05-libero-server.sh
```

By default this caches OpenPI checkpoints under
`/data/L202500340/data/openpi`. Override with `OPENPI_DATA_HOME=...` if needed.
The first run downloads `gs://openpi-assets/checkpoints/pi05_libero`, about
11.6 GB, before the websocket server starts.

2. Run one New-IL controlled LIBERO smoke episode:

```bash
HOST=127.0.0.1 PORT=8000 OUTPUT_DIR=runs/openpi_libero_smoke scripts/eval-openpi-libero-smoke.sh
```

This path uses `new_il.libero.rollout.run_one_episode`, the same canonical
Mem-style rollout core used by queue workers. The produced NPZ files use the
Mem-compatible keys `observation.images.image`, `observation.images.image2`,
`observation.state`, `actions`, `language`, and `success`.

3. For parallel collection, create a queue in the New-IL environment:

```bash
QUEUE_DIR=runs/openpi_libero_queue scripts/make-openpi-libero-queue.sh \
  --task-suite-name libero_spatial \
  --task-ids 0-9 \
  --attempts-per-task 50
```

4. Start one or more New-IL workers. Each worker owns LIBERO rollout execution
   and calls the OpenPI server for action chunks:

```bash
WORKER_ID=0 QUEUE_DIR=runs/openpi_libero_queue OUTPUT_DIR=runs/openpi_libero_collect \
  scripts/start-openpi-libero-worker.sh --per-task-target 10 --total-target 40
```

5. Rebuild PATCS artifacts directly from collected rollout NPZ files:

```bash
new-il-build-patcs-artifacts \
  --source-format rollout_npz \
  --input runs/openpi_libero_collect \
  --output data/patcs_artifacts/openpi_libero_spatial_task0_patcs.npz \
  --num-demos 16 \
  --num-phase 64 \
  --obs-key ee_pos
```

## Ownership Boundary

- OpenPI server:
  - loads `pi05_libero`,
  - serves websocket inference,
  - returns action chunks.
- New-IL worker:
  - claims jobs from `pending/`,
  - moves jobs to `running/`,
  - records `ledger/done` and `ledger/success`,
  - steps LIBERO,
  - converts observations to OpenPI payloads,
  - saves Mem-compatible rollout NPZ files, videos, and worker summaries.

The OpenPI payload adapter is `new_il.integrations.openpi`. It matches the
upstream LIBERO convention: render at 256, rotate agent/wrist images 180
degrees, resize/pad to 224, send 8D state plus prompt, and execute 5 actions
from each 10-step OpenPI chunk.
