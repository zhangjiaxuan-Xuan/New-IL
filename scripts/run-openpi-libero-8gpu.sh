#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/L202500340/New-IL}"
if ! command -v conda >/dev/null 2>&1; then
  for _conda_bin in /root/miniconda3/bin "$HOME/miniconda3/bin" /opt/conda/bin; do
    if [[ -x "${_conda_bin}/conda" ]]; then
      export PATH="${_conda_bin}:${PATH}"
      break
    fi
  done
fi
TASK_SUITES="${TASK_SUITES:-libero_spatial,libero_object,libero_goal,libero_10}"
TASK_IDS="${TASK_IDS:-0-9}"
FULL_PER_TASK_TARGET="${FULL_PER_TASK_TARGET:-${PER_TASK_TARGET:-100}}"
TARGET_FRACTION="${TARGET_FRACTION:-${TARGET_PERCENT:-100}}"
OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-pi}"
NEWIL_CONDA_ENV="${NEWIL_CONDA_ENV:-newil}"

run_newil_python() {
  conda run --no-capture-output -n "$NEWIL_CONDA_ENV" python "$@"
}

_AUTO_CONFIG="$(run_newil_python - <<'PY'
from __future__ import annotations

import os
import re
import subprocess

def run_nvidia_smi(args: list[str]) -> str:
    try:
        return subprocess.check_output(["nvidia-smi", *args], text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

def parse_mem_mib_from_name(text: str) -> int | None:
    lower = text.lower()
    match = re.search(r"(\d+)\s*gb", lower)
    if match:
        return int(match.group(1)) * 1024
    return None

def detect() -> tuple[list[int], int]:
    query = run_nvidia_smi(["--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    gpus: list[int] = []
    mem_values: list[int] = []
    for line in query.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        gpus.append(int(parts[0]))
        if len(parts) >= 3 and parts[2].isdigit():
            mem_values.append(int(parts[2]))

    listing = run_nvidia_smi(["-L"])
    if not gpus:
        for match in re.finditer(r"^GPU\s+(\d+):", listing, re.MULTILINE):
            gpus.append(int(match.group(1)))

    mig_mems = [int(match.group(1)) * 1024 for match in re.finditer(r"MIG\s+\S+\.(\d+)gb", listing, re.IGNORECASE)]
    if mig_mems:
        mem_values = mig_mems
    elif not mem_values:
        parsed = parse_mem_mib_from_name(listing)
        if parsed:
            mem_values = [parsed]

    if not gpus:
        gpus = [0]
    mem_mib = min(mem_values) if mem_values else int(os.environ.get("GPU_MEMORY_TOTAL_MIB", "40960"))
    return sorted(set(gpus)), mem_mib

gpus, mem_mib = detect()
server_fraction = float(os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85"))
safe_fraction = float(os.environ.get("GPU_SAFE_MEM_FRACTION", "0.95"))
reserve_mib = int(os.environ.get("GPU_MEM_RESERVE_MIB", "2048"))
worker_mem_mib = int(os.environ.get("LIBERO_WORKER_MEM_MIB", "768"))
max_workers = int(os.environ.get("MAX_WORKERS_PER_GPU", "6"))
max_servers = int(os.environ.get("MAX_SERVERS_PER_GPU", "1"))

server_mem_mib = int(mem_mib * server_fraction)
server_budget_mib = max(0, int(mem_mib * safe_fraction) - reserve_mib)
safe_servers = max(1, min(max_servers, server_budget_mib // max(1, server_mem_mib)))
remaining_mib = max(0, server_budget_mib - safe_servers * server_mem_mib)
safe_workers = max(1, min(max_workers, remaining_mib // max(1, worker_mem_mib)))

print(f"AUTO_GPUS_CSV={','.join(str(gpu) for gpu in gpus)}")
print(f"AUTO_GPU_MEMORY_TOTAL_MIB={mem_mib}")
print(f"AUTO_OPENPI_SERVER_MEM_MIB={server_mem_mib}")
print(f"AUTO_GPU_SERVER_BUDGET_MIB={server_budget_mib}")
print(f"AUTO_GPU_REMAINING_FOR_WORKERS_MIB={remaining_mib}")
print(f"AUTO_SERVERS_PER_GPU={safe_servers}")
print(f"AUTO_WORKERS_PER_GPU={safe_workers}")
PY
)"
eval "$_AUTO_CONFIG"

GPUS_CSV="${GPUS_CSV:-${AUTO_GPUS_CSV}}"
GPU_MEMORY_TOTAL_MIB="${GPU_MEMORY_TOTAL_MIB:-${AUTO_GPU_MEMORY_TOTAL_MIB}}"
OPENPI_SERVER_MEM_MIB="${OPENPI_SERVER_MEM_MIB:-${AUTO_OPENPI_SERVER_MEM_MIB}}"
GPU_SERVER_BUDGET_MIB="${GPU_SERVER_BUDGET_MIB:-${AUTO_GPU_SERVER_BUDGET_MIB}}"
GPU_REMAINING_FOR_WORKERS_MIB="${GPU_REMAINING_FOR_WORKERS_MIB:-${AUTO_GPU_REMAINING_FOR_WORKERS_MIB}}"
SERVERS_PER_GPU="${SERVERS_PER_GPU:-${AUTO_SERVERS_PER_GPU}}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-${AUTO_WORKERS_PER_GPU}}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/openpi_libero_no90_${GPUS_CSV//,/_}gpu_${SERVERS_PER_GPU}s${WORKERS_PER_GPU}w}"

IFS=',' read -r -a _TASK_SUITE_ARRAY <<< "$TASK_SUITES"
if [[ "$TASK_IDS" == *","* ]]; then
  IFS=',' read -r -a _TASK_ID_ARRAY <<< "$TASK_IDS"
  _TASK_COUNT="${#_TASK_ID_ARRAY[@]}"
elif [[ "$TASK_IDS" == *"-"* ]]; then
  _TASK_ID_START="${TASK_IDS%%-*}"
  _TASK_ID_END="${TASK_IDS##*-}"
  _TASK_COUNT=$(( _TASK_ID_END - _TASK_ID_START + 1 ))
else
  _TASK_COUNT=1
fi
FULL_TOTAL_TARGET="${FULL_TOTAL_TARGET:-$(( ${#_TASK_SUITE_ARRAY[@]} * _TASK_COUNT * FULL_PER_TASK_TARGET ))}"

read -r PER_TASK_TARGET TOTAL_TARGET < <(
  run_newil_python - "$FULL_PER_TASK_TARGET" "$FULL_TOTAL_TARGET" "$TARGET_FRACTION" <<'PY'
import math
import sys

full_per_task = int(sys.argv[1])
full_total = int(sys.argv[2])
raw_fraction = sys.argv[3].strip()
if raw_fraction.endswith("%"):
    fraction = float(raw_fraction[:-1]) / 100.0
else:
    value = float(raw_fraction)
    fraction = value / 100.0 if value > 1.0 else value
fraction = max(0.0, min(1.0, fraction))
per_task = max(1, math.ceil(full_per_task * fraction))
total = max(1, math.ceil(full_total * fraction))
print(per_task, total)
PY
)

export PROJECT_ROOT
export TASK_SUITES
export TASK_IDS
export ATTEMPTS_PER_TASK="${ATTEMPTS_PER_TASK:-500}"
export PER_TASK_TARGET
export TOTAL_TARGET
export GPUS_CSV
export GPU_MEMORY_TOTAL_MIB
export OPENPI_SERVER_MEM_MIB
export GPU_SERVER_BUDGET_MIB
export GPU_REMAINING_FOR_WORKERS_MIB
export SERVERS_PER_GPU
export WORKERS_PER_GPU
export BASE_PORT="${BASE_PORT:-8000}"
export HOST="${HOST:-127.0.0.1}"
export MAX_STEPS="${MAX_STEPS:-auto}"
export OPENPI_CONDA_ENV
export NEWIL_CONDA_ENV
export RESET_QUEUE="${RESET_QUEUE:-0}"
export RESET_OUTPUT="${RESET_OUTPUT:-0}"
export REQUEUE_RUNNING="${REQUEUE_RUNNING:-1}"
export RUN_ROOT
export STOP_SERVERS_ON_EXIT="${STOP_SERVERS_ON_EXIT:-1}"
export BACKGROUND="${BACKGROUND:-0}"

mkdir -p "$RUN_ROOT"

cat > "${RUN_ROOT}/launch.env" <<EOF
PROJECT_ROOT=${PROJECT_ROOT}
TASK_SUITES=${TASK_SUITES}
TASK_IDS=${TASK_IDS}
ATTEMPTS_PER_TASK=${ATTEMPTS_PER_TASK}
FULL_PER_TASK_TARGET=${FULL_PER_TASK_TARGET}
FULL_TOTAL_TARGET=${FULL_TOTAL_TARGET}
TARGET_FRACTION=${TARGET_FRACTION}
PER_TASK_TARGET=${PER_TASK_TARGET}
TOTAL_TARGET=${TOTAL_TARGET}
GPUS_CSV=${GPUS_CSV}
GPU_MEMORY_TOTAL_MIB=${GPU_MEMORY_TOTAL_MIB}
OPENPI_SERVER_MEM_MIB=${OPENPI_SERVER_MEM_MIB}
GPU_SERVER_BUDGET_MIB=${GPU_SERVER_BUDGET_MIB}
GPU_REMAINING_FOR_WORKERS_MIB=${GPU_REMAINING_FOR_WORKERS_MIB}
SERVERS_PER_GPU=${SERVERS_PER_GPU}
WORKERS_PER_GPU=${WORKERS_PER_GPU}
BASE_PORT=${BASE_PORT}
HOST=${HOST}
MAX_STEPS=${MAX_STEPS}
OPENPI_CONDA_ENV=${OPENPI_CONDA_ENV}
NEWIL_CONDA_ENV=${NEWIL_CONDA_ENV}
RUN_ROOT=${RUN_ROOT}
RESET_QUEUE=${RESET_QUEUE}
RESET_OUTPUT=${RESET_OUTPUT}
REQUEUE_RUNNING=${REQUEUE_RUNNING}
STOP_SERVERS_ON_EXIT=${STOP_SERVERS_ON_EXIT}
BACKGROUND=${BACKGROUND}
EOF

if [[ "${FOREGROUND:-0}" == "1" || "$BACKGROUND" != "1" ]]; then
  exec "$PROJECT_ROOT/scripts/run-openpi-libero-collect-8gpu.sh" "$@"
fi

if command -v setsid >/dev/null 2>&1; then
  setsid "$PROJECT_ROOT/scripts/run-openpi-libero-collect-8gpu.sh" "$@" > "${RUN_ROOT}/nohup.log" 2>&1 &
else
  nohup "$PROJECT_ROOT/scripts/run-openpi-libero-collect-8gpu.sh" "$@" > "${RUN_ROOT}/nohup.log" 2>&1 &
fi
pid=$!
echo "$pid" > "${RUN_ROOT}/orchestrator.pid"

sleep "${BACKGROUND_STARTUP_GRACE:-3}"
if ! kill -0 "$pid" 2>/dev/null; then
  cat >&2 <<EOF
ERROR: OpenPI LIBERO background orchestrator exited during startup.
pid: ${pid}
run_root: ${RUN_ROOT}
log: ${RUN_ROOT}/nohup.log
EOF
  if [[ -s "${RUN_ROOT}/nohup.log" ]]; then
    cat >&2 <<EOF
--- ${RUN_ROOT}/nohup.log ---
EOF
    tail -n 80 "${RUN_ROOT}/nohup.log" >&2 || true
  fi
  exit 1
fi

cat <<EOF
OpenPI LIBERO collection started in background.
pid: ${pid}
run_root: ${RUN_ROOT}
log: ${RUN_ROOT}/nohup.log
orchestrator_log: ${RUN_ROOT}/orchestrator.log
task_suites: ${TASK_SUITES}
target: ${TOTAL_TARGET} successes (${PER_TASK_TARGET} per suite/task)
EOF
