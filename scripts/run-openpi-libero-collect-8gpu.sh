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
NEW_IL_DATA_ROOT="${NEW_IL_DATA_ROOT:-/data/L202500340/data}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${NEW_IL_DATA_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME}/lerobot}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-${NEW_IL_DATA_ROOT}/openpi}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${NEW_IL_DATA_ROOT}/cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${NEW_IL_DATA_ROOT}/uv-cache}"

cd "$PROJECT_ROOT"

GPUS_CSV="${GPUS_CSV:-0}"
SERVERS_PER_GPU="${SERVERS_PER_GPU:-1}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-6}"
BASE_PORT="${BASE_PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

TASK_SUITES="${TASK_SUITES:-${TASK_SUITE:-libero_spatial}}"
TASK_IDS="${TASK_IDS:-0-9}"
ATTEMPTS_PER_TASK="${ATTEMPTS_PER_TASK:-500}"
PER_TASK_TARGET="${PER_TASK_TARGET:-100}"
TOTAL_TARGET="${TOTAL_TARGET:-1000}"
MAX_STEPS="${MAX_STEPS:-auto}"
SETTLE_STEPS="${SETTLE_STEPS:-10}"
REPLAN_STEPS="${REPLAN_STEPS:-5}"
CAMERA_SIZE="${CAMERA_SIZE:-256}"
RESIZE_SIZE="${RESIZE_SIZE:-224}"
FPS="${FPS:-10}"
SEED="${SEED:-7}"

OPENPI_CONDA_ENV="${OPENPI_CONDA_ENV:-pi}"
NEWIL_CONDA_ENV="${NEWIL_CONDA_ENV:-newil}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/data/L202500340/data/openpi}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
LIBERO_ROOT="${LIBERO_ROOT:-${PROJECT_ROOT}/third_party/LIBERO}"
GPU_MEMORY_TOTAL_MIB="${GPU_MEMORY_TOTAL_MIB:-unknown}"
OPENPI_SERVER_MEM_MIB="${OPENPI_SERVER_MEM_MIB:-unknown}"
GPU_SERVER_BUDGET_MIB="${GPU_SERVER_BUDGET_MIB:-unknown}"
GPU_REMAINING_FOR_WORKERS_MIB="${GPU_REMAINING_FOR_WORKERS_MIB:-unknown}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/runs/openpi_libero_8gpu}"
QUEUE_DIR="${QUEUE_DIR:-${RUN_ROOT}/queue}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_ROOT}/collect}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-${RUN_ROOT}/logs/servers}"
WORKER_LOG_DIR="${WORKER_LOG_DIR:-${RUN_ROOT}/logs/workers}"
PID_DIR="${PID_DIR:-${RUN_ROOT}/pids}"
MONITOR_INTERVAL="${MONITOR_INTERVAL:-30}"
SERVER_WAIT_TIMEOUT="${SERVER_WAIT_TIMEOUT:-900}"
RESET_QUEUE="${RESET_QUEUE:-0}"
RESET_OUTPUT="${RESET_OUTPUT:-0}"
REQUEUE_RUNNING="${REQUEUE_RUNNING:-1}"
START_SERVERS="${START_SERVERS:-1}"
START_WORKERS="${START_WORKERS:-1}"
STOP_SERVERS_ON_EXIT="${STOP_SERVERS_ON_EXIT:-1}"

export OPENPI_DATA_HOME
export XLA_PYTHON_CLIENT_MEM_FRACTION
export NEWIL_CONDA_ENV
export GPU_MEMORY_TOTAL_MIB
export OPENPI_SERVER_MEM_MIB
export GPU_SERVER_BUDGET_MIB
export GPU_REMAINING_FOR_WORKERS_MIB
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
TOTAL_WORKERS=$(( ${#GPUS[@]} * WORKERS_PER_GPU ))
TOTAL_SERVERS=$(( ${#GPUS[@]} * SERVERS_PER_GPU ))

WORKER_MAX_STEPS_ARGS=()
if [[ "$MAX_STEPS" != "auto" ]]; then
  WORKER_MAX_STEPS_ARGS=(--max-steps "$MAX_STEPS")
fi

mkdir -p "$RUN_ROOT" "$SERVER_LOG_DIR" "$WORKER_LOG_DIR" "$PID_DIR"

if [[ "$RESET_QUEUE" == "1" ]]; then
  rm -rf "$QUEUE_DIR"
fi
if [[ "$RESET_OUTPUT" == "1" ]]; then
  rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$QUEUE_DIR" "$OUTPUT_DIR"

SERVER_PIDS=()
WORKER_PIDS=()

log() {
  printf '[%(%F %T)T] %s\n' -1 "$*" | tee -a "$RUN_ROOT/orchestrator.log"
}

cleanup() {
  local code=$?
  log "cleanup start exit_code=${code}"
  if [[ "$START_WORKERS" == "1" ]]; then
    for pid in "${WORKER_PIDS[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
  fi
  if [[ "$STOP_SERVERS_ON_EXIT" == "1" && "$START_SERVERS" == "1" ]]; then
    for pid in "${SERVER_PIDS[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
      fi
    done
  fi
  log "cleanup done"
  exit "$code"
}
trap cleanup INT TERM EXIT

wait_for_port() {
  local host="$1"
  local port="$2"
  local timeout="$3"
  local started
  started="$(date +%s)"
  while true; do
    if conda run --no-capture-output -n "$NEWIL_CONDA_ENV" python - "$host" "$port" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(1.0)
    sock.connect((host, port))
PY
    then
      return 0
    fi
    if (( $(date +%s) - started > timeout )); then
      return 1
    fi
    sleep 5
  done
}

count_files() {
  local path="$1"
  if [[ -d "$path" ]]; then
    find "$path" -type f 2>/dev/null | wc -l
  else
    echo 0
  fi
}

start_servers() {
  log "starting ${TOTAL_SERVERS} OpenPI servers (${SERVERS_PER_GPU}/gpu) on ports ${BASE_PORT}..$((BASE_PORT + TOTAL_SERVERS - 1))"
  local server_index=0
  for gpu in "${GPUS[@]}"; do
    for local_server in $(seq 0 $((SERVERS_PER_GPU - 1))); do
      local port=$((BASE_PORT + server_index))
      local log_path="${SERVER_LOG_DIR}/server_${server_index}_gpu${gpu}_local${local_server}_port${port}.log"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export OPENPI_CONDA_ENV
        export PORT="$port"
        export OPENPI_DATA_HOME
        export XLA_PYTHON_CLIENT_MEM_FRACTION
        exec scripts/start-openpi-pi05-libero-server.sh
      ) >"$log_path" 2>&1 &
      local pid=$!
      SERVER_PIDS+=("$pid")
      echo "$pid" > "${PID_DIR}/server_${server_index}_gpu${gpu}_local${local_server}_port${port}.pid"
      log "server index=${server_index} gpu=${gpu} local=${local_server} port=${port} pid=${pid} log=${log_path}"
      server_index=$((server_index + 1))
    done
  done

  server_index=0
  for gpu in "${GPUS[@]}"; do
    for local_server in $(seq 0 $((SERVERS_PER_GPU - 1))); do
      local port=$((BASE_PORT + server_index))
      log "waiting for OpenPI server index=${server_index} gpu=${gpu} local=${local_server} port=${port}"
      if ! wait_for_port "$HOST" "$port" "$SERVER_WAIT_TIMEOUT"; then
        log "ERROR server port ${port} did not become ready within ${SERVER_WAIT_TIMEOUT}s"
        exit 1
      fi
      log "server ready index=${server_index} gpu=${gpu} local=${local_server} port=${port}"
      server_index=$((server_index + 1))
    done
  done
}

make_queue() {
  if [[ -f "${QUEUE_DIR}/queue_meta.json" ]]; then
    log "queue exists: ${QUEUE_DIR}/queue_meta.json"
    return
  fi
  log "creating queue dir=${QUEUE_DIR} task_suites=${TASK_SUITES} task_ids=${TASK_IDS} attempts_per_task=${ATTEMPTS_PER_TASK}"
  QUEUE_DIR="$QUEUE_DIR" scripts/make-openpi-libero-queue.sh \
    --task-suite-names "$TASK_SUITES" \
    --task-ids "$TASK_IDS" \
    --attempts-per-task "$ATTEMPTS_PER_TASK" \
    --seed "$SEED" | tee -a "$RUN_ROOT/orchestrator.log"
}

requeue_running() {
  if [[ "$REQUEUE_RUNNING" != "1" ]]; then
    log "REQUEUE_RUNNING=0, leaving existing running claims unchanged"
    return
  fi
  log "requeueing stale running jobs from ${QUEUE_DIR}/running"
  conda run --no-capture-output -n "$NEWIL_CONDA_ENV" \
    python -c "from new_il.libero.openpi_queue import requeue_running_main; requeue_running_main()" \
    --queue-dir "$QUEUE_DIR" | tee -a "$RUN_ROOT/orchestrator.log"
}

start_workers() {
  log "starting ${TOTAL_WORKERS} New-IL workers (${WORKERS_PER_GPU}/gpu), binding workers round-robin to ${SERVERS_PER_GPU}/gpu servers"
  local gpu_index=0
  for gpu in "${GPUS[@]}"; do
    for local_worker in $(seq 0 $((WORKERS_PER_GPU - 1))); do
      local worker_id=$((gpu_index * WORKERS_PER_GPU + local_worker))
      local local_server=$((local_worker % SERVERS_PER_GPU))
      local server_index=$((gpu_index * SERVERS_PER_GPU + local_server))
      local port=$((BASE_PORT + server_index))
      local log_path="${WORKER_LOG_DIR}/worker_${worker_id}_gpu${gpu}_server${server_index}_port${port}.log"
      (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export MUJOCO_GL
        export MUJOCO_EGL_DEVICE_ID="0"
        export PYOPENGL_PLATFORM
        export NEWIL_CONDA_ENV
        export WORKER_ID="$worker_id"
        export QUEUE_DIR
        export OUTPUT_DIR
        export HOST
        export PORT="$port"
        export LIBERO_ROOT
          exec scripts/start-openpi-libero-worker.sh \
          --per-task-target "$PER_TASK_TARGET" \
          --total-target "$TOTAL_TARGET" \
          "${WORKER_MAX_STEPS_ARGS[@]}" \
          --settle-steps "$SETTLE_STEPS" \
          --replan-steps "$REPLAN_STEPS" \
          --camera-size "$CAMERA_SIZE" \
          --resize-size "$RESIZE_SIZE" \
          --fps "$FPS"
      ) >"$log_path" 2>&1 &
      local pid=$!
      WORKER_PIDS+=("$pid")
      echo "$pid" > "${PID_DIR}/worker_${worker_id}.pid"
    done
    log "started workers gpu=${gpu} ids=$((gpu_index * WORKERS_PER_GPU))-$((gpu_index * WORKERS_PER_GPU + WORKERS_PER_GPU - 1))"
    gpu_index=$((gpu_index + 1))
  done
}

monitor() {
  log "monitoring run_root=${RUN_ROOT}"
  while true; do
    local done_count
    local success_count
    local success_npz
    local failed_npz
    done_count="$(count_files "${QUEUE_DIR}/ledger/done")"
    success_count="$(count_files "${QUEUE_DIR}/ledger/success")"
    success_npz="$(find "$OUTPUT_DIR" -path '*/success_rollouts/*.npz' -type f 2>/dev/null | wc -l)"
    failed_npz="$(find "$OUTPUT_DIR" -path '*/failed_rollouts/*.npz' -type f 2>/dev/null | wc -l)"
    log "progress done=${done_count} success_ledger=${success_count}/${TOTAL_TARGET} success_npz=${success_npz} failed_npz=${failed_npz}"

    if (( success_count >= TOTAL_TARGET )); then
      log "target reached success_ledger=${success_count}/${TOTAL_TARGET}"
      return 0
    fi

    local alive=0
    for pid in "${WORKER_PIDS[@]:-}"; do
      if kill -0 "$pid" 2>/dev/null; then
        alive=$((alive + 1))
      fi
    done
    if [[ "$START_WORKERS" == "1" && "$alive" == "0" ]]; then
      log "all workers exited before reaching target"
      return 0
    fi
    sleep "$MONITOR_INTERVAL"
  done
}

cat > "${RUN_ROOT}/run_config.json" <<EOF
{
  "gpus": "${GPUS_CSV}",
  "servers_per_gpu": ${SERVERS_PER_GPU},
  "workers_per_gpu": ${WORKERS_PER_GPU},
  "total_servers": ${TOTAL_SERVERS},
  "total_workers": ${TOTAL_WORKERS},
  "base_port": ${BASE_PORT},
  "gpu_memory_total_mib": "${GPU_MEMORY_TOTAL_MIB}",
  "openpi_server_mem_mib": "${OPENPI_SERVER_MEM_MIB}",
  "gpu_server_budget_mib": "${GPU_SERVER_BUDGET_MIB}",
  "gpu_remaining_for_workers_mib": "${GPU_REMAINING_FOR_WORKERS_MIB}",
  "xla_python_client_mem_fraction": "${XLA_PYTHON_CLIENT_MEM_FRACTION}",
  "task_suites": "${TASK_SUITES}",
  "task_ids": "${TASK_IDS}",
  "attempts_per_task": ${ATTEMPTS_PER_TASK},
  "per_task_target": ${PER_TASK_TARGET},
  "total_target": ${TOTAL_TARGET},
  "requeue_running": "${REQUEUE_RUNNING}",
  "max_steps": "${MAX_STEPS}",
  "queue_dir": "${QUEUE_DIR}",
  "output_dir": "${OUTPUT_DIR}",
  "openpi_data_home": "${OPENPI_DATA_HOME}"
}
EOF

log "run config written to ${RUN_ROOT}/run_config.json"

if [[ "$START_SERVERS" == "1" ]]; then
  start_servers
else
  log "START_SERVERS=0, assuming OpenPI servers are already running"
fi

make_queue
requeue_running

if [[ "$START_WORKERS" == "1" ]]; then
  start_workers
else
  log "START_WORKERS=0, queue prepared only"
  log "orchestration finished"
  trap - INT TERM EXIT
  exit 0
fi

monitor
log "orchestration finished"
