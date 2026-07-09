#!/usr/bin/env bash
set -euo pipefail

QUEUE_DIR="${QUEUE_DIR:-runs/openpi_libero_queue}"
NEWIL_CONDA_ENV="${NEWIL_CONDA_ENV:-newil}"
if ! command -v conda >/dev/null 2>&1; then
  for _conda_bin in /root/miniconda3/bin "$HOME/miniconda3/bin" /opt/conda/bin; do
    if [[ -x "${_conda_bin}/conda" ]]; then
      export PATH="${_conda_bin}:${PATH}"
      break
    fi
  done
fi

exec conda run --no-capture-output -n "$NEWIL_CONDA_ENV" \
  python -c "from new_il.libero.openpi_queue import make_queue_main; make_queue_main()" \
  --queue-dir "$QUEUE_DIR" \
  "$@"
