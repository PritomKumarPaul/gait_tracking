#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
GPU_COUNT=$(python - <<'PY'
import os
print(len([gpu for gpu in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if gpu.strip()]))
PY
)

cd "$REPO_ROOT"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate allinonegait 2>/dev/null || true
fi

TRAIN_LOG="$LOG_DIR/gaitgl_casiab_train.log"
TEST_LOG="$LOG_DIR/gaitgl_casiab_test.log"

python -m torch.distributed.launch \
  --master_port 29503 \
  --nproc_per_node="$GPU_COUNT" opengait/main.py \
  --cfgs ./configs/local/gaitgl_casiab_local.yaml \
  --phase train 2>&1 | tee "$TRAIN_LOG"

python -m torch.distributed.launch \
  --master_port 29503 \
  --nproc_per_node="$GPU_COUNT" opengait/main.py \
  --cfgs ./configs/local/gaitgl_casiab_local.yaml \
  --phase test \
  --iter 10000 2>&1 | tee "$TEST_LOG"
