#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TRAINER=${TRAINER:-${SCRIPT_DIR}/train_xyzflow_teacher_student_teacherforcing_novelocity_patchschedule.py}
NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

TRAJECTORY_DIR=${TRAJECTORY_DIR:-}
OUTPUT_DIR=${OUTPUT_DIR:-output_teacher_student_patchschedule_$(date +"%Y%m%d_%H%M%S")}
CHECKPOINT=${CHECKPOINT:-}
MODEL_SIZE=${MODEL_SIZE:-base}
BATCH_SIZE=${BATCH_SIZE:-16}
EPOCHS=${EPOCHS:-1000000}
LR=${LR:-0.0001}
CFG=${CFG:-1.0}
IMAGE_CFG=${IMAGE_CFG:-2.3}
RETAIN_ENCODER=${RETAIN_ENCODER:-6}
RETAIN_DECODER=${RETAIN_DECODER:-0}
WORKERS=${WORKERS:-4}
MAX_SAMPLES=${MAX_SAMPLES:-}
LOG_DIR=${LOG_DIR:-}
IMAGE_LOG_INTERVAL=${IMAGE_LOG_INTERVAL:-100}
IMAGE_LOG_COUNT=${IMAGE_LOG_COUNT:-4}
IMAGE_LOG_LABELS=${IMAGE_LOG_LABELS:-0}
IMAGE_LOG_SEED=${IMAGE_LOG_SEED:-"42,43,44,45"}
VAE_PATH=${VAE_PATH:-}
GRAD_ACCUMULATION_STEPS=${GRAD_ACCUMULATION_STEPS:-1}
CLIP_GRAD_NORM=${CLIP_GRAD_NORM:-1.0}
LOG_INTERVAL=${LOG_INTERVAL:-100}
EMA_DECAY=${EMA_DECAY:-0.9999}
MIXED_PRECISION=${MIXED_PRECISION:-bf16}
SAVE_EVERY_STEPS=${SAVE_EVERY_STEPS:-2000}
LABEL_FILTER=${LABEL_FILTER:-}

if [[ -z "${TRAJECTORY_DIR}" || ! -d "${TRAJECTORY_DIR}" ]]; then
  echo "ERROR: Set TRAJECTORY_DIR to a generated trajectory directory." >&2
  exit 1
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
  echo "ERROR: Set CHECKPOINT to the pretrained XYZFlow teacher weights." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

if [[ -n "${LABEL_FILTER}" ]]; then
  FILTER_ROOT="${OUTPUT_DIR}/filtered_trajectories"
  rm -rf "${FILTER_ROOT}"
  mkdir -p "${FILTER_ROOT}"
  IFS=',' read -ra LABEL_IDS <<< "${LABEL_FILTER}"
  for patch_dir in "${TRAJECTORY_DIR}"/patch_*; do
    [[ -d "${patch_dir}" ]] || continue
    patch_name=$(basename "${patch_dir}")
    for step_dir in "${patch_dir}"/step_*; do
      [[ -d "${step_dir}" ]] || continue
      step_name=$(basename "${step_dir}")
      mkdir -p "${FILTER_ROOT}/${patch_name}/${step_name}"
      for label_id in "${LABEL_IDS[@]}"; do
        printf -v label_dir "label%03d" "${label_id}"
        src="${step_dir}/${label_dir}"
        dst="${FILTER_ROOT}/${patch_name}/${step_name}/${label_dir}"
        if [[ -d "${src}" ]] ; then
          ln -s "${src}" "${dst}"
        fi
      done
    done
  done
  TRAJECTORY_DIR="${FILTER_ROOT}"
fi

EXTRA_ARGS=()
[[ -n "${MAX_SAMPLES}" ]] && EXTRA_ARGS+=(--max-samples "${MAX_SAMPLES}")
[[ -n "${LOG_DIR}" ]] && EXTRA_ARGS+=(--log-dir "${LOG_DIR}")
[[ -n "${VAE_PATH}" ]] && EXTRA_ARGS+=(--vae-path "${VAE_PATH}")
[[ -n "${MIXED_PRECISION}" ]] && EXTRA_ARGS+=(--mixed-precision "${MIXED_PRECISION}")
EXTRA_ARGS+=(--image-log-interval "${IMAGE_LOG_INTERVAL}")
EXTRA_ARGS+=(--image-log-count "${IMAGE_LOG_COUNT}")
EXTRA_ARGS+=(--image-log-labels "${IMAGE_LOG_LABELS}")
EXTRA_ARGS+=(--image-log-seed "${IMAGE_LOG_SEED}")
EXTRA_ARGS+=(--grad-accumulation-steps "${GRAD_ACCUMULATION_STEPS}")
EXTRA_ARGS+=(--clip-grad-norm "${CLIP_GRAD_NORM}")
EXTRA_ARGS+=(--log-interval "${LOG_INTERVAL}")
EXTRA_ARGS+=(--ema-decay "${EMA_DECAY}")
EXTRA_ARGS+=(--image-cfg "${IMAGE_CFG}")
EXTRA_ARGS+=(--save-every-steps "${SAVE_EVERY_STEPS}")

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
port = s.getsockname()[1]
s.close()
print(port)
PY
)}

echo "Trainer                  : ${TRAINER}"
echo "Output dir              : ${OUTPUT_DIR}"
echo "Trajectory dir          : ${TRAJECTORY_DIR}"
echo "Checkpoint              : ${CHECKPOINT}"

torchrun \
  --nproc_per_node="${NUM_GPUS}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  "${TRAINER}" \
  --trajectory-dir "${TRAJECTORY_DIR}" \
  --checkpoint "${CHECKPOINT}" \
  --model-size "${MODEL_SIZE}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size "${BATCH_SIZE}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --cfg "${CFG}" \
  --retain-encoder-layers "${RETAIN_ENCODER}" \
  --retain-decoder-layers "${RETAIN_DECODER}" \
  --num-workers "${WORKERS}" \
  "${EXTRA_ARGS[@]}"
