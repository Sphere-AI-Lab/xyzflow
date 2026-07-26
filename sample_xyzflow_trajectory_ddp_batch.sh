#!/usr/bin/env bash
#
# Multi-GPU trajectory sampling with BATCH support for XYZFlow models (Standard Teacher).
#
# This script uses the standard XYZFlow inference pipeline without KV cache modifications.
# Batch processing allows faster generation by processing multiple labels in parallel per GPU.
#
# Usage:
#   bash sample_xyzflow_trajectory_ddp_batch.sh
#
# Environment variables (optional):
#   NUM_GPUS          - Number of GPUs to use (default: 8)
#   CHECKPOINT        - Path to XYZFlow checkpoint
#   MODEL_SIZE        - Model size: base/large/huge (default: huge)
#   OUTPUT_ROOT       - Parent directory for trajectory runs (default: trajectories_runs)
#   OUTPUT_PREFIX     - Base name for each run (default: trajectories)
#   NUM_SAMPLES       - Samples per label (default: 1)
#   NUM_CLASSES       - Number of classes (default: 1000)
#   BATCH_SIZE        - Labels to process in parallel per GPU (default: 32)
#   STEPS_PER_PATCH   - Denoising steps per patch (default: 50)
#   SAVE_STEPS        - Steps to save (default: "0 10 20 30 40 50")
#   CFG               - Classifier-free guidance scale (default: 2.3)
#   MASTER_PORT       - DDP master port (default: auto-assign)
#   SEED              - Base random seed (default: 42; negative -> random per run)

set -euo pipefail

# Configuration
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NUM_GPUS=${NUM_GPUS:-8}
CHECKPOINT=${CHECKPOINT:-}
MODEL_SIZE=${MODEL_SIZE:-large}
OUTPUT_ROOT=${OUTPUT_ROOT:-trajectories_runs}
OUTPUT_PREFIX=${OUTPUT_PREFIX:-trajectories}
NUM_SAMPLES=${NUM_SAMPLES:-2500}
LABEL_START=${LABEL_START:-0}
LABEL_END=${LABEL_END:-999}
NUM_CLASSES=${NUM_CLASSES:-$((LABEL_END - LABEL_START + 1))}
BATCH_SIZE=${BATCH_SIZE:-16}
STEPS_PER_PATCH=${STEPS_PER_PATCH:-50}
SAVE_STEPS=${SAVE_STEPS:-"0 10 20 30 40 50"}
CFG=${CFG:-2.3}
SEED=${SEED:--1}
WRITE_WORKERS=${WRITE_WORKERS:-16}
MASTER_PORT=${MASTER_PORT:-29666}


TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_ROOT%/}/${OUTPUT_PREFIX}_${TIMESTAMP}"

# Logging
log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

log "=== XYZFlow Trajectory Sampling with BATCH (Standard Teacher) ==="
log "GPUs: ${NUM_GPUS}"
log "Batch size per GPU: ${BATCH_SIZE}"
log "Checkpoint: ${CHECKPOINT}"
log "Model size: ${MODEL_SIZE}"
log "Output directory: ${OUTPUT_DIR}"
log "Num samples per label: ${NUM_SAMPLES}"
log "Label range: ${LABEL_START}-${LABEL_END} (${NUM_CLASSES} classes)"
log "Steps per patch: ${STEPS_PER_PATCH}"
log "Save steps: ${SAVE_STEPS}"
log "CFG scale: ${CFG}"
log "Writer threads per rank: ${WRITE_WORKERS}"
log "=============================================================="

# Check checkpoint exists
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    log "ERROR: Checkpoint not found: ${CHECKPOINT}"
    log "Set CHECKPOINT=/path/to/XYZFlow-{B,L,H}.pth"
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" >/dev/null 2>&1

# Run DDP sampling with batch support
torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/sample_xyzflow_trajectory_ddp_batch.py" \
    --checkpoint "${CHECKPOINT}" \
    --model-size "${MODEL_SIZE}" \
    --output-dir "${OUTPUT_DIR}" \
    --num-samples "${NUM_SAMPLES}" \
    --num-classes "${NUM_CLASSES}" \
    --label-start "${LABEL_START}" \
    --label-end "${LABEL_END}" \
    --batch-size "${BATCH_SIZE}" \
    --steps-per-patch "${STEPS_PER_PATCH}" \
    --save-steps ${SAVE_STEPS} \
    --cfg "${CFG}" \
    --seed "${SEED}" \
    --write-workers "${WRITE_WORKERS}"

log ""
log "Batch trajectory sampling completed!"
log "Output saved to: ${OUTPUT_DIR}"
log ""
log "Directory structure:"
log "  ${OUTPUT_DIR}/"
log "  ├── patch_A/"
log "  │   ├── step_000/  # First denoising step"
log "  │   ├── step_010/"
log "  │   └── ..."
log "  ├── patch_B/"
log "  ├── patch_C/"
log "  ├── patch_D/"
log "  └── metadata.json  # File index"
log ""
log "Performance improvement vs non-batch version:"
log "  Batch size ${BATCH_SIZE}: ~${BATCH_SIZE}x faster per GPU"
