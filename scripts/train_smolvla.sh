#!/usr/bin/env bash
set -euo pipefail

# Fine-tune HuggingFace SmolVLA on the local PiperXPickFoam LeRobotDataset (RTX 4080 16GB).
# Run inside the `lerobot` conda environment:
#   conda activate lerobot
#   bash scripts/train_smolvla.sh
#
# Optional overrides:
#   STEPS=10 SAVE_CKPT=false bash scripts/train_smolvla.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STEPS="${STEPS:-5000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
SAVE_CKPT="${SAVE_CKPT:-true}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/piperx_pick_foam_aligned_lerobot}"
REPO_ID="${REPO_ID:-local/piperx_pick_foam_aligned}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/outputs/smolvla_pick_foam_aligned}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-$(command -v lerobot-train || echo lerobot-train)}"

"${LEROBOT_TRAIN}" \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --dataset.repo_id="${REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.video_backend=pyav \
  --dataset.image_transforms.enable=true \
  --batch_size="${BATCH_SIZE}" \
  --accelerator.mixed_precision=bf16 \
  --num_workers=4 \
  --steps="${STEPS}" \
  --log_freq=50 \
  --save_checkpoint="${SAVE_CKPT}" \
  --save_freq="${SAVE_FREQ}" \
  --output_dir="${OUT_DIR}" \
  --trackio.enable=false
