#!/bin/bash

# ============================================================
# Local Training Script for LeRobot
# ============================================================
#
# Usage: ./run_training.sh
#
# This script runs training locally (not on a cluster).
# Modify the variables below to customize your training.
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "Starting LeRobot Training"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# ============================================================
# Configuration
# ============================================================

# Set your HuggingFace token (get from https://huggingface.co/settings/tokens)
export HF_TOKEN="${HF_TOKEN:-hf_your_token_here}"

# Optional: WandB API key for logging (get from https://wandb.ai/authorize)
# export WANDB_API_KEY="your_wandb_key_here"

# Create logs directory
mkdir -p logs

# Redirect output to log file while also showing on screen
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
exec &> >(tee -a "$LOG_FILE")

# ============================================================
# Run Training
# ============================================================

# Option 1: Use config file (recommended)
python standalone_train.py --config train_config.yaml

# Option 2: Use command-line arguments
# Uncomment and modify as needed:
# python standalone_train.py \
#     --dataset-repo-id lerobot/svla_so101_pickplace \
#     --policy-type smolvla \
#     --output-dir outputs/my_model \
#     --training-steps 20000 \
#     --batch-size 8 \
#     --save-freq 1000

# Option 3: Quick test run (100 steps)
# python standalone_train.py \
#     --dataset-repo-id lerobot/svla_so101_pickplace \
#     --policy-type smolvla \
#     --output-dir outputs/test \
#     --training-steps 100 \
#     --batch-size 4

# ============================================================
# Completion
# ============================================================

EXIT_STATUS=$?

echo ""
echo "============================================================"
if [ $EXIT_STATUS -eq 0 ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed with exit status: $EXIT_STATUS"
fi
echo "End time: $(date)"
echo "Log saved to: $LOG_FILE"
echo "============================================================"

exit $EXIT_STATUS
