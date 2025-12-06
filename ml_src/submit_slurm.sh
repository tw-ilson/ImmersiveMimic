#!/bin/bash
#SBATCH --job-name=lerobot_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# SLURM Job Script for LeRobot Training
# ============================================================
#
# Usage: sbatch submit_slurm.sh
#
# Customize the SBATCH directives above based on your cluster:
# - Adjust GPU type: --gres=gpu:a100:1
# - Adjust memory: --mem=64G
# - Adjust time limit: --time=48:00:00
# ============================================================

echo "Starting LeRobot training job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Load modules (customize for your cluster)
module load python/3.10
module load cuda/12.1

# Optional: Activate virtual environment
# source ~/venv/bin/activate

# Set environment variables
export HF_TOKEN="hf_your_token_here"  # Replace with your token
export WANDB_API_KEY="your_wandb_key_here"  # Optional: for WandB logging

# Create logs directory if it doesn't exist
mkdir -p logs

# Run training
# Option 1: Use config file (recommended)
python standalone_train.py --config train_config.yaml

# Option 2: Use command-line arguments
# python standalone_train.py \
#     --dataset-repo-id lerobot/svla_so101_pickplace \
#     --policy-type smolvla \
#     --output-dir $SCRATCH/outputs/my_model \
#     --training-steps 20000 \
#     --batch-size 8 \
#     --use-wandb \
#     --wandb-project robot-training

echo "Job completed at: $(date)"
echo "Exit status: $?"
