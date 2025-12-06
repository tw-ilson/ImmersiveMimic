#!/bin/bash

# ============================================================
# Setup Script for LeRobot Standalone Training
# ============================================================
#
# This script sets up the environment for standalone training.
# Run this once before running training scripts.
#
# Usage: ./setup_training.sh
# ============================================================

set -e

echo "============================================================"
echo "Setting up LeRobot Standalone Training Environment"
echo "============================================================"
echo ""

# ============================================================
# Step 1: Check Python version
# ============================================================

echo "Step 1: Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
    echo "❌ Python 3.8+ required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✅ Python $PYTHON_VERSION"
echo ""

# ============================================================
# Step 2: Create virtual environment (optional but recommended)
# ============================================================

echo "Step 2: Virtual Environment Setup"
read -p "Create a new virtual environment? (y/n) [y]: " CREATE_VENV
CREATE_VENV=${CREATE_VENV:-y}

if [ "$CREATE_VENV" = "y" ] || [ "$CREATE_VENV" = "Y" ]; then
    if [ -d "training_venv" ]; then
        echo "⚠️  Virtual environment 'training_venv' already exists"
        read -p "Remove and recreate? (y/n) [n]: " RECREATE
        RECREATE=${RECREATE:-n}
        if [ "$RECREATE" = "y" ] || [ "$RECREATE" = "Y" ]; then
            rm -rf training_venv
            python3 -m venv training_venv
            echo "✅ Created new virtual environment"
        fi
    else
        python3 -m venv training_venv
        echo "✅ Created virtual environment"
    fi

    echo "Activating virtual environment..."
    source training_venv/bin/activate
else
    echo "ℹ️  Skipping virtual environment creation"
fi
echo ""

# ============================================================
# Step 3: Install dependencies
# ============================================================

echo "Step 3: Installing dependencies..."
echo "This may take several minutes..."
echo ""

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install core dependencies
echo "Installing LeRobot..."
pip install lerobot

echo "Installing PyTorch (if not already installed)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optional dependencies
echo "Installing optional dependencies..."
pip install wandb huggingface_hub pyyaml

echo ""
echo "✅ Dependencies installed"
echo ""

# ============================================================
# Step 4: Verify installation
# ============================================================

echo "Step 4: Verifying installation..."
python3 -c "
import sys
try:
    import lerobot
    print(f'✅ LeRobot version: {lerobot.__version__}')
except ImportError:
    print('❌ LeRobot not found')
    sys.exit(1)

try:
    import torch
    print(f'✅ PyTorch version: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'✅ CUDA devices: {torch.cuda.device_count()}')
except ImportError:
    print('❌ PyTorch not found')
    sys.exit(1)

try:
    import wandb
    print('✅ WandB installed')
except ImportError:
    print('⚠️  WandB not found (optional)')

try:
    import huggingface_hub
    print('✅ HuggingFace Hub installed')
except ImportError:
    print('❌ HuggingFace Hub not found')
    sys.exit(1)
"

echo ""

# ============================================================
# Step 5: Create configuration template
# ============================================================

echo "Step 5: Creating configuration template..."

if [ ! -f "train_config.yaml" ]; then
    cp train_config.example.yaml train_config.yaml
    echo "✅ Created train_config.yaml from example"
    echo "📝 Edit train_config.yaml to customize your training"
else
    echo "ℹ️  train_config.yaml already exists (not overwriting)"
fi

echo ""

# ============================================================
# Step 6: Set up directories
# ============================================================

echo "Step 6: Setting up directories..."
mkdir -p logs
mkdir -p outputs
echo "✅ Created logs/ and outputs/ directories"
echo ""

# ============================================================
# Step 7: HuggingFace authentication
# ============================================================

echo "Step 7: HuggingFace Authentication"
echo ""
echo "To use HuggingFace datasets and models, you need an API token."
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""
read -p "Configure HuggingFace token now? (y/n) [y]: " SETUP_HF
SETUP_HF=${SETUP_HF:-y}

if [ "$SETUP_HF" = "y" ] || [ "$SETUP_HF" = "Y" ]; then
    echo ""
    echo "Enter your HuggingFace token (or press Enter to skip):"
    read -s HF_TOKEN

    if [ -n "$HF_TOKEN" ]; then
        # Save to environment file
        echo "export HF_TOKEN=\"$HF_TOKEN\"" >> .env
        echo "✅ HuggingFace token saved to .env"
        echo "ℹ️  Source this file before running: source .env"
    else
        echo "⚠️  Skipped HuggingFace token setup"
        echo "💡 Set manually later: export HF_TOKEN='hf_your_token_here'"
    fi
else
    echo "⚠️  Skipped HuggingFace authentication"
    echo "💡 Set token manually: export HF_TOKEN='hf_your_token_here'"
fi

echo ""

# ============================================================
# Step 8: Optional WandB setup
# ============================================================

echo "Step 8: Weights & Biases (WandB) Setup (optional)"
read -p "Configure WandB for logging? (y/n) [n]: " SETUP_WANDB
SETUP_WANDB=${SETUP_WANDB:-n}

if [ "$SETUP_WANDB" = "y" ] || [ "$SETUP_WANDB" = "Y" ]; then
    echo ""
    echo "Get your WandB API key from: https://wandb.ai/authorize"
    echo "Enter your WandB API key (or press Enter to skip):"
    read -s WANDB_KEY

    if [ -n "$WANDB_KEY" ]; then
        echo "export WANDB_API_KEY=\"$WANDB_KEY\"" >> .env
        echo "✅ WandB API key saved to .env"
    else
        echo "⚠️  Skipped WandB setup"
    fi
else
    echo "ℹ️  Skipped WandB setup"
    echo "💡 Set later: export WANDB_API_KEY='your_key_here'"
fi

echo ""

# ============================================================
# Completion
# ============================================================

echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate virtual environment (if created):"
echo "   source training_venv/bin/activate"
echo ""
echo "2. Source environment file (if created):"
echo "   source .env"
echo ""
echo "3. Edit configuration file:"
echo "   nano train_config.yaml"
echo ""
echo "4. Run training:"
echo "   python standalone_train.py --config train_config.yaml"
echo ""
echo "   OR use the helper script:"
echo "   ./run_training.sh"
echo ""
echo "5. For cluster submission:"
echo "   - SLURM: sbatch submit_slurm.sh"
echo "   - PBS:   qsub submit_pbs.sh"
echo ""
echo "For more information, see: STANDALONE_TRAINING.md"
echo "============================================================"
