#!/usr/bin/env python3
"""
Standalone LeRobot Training Script
===================================

This script can be run on any machine with LeRobot installed, including compute clusters.
It does NOT require the Solo CLI, Ollama, or Docker.

Usage:
    python standalone_train.py --config train_config.yaml

    OR

    python standalone_train.py \
        --dataset-repo-id lerobot/svla_so101_pickplace \
        --policy-type smolvla \
        --output-dir outputs/train/my_model \
        --training-steps 20000 \
        --batch-size 8

Requirements:
    pip install lerobot wandb huggingface_hub
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional
import yaml


def setup_environment():
    """Set up environment variables for training"""
    os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

    # Suppress common warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
    warnings.filterwarnings("ignore", message=".*torch_dtype.*")
    warnings.filterwarnings("ignore", message=".*video decoding.*")


def check_dependencies():
    """Check if required packages are installed"""
    try:
        import lerobot
        print(f"✅ LeRobot version: {lerobot.__version__}")
    except ImportError:
        print("❌ LeRobot not found. Please install: pip install lerobot")
        sys.exit(1)

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA devices: {torch.cuda.device_count()}")
    except ImportError:
        print("❌ PyTorch not found. Please install: pip install torch")
        sys.exit(1)


def authenticate_huggingface(token: Optional[str] = None):
    """Authenticate with HuggingFace Hub"""
    from huggingface_hub import login, whoami

    try:
        # Try to get current user
        user_info = whoami()
        username = user_info.get('name', 'unknown')
        print(f"✅ Already logged into HuggingFace as: {username}")
        return username
    except Exception:
        # Not logged in, try to log in
        if token:
            try:
                login(token=token)
                user_info = whoami()
                username = user_info.get('name', 'unknown')
                print(f"✅ Logged into HuggingFace as: {username}")
                return username
            except Exception as e:
                print(f"❌ Failed to login with provided token: {e}")
                return None
        else:
            print("⚠️  Not logged into HuggingFace. Some features may be limited.")
            print("💡 Set HF_TOKEN environment variable or use --hf-token")
            return None


def setup_wandb(project_name: str, enable: bool = True):
    """Set up Weights & Biases logging"""
    if not enable:
        print("ℹ️  WandB logging disabled")
        return False

    try:
        import wandb

        # Check if already logged in
        if wandb.api.api_key is None:
            print("⚠️  WandB not configured. Set WANDB_API_KEY environment variable")
            return False

        print(f"✅ WandB configured for project: {project_name}")
        return True
    except ImportError:
        print("❌ WandB not found. Install with: pip install wandb")
        return False


def check_dataset_exists(repo_id: str) -> bool:
    """Check if dataset exists locally or on HuggingFace Hub"""
    from huggingface_hub import HfApi

    # Check local cache first
    cache_dir = os.path.expanduser("~/.cache/huggingface/lerobot")
    local_path = os.path.join(cache_dir, repo_id)

    if os.path.exists(local_path):
        print(f"✅ Found local dataset: {repo_id}")
        return True

    # Check on HuggingFace Hub
    try:
        api = HfApi()
        api.dataset_info(repo_id)
        print(f"✅ Found dataset on HuggingFace Hub: {repo_id}")
        return True
    except Exception:
        print(f"⚠️  Dataset not found: {repo_id}")
        return False


def get_policy_config(policy_type: str, pretrained_path: Optional[str] = None):
    """Get policy configuration"""
    from lerobot.configs.policies import PreTrainedConfig

    if pretrained_path:
        print(f"📥 Loading pretrained policy config from {pretrained_path}")
        policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
        policy_config.pretrained_path = pretrained_path
        return policy_config, policy_config.type

    # Create new policy config
    if policy_type == "diffusion":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        return DiffusionConfig(), policy_type
    elif policy_type == "act":
        from lerobot.policies.act.configuration_act import ACTConfig
        return ACTConfig(), policy_type
    elif policy_type == "tdmpc":
        from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
        return TDMPCConfig(), policy_type
    elif policy_type == "smolvla":
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        # Use default pretrained checkpoint for SmolVLA
        if not pretrained_path:
            pretrained_path = "lerobot/smolvla_base"
            print(f"ℹ️  Using default pretrained SmolVLA checkpoint: {pretrained_path}")
            policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
            policy_config.pretrained_path = pretrained_path
            return policy_config, policy_type
        return SmolVLAConfig(), policy_type
    elif policy_type == "pi0":
        from lerobot.policies.pi0.configuration_pi0 import PI0Config
        return PI0Config(), policy_type
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")


def handle_output_directory(output_dir: str, resume: bool = False):
    """Handle output directory validation and conflicts"""
    output_path = Path(output_dir)

    if output_path.exists() and output_path.is_dir():
        checkpoint_files = list(output_path.glob("**/*checkpoint*")) + list(output_path.glob("**/*.pt"))
        has_checkpoints = len(checkpoint_files) > 0

        if has_checkpoints and resume:
            print(f"🔄 Resuming training from: {output_dir}")
            return output_path, True
        elif has_checkpoints and not resume:
            print(f"⚠️  Output directory exists with checkpoints: {output_dir}")
            print("💡 Use --resume to continue training or --force to overwrite")
            sys.exit(1)

    # Don't create the directory - let LeRobot create it to avoid conflicts
    return output_path, resume


def train_model(
    dataset_repo_id: str,
    policy_type: str,
    output_dir: str,
    training_steps: int = 20000,
    batch_size: int = 8,
    save_freq: int = 1000,
    pretrained_path: Optional[str] = None,
    push_to_hub: bool = False,
    policy_repo_id: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = "lerobot-training",
    resume: bool = False,
    force: bool = False,
    seed: int = 1000,
    rename_map: Optional[dict] = None,
):
    """Main training function"""

    print("\n" + "="*60)
    print("🎓 LeRobot Standalone Training")
    print("="*60)

    # Check dependencies
    check_dependencies()

    # Set up environment
    setup_environment()

    # Check dataset
    if not check_dataset_exists(dataset_repo_id):
        print(f"❌ Dataset not found: {dataset_repo_id}")
        print("💡 Download dataset first or check the repo ID")
        sys.exit(1)

    # Handle output directory
    output_path, resume = handle_output_directory(output_dir, resume)

    # Import LeRobot components
    from lerobot.scripts.lerobot_train import train
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.default import DatasetConfig, WandBConfig

    # Create dataset config
    dataset_config = DatasetConfig(repo_id=dataset_repo_id)

    # Handle video backend
    if dataset_config.video_backend == "torchcodec":
        try:
            import torchcodec  # noqa: F401
        except Exception as torchcodec_error:
            print(f"⚠️  TorchCodec unavailable ({torchcodec_error}) — falling back to PyAV")
            dataset_config.video_backend = "pyav"
    print(f"ℹ️  Video backend: {dataset_config.video_backend}")

    # Get policy config
    policy_config, actual_policy_type = get_policy_config(policy_type, pretrained_path)

    # Set up hub pushing
    if push_to_hub:
        if not policy_repo_id:
            print("❌ --policy-repo-id required when --push-to-hub is enabled")
            sys.exit(1)
        policy_config.repo_id = policy_repo_id
        policy_config.push_to_hub = True
        print(f"🚀 Will push model to: {policy_repo_id}")

    # Set up WandB
    wandb_enabled = setup_wandb(wandb_project, use_wandb) if use_wandb else False
    wandb_config = WandBConfig(
        enable=wandb_enabled,
        project=wandb_project if wandb_enabled else None
    )

    # Create training config
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        policy=policy_config,
        output_dir=output_path,
        steps=training_steps,
        batch_size=batch_size,
        save_freq=save_freq,
        save_checkpoint=True,
        wandb=wandb_config,
        seed=seed,
        resume=resume,
        rename_map=rename_map or {},
    )

    # Print configuration
    print("\n📋 Training Configuration:")
    print(f"   • Dataset: {dataset_repo_id}")
    print(f"   • Policy: {actual_policy_type}")
    print(f"   • Training steps: {training_steps}")
    print(f"   • Batch size: {batch_size}")
    print(f"   • Save frequency: {save_freq}")
    print(f"   • Output directory: {output_dir}")
    print(f"   • Resume training: {resume}")
    print(f"   • Push to Hub: {push_to_hub}")
    if push_to_hub:
        print(f"   • Policy repository: {policy_repo_id}")
    print(f"   • WandB logging: {wandb_enabled}")
    if wandb_enabled:
        print(f"   • WandB project: {wandb_project}")
    if pretrained_path:
        print(f"   • Pretrained checkpoint: {pretrained_path}")

    print("\n💡 Training Tips:")
    print("   • Training progress will be logged to console")
    print(f"   • Checkpoints saved every {save_freq} steps to {output_dir}")
    if wandb_enabled:
        print(f"   • Monitor at https://wandb.ai/{wandb_project}")
    print("   • Press Ctrl+C to stop training early")

    # Start training
    try:
        print("\n🚀 Starting training...")
        print("="*60 + "\n")

        train(train_config)

        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print("="*60)
        print(f"📊 Dataset: {dataset_repo_id}")
        print(f"🤖 Policy: {actual_policy_type}")
        print(f"💾 Checkpoints saved to: {output_dir}")

        if push_to_hub and policy_repo_id:
            print(f"🚀 Model pushed to: https://huggingface.co/{policy_repo_id}")

        if wandb_enabled:
            print(f"📈 Training logs: https://wandb.ai/{wandb_project}")

        return True

    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user")
        print(f"💾 Partial checkpoints saved to: {output_dir}")
        return False
    except Exception as e:
        import traceback
        print(f"\n❌ Training failed: {e}")
        print("\n🔍 Full error traceback:")
        print(traceback.format_exc())
        return False


def load_config_file(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description="Standalone LeRobot Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with command-line arguments
  python standalone_train.py \\
      --dataset-repo-id lerobot/svla_so101_pickplace \\
      --policy-type smolvla \\
      --output-dir outputs/my_model \\
      --training-steps 20000 \\
      --batch-size 8

  # Train with config file
  python standalone_train.py --config train_config.yaml

  # Resume training
  python standalone_train.py --config train_config.yaml --resume

  # Train and push to HuggingFace Hub
  python standalone_train.py \\
      --dataset-repo-id lerobot/svla_so101_pickplace \\
      --policy-type smolvla \\
      --output-dir outputs/my_model \\
      --push-to-hub \\
      --policy-repo-id username/my-smolvla-model
        """
    )

    # Config file option
    parser.add_argument(
        '--config', type=str,
        help='Path to YAML config file (overrides other arguments)'
    )

    # Required arguments (unless using config file)
    parser.add_argument(
        '--dataset-repo-id', type=str,
        help='HuggingFace dataset repository ID (e.g., lerobot/svla_so101_pickplace)'
    )
    parser.add_argument(
        '--policy-type', type=str,
        choices=['smolvla', 'act', 'pi0', 'tdmpc', 'diffusion'],
        help='Policy type to train'
    )
    parser.add_argument(
        '--output-dir', type=str,
        help='Output directory for checkpoints'
    )

    # Optional training arguments
    parser.add_argument(
        '--training-steps', type=int, default=20000,
        help='Number of training steps (default: 20000)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8,
        help='Training batch size (default: 8)'
    )
    parser.add_argument(
        '--save-freq', type=int, default=1000,
        help='Checkpoint save frequency (default: 1000)'
    )
    parser.add_argument(
        '--seed', type=int, default=1000,
        help='Random seed (default: 1000)'
    )

    # Pretrained model
    parser.add_argument(
        '--pretrained-path', type=str,
        help='Path to pretrained checkpoint (HuggingFace model ID or local path)'
    )

    # HuggingFace Hub
    parser.add_argument(
        '--push-to-hub', action='store_true',
        help='Push trained model to HuggingFace Hub'
    )
    parser.add_argument(
        '--policy-repo-id', type=str,
        help='HuggingFace repository ID for trained model (required with --push-to-hub)'
    )
    parser.add_argument(
        '--hf-token', type=str,
        help='HuggingFace API token (or set HF_TOKEN env var)'
    )

    # WandB
    parser.add_argument(
        '--use-wandb', action='store_true',
        help='Enable Weights & Biases logging'
    )
    parser.add_argument(
        '--wandb-project', type=str, default='lerobot-training',
        help='WandB project name (default: lerobot-training)'
    )

    # Training control
    parser.add_argument(
        '--resume', action='store_true',
        help='Resume training from existing checkpoint'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite existing output directory'
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        config = load_config_file(args.config)

        # Override with command-line arguments
        dataset_repo_id = args.dataset_repo_id or config.get('dataset_repo_id')
        policy_type = args.policy_type or config.get('policy_type')
        output_dir = args.output_dir or config.get('output_dir')
        training_steps = args.training_steps if args.training_steps != 20000 else config.get('training_steps', 20000)
        batch_size = args.batch_size if args.batch_size != 8 else config.get('batch_size', 8)
        save_freq = args.save_freq if args.save_freq != 1000 else config.get('save_freq', 1000)
        pretrained_path = args.pretrained_path or config.get('pretrained_path')
        push_to_hub = args.push_to_hub or config.get('push_to_hub', False)
        policy_repo_id = args.policy_repo_id or config.get('policy_repo_id')
        use_wandb = args.use_wandb or config.get('use_wandb', False)
        wandb_project = args.wandb_project if args.wandb_project != 'lerobot-training' else config.get('wandb_project', 'lerobot-training')
        seed = args.seed if args.seed != 1000 else config.get('seed', 1000)
        rename_map = config.get('rename_map')
    else:
        # Use command-line arguments
        dataset_repo_id = args.dataset_repo_id
        policy_type = args.policy_type
        output_dir = args.output_dir
        training_steps = args.training_steps
        batch_size = args.batch_size
        save_freq = args.save_freq
        pretrained_path = args.pretrained_path
        push_to_hub = args.push_to_hub
        policy_repo_id = args.policy_repo_id
        use_wandb = args.use_wandb
        wandb_project = args.wandb_project
        seed = args.seed
        rename_map = None

    # Validate required arguments
    if not dataset_repo_id:
        parser.error("--dataset-repo-id is required")
    if not policy_type:
        parser.error("--policy-type is required")
    if not output_dir:
        parser.error("--output-dir is required")

    # Authenticate with HuggingFace
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    authenticate_huggingface(hf_token)

    # Run training
    success = train_model(
        dataset_repo_id=dataset_repo_id,
        policy_type=policy_type,
        output_dir=output_dir,
        training_steps=training_steps,
        batch_size=batch_size,
        save_freq=save_freq,
        pretrained_path=pretrained_path,
        push_to_hub=push_to_hub,
        policy_repo_id=policy_repo_id,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        resume=args.resume,
        force=args.force,
        seed=seed,
        rename_map=rename_map,
    )

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
