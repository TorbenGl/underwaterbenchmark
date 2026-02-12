#!/usr/bin/env python3
"""
Training script for Image Classification.

Uses dataset and model presets for streamlined configuration.

Usage:
    python train_classification.py --dataset cou --model cls_dinov3_base
    python train_classification.py --dataset suim --model cls_dinov3_small --batch_size 8
"""
import argparse
import os
import shutil
import sys
from datetime import datetime
from typing import List

import torch
import lightning as L
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
    RichProgressBar,
)
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
import wandb

from huggingface_hub import login as hf_login

from modules.ClassificationLightningModule import ClassificationLightningModule
from presets.DataModules import get_datamodule, AVAILABLE_DATASETS, set_data_root, get_data_root, _DATASET_REGISTRY
from presets.Models import get_model, AVAILABLE_MODELS
from preprocessors import PaddingPreprocessor
from src.augmentation.kornia_augs import (
    LightAugmentation,
    MediumAugmentation,
    HeavyAugmentation,
    UnderwaterAugmentation,
    SemanticSegmentationAugmentation,
)

try:
    from rich.console import Console
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Image Classification Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset & Model (required)
    parser.add_argument("--dataset", type=str, required=True, choices=AVAILABLE_DATASETS,
                        help=f"Dataset preset: {', '.join(AVAILABLE_DATASETS)}")
    parser.add_argument("--model", type=str, default="cls_dinov3_base", choices=AVAILABLE_MODELS,
                        help=f"Model preset: {', '.join(AVAILABLE_MODELS)}")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Global data root directory")
    parser.add_argument("--img_size", type=int, nargs=2, default=None,
                        help="Override image size (height width)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "step", "polynomial", "none"])
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--early_stopping", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")

    # Augmentation
    parser.add_argument("--augmentation", type=str, default="medium",
                        choices=["none", "light", "medium", "heavy", "underwater"])

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cache_dataset", action="store_true")

    # Hardware
    parser.add_argument("--gpus", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="32",
                        choices=["16-mixed", "bf16-mixed", "32"])
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)

    # Logging
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="./logs")

    # Checkpoints
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_top_k", type=int, default=3)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fast_dev_run", action="store_true")
    parser.add_argument("--hf_login", action="store_true")

    return parser.parse_args()


def setup_callbacks(args, run_dir: str) -> List[L.Callback]:
    """Setup training callbacks."""
    callbacks = []

    callbacks.append(ModelCheckpoint(
        dirpath=os.path.join(run_dir, "checkpoints"),
        filename="epoch={epoch:02d}-val_acc={val/acc:.4f}",
        monitor="val/acc",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    ))

    callbacks.append(LearningRateMonitor(logging_interval="step"))

    if args.early_stopping:
        callbacks.append(EarlyStopping(
            monitor="val/acc",
            mode="max",
            patience=args.early_stopping,
        ))

    try:
        callbacks.append(RichProgressBar())
    except Exception:
        from lightning.pytorch.callbacks import TQDMProgressBar
        callbacks.append(TQDMProgressBar())

    return callbacks


def setup_logger(args):
    """Setup experiment logger and run directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name:
        run_name = f"{args.experiment_name}_{args.model}_aug-{args.augmentation}_{timestamp}"
    else:
        run_name = f"{args.dataset}_{args.model}_aug-{args.augmentation}_{timestamp}"

    if args.wandb_project:
        project_name = f"{args.wandb_project}_{args.dataset}"
        run_dir = os.path.join(args.log_dir, project_name, run_name)
        os.makedirs(run_dir, exist_ok=True)
        logger = WandbLogger(
            project=project_name,
            name=run_name,
            save_dir=run_dir,
            log_model=True,
        )
        logger.experiment.config.update(vars(args))
    else:
        run_dir = os.path.join(args.log_dir, run_name)
        os.makedirs(run_dir, exist_ok=True)
        logger = TensorBoardLogger(save_dir=run_dir, name="tensorboard")

    return logger, run_dir


def print_config(args, datamodule, model_info):
    """Print training configuration."""
    num_classes = datamodule.get_num_classes()
    img_size = datamodule.img_size
    id2label = datamodule.get_id2label() if hasattr(datamodule, 'get_id2label') else None

    if RICH_AVAILABLE:
        console = Console()

        table = Table(title="Classification Training Configuration", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Task", "Classification")
        table.add_row("Dataset", args.dataset)
        table.add_row("Model", args.model)
        table.add_row("Backbone", model_info.get("backbone", "N/A"))
        table.add_row("Pooling Mode", model_info.get("pooling_mode", "N/A"))
        table.add_row("Data Root", get_data_root())
        table.add_row("Image Size", str(img_size))
        table.add_row("Num Classes", str(num_classes))
        table.add_row("Augmentation", args.augmentation)
        table.add_row("Batch Size", str(args.batch_size))
        table.add_row("Learning Rate", str(args.lr))
        table.add_row("Epochs", str(args.epochs))
        table.add_row("Freeze Backbone", str(args.freeze_backbone))
        table.add_row("GPUs", str(args.gpus))
        table.add_row("Precision", args.precision)

        console.print(table)

        if id2label:
            class_table = Table(title="Classes", show_header=True)
            class_table.add_column("ID", style="cyan")
            class_table.add_column("Label", style="green")
            for idx, label in sorted(id2label.items()):
                class_table.add_row(str(idx), label)
            console.print(class_table)
    else:
        print("\n" + "=" * 60)
        print("Classification Training Configuration")
        print("=" * 60)
        print(f"  Dataset: {args.dataset}")
        print(f"  Model: {args.model}")
        print(f"  Backbone: {model_info.get('backbone', 'N/A')}")
        print(f"  Pooling Mode: {model_info.get('pooling_mode', 'N/A')}")
        print(f"  Image Size: {img_size}")
        print(f"  Num Classes: {num_classes}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Epochs: {args.epochs}")
        print("=" * 60 + "\n")


def create_augmentations(args, img_size):
    """Create Kornia augmentation modules."""
    if args.augmentation == "none":
        return None, None

    aug_map = {
        "light": LightAugmentation,
        "medium": MediumAugmentation,
        "heavy": HeavyAugmentation,
        "underwater": UnderwaterAugmentation,
    }

    train_aug = aug_map[args.augmentation](img_size=img_size)
    val_aug = SemanticSegmentationAugmentation(img_size=img_size, train=False)

    print(f"Using {args.augmentation} augmentation")
    return train_aug, val_aug


def main():
    args = parse_args()

    # HuggingFace auth
    if args.hf_login:
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            hf_login(token=hf_token)
            print("Logged in to HuggingFace")
        else:
            print("HF_TOKEN not found")
            sys.exit(1)

    # Set seed
    L.seed_everything(args.seed, workers=True)

    # Set data root
    if args.data_root:
        set_data_root(args.data_root)
    print(f"Data root: {get_data_root()}")

    # Setup logger and run directory
    logger, run_dir = setup_logger(args)
    print(f"Run directory: {run_dir}")

    # Copy trainingjob.sh to run directory if it exists
    trainingjob_path = os.path.join(os.path.dirname(__file__), "trainingjob.sh")
    if os.path.exists(trainingjob_path):
        shutil.copy(trainingjob_path, os.path.join(run_dir, "trainingjob.sh"))
        print(f"Copied trainingjob.sh to {run_dir}")

    # Get dataset metadata
    print(f"\nLoading dataset: {args.dataset}")
    dataset_cls = _DATASET_REGISTRY[args.dataset]
    metadata = dataset_cls.get_metadata(
        img_size=tuple(args.img_size) if args.img_size else None
    )

    num_classes = metadata["num_classes"]
    img_size = metadata["img_size"]
    id2label = metadata.get("id2label", {})

    print(f"  Classes: {num_classes}")
    print(f"  Image size: {img_size}")

    # Create model from preset
    print(f"\nLoading model: {args.model}")
    model = get_model(
        name=args.model,
        num_classes=num_classes,
        img_size=img_size,
    )

    # Get model info for logging
    from presets.Models import _MODEL_REGISTRY
    model_info = _MODEL_REGISTRY[args.model].get_info()
    print(f"  Backbone: {model_info.get('backbone', 'N/A')}")

    # Create padding preprocessor
    preprocessor = PaddingPreprocessor(
        target_size=img_size,
        mask_pad_value=255,
    )

    # Create datamodule with preprocessor
    datamodule = get_datamodule(
        name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=tuple(args.img_size) if args.img_size else None,
        preprocessor=preprocessor,
        cache_dataset=args.cache_dataset,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    datamodule.setup(stage="fit")

    print(f"  Train samples: {len(datamodule.train_dataset)}")
    print(f"  Val samples: {len(datamodule.val_dataset)}")

    # Print config
    print_config(args, datamodule, model_info)

    # Create augmentations
    train_aug, val_aug = create_augmentations(args, img_size)

    # Create Lightning module
    lightning_module = ClassificationLightningModule(
        model=model,
        num_classes=num_classes,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler=args.scheduler if args.scheduler != "none" else None,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        freeze_backbone=args.freeze_backbone,
        train_augmentation=train_aug,
        val_augmentation=val_aug,
        id2label=id2label,
    )

    torch.set_float32_matmul_precision('medium')

    # Print model info
    total_params = sum(p.numel() for p in lightning_module.parameters())
    trainable_params = sum(p.numel() for p in lightning_module.parameters() if p.requires_grad)
    print(f"\nModel: {total_params:,} params ({trainable_params:,} trainable)\n")

    # Setup callbacks
    callbacks = setup_callbacks(args, run_dir)

    # Parse devices
    if args.gpus == "auto":
        accelerator, devices = "auto", "auto"
    else:
        accelerator = "gpu"
        devices = [int(g) for g in args.gpus.split(",")]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        accumulate_grad_batches=args.accumulate_grad_batches,
        fast_dev_run=args.fast_dev_run,
        deterministic=False,
        benchmark=True,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting Classification Training")
    print("=" * 60 + "\n")

    trainer.fit(lightning_module, datamodule=datamodule, ckpt_path=args.resume_from)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val/acc: {callbacks[0].best_model_score:.4f}")

    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
