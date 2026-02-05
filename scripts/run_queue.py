#!/usr/bin/env python3
"""
Job Queue Runner for Underwater Benchmark Training

Reads YAML job configs from a directory and executes them sequentially.
Designed to run inside a Docker container with a single GPU.

Usage:
    python run_queue.py --jobs-dir /workspace/jobs/gpu0
    python run_queue.py --jobs-dir /workspace/jobs/gpu0 --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def load_job_config(config_path: Path) -> dict[str, Any]:
    """Load a YAML job configuration file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_command(config: dict[str, Any], base_script: str = "train_semantic.py") -> list[str]:
    """Build the training command from a job config."""
    cmd = ["python", base_script]

    # Required arguments
    cmd.extend(["--dataset", config["dataset"]])
    cmd.extend(["--model", config["model"]])

    # Data paths
    if "data_root" in config:
        cmd.extend(["--data_root", config["data_root"]])

    # Training parameters
    if "batch_size" in config:
        cmd.extend(["--batch_size", str(config["batch_size"])])
    if "epochs" in config:
        cmd.extend(["--epochs", str(config["epochs"])])
    if "lr" in config:
        cmd.extend(["--lr", str(config["lr"])])
    if "weight_decay" in config:
        cmd.extend(["--weight_decay", str(config["weight_decay"])])

    # Scheduler
    if "scheduler" in config:
        cmd.extend(["--scheduler", config["scheduler"]])
    if "warmup_epochs" in config:
        cmd.extend(["--warmup_epochs", str(config["warmup_epochs"])])

    # Augmentation
    if "augmentation" in config:
        cmd.extend(["--augmentation", config["augmentation"]])

    # Hardware
    if "gpus" in config:
        cmd.extend(["--gpus", config["gpus"]])
    if "precision" in config:
        cmd.extend(["--precision", config["precision"]])
    if "num_workers" in config:
        cmd.extend(["--num_workers", str(config["num_workers"])])
    if "accumulate_grad_batches" in config:
        cmd.extend(["--accumulate_grad_batches", str(config["accumulate_grad_batches"])])

    # Logging
    if "wandb_project" in config:
        cmd.extend(["--wandb_project", config["wandb_project"]])
    if "experiment_name" in config:
        cmd.extend(["--experiment_name", config["experiment_name"]])
    if "log_dir" in config:
        cmd.extend(["--log_dir", config["log_dir"]])

    # Boolean flags
    if config.get("freeze_backbone", False):
        cmd.append("--freeze_backbone")
    if config.get("hf_login", False):
        cmd.append("--hf_login")
    if config.get("cache_dataset", False):
        cmd.append("--cache_dataset")
    if "early_stopping" in config:
        cmd.extend(["--early_stopping", str(config["early_stopping"])])

    # Resume from checkpoint
    if "resume_from" in config:
        cmd.extend(["--resume_from", config["resume_from"]])

    # Additional arguments (passthrough)
    if "extra_args" in config:
        cmd.extend(config["extra_args"])

    return cmd


def run_job(config_path: Path, dry_run: bool = False, workdir: Path | None = None) -> bool:
    """Run a single training job. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Job: {config_path.name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    try:
        config = load_job_config(config_path)
        cmd = build_command(config)

        print(f"Command: {' '.join(cmd)}\n")

        if dry_run:
            print("[DRY RUN] Would execute the above command")
            return True

        # Run the training
        result = subprocess.run(
            cmd,
            cwd=workdir or Path(__file__).parent.parent,
            env=os.environ.copy(),
        )

        success = result.returncode == 0
        status = "COMPLETED" if success else f"FAILED (exit code: {result.returncode})"

    except Exception as e:
        print(f"ERROR: {e}")
        success = False
        status = f"ERROR: {e}"

    print(f"\n{'='*60}")
    print(f"Job: {config_path.name} - {status}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    return success


def get_job_files(jobs_dir: Path) -> list[Path]:
    """Get all YAML job files from directory, sorted by name."""
    patterns = ["*.yaml", "*.yml"]
    jobs = []
    for pattern in patterns:
        jobs.extend(jobs_dir.glob(pattern))
    return sorted(jobs, key=lambda p: p.name)


def main():
    parser = argparse.ArgumentParser(description="Run training jobs from YAML configs")
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        required=True,
        help="Directory containing YAML job configs",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Working directory for running training (default: parent of this script)",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running remaining jobs even if one fails",
    )
    args = parser.parse_args()

    if not args.jobs_dir.exists():
        print(f"ERROR: Jobs directory does not exist: {args.jobs_dir}")
        sys.exit(1)

    jobs = get_job_files(args.jobs_dir)

    if not jobs:
        print(f"No job files found in {args.jobs_dir}")
        sys.exit(0)

    print(f"\n{'#'*60}")
    print(f"# Job Queue Runner")
    print(f"# Jobs directory: {args.jobs_dir}")
    print(f"# Found {len(jobs)} job(s)")
    print(f"# Mode: {'DRY RUN' if args.dry_run else 'EXECUTE'}")
    print(f"{'#'*60}\n")

    for i, job_path in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {job_path.name}")

    print()

    # Track results
    results = {}
    start_time = time.time()

    for i, job_path in enumerate(jobs, 1):
        print(f"\n>>> Running job {i}/{len(jobs)}: {job_path.name}")

        success = run_job(job_path, dry_run=args.dry_run, workdir=args.workdir)
        results[job_path.name] = success

        if not success and not args.continue_on_failure:
            print(f"\nStopping queue due to job failure. Use --continue-on-failure to override.")
            break

    # Summary
    elapsed = time.time() - start_time
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)

    print(f"\n{'#'*60}")
    print(f"# Queue Summary")
    print(f"# Total jobs: {len(jobs)}")
    print(f"# Completed: {len(results)}")
    print(f"# Succeeded: {succeeded}")
    print(f"# Failed: {failed}")
    print(f"# Elapsed: {elapsed/3600:.2f} hours")
    print(f"{'#'*60}\n")

    # Exit with error if any job failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
