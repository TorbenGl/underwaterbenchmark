# Job Queue Configuration

This directory contains YAML job configs organized by GPU.

## Structure

```
jobs/
├── job_template.yaml    # Template to copy from
├── gpu0/                # Jobs for GPU 0 container
│   ├── 001_*.yaml       # First job to run
│   ├── 002_*.yaml       # Second job to run
│   └── ...
├── gpu1/                # Jobs for GPU 1 container
├── gpu2/                # Jobs for GPU 2 container
└── gpu3/                # Jobs for GPU 3 container
```

## Usage

1. Copy `job_template.yaml` to `gpuX/NNN_description.yaml`
2. Edit the config with your training parameters
3. Jobs run in alphabetical order (use numeric prefixes: 001_, 002_, etc.)

## Quick Start

```bash
# Copy template
cp job_template.yaml gpu0/001_my_experiment.yaml

# Edit the config
nano gpu0/001_my_experiment.yaml

# Start the queue
docker-compose up -d gpu0
```

## Tips

- Use numeric prefixes (001_, 002_) to control execution order
- Each container sees GPU as device 0 (set `gpus: "0"` in configs)
- Use descriptive names: `001_cou_dinov3_base_lr1e4.yaml`
- Jobs continue on failure by default (--continue-on-failure flag)
