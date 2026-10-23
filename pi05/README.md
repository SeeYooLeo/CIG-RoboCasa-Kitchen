# π0.5 on robocasa-kitchen

The CIG continuation recipe uses the [robocasa-kitchen settings in Appendix C](../docs/paper.md#continuation-training-configurations): four GPUs and 5,000 optimizer updates. Baseline imitation training has its own schedule below.

[Project overview](../README.md) · [Installation](../docs/installation.md) · [Data preparation](../data/README.md) · [Checkpoint usage](../docs/checkpoints.md) · [Evaluation](../evaluation/README.md)

We train the **π0.5 baseline using the official Physical Intelligence openpi JAX training script**, with the robocasa-kitchen configuration below. JAX/Orbax parameters are converted to PyTorch SafeTensors for inference and CIG continuation. The released `pi05-baseline` weights are already converted.

Install the π0.5 environment and prepare the LeRobot dataset before training.

Set `ROOT` to the absolute path of the release in each terminal. Commands use Bash.

```bash
export ROOT=/absolute/path/to/robocasa-kitchen
```

## Baseline: official JAX training

`pi05/scripts/train_jax.py` retains the official [openpi baseline training loop](https://github.com/Physical-Intelligence/openpi/blob/c2b6977d84a460f3f12ba3054ff04e169d2264bd/scripts/train.py). The robocasa-kitchen configuration is in `pi05/src/openpi/training/config.py`:

- `Pi0Config(pi05=True, action_horizon=10, action_dim=32)`.
- Three cameras, task text from LeRobot metadata, `exclude_base_movement=True`.
- Initial weights: `gs://openpi-assets/checkpoints/pi05_base/params`.
- Global batch size 64, learning rate `5e-5`, warmup 1,000 steps, AdamW, gradient clipping 1.0, EMA 0.999.
- The release recipe stops after 15,000 updates. Set `--num-train-steps` to train longer.

```bash
source "$ROOT/.venv-pi05/bin/activate"
export HF_LEROBOT_HOME="$ROOT/datasets/lerobot"
unset JAX_PLATFORMS
cd "$ROOT/pi05"
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train_jax.py pi05-baseline \
  --exp-name baseline --data.repo-id robocasa-kitchen --batch-size 64 \
  --num-train-steps 15000 --fsdp-devices 1
```

Use `--fsdp-devices` to shard across multiple visible GPUs when necessary. The trainer writes JAX/Orbax checkpoints to `checkpoints/pi05-baseline/baseline/<step>/`. The upstream loop uses zero-based step labels: after 15,000 updates, the final directory is normally `14999`. `--resume` resumes a full training checkpoint; the released PyTorch weights do not contain the JAX optimizer state.

### Convert JAX parameters to PyTorch

Pass the **`params` subdirectory** of the selected JAX checkpoint, not its optimizer directory:

```bash
JAX_PLATFORMS=cpu python scripts/convert_jax_to_pytorch.py \
  --checkpoint-dir checkpoints/pi05-baseline/baseline/14999/params \
  --config-name pi05-baseline \
  --output-path ../weights/pi05-baseline-new \
  --precision bfloat16
```

The converter maps JAX parameter names/layouts to the PyTorch architecture, checks unmatched parameters, preserves tied embeddings, writes `model.safetensors`, and copies the adjacent `assets/` directory. Use a new output directory. The resulting directory must include `assets/robocasa-kitchen/norm_stats.json` before serving. Existing released weights need no conversion.

## CIG continuation in PyTorch

The CIG implementations compare positive and negative instruction-conditioned action energies. OFT uses L1 energy; π0.5 uses flow-matching energy with shared noise/time. Negative instructions come from `data/instruction_pool.json`. The action expert receives the imitation-learning gradient, while CIG updates the conditioning model.

```bash
source "$ROOT/.venv-pi05/bin/activate"
export HF_LEROBOT_HOME="$ROOT/datasets/lerobot"
cd "$ROOT/pi05"
JAX_PLATFORMS=cpu PYTHONPATH=. torchrun --standalone --nproc_per_node=4 \
  scripts/train.py pi05-cig --exp-name cig \
  --pytorch-weight-path ../weights/pi05-baseline \
  --data.repo-id robocasa-kitchen --data.assets.assets-dir ../weights/pi05-baseline/assets \
  --batch-size 128 --num-train-steps 5000 \
  --cig.lambda-weight 0.1 --cig.tau-c 1.0 --cig.num-negatives 4 \
  --cig.cig-floor 0.5 --cig.hard-negatives \
  --cig.hard-neg-same-ratio 0.5 --cig.hard-neg-cross-ratio 0.25
```

Batch size is global and must be divisible by the number of processes. Reduce it to fit available memory; `--gradient-accumulation-steps` increases the effective batch size. Checkpoints are written under `pi05/checkpoints/pi05-cig/cig/<step>/`. CIG continuation initializes from the released baseline with a fresh optimizer. `--resume` is for full checkpoints created by this trainer. W&B is disabled by default.
