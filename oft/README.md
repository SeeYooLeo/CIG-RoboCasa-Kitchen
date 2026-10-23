# OpenVLA-OFT on robocasa-kitchen

The CIG continuation recipe uses the [robocasa-kitchen settings in Appendix C](../docs/paper.md#continuation-training-configurations): four GPUs and 5,000 optimizer updates. Baseline imitation training has its own schedule below.

[Project overview](../README.md) · [Installation](../docs/installation.md) · [Data preparation](../data/README.md) · [Checkpoint usage](../docs/checkpoints.md) · [Evaluation](../evaluation/README.md)

Install the OFT environment and build the robocasa-kitchen RLDS dataset before training. Released OFT checkpoints include merged LoRA weights, the action head, the proprio projector and dataset statistics.

Set `ROOT` to the absolute path of the release in each terminal. Commands use Bash.

```bash
export ROOT=/absolute/path/to/robocasa-kitchen
```

## Baseline and CIG training

The CIG implementations compare positive and negative instruction-conditioned action energies. OFT uses L1 energy; π0.5 uses flow-matching energy with shared noise/time. Negative instructions come from `data/instruction_pool.json`. The action expert receives the imitation-learning gradient, while CIG updates the conditioning model.

```bash
source "$ROOT/.venv-oft/bin/activate"
cd "$ROOT/oft"
torchrun --standalone --nnodes 1 --nproc-per-node 4 train.py \
  --vla_path openvla/openvla-7b --data_root_dir "$ROOT/datasets/rlds" \
  --dataset_name robocasa_no_noops --run_root_dir "$ROOT/runs" \
  --batch_size 8 --grad_accumulation_steps 1 --lora_rank 128 \
  --learning_rate 5e-4 --max_steps 100000 --save_freq 10000 \
  --run_id_override oft-baseline

# Example CIG continuation from the supplied merged baseline.
torchrun --standalone --nnodes 1 --nproc-per-node 4 train.py \
  --vla_path ../weights/oft-baseline --data_root_dir "$ROOT/datasets/rlds" \
  --dataset_name robocasa_no_noops --run_root_dir "$ROOT/runs" \
  --resume True --resume_step 100000 --max_steps 105000 --save_freq 1000 \
  --batch_size 4 --learning_rate 5e-5 --lora_rank 128 \
  --use_cig True --cig_pool_path ../data/instruction_pool.json \
  --cig_lambda 0.1 --cig_tau 1.0 --cig_num_negatives 4 \
  --cig_floor 0.5 --cig_hard_negatives True \
  --cig_same_ratio 0.5 --cig_cross_ratio 0.25 \
  --run_id_override oft-cig
```

OFT batch size is per GPU. `--resume True` loads the saved action head and proprio projector, uses the supplied merged model as its initialization and offsets step labels; it initializes a new LoRA adapter and optimizer. Training outputs use the upstream step-based component names, which the loader also accepts. The commands are configurable training recipes; they do not constitute a new measurement of the released checkpoints. Use the supplied four checkpoints for evaluation of this release.
