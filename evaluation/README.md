# robocasa-kitchen inference and evaluation

[Project overview](../README.md) · [Installation](../docs/installation.md#robocasa-kitchen-simulator) · [Checkpoint usage](../docs/checkpoints.md) · [Task definitions](tasks.json)

Install the matching model environment and the pinned simulator environment, download simulator assets, and place the selected checkpoint under `weights/`. Evaluation of the supplied weights does not require preparing the training dataset.

Set `ROOT` to the absolute path of the release in each terminal. Commands use Bash.

```bash
export ROOT=/absolute/path/to/robocasa-kitchen
```

## Start a policy server

Start one model server in its own terminal. Select either checkpoint of the same family by changing `--checkpoint`:

```bash
# π0.5 baseline or CIG
source "$ROOT/.venv-pi05/bin/activate"
cd "$ROOT/pi05"
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=0 python scripts/serve.py \
  --checkpoint ../weights/pi05-baseline --port 8000

# Alternatively, OFT baseline or CIG
source "$ROOT/.venv-oft/bin/activate"
cd "$ROOT/oft"
CUDA_VISIBLE_DEVICES=0 python serve.py --checkpoint ../weights/oft-cig --port 8000
```

## Evaluate all 24 tasks

In a separate terminal with the same `ROOT`, activate the simulator environment:

```bash
source "$ROOT/.venv-sim/bin/activate"
cd "$ROOT"
MUJOCO_GL=egl python evaluation/evaluate.py \
  --host 127.0.0.1 --port 8000 --task all --trials 50 --seed 42 \
  --replan-steps 5 --output results/pi05-baseline.json
```

Use `--task PnPCounterToCab --trials 1` for a short connectivity check and a new output filename per run. The fixed evaluation protocol uses held-out object split B; layout/style pairs `(1,1), (2,2), (4,4), (6,9), (7,10)`; no randomized cameras or generative textures; and task-specific horizons listed in `evaluation/tasks.json`. Each policy returns 10 actions, and evaluation executes the first 5 before replanning.

Results contain per-episode success, per-task rates, aggregate success and macro-average success. Rates are fractions in `[0,1]`; multiply by 100 for percent. A complete default run contains 1,200 episodes. Environment or policy exceptions stop the run and leave `complete: false` with an error, rather than silently counting an infrastructure failure as a task failure. Existing result files are never overwritten.
