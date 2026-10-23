# robocasa-kitchen data

[Project overview](../README.md) · [Installation](../docs/installation.md) · [π0.5 training](../pi05/README.md) · [OFT training](../oft/README.md)

Our training demonstrations come from the **official robocasa-kitchen v0.1 downloads**. Conversion changes storage format and assembles observation fields without generating new demonstrations. This recipe uses the 24-task v0.1 dataset.

Complete the model and simulator installation first.

Set `ROOT` to the absolute path of the release in each terminal. Commands use Bash.

```bash
export ROOT=/absolute/path/to/robocasa-kitchen
```

## Download and prepare

Download the image demonstrations through the [official robocasa-kitchen downloader](https://github.com/robocasa/robocasa/blob/3f5723029b0ecc8dc9793562c6fa129aa74d6e76/robocasa/scripts/download_datasets.py). The `human_im` files have names such as `demo_gentex_im128_randcams.hdf5`. Select only the tasks listed in [`evaluation/tasks.json`](../evaluation/tasks.json); navigation and multi-stage tasks are excluded.

In the simulator environment, run from the release root:

```bash
cd "$ROOT"
python - <<'PY'
import json
from robocasa.scripts.download_datasets import download_datasets
with open("evaluation/tasks.json") as stream:
    tasks = list(json.load(stream))
download_datasets(tasks=tasks, ds_types=["human_im"], overwrite=False, dryrun=False)
PY
```

Convert the downloaded `datasets/v0.1/single_stage` directory to the pinned LeRobot format. Set `HF_LEROBOT_HOME` consistently for conversion and training. The dataset id `robocasa-kitchen` is a local identifier, not a hosted repository that must be downloaded.

```bash
source "$ROOT/.venv-pi05/bin/activate"
export HF_LEROBOT_HOME="$ROOT/datasets/lerobot"
cd "$ROOT/pi05"
python scripts/convert_dataset.py \
  --data-dir "$HOME/robocasa-kitchen-deps/robocasa/datasets/v0.1/single_stage" \
  --repo-id robocasa-kitchen --num-workers 8
python scripts/compute_norm_stats.py --config-name pi05-baseline --repo-id robocasa-kitchen
```

The converter defaults to image storage, which is also required by the OFT RLDS builder. It refuses to replace an existing output dataset. Supply only the intended official image demonstration files under `--data-dir`, rather than mixing raw, image and generated variants. For the supplied checkpoints, always use their bundled statistics; newly computed statistics belong to a new training run.

For OFT, convert the same LeRobot dataset to RLDS:

```bash
source "$ROOT/.venv-oft/bin/activate"
cd "$ROOT/oft"
LEROBOT_ROOT="$ROOT/datasets/lerobot/robocasa-kitchen" \
DATA_DIR="$ROOT/datasets/rlds" NUM_WORKERS=8 python data/build_robocasa_rlds.py
```

### Action and state representation

All raw vectors are `float32`, and the data/control frequency is 20 Hz. Images are RGB from left agent view, right agent view and wrist view, stored at 128×128 and resized to 224×224 for the policy. Evaluation flips simulator images vertically to match stored demonstration images.

| Raw state slice | Size | Meaning |
|---|---:|---|
| `state[0:7]` | 7 | Arm joint positions |
| `state[7:14]` | 7 | Arm joint velocities |
| `state[14:16]` | 2 | Gripper joint positions |
| `state[16:18]` | 2 | Gripper joint velocities |
| `state[18:21]` | 3 | End-effector position relative to the robot base |
| `state[21:25]` | 4 | End-effector quaternion relative to the base, `(x,y,z,w)` |
| `state[25:27]` | 2 | `sin(base_yaw), cos(base_yaw)` |

| Raw action slice | Size | Meaning |
|---|---:|---|
| `action[0:3]` | 3 | End-effector translation control, base frame |
| `action[3:6]` | 3 | End-effector rotation control, axis-angle |
| `action[6]` | 1 | Gripper command |
| `action[7:10]` | 3 | Mobile-base velocity commands `(vx,vy,vyaw)` |
| `action[10]` | 1 | Torso command |
| `action[11]` | 1 | Arm/base mode flag (`>0` selects base mode) |

Both model families use `state[:25]` and `action[:7]` for these stationary manipulation tasks. π0.5 pads model tensors to 32 dimensions internally; that padding is not an extra robot actuator. Both policies predict 10-step chunks. The servers reconstruct the simulator's 12-dimensional actions, fixing base and torso commands to zero. π0.5 uses mode `-1`, and OFT preserves its original mode `0`; both select arm mode.

An illustrative example (not a recorded trajectory or a physical robot command):

```python
import numpy as np
state = np.concatenate([
    np.zeros(7),                   # joint positions
    np.zeros(7),                   # joint velocities
    [0.02, 0.02], [0.0, 0.0],     # gripper positions and velocities
    [0.40, 0.0, 0.25],             # base-relative end-effector position
    [0.0, 0.0, 0.0, 1.0],         # identity quaternion, xyzw
    [0.0, 1.0],                    # yaw = 0
]).astype(np.float32)              # (27,)
action = np.array([0.01, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1], dtype=np.float32)
assert action.shape == (12,)
model_state, model_action = state[:25], action[:7]
```

Actions retain the simulator controller's scaling; do not reinterpret a numeric command directly as a displacement in meters without the controller's scaling configuration. Quantile normalization is performed by each model's data transforms.
