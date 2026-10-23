# Installation

[Project overview](../README.md) · [Data preparation](../data/README.md) · [Evaluation](../evaluation/README.md)

Run the environment setup below from the release root. In subsequent terminals, set `ROOT` to the same absolute path before following a guide.

Use Linux and separate environments for OFT, π0.5 and the simulator. `ROOT` below is the absolute path of this release. Commands assume Bash. The dependencies pin the model-specific Transformers implementations; mixing the two environments is unsupported.

```bash
export ROOT="$PWD"
python3.11 -m venv "$ROOT/.venv-pi05"
source "$ROOT/.venv-pi05/bin/activate"
GIT_LFS_SKIP_SMUDGE=1 pip install -e "$ROOT/packages/openpi-client" -e "$ROOT/pi05[cuda]"
python "$ROOT/pi05/scripts/install_transformers_patch.py"
deactivate

python3.10 -m venv "$ROOT/.venv-oft"
source "$ROOT/.venv-oft/bin/activate"
pip install -e "$ROOT/packages/openpi-client" -e "$ROOT/oft"
deactivate
```

The π0.5 patch is the openpi adaptation for Transformers 4.53.2 and is installed only in the active environment. The PaliGemma tokenizer and initial JAX base checkpoint download from official public storage on first use. Model execution requires a CUDA-enabled PyTorch installation. JAX baseline training uses `jax[cuda12]`; PyTorch inference/CIG can set `JAX_PLATFORMS=cpu` to reserve GPU memory for PyTorch.

### robocasa-kitchen simulator

Use the robocasa-kitchen v0.1 task implementation, pinned below. Check out external dependencies **outside** this release directory:

```bash
python3.10 -m venv "$ROOT/.venv-sim"
source "$ROOT/.venv-sim/bin/activate"
mkdir -p "$HOME/robocasa-kitchen-deps"
cd "$HOME/robocasa-kitchen-deps"
git clone https://github.com/ARISE-Initiative/robosuite.git
git -C robosuite checkout 1a8701b90c07c6595ace4af9935d7c5ebe1baed3
git clone https://github.com/robocasa/robocasa.git
git -C robocasa checkout 3f5723029b0ecc8dc9793562c6fa129aa74d6e76
pip install -e robosuite -e robocasa
pip install -r "$ROOT/evaluation/requirements.txt" -e "$ROOT/packages/openpi-client"
python -m robocasa.scripts.download_kitchen_assets
```

Use `MUJOCO_GL=egl` for headless rendering. Download simulator assets before evaluation. These pinned versions retain the 24 single-stage task definitions, PandaMobile observations and 12-dimensional action interface used here.

## Next steps

- [Download and prepare the official data](../data/README.md).
- [Train π0.5 and convert JAX weights](../pi05/README.md) or [train OFT](../oft/README.md).
- [Use the released checkpoints](checkpoints.md) and [run evaluation](../evaluation/README.md).
- For Stretch 3 + Quest 3S, use the separate [robot installation and operation guide](../real_robot/stretch3_vr/README.md).
