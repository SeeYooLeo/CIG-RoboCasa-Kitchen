# Checkpoints and weight usage

[Project overview](../README.md) · [Installation](installation.md) · [Evaluation](../evaluation/README.md)

| Name | Model | Files required for inference |
|---|---|---|
| `oft-baseline` | OpenVLA-OFT, LoRA rank 128, merged | Model shards, index, processor/tokenizer/config, statistics, `action_head.pt`, `proprio_projector.pt` |
| `oft-cig` | OpenVLA-OFT + CIG, merged | Same layout as `oft-baseline` |
| `pi05-baseline` | π0.5 baseline | `model.safetensors`, `assets/robocasa-kitchen/norm_stats.json` |
| `pi05-cig` | π0.5 + CIG | Same layout as `pi05-baseline` |

## Download from Hugging Face

The four checkpoints are hosted in [SeeYooLeo/CIG-RoboCasa-Kitchen](https://huggingface.co/SeeYooLeo/CIG-RoboCasa-Kitchen). Run the following from the release root:

```bash
pip install -U huggingface_hub
hf download SeeYooLeo/CIG-RoboCasa-Kitchen \
  --revision 4bdc391cb1dd6df8faa978e55aebf31e40c39da4 \
  --include "oft-baseline/*" --include "oft-cig/*" \
  --include "pi05-baseline/*" --include "pi05-cig/*" \
  --local-dir weights
python verify_weights.py
```

The revision pins the verified release. To download only one checkpoint, keep only its pattern in `--include`; `verify_weights.py` checks all four checkpoints and should be run after downloading the complete set. If authentication is required, run `hf auth login` with an account that has access.

The download preserves the following layout:

```text
robocasa-kitchen/
├── oft/                   # OFT training, inference and RLDS preparation
├── pi05/                  # π0.5 JAX training, PyTorch CIG, conversion and inference
├── evaluation/            # Shared simulator client and exact 24-task list
├── real_robot/stretch3_vr/ # Stretch 3 VR teleoperation, recording and replay
├── packages/openpi-client/
├── data/instruction_pool.json
├── weights/{oft-baseline,oft-cig,pi05-baseline,pi05-cig}/
├── weights_manifest.json
└── verify_weights.py
```

Run `python verify_weights.py` from the release root to check every weight file's size and SHA-256. The tensors are preserved from the selected checkpoints. OFT LoRA updates are already merged: do not apply a second adapter. Optimizer states and original run metadata are omitted; the supplied weights support inference and initialization of a new training run, not exact optimizer-state resumption. Weights are distributed separately from Git and are ignored by `.gitignore`.

## Select a checkpoint

Use `weights/oft-baseline` or `weights/oft-cig` with `oft/serve.py`; use `weights/pi05-baseline` or `weights/pi05-cig` with `pi05/scripts/serve.py`. The exact server commands and simulator protocol are in the [evaluation guide](../evaluation/README.md).

Use the statistics bundled with the selected checkpoint. Do not substitute statistics recomputed from another dataset. The π0.5 baseline is already in PyTorch format; the [JAX conversion instructions](../pi05/README.md#convert-jax-parameters-to-pytorch) apply to newly trained JAX checkpoints.

The checkpoints are robocasa-kitchen simulator policies. The [Stretch 3 VR module](../real_robot/stretch3_vr/README.md) provides teleoperation, recording and replay, and does not deploy these policies on hardware. Upstream model terms apply; see [third-party notices](../THIRD_PARTY_NOTICES.md).
