# Paper and implementation notes

[Project overview](../README.md) · [π0.5 training](../pi05/README.md) · [OFT training](../oft/README.md)

## Manuscript identity and reported results

The camera-ready manuscript is titled **Counterfactual Instruction Grounding for Vision-Language-Action Models**, by **Shiyu Liu, Yu Zhou, Meng Liu, Xuanming Guo and Liqiang Nie**. Yu Zhou and Liqiang Nie are the corresponding authors. The homepage citation follows this author order. A public paper URL and proceedings identifier have not yet been supplied.

The homepage's robocasa-kitchen category results come from the main simulation-results table; ablations and the grounding diagnostic come from the main experiments section. Diagnostic sampling is described in Appendix E.2. Stretch 3 counts and collection/deployment settings come from Appendix F, “Stretch 3 Real-World Protocol.” They are paper-reported results, not new evaluations performed while preparing this release.

The homepage retains the robocasa-kitchen and supplementary Stretch 3 scope of this release. Its four checkpoints are simulator policies.

## Scoring and optimization

The method covers continuous-regression (CR) and flow-matching (FM) policies. Both score the same demonstrated action chunk under the commanded instruction and alternative instructions, then minimize the commanded instruction's negative log probability under the contrastive softmax.

OpenVLA-OFT is the CR backbone: it predicts continuous action chunks in parallel and uses negative mean L1 action error as the score. π0.5 is the FM backbone: it uses negative flow-matching residual energy, sharing one noise and interpolation-time draw across candidate instructions during training and averaging squared residuals over the 10 × 32 model coordinates (including padding). These scores define energy-based instruction comparisons. The paper gives an auxiliary Laplace observation-model interpretation for CR; it does not claim a calibrated rollout density.

Both continuation recipes use VLM-only CIG gradient routing and a saturation floor of 0.5. CIG contributes a gradient when its loss is at least this floor; imitation learning remains active at every update. The gate uses the mean CIG loss on each GPU microbatch; accumulated IL and routed CIG gradients are averaged across GPUs before one optimizer step. In OFT, the action head and proprioceptive projector receive only IL gradients.

## Continuation training configurations

The robocasa-kitchen continuation settings in Appendix C and the release recipes are:

| Setting | OpenVLA-OFT + CIG | π0.5 + CIG |
|---|---|---|
| GPUs | 4 × A100 | 4 × A100 |
| Effective batch size | 16 (4 per GPU, accumulation 1) | 128 (global, accumulation 1) |
| Learning rate | 5e-5 | 1,000-step warmup to 5e-5; peak and end rate both 5e-5 |
| Continuation updates | 5,000 (update labels 100,001–105,000) | 5,000 |
| CIG weight λ | 0.1 | 0.1 |
| Temperature κ | 1.0 | 1.0 |
| Saturation floor δ | 0.5 | 0.5 |
| Negative instructions K | 4 | 4 |
| Sampling | 50% same-category / 25% cross-category / 25% random | Same proportions |

With four negatives and an available category, the sampler draws two same-category negatives, one cross-category negative and one from the remaining pool. For a non-pick-and-place positive, the cross-category pool excludes the dominant pick-and-place category. Draws exclude the positive and already selected instructions; the supplied pool is large enough to avoid replacement. These are sampling-source proportions; the random draw can belong to any category. When category metadata or enough category candidates are unavailable, the sampler fills from the remaining pool.

The paper states the `log(K+1) - L_CIG` information bound under the conditional candidate-prior assumption in Appendix A.2. Category-aware sampling alone does not establish that assumption. The grounding diagnostic separately uses eight noise samples and temperature 0.01; training uses temperature 1.0.

The settings are defined in [π0.5 configuration](../pi05/src/openpi/training/config.py), the [π0.5 trainer](../pi05/scripts/train.py), the [OFT trainer](../oft/train.py), and the training guides. They apply to CIG continuation; initial imitation training uses its separate baseline recipe. Updating a recipe does not retrain or relabel an existing checkpoint. The four supplied checkpoint tensors and the paper-reported evaluation numbers are unchanged.

## VR data collection and real-robot policy settings

The manuscript's Stretch 3 demonstrations are collected through the Quest 3S / WebXR VR interface: 50 episodes per task, with an average duration of approximately two minutes. Timestamped head/wrist RGB observations, robot states and actions are aligned to a 15 Hz policy-data timeline. RGB frames are center-cropped and resized to 224 × 224; the processed policy representation uses a 14D state and an 18D action, with H.264 video encoding at CRF 15. These are the paper's VR demonstration preprocessing settings.

The paper reports 20 held-out evaluation rollouts per task and method. Policy deployment uses an RTX 5090 at 15 Hz, a 12-step horizon, replanning every 5 steps and skipping the first 3 predictions for latency compensation. The resulting horizon and replanning period are 0.8 s and approximately 0.33 s.

The released VR module provides the collection and command-replay interface for this workflow. It retains raw timestamped ROS events and JPEG camera samples at their available resolution and configured rotation. Its browser input rate (approximately 30 Hz), camera encoding limit (15 Hz), browser preview limit (12 Hz) and joint-goal submission limit (4 Hz) serve different stages of the live interface. The paper's aligned 15 Hz dataset is produced during downstream preprocessing; the raw recorder does not automatically create the 14D/18D policy representation or H.264 training videos.

The module includes a legacy gamepad-format replay adapter for compatibility with older files; that adapter does not describe the paper's VR data collection. The real-robot policy checkpoint, training-data converter and policy deployment loop are outside this release. See the [robot guide](../real_robot/stretch3_vr/README.md#paper-experiment-settings) for the parameter mapping and [recording/replay workflow](../real_robot/stretch3_vr/README.md#recording-and-replay).
