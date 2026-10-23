# Third-party code and model terms

- `oft/prismatic` and the OFT training/inference implementation derive from [OpenVLA-OFT](https://github.com/moojink/openvla-oft) and [Prismatic VLMs](https://github.com/TRI-ML/prismatic-vlms). The MIT license is retained in `oft/LICENSE`.
- `pi05/src/openpi`, the baseline JAX trainer, weight converter, and `packages/openpi-client` derive from [Physical Intelligence openpi](https://github.com/Physical-Intelligence/openpi). The Apache 2.0 license is retained in `pi05/LICENSE`. The baseline JAX training loop is based on upstream commit `c2b6977d84a460f3f12ba3054ff04e169d2264bd`.
- The Gemma/SigLIP code includes work by Google, Big Vision and Hugging Face. Copyright and license notices are retained in the affected files. Gemma model terms are included in `pi05/LICENSE_GEMMA.txt`.
- This release adapts the upstream implementations to robocasa-kitchen, adds CIG training, and removes unrelated datasets, experimental utilities and configurations. Public entry points and checkpoint names are simplified.
- The four fine-tuned checkpoints remain subject to the terms of their upstream base models. The code licenses do not replace those model terms. Official robocasa-kitchen data and simulator assets are obtained separately under their respective upstream terms.
- The supplementary Stretch 3 VR module retains its Apache-2.0 notice in `real_robot/stretch3_vr/LICENSE`. It interfaces with the separately installed ROS 2, Stretch ROS 2 and Stretch AI packages; those dependencies and robot-specific assets are not vendored here.
