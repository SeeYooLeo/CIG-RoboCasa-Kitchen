import argparse
import pathlib
import sys
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from evaluation.server import WebsocketPolicyServer
from policy_utils import (
    get_action_head,
    get_processor,
    get_proprio_projector,
    get_vla,
    get_vla_action,
)


class Policy:
    def __init__(self, checkpoint):
        self.cfg = SimpleNamespace(
            pretrained_checkpoint=str(checkpoint),
            num_images_in_input=3,
            use_proprio=True,
            use_l1_regression=True,
            use_diffusion=False,
            load_in_8bit=False,
            load_in_4bit=False,
            center_crop=True,
            unnorm_key="robocasa_no_noops",
        )
        self.model = get_vla(self.cfg)
        self.processor = get_processor(self.cfg)
        self.head = get_action_head(self.cfg, self.model.llm_dim)
        self.proprio = get_proprio_projector(self.cfg, self.model.llm_dim, 25)

    def infer(self, obs):
        state = np.asarray(obs["observation/state"], dtype=np.float32)
        if state.shape != (27,):
            raise ValueError(f"Expected raw state (27,), got {state.shape}")
        inputs = {
            "full_image": obs["observation/image"],
            "wrist_image_right": obs["observation/image_right"],
            "wrist_image": obs["observation/wrist_image"],
            "state": state[:25].copy(),
        }
        actions = np.asarray(
            get_vla_action(
                self.cfg,
                self.model,
                self.processor,
                inputs,
                obs["prompt"],
                action_head=self.head,
                proprio_projector=self.proprio,
            ),
            dtype=np.float32,
        )
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid predicted actions: {actions.shape}")
        return {"actions": np.pad(actions, ((0, 0), (0, 5)))}


def main():
    parser = argparse.ArgumentParser(description="Serve robocasa-kitchen OFT weights")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    WebsocketPolicyServer(
        Policy(args.checkpoint),
        args.host,
        args.port,
        metadata={
            "model": args.checkpoint.name,
            "action_horizon": 10,
            "action_dim": 12,
        },
    ).serve_forever()


if __name__ == "__main__":
    main()
