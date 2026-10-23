import argparse
import dataclasses
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from evaluation.server import WebsocketPolicyServer
from openpi.policies.policy_config import create_trained_policy
from openpi.training.config import AssetsConfig, get_config


def make_policy(checkpoint, device="cuda"):
    checkpoint = pathlib.Path(checkpoint).resolve()
    config = get_config("pi05-baseline")
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(
            config.data,
            assets=AssetsConfig(
                assets_dir=str(checkpoint / "assets"), asset_id="robocasa-kitchen"
            ),
        ),
    )
    return create_trained_policy(config, checkpoint, pytorch_device=device)


def main():
    parser = argparse.ArgumentParser(description="Serve robocasa-kitchen pi0.5 weights")
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    WebsocketPolicyServer(
        make_policy(args.checkpoint, args.device),
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
