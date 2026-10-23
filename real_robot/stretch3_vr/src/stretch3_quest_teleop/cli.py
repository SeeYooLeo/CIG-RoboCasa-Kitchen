from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .backends.mock import MockRobotBackend
from .config import TeleopConfig, default_config_path
from .doctor import run_doctor
from .server import TeleopServer, make_ssl_context
from .smoke import run_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stretch3-vr",
        description="Quest 3S WebXR teleoperation bridge for Stretch 3",
    )
    parser.add_argument(
        "--config",
        default=str(default_config_path()),
        help="path to JSON configuration",
    )
    parser.add_argument(
        "--config-overlay",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "JSON file deep-merged over --config; repeatable, applied in order. "
            "Use config/overlay_base_locked.json to bring the robot up with every "
            "base speed at zero, and config/overlay_slow.json for the first "
            "moving runs, instead of editing the shipped configuration."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="serve Quest UI and teleoperation")
    serve.add_argument("--backend", choices=("mock", "ros2"), default="mock")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--cert", help="TLS certificate PEM")
    serve.add_argument("--key", help="TLS private key PEM")
    serve.add_argument("--record-dir", type=Path,
                       help="enable episode recording beneath this directory")
    serve.add_argument(
        "--insecure-http",
        action="store_true",
        help="serve HTTP for desktop debugging; WebXR on Quest requires HTTPS",
    )
    serve.add_argument(
        "--camera-dir",
        help=(
            "mock backend only: directory of JPEGs to stream instead of a robot "
            "camera. Use navigation/, head/ and gripper/ subdirectories to give "
            "each mode its own picture, or a flat directory for all of them."
        ),
    )

    subparsers.add_parser("smoke", help="run hardware-free control smoke test")
    doctor = subparsers.add_parser("doctor", help="check local runtime dependencies")
    doctor.add_argument("--ros2", action="store_true", help="also check ROS 2/IK imports")
    doctor.add_argument(
        "--camera-probe-s",
        type=float,
        default=3.0,
        help=(
            "with --ros2, subscribe to the configured topics for this long and "
            "report each camera's actual width, height, encoding and rate. "
            "0 disables the probe. Read-only: it never commands the robot."
        ),
    )
    doctor.add_argument("--cert")
    doctor.add_argument("--key")
    for name in ("replay", "legacy-replay"):
        command = subparsers.add_parser(name, help=f"inspect or execute an episode with {name}")
        command.add_argument("arguments", nargs=argparse.REMAINDER)
    return parser


def _backend(name: str, config: TeleopConfig, camera_dir: str | None = None):
    if name == "mock":
        return MockRobotBackend(camera_dir)
    if camera_dir:
        raise SystemExit("--camera-dir only applies to the mock backend")
    from .backends.ros2 import Ros2StretchBackend

    return Ros2StretchBackend(config)


async def _serve(args: argparse.Namespace, config: TeleopConfig) -> None:
    ssl_context = None
    if not args.insecure_http:
        if not args.cert or not args.key:
            raise SystemExit(
                "Quest WebXR requires HTTPS. Pass --cert and --key, or use "
                "--insecure-http only for desktop debugging."
            )
        ssl_context = make_ssl_context(args.cert, args.key)
    server = TeleopServer(
        config,
        _backend(args.backend, config, args.camera_dir),
        host=args.host,
        port=args.port,
        ssl_context=ssl_context,
        record_dir=args.record_dir,
    )
    try:
        await server.serve_forever()
    finally:
        await server.close()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in ("replay", "legacy-replay"):
        if args.command == "replay":
            from .replay import main as replay_main
        else:
            from .legacy_replay import main as replay_main
        return replay_main(args.arguments)
    config_path = Path(args.config).resolve()
    overlays = [Path(item).resolve() for item in args.config_overlay]
    for overlay in overlays:
        if not overlay.is_file():
            raise SystemExit(f"configuration overlay not found: {overlay}")
    config = TeleopConfig.load(config_path, overlays)
    if args.command == "smoke":
        print(json.dumps(run_smoke(config), indent=2))
        return 0
    if args.command == "doctor":
        result = run_doctor(
            config,
            check_ros2=args.ros2,
            cert=args.cert,
            key=args.key,
            camera_probe_s=args.camera_probe_s,
        )
        print(json.dumps(result, indent=2))
        return 0 if result["ok"] else 1
    asyncio.run(_serve(args, config))
    return 0
