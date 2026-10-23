import argparse
import collections
import json
import pathlib
import time
import numpy as np
from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy
from environment import CAMERA_NAMES, build_state, create_env, get_task_prompt

TASKS = json.loads(pathlib.Path(__file__).with_name("tasks.json").read_text())


def policy_observation(obs, prompt):
    images = [
        image_tools.convert_to_uint8(
            image_tools.resize_with_pad(
                np.ascontiguousarray(obs[name + "_image"][::-1]), 224, 224
            )
        )
        for name in CAMERA_NAMES
    ]
    return {
        "observation/image": images[0],
        "observation/image_right": images[1],
        "observation/wrist_image": images[2],
        "observation/state": build_state(obs),
        "prompt": prompt,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the 24 robocasa-kitchen single-stage tasks"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", choices=["all", *TASKS], default="all")
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.trials < 1 or not 1 <= args.replan_steps <= 10:
        parser.error("trials must be positive; replan-steps must be in [1, 10]")
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    client = WebsocketClientPolicy(args.host, args.port)
    results = {
        "protocol": {
            "seed": args.seed,
            "trials_per_task": args.trials,
            "replan_steps": args.replan_steps,
            "object_split": "B",
            "layout_style_pairs": [[1, 1], [2, 2], [4, 4], [6, 9], [7, 10]],
        },
        "policy": client.get_server_metadata(),
        "tasks": {},
        "complete": False,
    }
    tasks = TASKS if args.task == "all" else {args.task: TASKS[args.task]}
    try:
        for task, horizon in tasks.items():
            env = create_env(task, seed=args.seed)
            episodes = []
            results["tasks"][task] = {"horizon": horizon, "episodes": episodes}
            try:
                for trial in range(args.trials):
                    obs = env.reset()
                    prompt = get_task_prompt(task, env)
                    plan = collections.deque()
                    success = False
                    start = time.monotonic()
                    for step in range(horizon):
                        if not plan:
                            actions = np.asarray(
                                client.infer(policy_observation(obs, prompt))["actions"]
                            )
                            if (
                                actions.ndim != 2
                                or actions.shape[1] != 12
                                or len(actions) < args.replan_steps
                                or (not np.isfinite(actions).all())
                            ):
                                raise ValueError(
                                    f"Invalid policy actions: {actions.shape}"
                                )
                            plan.extend(actions[: args.replan_steps])
                        obs, _, _, _ = env.step(plan.popleft().tolist())
                        if env._check_success():
                            success = True
                            break
                    episodes.append(
                        {
                            "trial": trial,
                            "success": success,
                            "steps": step + 1,
                            "seconds": round(time.monotonic() - start, 3),
                        }
                    )
                    print(
                        f"{task} {trial + 1}/{args.trials}: success={success}",
                        flush=True,
                    )
                    args.output.write_text(json.dumps(results, indent=2) + "\n")
                successes = sum((e["success"] for e in episodes))
                results["tasks"][task].update(
                    successes=successes,
                    total=len(episodes),
                    success_rate=successes / len(episodes),
                )
            finally:
                env.close()
        total = sum((t["total"] for t in results["tasks"].values()))
        successes = sum((t["successes"] for t in results["tasks"].values()))
        results.update(
            complete=True,
            total=total,
            successes=successes,
            success_rate=successes / total,
            macro_success_rate=float(
                np.mean([t["success_rate"] for t in results["tasks"].values()])
            ),
        )
    except Exception as error:
        results["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
