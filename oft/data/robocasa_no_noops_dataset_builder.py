from __future__ import annotations
import glob
import io
import json
import multiprocessing as mp
import os
from typing import Any, Dict, Iterator, Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image as PILImage

LEROBOT_ROOT = os.environ.get("LEROBOT_ROOT", "datasets/robocasa-kitchen")
IMAGE_SIZE = 128
STATE_DIM = 27
ACTION_DIM = 12


def _extract_png_bytes(cell: Any) -> bytes:
    if isinstance(cell, dict) and "bytes" in cell:
        return cell["bytes"]
    if isinstance(cell, (bytes, bytearray)):
        return bytes(cell)
    arr = np.asarray(cell, dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _load_task_map(root: str) -> Dict[int, str]:
    mp_ = {}
    with open(os.path.join(root, "meta", "tasks.jsonl"), "r") as f:
        for line in f:
            row = json.loads(line)
            mp_[int(row["task_index"])] = row["task"]
    return mp_


def _load_episodes(root: str) -> Dict[int, Dict[str, Any]]:
    mp_ = {}
    with open(os.path.join(root, "meta", "episodes.jsonl"), "r") as f:
        for line in f:
            row = json.loads(line)
            mp_[int(row["episode_index"])] = row
    return mp_


def _process_episode(args):
    path, instruction, ep_idx = args
    df = pd.read_parquet(path)
    N = len(df)
    steps = []
    img_col = df["observation.images.base_0_rgb"].to_list()
    img2_col = df["observation.images.base_1_rgb"].to_list()
    wrist_col = df["observation.images.wrist_0_rgb"].to_list()
    state_col = df["observation.state"].to_list()
    action_col = df["action"].to_list()
    for i in range(N):
        is_last = i == N - 1
        steps.append(
            {
                "observation": {
                    "image": _extract_png_bytes(img_col[i]),
                    "secondary_image": _extract_png_bytes(img2_col[i]),
                    "wrist_image": _extract_png_bytes(wrist_col[i]),
                    "state": np.asarray(state_col[i], dtype=np.float32).reshape(-1),
                },
                "action": np.asarray(action_col[i], dtype=np.float32).reshape(-1),
                "discount": np.float32(1.0),
                "reward": np.float32(1.0 if is_last else 0.0),
                "is_first": i == 0,
                "is_last": is_last,
                "is_terminal": is_last,
                "language_instruction": instruction,
            }
        )
    return (
        f"ep_{ep_idx:06d}",
        {
            "steps": steps,
            "episode_metadata": {"file_path": path, "episode_index": ep_idx},
        },
    )


class RobocasaNoNoops(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "robocasa-kitchen demonstrations."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description="robocasa-kitchen single-stage 24-task Panda-mobile demonstrations.",
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                        dtype=tf.uint8,
                                        encoding_format="png",
                                    ),
                                    "secondary_image": tfds.features.Image(
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                        dtype=tf.uint8,
                                        encoding_format="png",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                                        dtype=tf.uint8,
                                        encoding_format="png",
                                    ),
                                    "state": tfds.features.Tensor(
                                        shape=(STATE_DIM,), dtype=tf.float32
                                    ),
                                }
                            ),
                            "action": tfds.features.Tensor(
                                shape=(ACTION_DIM,), dtype=tf.float32
                            ),
                            "discount": tfds.features.Scalar(dtype=tf.float32),
                            "reward": tfds.features.Scalar(dtype=tf.float32),
                            "is_first": tfds.features.Scalar(dtype=tf.bool),
                            "is_last": tfds.features.Scalar(dtype=tf.bool),
                            "is_terminal": tfds.features.Scalar(dtype=tf.bool),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "episode_index": tfds.features.Scalar(dtype=tf.int64),
                        }
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://robocasa.ai",
        )

    def _split_generators(self, dl_manager):
        paths = sorted(
            glob.glob(
                os.path.join(LEROBOT_ROOT, "data", "chunk-*", "episode_*.parquet")
            )
        )
        task_map = _load_task_map(LEROBOT_ROOT)
        episodes = _load_episodes(LEROBOT_ROOT)
        tasks = []
        for p in paths:
            ep_idx = int(os.path.basename(p).split("_")[1].split(".")[0])
            meta = episodes.get(ep_idx, {})
            instruction = None
            if meta.get("tasks"):
                instruction = meta["tasks"][0]
            if not instruction:
                instruction = task_map.get(0, "")
            tasks.append((p, instruction, ep_idx))
        return {"train": self._generate_examples(tasks)}

    def _generate_examples(self, tasks) -> Iterator[Tuple[str, Dict[str, Any]]]:
        n_workers = int(os.environ.get("NUM_WORKERS", "16"))
        if n_workers <= 1:
            for t in tasks:
                yield _process_episode(t)
            return
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for key, ex in pool.imap_unordered(_process_episode, tasks, chunksize=1):
                yield (key, ex)
