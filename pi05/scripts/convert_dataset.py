from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import cv2
import h5py
import numpy as np
import tqdm
import tyro

DEFAULT_HF_HOME = Path.home() / ".cache" / "huggingface"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

EXCLUDED_TASKS = {"NavigateKitchen"}
CAMERAS = [
    ("robot0_agentview_left_image", "base_0_rgb"),
    ("robot0_agentview_right_image", "base_1_rgb"),
    ("robot0_eye_in_hand_image", "wrist_0_rgb"),
]
STATE_DIM = 27
ACTION_DIM = 12
TARGET_IMAGE_SIZE = 128
FPS = 20
NUM_WORKERS = 16
DEFAULT_DATA_DIR = os.environ.get(
    "ROBOCASA_SINGLE_STAGE_DATA_DIR",
    str(Path.home() / "robocasa" / "datasets" / "v0.1" / "single_stage"),
)


def resize_images_batch(
    images: np.ndarray, target_size: int, num_workers: int = NUM_WORKERS
) -> np.ndarray:
    def resize_single(img):
        return cv2.resize(
            img, (target_size, target_size), interpolation=cv2.INTER_LINEAR
        )

    with ThreadPoolExecutor(max_workers=max(1, num_workers)) as executor:
        resized = list(executor.map(resize_single, images))
    return np.array(resized)


def extract_states_batch(obs: dict, num_steps: int) -> np.ndarray:
    joint_pos = obs["robot0_joint_pos"][:num_steps]
    joint_vel = obs["robot0_joint_vel"][:num_steps]
    gripper_qpos = obs["robot0_gripper_qpos"][:num_steps]
    gripper_qvel = obs["robot0_gripper_qvel"][:num_steps]
    base_to_eef_pos = obs["robot0_base_to_eef_pos"][:num_steps]
    base_to_eef_quat = obs["robot0_base_to_eef_quat"][:num_steps]
    base_quat = obs["robot0_base_quat"][:num_steps]
    qz = base_quat[:, 2]
    qw = base_quat[:, 3]
    yaw = 2 * np.arctan2(qz, qw)
    base_yaw_sin = np.sin(yaw)[:, np.newaxis]
    base_yaw_cos = np.cos(yaw)[:, np.newaxis]
    states = np.concatenate(
        [
            joint_pos,
            joint_vel,
            gripper_qpos,
            gripper_qvel,
            base_to_eef_pos,
            base_to_eef_quat,
            base_yaw_sin,
            base_yaw_cos,
        ],
        axis=1,
    ).astype(np.float32)
    return states


def extract_actions_batch(actions: np.ndarray) -> np.ndarray:
    return actions[:, :ACTION_DIM].astype(np.float32)


def create_empty_dataset(
    repo_id: str, num_workers: int = 8, use_videos: bool = True
) -> LeRobotDataset:
    image_dtype = "video" if use_videos else "image"
    features = {
        "observation.images.base_0_rgb": {
            "dtype": image_dtype,
            "shape": (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.base_1_rgb": {
            "dtype": image_dtype,
            "shape": (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.images.wrist_0_rgb": {
            "dtype": image_dtype,
            "shape": (TARGET_IMAGE_SIZE, TARGET_IMAGE_SIZE, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": ["state"],
        },
        "action": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["action"]},
    }
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        raise FileExistsError(output_path)
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=FPS,
        robot_type="panda_mobile",
        features=features,
        use_videos=use_videos,
        image_writer_processes=num_workers,
        image_writer_threads=num_workers * 2,
    )


def find_all_hdf5_files(data_dir: Path) -> list[Path]:
    all_files = sorted(data_dir.rglob("*.hdf5"))
    filtered_files = []
    for f in all_files:
        skip = False
        for excluded in EXCLUDED_TASKS:
            if excluded in str(f):
                skip = True
                break
        if not skip:
            filtered_files.append(f)
    return filtered_files


def process_episode_fast(ep: h5py.Group, num_workers: int) -> dict:
    obs = ep["obs"]
    actions_raw = ep["actions"][:]
    num_steps = actions_raw.shape[0]
    ep_meta = json.loads(ep.attrs.get("ep_meta", "{}"))
    task = ep_meta.get("lang", "perform the task")
    images = {}
    for robocasa_name, lerobot_name in CAMERAS:
        imgs = obs[robocasa_name][:num_steps]
        if imgs.shape[1] != TARGET_IMAGE_SIZE or imgs.shape[2] != TARGET_IMAGE_SIZE:
            imgs = resize_images_batch(imgs, TARGET_IMAGE_SIZE, num_workers)
        images[lerobot_name] = imgs
    states = extract_states_batch(obs, num_steps)
    actions = extract_actions_batch(actions_raw)
    return {
        "images": images,
        "states": states,
        "actions": actions,
        "task": task,
        "num_steps": num_steps,
    }


def convert_robocasa_to_lerobot(
    data_dir: Path, repo_id: str, num_workers: int = 8, use_videos: bool = True
) -> None:
    hdf5_files = find_all_hdf5_files(data_dir)
    print(f"Found {len(hdf5_files)} HDF5 files (after excluding: {EXCLUDED_TASKS})")
    print(f"Using {num_workers} workers for parallel processing")
    print(f"Storage mode: {('video' if use_videos else 'images')}")
    if not hdf5_files:
        raise FileNotFoundError(f"No HDF5 demonstrations in {data_dir}")
    dataset = create_empty_dataset(repo_id, num_workers, use_videos)
    total_episodes = 0
    total_frames = 0
    for hdf5_path in tqdm.tqdm(hdf5_files, desc="Processing files"):
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"Skipping {hdf5_path}: no 'data' group found")
                continue
            demo_keys = sorted(
                [k for k in f["data"].keys()],
                key=lambda x: int(x.split("_")[-1]) if "_" in x else 0,
            )
            for demo_key in tqdm.tqdm(
                demo_keys, desc=f"  {hdf5_path.stem}", leave=False
            ):
                ep = f["data"][demo_key]
                ep_data = process_episode_fast(ep, num_workers)
                num_steps = ep_data["num_steps"]
                task = ep_data["task"]
                for i in range(num_steps):
                    frame = {
                        "observation.images.base_0_rgb": ep_data["images"][
                            "base_0_rgb"
                        ][i],
                        "observation.images.base_1_rgb": ep_data["images"][
                            "base_1_rgb"
                        ][i],
                        "observation.images.wrist_0_rgb": ep_data["images"][
                            "wrist_0_rgb"
                        ][i],
                        "observation.state": ep_data["states"][i],
                        "action": ep_data["actions"][i],
                        "task": task,
                    }
                    dataset.add_frame(frame)
                dataset.save_episode()
                total_episodes += 1
                total_frames += num_steps
    print("\nConversion complete!")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Total frames: {total_frames}")
    print(f"  Dataset saved to: {HF_LEROBOT_HOME / repo_id}")


def main(
    data_dir: str = DEFAULT_DATA_DIR,
    repo_id: str = "robocasa-kitchen",
    num_workers: int = 16,
    use_videos: bool = False,
):
    convert_robocasa_to_lerobot(
        Path(data_dir), repo_id, num_workers=num_workers, use_videos=use_videos
    )


if __name__ == "__main__":
    tyro.cli(main)
