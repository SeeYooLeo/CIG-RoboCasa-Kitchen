import numpy as np
import robocasa
import robosuite

ROBOCASA_ENV_RESOLUTION = 128
CAMERA_NAMES = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]


def build_state(obs: dict) -> np.ndarray:
    if "robot0_joint_pos" in obs:
        joint_pos = obs["robot0_joint_pos"]
    else:
        joint_pos = np.arctan2(obs["robot0_joint_pos_sin"], obs["robot0_joint_pos_cos"])
    joint_vel = obs["robot0_joint_vel"]
    gripper_qpos = obs["robot0_gripper_qpos"]
    gripper_qvel = obs["robot0_gripper_qvel"]
    base_to_eef_pos = obs["robot0_base_to_eef_pos"]
    base_to_eef_quat = obs["robot0_base_to_eef_quat"]
    base_quat = obs["robot0_base_quat"]
    qz, qw = (base_quat[2], base_quat[3])
    yaw = 2 * np.arctan2(qz, qw)
    base_yaw_sin = np.array([np.sin(yaw)])
    base_yaw_cos = np.array([np.cos(yaw)])
    state = np.concatenate(
        [
            joint_pos,
            joint_vel,
            gripper_qpos,
            gripper_qvel,
            base_to_eef_pos,
            base_to_eef_quat,
            base_yaw_sin,
            base_yaw_cos,
        ]
    ).astype(np.float32)
    return state


def create_env(
    task_name: str,
    seed: int = 42,
    obj_instance_split: str = "B",
    layout_and_style_ids: tuple[tuple[int, int], ...] | None = (
        (1, 1),
        (2, 2),
        (4, 4),
        (6, 9),
        (7, 10),
    ),
    generative_textures: str | None = None,
    randomize_cameras: bool = False,
    layout_ids: int | None = None,
    style_ids: list[int] | None = None,
) -> robocasa.environments.kitchen.kitchen.Kitchen:
    controller_configs = robosuite.load_composite_controller_config(robot="PandaOmron")
    env_kwargs = dict(
        env_name=task_name,
        robots="PandaMobile",
        controller_configs=controller_configs,
        camera_names=CAMERA_NAMES,
        camera_widths=ROBOCASA_ENV_RESOLUTION,
        camera_heights=ROBOCASA_ENV_RESOLUTION,
        has_renderer=False,
        has_offscreen_renderer=True,
        ignore_done=True,
        use_object_obs=True,
        use_camera_obs=True,
        camera_depths=False,
        seed=seed,
        obj_instance_split=obj_instance_split,
        translucent_robot=False,
        generative_textures=generative_textures,
        randomize_cameras=randomize_cameras,
    )
    if layout_and_style_ids is not None:
        env_kwargs["layout_and_style_ids"] = list(layout_and_style_ids)
    else:
        env_kwargs["layout_ids"] = layout_ids
        env_kwargs["style_ids"] = style_ids
    env = robocasa.make(**env_kwargs)
    return env


def get_task_prompt(task_name: str, env) -> str:
    if hasattr(env, "get_ep_meta"):
        try:
            ep_meta = env.get_ep_meta()
            if "lang" in ep_meta:
                return ep_meta["lang"]
        except Exception:
            pass
    task_prompts = {
        "PnPCounterToCab": "pick the object from the counter and place it in the cabinet",
        "PnPCabToCounter": "pick the object from the cabinet and place it on the counter",
        "PnPCounterToSink": "pick the object from the counter and place it in the sink",
        "PnPSinkToCounter": "pick the object from the sink and place it on the counter",
        "PnPCounterToMicrowave": "pick the object from the counter and place it in the microwave",
        "PnPMicrowaveToCounter": "pick the object from the microwave and place it on the counter",
        "PnPCounterToStove": "pick the object from the counter and place it on the stove",
        "PnPStoveToCounter": "pick the object from the stove and place it on the counter",
        "OpenSingleDoor": "open the door",
        "CloseSingleDoor": "close the door",
        "OpenDoubleDoor": "open both doors",
        "CloseDoubleDoor": "close both doors",
        "OpenDrawer": "open the drawer",
        "CloseDrawer": "close the drawer",
        "TurnOnSinkFaucet": "turn on the sink faucet",
        "TurnOffSinkFaucet": "turn off the sink faucet",
        "TurnSinkSpout": "turn the sink spout",
        "TurnOnStove": "turn on the stove",
        "TurnOffStove": "turn off the stove",
        "TurnOnMicrowave": "turn on the microwave",
        "TurnOffMicrowave": "turn off the microwave",
        "CoffeeSetupMug": "place the mug under the coffee machine",
        "CoffeeServeMug": "serve the coffee mug",
        "CoffeePressButton": "press the coffee machine button",
    }
    return task_prompts.get(task_name, f"perform the {task_name} task")
