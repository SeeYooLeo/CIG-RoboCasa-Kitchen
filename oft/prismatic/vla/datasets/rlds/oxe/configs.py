from enum import IntEnum


class StateEncoding(IntEnum):
    NONE = -1
    POS_EULER = 1
    POS_QUAT = 2
    JOINT = 3
    JOINT_BIMANUAL = 4


class ActionEncoding(IntEnum):
    EEF_POS = 1
    JOINT_POS = 2
    JOINT_POS_BIMANUAL = 3
    EEF_R6 = 4
    ROBOCASA = 5


OXE_DATASET_CONFIGS = {
    "robocasa_no_noops": {
        "image_obs_keys": {
            "primary": "image",
            "secondary": "secondary_image",
            "wrist": "wrist_image",
        },
        "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
        "state_obs_keys": ["state"],
        "state_encoding": StateEncoding.NONE,
        "action_encoding": ActionEncoding.ROBOCASA,
    }
}
