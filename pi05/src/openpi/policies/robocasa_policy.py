import dataclasses
import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

STATE_DIM = 27
ACTION_DIM = 12
ACTION_DIM_NAMES = {
    0: "rel_pos_x",
    1: "rel_pos_y",
    2: "rel_pos_z",
    3: "rel_rot_x",
    4: "rel_rot_y",
    5: "rel_rot_z",
    6: "gripper",
    7: "base_vel_vx",
    8: "base_vel_vy",
    9: "base_vel_vyaw",
    10: "torso",
    11: "flag",
}
SAFE_OUTPUT_VALUES = {7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: -1.0}
NO_BASE_ACTION_DIMS = tuple(range(7))
NO_BASE_STATE_DIMS = tuple(range(25))
NO_BASE_FIXED_ACTION_VALUES = {7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: -1.0}


def detect_constant_action_dims(
    norm_stats: dict, threshold: float = 1e-05
) -> tuple[dict[int, float], dict[int, float]]:
    import logging

    if "actions" not in norm_stats:
        logging.warning("No 'actions' in norm_stats, skipping constant dim detection")
        return ({}, {})
    actions_stats = norm_stats["actions"]
    if not hasattr(actions_stats, "q01") or not hasattr(actions_stats, "q99"):
        logging.warning(
            "No quantile stats (q01/q99) in norm_stats, skipping constant dim detection"
        )
        return ({}, {})
    if actions_stats.q01 is None or actions_stats.q99 is None:
        logging.warning("Quantile stats are None, skipping constant dim detection")
        return ({}, {})
    q01 = np.asarray(actions_stats.q01)
    q99 = np.asarray(actions_stats.q99)
    constant_action_dims = {}
    output_values = {}
    for dim in range(min(len(q01), ACTION_DIM)):
        diff = abs(q99[dim] - q01[dim])
        if diff < threshold:
            constant_action_dims[dim] = float(q01[dim])
            output_values[dim] = SAFE_OUTPUT_VALUES.get(dim, 0.0)
            dim_name = ACTION_DIM_NAMES.get(dim, f"dim{dim}")
            logging.info(
                f"Detected constant action dim {dim} ({dim_name}): q01={q01[dim]:.6g}, q99={q99[dim]:.6g}, diff={diff:.2e}, train_value={constant_action_dims[dim]:.6g}, output_value={output_values[dim]:.6g}"
            )
    return (constant_action_dims, output_values)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    model_type: _model.ModelType
    constant_action_dims: dict[int, float] = dataclasses.field(default_factory=dict)
    exclude_base_movement: bool = False

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        if "observation/image_right" in data:
            right_image = _parse_image(data["observation/image_right"])
            has_right_image = True
        else:
            right_image = np.zeros_like(base_image)
            has_right_image = False
        state = np.asarray(data["observation/state"]).astype(np.float32)
        if self.exclude_base_movement:
            state = state[..., NO_BASE_STATE_DIMS]
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, right_image)
                image_masks = (
                    np.True_,
                    np.True_,
                    np.True_ if has_right_image else np.False_,
                )
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                images = (base_image, right_image, wrist_image)
                image_masks = (
                    np.True_,
                    np.True_ if has_right_image else np.False_,
                    np.True_,
                )
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")
        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }
        if "actions" in data:
            actions = np.asarray(data["actions"]).copy()
            for dim, value in self.constant_action_dims.items():
                if dim >= actions.shape[-1]:
                    continue
                if actions.ndim == 2:
                    actions[:, dim] = value
                else:
                    actions[dim] = value
            if self.exclude_base_movement:
                actions = actions[..., NO_BASE_ACTION_DIMS]
            inputs["actions"] = actions
        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt
        if "counterfactual_instructions" in data:
            inputs["counterfactual_instructions"] = data["counterfactual_instructions"]
        if "counterfactual_levels" in data:
            inputs["counterfactual_levels"] = data["counterfactual_levels"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    output_values: dict[int, float] = dataclasses.field(default_factory=dict)
    exclude_base_movement: bool = False

    def __call__(self, data: dict) -> dict:
        model_actions = np.asarray(data["actions"])
        if self.exclude_base_movement:
            reduced_actions = np.asarray(
                model_actions[:, : len(NO_BASE_ACTION_DIMS)]
            ).copy()
            actions = np.zeros(
                (reduced_actions.shape[0], ACTION_DIM), dtype=reduced_actions.dtype
            )
            actions[:, NO_BASE_ACTION_DIMS] = reduced_actions
            for dim, value in NO_BASE_FIXED_ACTION_VALUES.items():
                actions[:, dim] = value
        else:
            actions = np.asarray(model_actions[:, :ACTION_DIM]).copy()
        for dim, value in self.output_values.items():
            if dim >= actions.shape[-1]:
                continue
            if self.exclude_base_movement and dim in NO_BASE_FIXED_ACTION_VALUES:
                continue
            actions[:, dim] = value
        return {"actions": actions}
