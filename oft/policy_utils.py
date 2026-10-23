import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import json_numpy
import numpy as np
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)

json_numpy.patch()
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import ACTION_DIM, ACTION_PROPRIO_NORMALIZATION_TYPE
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    if os.path.isdir(model_path):
        return False
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    assert os.path.isdir(pretrained_checkpoint), (
        f"Checkpoint path must be a directory: {pretrained_checkpoint}"
    )
    released = os.path.join(pretrained_checkpoint, file_pattern + ".pt")
    if os.path.isfile(released):
        return released
    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)
    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )
    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    state_dict = torch.load(
        checkpoint_path,
        weights_only=True,
        map_location="cpu" if not torch.cuda.is_available() else None,
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    print("Instantiating pretrained VLA policy...")
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    vla.eval()
    if not cfg.load_in_8bit and (not cfg.load_in_4bit):
        vla = vla.to(DEVICE)
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)
    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    if model_is_on_hf_hub(checkpoint_path):
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path, filename="dataset_statistics.json"
        )
    else:
        dataset_statistics_path = os.path.join(
            checkpoint_path, "dataset_statistics.json"
        )
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\nYou can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint.Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    return AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint, trust_remote_code=False
    )


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    proprio_projector = ProprioProjector(llm_dim=llm_dim, proprio_dim=proprio_dim).to(
        DEVICE
    )
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()
    checkpoint_path = find_checkpoint_file(
        cfg.pretrained_checkpoint, "proprio_projector"
    )
    state_dict = load_component_state_dict(checkpoint_path)
    proprio_projector.load_state_dict(state_dict)
    return proprio_projector


def get_action_head(cfg: Any, llm_dim: int) -> L1RegressionActionHead:
    assert not (cfg.use_l1_regression and cfg.use_diffusion), (
        "Cannot use both L1 regression and diffusion action head!"
    )
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM
        )
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")
    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
    state_dict = load_component_state_dict(checkpoint_path)
    action_head.load_state_dict(state_dict)
    return action_head


def resize_image_for_policy(
    img: np.ndarray, resize_size: Union[int, Tuple[int, int]]
) -> np.ndarray:
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = tf.image.encode_jpeg(img)
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True
    new_heights = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,)
    )
    new_widths = tf.reshape(
        tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,)
    )
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )
    image = tf.image.crop_and_resize(
        image,
        bounding_boxes,
        tf.range(batch_size),
        (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE),
    )
    if expanded_dims:
        image = image[0]
    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    batch_size = 1
    crop_scale = 0.9
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))
    orig_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = crop_and_resize(image, crop_scale, batch_size)
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8
    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = (
            np.array(norm_stats["max"]),
            np.array(norm_stats["min"]),
        )
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = (
            np.array(norm_stats["q99"]),
            np.array(norm_stats["q01"]),
        )
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")
    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-08) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )
    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    processed_images = []
    for image in images:
        check_image_format(image)
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)
        pil_image = Image.fromarray(image).convert("RGB")
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)
        processed_images.append(pil_image)
    return processed_images


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
) -> List[np.ndarray]:
    with torch.inference_mode():
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])
        all_images = prepare_images_for_vla(all_images, cfg)
        primary_image = all_images.pop(0)
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"
        inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)
        if all_images:
            all_wrist_inputs = [
                processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16)
                for image_wrist in all_images
            ]
            primary_pixel_values = inputs["pixel_values"]
            all_wrist_pixel_values = [
                wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs
            ]
            inputs["pixel_values"] = torch.cat(
                [primary_pixel_values] + all_wrist_pixel_values, dim=1
            )
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
            proprio = obs["state"]
        if action_head is None:
            action, _ = vla.predict_action(
                **inputs, unnorm_key=cfg.unnorm_key, do_sample=False
            )
        else:
            action, _ = vla.predict_action(
                **inputs,
                unnorm_key=cfg.unnorm_key,
                do_sample=False,
                proprio=proprio,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                action_head=action_head,
                use_film=use_film,
            )
    single = action[0]
    return [single[i] for i in range(single.shape[0])]
