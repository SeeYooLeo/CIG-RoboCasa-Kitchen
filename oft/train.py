import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Type
import draccus
import torch
import torch.distributed as dist
import torch.nn as nn
import tqdm
from accelerate import PartialState
from huggingface_hub import snapshot_download
from peft import LoraConfig, PeftModel, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForVision2Seq,
    AutoProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
from policy_utils import find_checkpoint_file, model_is_on_hf_hub
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor,
    PrismaticProcessor,
)
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.cig_loss import compute_cig_loss, compute_l1_energy
from prismatic.training.negative_pool import NegativeInstructionPool
from prismatic.training.train_utils import (
    get_current_action_mask,
    get_next_actions_mask,
)
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("WANDB_MODE", "disabled")


@dataclass
class FinetuneConfig:
    vla_path: str = "openvla/openvla-7b"
    data_root_dir: Path = Path("datasets/rlds")
    dataset_name: str = "robocasa_no_noops"
    run_root_dir: Path = Path("runs")
    shuffle_buffer_size: int = 100000
    num_images_in_input: int = 3
    batch_size: int = 8
    learning_rate: float = 0.0005
    lr_warmup_steps: int = 0
    num_steps_before_decay: int = 100000
    grad_accumulation_steps: int = 1
    max_steps: int = 200000
    use_val_set: bool = False
    val_freq: int = 10000
    val_time_limit: int = 180
    save_freq: int = 10000
    save_latest_checkpoint_only: bool = False
    resume: bool = False
    resume_step: Optional[int] = None
    image_aug: bool = True
    lora_rank: int = 128
    lora_dropout: float = 0.0
    merge_lora_during_training: bool = True
    use_cig: bool = False
    cig_lambda: float = 0.1
    cig_tau: float = 1.0
    cig_num_negatives: int = 4
    cig_floor: float = 0.5
    cig_hard_negatives: bool = True
    cig_same_ratio: float = 0.5
    cig_cross_ratio: float = 0.25
    cig_pool_path: str = "../data/instruction_pool.json"
    cig_restore_expert_grad: bool = True
    wandb_entity: Optional[str] = None
    wandb_project: str = "robocasa-kitchen"
    run_id_note: Optional[str] = None
    run_id_override: Optional[str] = None
    wandb_log_freq: int = 10


def remove_ddp_in_checkpoint(state_dict) -> dict:
    new_state_dict = {}
    for k, v in state_dict.items():
        if k[:7] == "module.":
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


def get_run_id(cfg) -> str:
    if cfg.run_id_override is not None:
        run_id = cfg.run_id_override
    elif cfg.resume:
        run_id = cfg.vla_path.split("/")[-1]
        if "chkpt" in run_id.split("--")[-1]:
            run_id = "--".join(run_id.split("--")[:-1])
    else:
        run_id = f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}+b{cfg.batch_size * cfg.grad_accumulation_steps}+lr-{cfg.learning_rate}"
        run_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
        if cfg.image_aug:
            run_id += "--image_aug"
        if cfg.run_id_note is not None:
            run_id += f"--{cfg.run_id_note}"
    return run_id


def load_checkpoint(
    module_name: str, path: str, step: int, device: str = "cpu"
) -> dict:
    checkpoint_path = find_checkpoint_file(path, module_name)
    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, weights_only=True, map_location=device)
    return remove_ddp_in_checkpoint(state_dict)


def wrap_ddp(module: nn.Module, device_id: int, find_unused: bool = False) -> DDP:
    return DDP(
        module,
        device_ids=[device_id],
        find_unused_parameters=find_unused,
        gradient_as_bucket_view=True,
    )


def count_parameters(module: nn.Module, name: str) -> None:
    num_params = sum((p.numel() for p in module.parameters() if p.requires_grad))
    print(f"# trainable params in {name}: {num_params}")


def synchronize_gradients(parameters) -> None:
    """Average accumulated IL and routed CIG gradients once per optimizer update."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return
    parameters = list(parameters)
    if not parameters:
        return
    present = torch.tensor(
        [p.grad is not None for p in parameters],
        dtype=torch.int32,
        device=parameters[0].device,
    )
    dist.all_reduce(present, op=dist.ReduceOp.SUM)
    for p, count in zip(parameters, present.tolist()):
        if count == 0:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad.div_(dist.get_world_size())


def init_module(
    module_class: Type[nn.Module],
    module_name: str,
    cfg: FinetuneConfig,
    device_id: int,
    module_args: dict,
    to_bf16: bool = False,
    find_unused_params: bool = False,
) -> DDP:
    module = module_class(**module_args)
    count_parameters(module, module_name)
    if cfg.resume:
        state_dict = load_checkpoint(module_name, cfg.vla_path, cfg.resume_step)
        module.load_state_dict(state_dict)
    if to_bf16:
        module = module.to(torch.bfloat16)
    module = module.to(device_id)
    return wrap_ddp(module, device_id, find_unused_params)


def build_cig_negative_batch(
    neg_instructions,
    action_tokenizer,
    base_tokenizer,
    prompt_builder_fn,
    pad_token_id,
    model_max_length,
    gt_actions_np,
):
    from torch.nn.utils.rnn import pad_sequence
    from prismatic.vla.constants import IGNORE_INDEX

    B = len(neg_instructions)
    input_ids_list, labels_list = ([], [])
    for b in range(B):
        lang = neg_instructions[b].lower()
        current_action = gt_actions_np[b, 0]
        future_actions = gt_actions_np[b, 1:]
        current_action_string = action_tokenizer(current_action)
        future_actions_string = "".join(action_tokenizer(future_actions))
        action_chunk_string = current_action_string + future_actions_string
        action_chunk_len = len(action_chunk_string)
        pb = prompt_builder_fn("openvla")
        pb.add_turn("human", f"What action should the robot take to {lang}?")
        pb.add_turn("gpt", action_chunk_string)
        ids = base_tokenizer(pb.get_prompt(), add_special_tokens=True).input_ids
        ids = torch.tensor(ids, dtype=torch.long)
        lbl = ids.clone()
        lbl[: -(action_chunk_len + 1)] = IGNORE_INDEX
        input_ids_list.append(ids)
        labels_list.append(lbl)
    input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    labels = pad_sequence(labels_list, batch_first=True, padding_value=IGNORE_INDEX)
    input_ids = input_ids[:, :model_max_length]
    labels = labels[:, :model_max_length]
    attention_mask = input_ids.ne(pad_token_id)
    return (input_ids, attention_mask, labels)


def compute_cig_energy_with_cached_vision(
    vla,
    action_head,
    projected_patch_embeddings,
    input_ids,
    attention_mask,
    labels,
    ground_truth_actions,
    num_patches,
    device_id,
):
    input_ids = input_ids.to(device_id)
    attention_mask = attention_mask.to(device_id)
    labels = labels.to(device_id)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = vla.module.forward_with_cached_vision(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            projected_patch_embeddings=projected_patch_embeddings,
            output_hidden_states=True,
            return_dict=True,
        )
    ground_truth_token_ids = labels[:, 1:]
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = input_ids.shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    energy = compute_l1_energy(predicted_actions, ground_truth_actions)
    return energy


def run_forward_pass(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    batch,
    action_tokenizer,
    device_id,
    use_l1_regression,
    use_diffusion,
    use_proprio,
    use_film,
    num_patches,
    compute_diffusion_l1=False,
    num_diffusion_steps_train=None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics = {}
    ground_truth_actions = batch["actions"].to(device_id).float()
    noise, noisy_actions, diffusion_timestep_embeddings = (None, None, None)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output: CausalLMOutputWithPast = vla(
            input_ids=batch["input_ids"].to(device_id),
            attention_mask=batch["attention_mask"].to(device_id),
            pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device_id),
            labels=batch["labels"],
            output_hidden_states=True,
            proprio=batch["proprio"],
            proprio_projector=proprio_projector,
            noisy_actions=None,
            noisy_action_projector=None,
            diffusion_timestep_embeddings=None,
            use_film=False,
        )
    ground_truth_token_ids = batch["labels"][:, 1:].to(device_id)
    current_action_mask = get_current_action_mask(ground_truth_token_ids)
    next_actions_mask = get_next_actions_mask(ground_truth_token_ids)
    last_hidden_states = output.hidden_states[-1]
    text_hidden_states = last_hidden_states[:, num_patches:-1]
    batch_size = batch["input_ids"].shape[0]
    actions_hidden_states = (
        text_hidden_states[current_action_mask | next_actions_mask]
        .reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        .to(torch.bfloat16)
    )
    predicted_actions = action_head.module.predict_action(actions_hidden_states)
    loss = compute_l1_energy(predicted_actions, ground_truth_actions).mean()
    metrics.update({"loss_value": loss.item()})
    should_log_l1_loss = True
    if should_log_l1_loss:
        ground_truth_curr_action = ground_truth_actions[:, 0]
        predicted_curr_action = predicted_actions[:, 0]
        ground_truth_next_actions = ground_truth_actions[:, 1:]
        predicted_next_actions = predicted_actions[:, 1:]
        curr_action_l1_loss = torch.nn.L1Loss()(
            ground_truth_curr_action, predicted_curr_action
        )
        next_actions_l1_loss = torch.nn.L1Loss()(
            ground_truth_next_actions, predicted_next_actions
        )
        metrics.update(
            {
                "curr_action_l1_loss": curr_action_l1_loss.item(),
                "next_actions_l1_loss": next_actions_l1_loss.item(),
            }
        )
    return (loss, metrics)


def compute_smoothened_metrics(metrics_deques) -> dict:
    smoothened_metrics = {}
    for name, deque in metrics_deques.items():
        if deque and len(deque) > 0:
            smoothened_metrics[name] = sum(deque) / len(deque)
    return smoothened_metrics


def log_metrics_to_wandb(metrics, prefix, step, wandb_entity) -> None:
    log_dict = {}
    for name, value in metrics.items():
        if name == "loss_value":
            log_dict[f"{prefix}/Loss"] = value
        else:
            log_dict[f"{prefix}/{name.replace('_', ' ').title()}"] = value
    wandb_entity.log(log_dict, step=step)


def save_training_checkpoint(
    cfg,
    run_dir,
    log_step,
    vla,
    processor,
    proprio_projector,
    noisy_action_projector,
    action_head,
    train_dataset,
    distributed_state,
) -> None:
    if cfg.save_latest_checkpoint_only:
        checkpoint_dir = run_dir
        checkpoint_name_suffix = "latest_checkpoint.pt"
    else:
        checkpoint_dir = Path(str(run_dir) + f"--{log_step}_chkpt")
        checkpoint_name_suffix = f"{log_step}_checkpoint.pt"
    adapter_dir = checkpoint_dir / "lora_adapter"
    if distributed_state.is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(adapter_dir, exist_ok=True)
        save_dataset_statistics(train_dataset.dataset_statistics, checkpoint_dir)
        print(f"Saving Model Checkpoint for Step {log_step}")
    dist.barrier()
    if distributed_state.is_main_process:
        processor.save_pretrained(checkpoint_dir)
        vla.module.save_pretrained(adapter_dir)
        if proprio_projector is not None:
            torch.save(
                proprio_projector.state_dict(),
                checkpoint_dir / f"proprio_projector--{checkpoint_name_suffix}",
            )
        if action_head is not None:
            torch.save(
                action_head.state_dict(),
                checkpoint_dir / f"action_head--{checkpoint_name_suffix}",
            )
    dist.barrier()
    if cfg.merge_lora_during_training:
        base_vla = AutoModelForVision2Seq.from_pretrained(
            cfg.vla_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
        merged_vla = merged_vla.merge_and_unload()
        if distributed_state.is_main_process:
            merged_vla.save_pretrained(checkpoint_dir)
            print(f"Saved merged model for Step {log_step} at: {checkpoint_dir}")
        dist.barrier()


def run_validation(
    vla,
    action_head,
    noisy_action_projector,
    proprio_projector,
    val_dataloader,
    action_tokenizer,
    device_id,
    cfg,
    num_patches,
    log_step,
    distributed_state,
    val_time_limit,
) -> None:
    val_start_time = time.time()
    vla.eval()
    val_batches_count = 0
    all_val_metrics = []
    with torch.no_grad():
        for batch in val_dataloader:
            _, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=noisy_action_projector,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_proprio=True,
                use_film=False,
                num_patches=num_patches,
                compute_diffusion_l1=True,
                num_diffusion_steps_train=None,
            )
            metrics["loss"] = metrics["loss_value"]
            all_val_metrics.append(metrics)
            val_batches_count += 1
            if time.time() - val_start_time > val_time_limit:
                break
    avg_val_metrics = {}
    for metric_name in all_val_metrics[0].keys():
        values = [
            metrics[metric_name]
            for metrics in all_val_metrics
            if metric_name in metrics
        ]
        if values:
            avg_val_metrics[metric_name] = sum(values) / len(values)
    avg_val_metrics["val_batches_count"] = val_batches_count
    if distributed_state.is_main_process:
        log_metrics_to_wandb(avg_val_metrics, "VLA Val", log_step, wandb)


@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    cfg.vla_path = cfg.vla_path.rstrip("/")
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    run_id = get_run_id(cfg)
    run_dir = cfg.run_root_dir / run_id
    os.makedirs(run_dir, exist_ok=True)
    distributed_state = PartialState()
    device_id = distributed_state.local_process_index
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    if distributed_state.is_main_process:
        wandb.init(
            entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{run_id}"
        )
    print(
        f"Detected constants:\n\tNUM_ACTIONS_CHUNK: {NUM_ACTIONS_CHUNK}\n\tACTION_DIM: {ACTION_DIM}\n\tPROPRIO_DIM: {PROPRIO_DIM}\n\tACTION_PROPRIO_NORMALIZATION_TYPE: {ACTION_PROPRIO_NORMALIZATION_TYPE}"
    )
    if model_is_on_hf_hub(cfg.vla_path):
        vla_download_path = snapshot_download(repo_id=cfg.vla_path)
        cfg.vla_path = vla_download_path
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    dist.barrier()
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=False)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    ).to(device_id)
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=cfg.lora_dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()
    vla = wrap_ddp(vla, device_id, find_unused=True)
    proprio_projector = init_module(
        ProprioProjector,
        "proprio_projector",
        cfg,
        device_id,
        {"llm_dim": vla.module.llm_dim, "proprio_dim": PROPRIO_DIM},
    )
    action_head = init_module(
        L1RegressionActionHead,
        "action_head",
        cfg,
        device_id,
        {
            "input_dim": vla.module.llm_dim,
            "hidden_dim": vla.module.llm_dim,
            "action_dim": ACTION_DIM,
        },
        to_bf16=True,
    )
    NUM_PATCHES = (
        vla.module.vision_backbone.get_num_patches()
        * vla.module.vision_backbone.get_num_images_in_input()
    )
    NUM_PATCHES += 1
    negative_pool = None
    expert_param_ids: set = set()
    if cfg.use_cig:
        negative_pool = NegativeInstructionPool(cfg.cig_pool_path)
        try:
            vla.module.language_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if distributed_state.is_main_process:
                print("[CIG] Enabled gradient checkpointing on language_model.")
        except Exception as e:
            if distributed_state.is_main_process:
                print(f"[CIG] WARNING: failed to enable gradient checkpointing: {e}")
        for p in action_head.module.parameters():
            expert_param_ids.add(id(p))
        for p in proprio_projector.module.parameters():
            expert_param_ids.add(id(p))
        if distributed_state.is_main_process:
            print(f"[CIG] expert params (IL-only) count: {len(expert_param_ids)}")
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    trainable_params += [
        param for param in action_head.parameters() if param.requires_grad
    ]
    trainable_params += [
        param for param in proprio_projector.parameters() if param.requires_grad
    ]
    print(f"# total trainable params: {sum((p.numel() for p in trainable_params))}")
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)
    original_lr = optimizer.param_groups[0]["lr"]
    scheduler = MultiStepLR(
        optimizer, milestones=[cfg.num_steps_before_decay], gamma=0.1
    )
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    use_wrist_image = cfg.num_images_in_input > 1
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=use_wrist_image,
        use_proprio=True,
    )
    train_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.module.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )
    if cfg.use_val_set:
        val_dataset = RLDSDataset(
            cfg.data_root_dir,
            cfg.dataset_name,
            batch_transform,
            resize_resolution=tuple(vla.module.config.image_sizes),
            shuffle_buffer_size=cfg.shuffle_buffer_size // 10,
            image_aug=cfg.image_aug,
            train=False,
        )
    if distributed_state.is_main_process:
        save_dataset_statistics(train_dataset.dataset_statistics, run_dir)
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )
    if cfg.use_val_set:
        val_batch_size = cfg.batch_size
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            sampler=None,
            collate_fn=collator,
            num_workers=0,
        )
    recent_metrics = {
        "loss_value": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "curr_action_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_accuracy": deque(maxlen=cfg.grad_accumulation_steps),
        "next_actions_l1_loss": deque(maxlen=cfg.grad_accumulation_steps),
    }
    if cfg.use_cig:
        for k in (
            "loss_il",
            "loss_cig",
            "pos_energy",
            "neg_energy_mean",
            "energy_gap",
            "cig_active",
        ):
            recent_metrics[k] = deque(maxlen=cfg.grad_accumulation_steps)
    start_step = cfg.resume_step if cfg.resume else 0
    if start_step is None or cfg.max_steps <= start_step:
        raise ValueError("max_steps must exceed the starting update count")
    if cfg.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be positive")
    # Cached CIG forwards and action prediction call .module directly. Defer
    # reduction until both backward passes and expert-gradient restoration finish.
    with (
        vla.no_sync(),
        action_head.no_sync(),
        proprio_projector.no_sync(),
        tqdm.tqdm(total=cfg.max_steps - start_step, leave=False) as progress,
    ):
        vla.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            compute_diffusion_l1 = False
            loss, metrics = run_forward_pass(
                vla=vla,
                action_head=action_head,
                noisy_action_projector=None,
                proprio_projector=proprio_projector,
                batch=batch,
                action_tokenizer=action_tokenizer,
                device_id=device_id,
                use_l1_regression=True,
                use_diffusion=False,
                use_proprio=True,
                use_film=False,
                num_patches=NUM_PATCHES,
                compute_diffusion_l1=compute_diffusion_l1,
                num_diffusion_steps_train=None,
            )
            normalized_loss = loss / cfg.grad_accumulation_steps
            if cfg.use_cig:
                gt_actions = batch["actions"].to(device_id).float()
                proprio_tensor = batch["proprio"].to(device_id)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    projected_patch_embeddings = vla.module._process_vision_features(
                        pixel_values=batch["pixel_values"]
                        .to(torch.bfloat16)
                        .to(device_id),
                        language_embeddings=None,
                        use_film=False,
                    )
                    projected_patch_embeddings = vla.module._process_proprio_features(
                        projected_patch_embeddings,
                        proprio_tensor.to(torch.bfloat16),
                        proprio_projector,
                    )
                pos_input_ids = batch["input_ids"].to(device_id)
                pos_attention_mask = batch["attention_mask"].to(device_id)
                pos_labels = batch["labels"]
                pos_energy = compute_cig_energy_with_cached_vision(
                    vla=vla,
                    action_head=action_head,
                    projected_patch_embeddings=projected_patch_embeddings,
                    input_ids=pos_input_ids,
                    attention_mask=pos_attention_mask,
                    labels=pos_labels,
                    ground_truth_actions=gt_actions,
                    num_patches=NUM_PATCHES,
                    device_id=device_id,
                )
                positive_instructions = batch["language_instructions"]
                gt_actions_np = batch["actions"].cpu().numpy()
                same_r = cfg.cig_same_ratio if cfg.cig_hard_negatives else 0.0
                cross_r = cfg.cig_cross_ratio if cfg.cig_hard_negatives else 0.0
                neg_lists = negative_pool.sample_negatives(
                    positive_instructions,
                    K=cfg.cig_num_negatives,
                    same_ratio=same_r,
                    cross_ratio=cross_r,
                )
                neg_energies = []
                for k_idx in range(cfg.cig_num_negatives):
                    neg_input_ids, neg_attention_mask, neg_labels = (
                        build_cig_negative_batch(
                            neg_instructions=neg_lists[k_idx],
                            action_tokenizer=action_tokenizer,
                            base_tokenizer=processor.tokenizer,
                            prompt_builder_fn=PurePromptBuilder,
                            pad_token_id=processor.tokenizer.pad_token_id,
                            model_max_length=processor.tokenizer.model_max_length,
                            gt_actions_np=gt_actions_np,
                        )
                    )
                    neg_E = compute_cig_energy_with_cached_vision(
                        vla=vla,
                        action_head=action_head,
                        projected_patch_embeddings=projected_patch_embeddings,
                        input_ids=neg_input_ids,
                        attention_mask=neg_attention_mask,
                        labels=neg_labels,
                        ground_truth_actions=gt_actions,
                        num_patches=NUM_PATCHES,
                        device_id=device_id,
                    )
                    neg_energies.append(neg_E)
                loss_cig = compute_cig_loss(pos_energy, neg_energies, tau_c=cfg.cig_tau)
                neg_E_stack = torch.stack(neg_energies, dim=0)
                pos_E_mean = pos_energy.detach().float().mean().item()
                neg_E_mean = neg_E_stack.detach().float().mean().item()
                energy_gap = neg_E_mean - pos_E_mean
                cig_active = float(loss_cig.item() >= cfg.cig_floor)
                metrics["loss_il"] = loss.item()
                metrics["loss_cig"] = loss_cig.item()
                metrics["pos_energy"] = pos_E_mean
                metrics["neg_energy_mean"] = neg_E_mean
                metrics["energy_gap"] = energy_gap
                metrics["cig_active"] = cig_active
                if cig_active > 0.5:
                    normalized_loss.backward()
                    expert_il_grads = {}
                    for p in action_head.module.parameters():
                        if p.grad is not None:
                            expert_il_grads[id(p)] = p.grad.detach().clone()
                    for p in proprio_projector.module.parameters():
                        if p.grad is not None:
                            expert_il_grads[id(p)] = p.grad.detach().clone()
                    (cfg.cig_lambda * loss_cig / cfg.grad_accumulation_steps).backward()
                    if cfg.cig_restore_expert_grad:
                        for p in action_head.module.parameters():
                            saved = expert_il_grads.get(id(p))
                            if saved is not None:
                                p.grad.copy_(saved)
                            else:
                                p.grad = None
                        for p in proprio_projector.module.parameters():
                            saved = expert_il_grads.get(id(p))
                            if saved is not None:
                                p.grad.copy_(saved)
                            else:
                                p.grad = None
                else:
                    normalized_loss.backward()
            else:
                normalized_loss.backward()
            for metric_name, value in metrics.items():
                if metric_name in recent_metrics:
                    recent_metrics[metric_name].append(value)
            # Count completed optimizer updates, not zero-based microbatch labels.
            if (batch_idx + 1) % cfg.grad_accumulation_steps != 0:
                continue
            gradient_step_idx = (batch_idx + 1) // cfg.grad_accumulation_steps
            smoothened_metrics = compute_smoothened_metrics(recent_metrics)
            log_step = start_step + gradient_step_idx
            if distributed_state.is_main_process and log_step % cfg.wandb_log_freq == 0:
                log_metrics_to_wandb(smoothened_metrics, "VLA Train", log_step, wandb)
                try:
                    _pf = {
                        k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
                        for k, v in smoothened_metrics.items()
                        if isinstance(v, (int, float))
                    }
                    progress.set_postfix(_pf)
                    if log_step % 50 == 0:
                        print(
                            f"[step {log_step}] "
                            + " ".join((f"{k}={v}" for k, v in _pf.items())),
                            flush=True,
                        )
                except Exception as _e:
                    pass
            if cfg.lr_warmup_steps > 0:
                lr_progress = min(gradient_step_idx / cfg.lr_warmup_steps, 1.0)
                current_lr = original_lr * (0.1 + 0.9 * lr_progress)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr
            if (
                distributed_state.is_main_process
                and gradient_step_idx % cfg.wandb_log_freq == 0
            ):
                wandb.log(
                    {"VLA Train/Learning Rate": scheduler.get_last_lr()[0]},
                    step=log_step,
                )
            synchronize_gradients(trainable_params)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            progress.update()
            if log_step % cfg.save_freq == 0 or log_step >= cfg.max_steps:
                save_training_checkpoint(
                    cfg=cfg,
                    run_dir=run_dir,
                    log_step=log_step,
                    vla=vla,
                    processor=processor,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=None,
                    action_head=action_head,
                    train_dataset=train_dataset,
                    distributed_state=distributed_state,
                )
            if cfg.use_val_set and log_step > 0 and (log_step % cfg.val_freq == 0):
                run_validation(
                    vla=vla,
                    action_head=action_head,
                    noisy_action_projector=None,
                    proprio_projector=proprio_projector,
                    val_dataloader=val_dataloader,
                    action_tokenizer=action_tokenizer,
                    device_id=device_id,
                    cfg=cfg,
                    num_patches=NUM_PATCHES,
                    log_step=log_step,
                    distributed_state=distributed_state,
                    val_time_limit=cfg.val_time_limit,
                )
                vla.train()
            if log_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
