import logging
import os
import platform
import random
import shutil
import time
import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
import openpi.models.pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import openpi.training.data_loader as _data
from openpi.training.cig_data_loader import create_cig_data_loader
from openpi.training.cig_loss import compute_cig_loss, compute_fm_energy_cached
from openpi.training.negative_pool import NegativeInstructionPool

EXPERT_PARAM_KEYS = (
    "gemma_expert",
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "state_proj",
)


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


from scripts.training_utils import (
    cleanup_ddp,
    get_latest_checkpoint_step,
    init_logging,
    init_wandb,
    load_checkpoint,
    log_memory_usage,
    save_checkpoint,
    set_seed,
    setup_ddp,
)


def _is_expert_param(name: str) -> bool:
    return any((k in name for k in EXPERT_PARAM_KEYS))


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = not use_ddp or dist.get_rank() == 0
    set_seed(config.seed, local_rank)
    cig_cfg = config.cig
    resuming = False
    if config.resume:
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(
                    f"No valid checkpoints found in {exp_checkpoint_dir} for resume"
                )
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume"
            )
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")
    if not resuming:
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total: {config.batch_size} across {world_size} GPUs)"
    )
    if cig_cfg.enabled:
        loader, data_config = create_cig_data_loader(config, shuffle=True)
        logging.info("Using CIG data loader with counterfactual instructions")
    else:
        loader_impl = _data.create_data_loader(
            config, framework="pytorch", shuffle=True
        )
        data_config = loader_impl.data_config()
        loader = loader_impl
        logging.info("Using standard data loader (CIG disabled)")
    tokenizer = None
    if cig_cfg.enabled:
        tokenizer = _tokenizer.PaligemmaTokenizer(config.model.max_token_len)
        logging.info("Initialized PaligemmaTokenizer for counterfactual tokenization")
    neg_pool = None
    if cig_cfg.enabled and cig_cfg.instruction_pool_path:
        neg_pool = NegativeInstructionPool(
            cig_cfg.instruction_pool_path, tokenizer, device
        )
        logging.info(
            f"Using global negative instruction pool: {cig_cfg.instruction_pool_path}"
        )
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(
                config.model, "action_expert_variant", "gemma_300m"
            ),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing")
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(model, model_path)
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")
    if is_main:
        expert_params = sum(
            (p.numel() for n, p in model.named_parameters() if _is_expert_param(n))
        )
        vlm_params = sum(
            (p.numel() for n, p in model.named_parameters() if not _is_expert_param(n))
        )
        logging.info(
            f"Parameter split: Expert={expert_params / 1000000.0:.1f}M (IL only), VLM={vlm_params / 1000000.0:.1f}M (IL+CIG)"
        )
    if use_ddp:
        for tensor in model.state_dict().values():
            dist.broadcast(tensor, src=0)
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    if config.batch_size % world_size:
        raise ValueError("batch_size must be divisible by world size")
    optim.zero_grad(set_to_none=True)
    model.train()
    start_time = time.time()
    infos = []
    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={world_size}")
        logging.info(
            f"Training: batch_size={config.batch_size}, effective={effective_batch_size}, steps={config.num_train_steps}"
        )
        logging.info(
            f"CIG: enabled={cig_cfg.enabled}, lambda={cig_cfg.lambda_weight}, tau_c={cig_cfg.tau_c}, K={cig_cfg.num_negatives}, CIG→VLM only (Expert frozen for CIG), pool={cig_cfg.instruction_pool_path}"
        )
        logging.info(
            f"LR: warmup={warmup_steps}, peak={peak_lr:.2e}, decay={decay_steps}, end={end_lr:.2e}"
        )
    accum_steps = config.gradient_accumulation_steps
    if accum_steps > 1 and is_main:
        logging.info(f"Gradient accumulation: {accum_steps} micro-steps")
    pbar = (
        tqdm.tqdm(
            total=config.num_train_steps,
            initial=global_step,
            desc="Training",
            disable=not is_main,
        )
        if is_main
        else None
    )
    micro_step = 0
    accum_loss_il = 0.0
    accum_loss_cig = 0.0
    accum_pos_energy = 0.0
    accum_neg_energy = 0.0
    while global_step < config.num_train_steps:
        for batch_data in loader:
            if global_step >= config.num_train_steps:
                break
            if cig_cfg.enabled:
                observation, actions, cf_instructions_batch, pos_instructions = (
                    batch_data
                )
            else:
                observation, actions = batch_data
                cf_instructions_batch = None
            observation = jax.tree.map(lambda x: x.to(device), observation)
            actions = actions.to(torch.float32).to(device)
            if micro_step == 0:
                for pg in optim.param_groups:
                    pg["lr"] = lr_schedule(global_step)
            model_raw = (
                model.module
                if isinstance(model, torch.nn.parallel.DistributedDataParallel)
                else model
            )
            losses_il = model(observation, actions)
            if isinstance(losses_il, list | tuple):
                losses_il = torch.stack(losses_il)
            loss_il = losses_il.mean()
            loss_cig = None
            pos_e_val = 0.0
            neg_e_val = 0.0
            if cig_cfg.enabled:
                B = actions.shape[0]
                K = cig_cfg.num_negatives
                noise = model_raw.sample_noise(actions.shape, actions.device)
                time_t = model_raw.sample_time(B, actions.device)
                images, img_masks, lang_tokens, lang_masks, state = (
                    model_raw._preprocess_observation(observation, train=True)
                )
                img_cache = model_raw.embed_images(images, img_masks)
                pos_energy = compute_fm_energy_cached(
                    model_raw,
                    img_cache,
                    lang_tokens,
                    lang_masks,
                    state,
                    actions,
                    noise,
                    time_t,
                    detach_prefix=False,
                )
                neg_energies = []
                if neg_pool is not None:
                    same_r = (
                        cig_cfg.hard_neg_same_ratio if cig_cfg.hard_negatives else 0.0
                    )
                    cross_r = (
                        cig_cfg.hard_neg_cross_ratio
                        if cig_cfg.hard_negatives
                        else 0.0
                    )
                    neg_tokens_list, neg_masks_list = neg_pool.sample_negatives(
                        pos_instructions, K, same_ratio=same_r, cross_ratio=cross_r
                    )
                    for k in range(K):
                        neg_e = compute_fm_energy_cached(
                            model_raw,
                            img_cache,
                            neg_tokens_list[k],
                            neg_masks_list[k],
                            state,
                            actions,
                            noise,
                            time_t,
                            detach_prefix=False,
                        )
                        neg_energies.append(neg_e)
                elif cf_instructions_batch is not None:
                    for k_idx in range(K):
                        cf_instructions_k = []
                        for b in range(B):
                            cf_list = cf_instructions_batch[b]
                            if cf_list:
                                idx = (
                                    k_idx % len(cf_list)
                                    if k_idx < len(cf_list)
                                    else random.randint(0, len(cf_list) - 1)
                                )
                                cf_instructions_k.append(cf_list[idx])
                            else:
                                cf_instructions_k.append("do nothing")
                        cf_tokens, cf_masks = _tokenize_instructions(
                            tokenizer, cf_instructions_k, device
                        )
                        neg_e = compute_fm_energy_cached(
                            model_raw,
                            img_cache,
                            cf_tokens,
                            cf_masks,
                            state,
                            actions,
                            noise,
                            time_t,
                            detach_prefix=False,
                        )
                        neg_energies.append(neg_e)
                loss_cig = compute_cig_loss(pos_energy, neg_energies, cig_cfg.tau_c)
                with torch.no_grad():
                    pos_e_val = pos_energy.mean().item()
                    neg_e_val = sum((ne.mean().item() for ne in neg_energies)) / max(
                        len(neg_energies), 1
                    )
                cig_active = loss_cig.item() >= cig_cfg.cig_floor
                if cig_active:
                    (loss_il / accum_steps).backward(retain_graph=True)
                    expert_il_grads = {}
                    for n, p in model_raw.named_parameters():
                        if _is_expert_param(n) and p.grad is not None:
                            expert_il_grads[n] = p.grad.detach().clone()
                    (cig_cfg.lambda_weight * loss_cig / accum_steps).backward()
                    for n, p in model_raw.named_parameters():
                        if _is_expert_param(n):
                            if n in expert_il_grads:
                                p.grad = expert_il_grads[n]
                            else:
                                p.grad = None
                else:
                    (loss_il / accum_steps).backward()
            else:
                (loss_il / accum_steps).backward()
            if (
                global_step < 3
                and micro_step == 0
                and is_main
                and torch.cuda.is_available()
            ):
                log_memory_usage(device, global_step, "after_backward")
            accum_loss_il += loss_il.item()
            if loss_cig is not None:
                accum_loss_cig += loss_cig.item()
                accum_pos_energy += pos_e_val
                accum_neg_energy += neg_e_val
            micro_step += 1
            if micro_step >= accum_steps:
                if use_ddp:
                    for p in model_raw.parameters():
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=config.optimizer.clip_gradient_norm
                )
                optim.step()
                optim.zero_grad(set_to_none=True)
                if is_main:
                    info = {
                        "loss_il": accum_loss_il / accum_steps,
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm)
                        if isinstance(grad_norm, torch.Tensor)
                        else grad_norm,
                    }
                    if cig_cfg.enabled:
                        info["loss_cig"] = accum_loss_cig / accum_steps
                        info["pos_energy"] = accum_pos_energy / accum_steps
                        info["neg_energy"] = accum_neg_energy / accum_steps
                        info["energy_gap"] = info["neg_energy"] - info["pos_energy"]
                        info["cig_active"] = (
                            1
                            if accum_loss_cig / accum_steps >= cig_cfg.cig_floor
                            else 0
                        )
                    infos.append(info)
                if is_main and global_step % config.log_interval == 0:
                    elapsed = time.time() - start_time
                    avg_loss_il = sum((i["loss_il"] for i in infos)) / len(infos)
                    avg_lr = sum((i["learning_rate"] for i in infos)) / len(infos)
                    log_msg = (
                        f"step={global_step} loss_il={avg_loss_il:.4f} lr={avg_lr:.2e}"
                    )
                    log_payload = {
                        "loss_il": avg_loss_il,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / max(1, len(infos)),
                    }
                    vals = [
                        i["grad_norm"]
                        for i in infos
                        if "grad_norm" in i and i["grad_norm"] is not None
                    ]
                    if vals:
                        avg_grad_norm = sum(vals) / len(vals)
                        log_msg += f" grad_norm={avg_grad_norm:.2f}"
                        log_payload["grad_norm"] = avg_grad_norm
                    if any(("loss_cig" in i for i in infos)):
                        avg_loss_cig = sum((i.get("loss_cig", 0) for i in infos)) / len(
                            infos
                        )
                        avg_pos_e = sum((i.get("pos_energy", 0) for i in infos)) / len(
                            infos
                        )
                        avg_neg_e = sum((i.get("neg_energy", 0) for i in infos)) / len(
                            infos
                        )
                        avg_gap = sum((i.get("energy_gap", 0) for i in infos)) / len(
                            infos
                        )
                        avg_cig_active = sum(
                            (i.get("cig_active", 0) for i in infos)
                        ) / len(infos)
                        log_msg += f" loss_cig={avg_loss_cig:.4f} pos_e={avg_pos_e:.4f} neg_e={avg_neg_e:.4f} gap={avg_gap:.4f} cig_on={avg_cig_active:.0%}"
                        log_payload["loss_cig"] = avg_loss_cig
                        log_payload["pos_energy"] = avg_pos_e
                        log_payload["neg_energy"] = avg_neg_e
                        log_payload["energy_gap"] = avg_gap
                    eff_batch = config.batch_size * accum_steps
                    log_msg += f" eff_batch={eff_batch} time={elapsed:.1f}s"
                    logging.info(log_msg)
                    if config.wandb_enabled:
                        wandb.log(log_payload, step=global_step)
                    start_time = time.time()
                    infos = []
                global_step += 1
                save_checkpoint(model, optim, global_step, config, is_main, data_config)
                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix(
                        {
                            "loss_il": f"{accum_loss_il / accum_steps:.4f}",
                            "lr": f"{optim.param_groups[0]['lr']:.2e}",
                            "step": global_step,
                        }
                    )
                micro_step = 0
                accum_loss_il = 0.0
                accum_loss_cig = 0.0
                accum_pos_energy = 0.0
                accum_neg_energy = 0.0
    if pbar is not None:
        pbar.close()
    if is_main and config.wandb_enabled:
        wandb.finish()
    cleanup_ddp()


def _tokenize_instructions(
    tokenizer, instructions: list[str], device
) -> tuple[torch.Tensor, torch.Tensor]:
    all_tokens = []
    all_masks = []
    for inst in instructions:
        tokens, masks = tokenizer.tokenize(inst)
        all_tokens.append(torch.from_numpy(tokens))
        all_masks.append(torch.from_numpy(masks))
    tokens_batch = torch.stack(all_tokens).to(device)
    masks_batch = torch.stack(all_masks).to(device)
    return (tokens_batch, masks_batch)


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
