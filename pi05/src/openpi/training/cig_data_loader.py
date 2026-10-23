import json
import logging
import jax
import numpy as np
import torch
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data


def _is_numeric_array(x):
    if isinstance(x, np.ndarray):
        return np.issubdtype(x.dtype, np.number) or np.issubdtype(x.dtype, np.bool_)
    return True


class CIGDataLoaderImpl:
    def __init__(
        self, data_config: _config.DataConfig, pytorch_dataloader, num_batches=None
    ):
        self._data_config = data_config
        self._pytorch_dataloader = pytorch_dataloader
        self._num_batches = num_batches

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._pytorch_dataloader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                num_items += 1
                positive_prompts = [str(p) for p in batch.pop("positive_prompt")]
                cf_instructions_raw = None
                if "counterfactual_instructions" in batch:
                    cf_instructions_raw = batch.pop("counterfactual_instructions")
                if "counterfactual_levels" in batch:
                    batch.pop("counterfactual_levels")

                def safe_to_tensor(x):
                    if (
                        isinstance(x, np.ndarray)
                        and (not np.issubdtype(x.dtype, np.number))
                        and (not np.issubdtype(x.dtype, np.bool_))
                    ):
                        return x
                    return torch.as_tensor(x)

                batch = jax.tree.map(safe_to_tensor, batch)
                cf_instructions_batch = []
                if cf_instructions_raw is not None:
                    for item in cf_instructions_raw:
                        if isinstance(item, bytes):
                            item = item.decode("utf-8")
                        if isinstance(item, np.ndarray):
                            item = str(item.item()) if item.ndim == 0 else str(item)
                        try:
                            instructions = json.loads(item)
                        except (json.JSONDecodeError, TypeError):
                            instructions = []
                        cf_instructions_batch.append(instructions)
                else:
                    bs = batch.get("state", batch.get("actions", None))
                    batch_size = bs.shape[0] if bs is not None else 1
                    cf_instructions_batch = [[] for _ in range(batch_size)]
                observation = _model.Observation.from_dict(batch)
                actions = batch["actions"]
                yield (observation, actions, cf_instructions_batch, positive_prompts)


def create_cig_data_loader(
    config: _config.TrainConfig, *, shuffle: bool = True
) -> tuple[CIGDataLoaderImpl, _config.DataConfig]:
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"CIG data_config: {data_config}")
    dataset = _data.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    dataset = _data.transform_dataset(dataset, data_config, preserve_prompt=True)
    sampler = None
    if torch.distributed.is_initialized():
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=shuffle,
            drop_last=True,
        )
        local_batch_size = config.batch_size // torch.distributed.get_world_size()
    else:
        local_batch_size = config.batch_size
    logging.info(f"CIG local_batch_size: {local_batch_size}")
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    pytorch_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=sampler is None and shuffle,
        sampler=sampler,
        collate_fn=_data._collate_fn,
        num_workers=config.num_workers,
        drop_last=True,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )
    return (CIGDataLoaderImpl(data_config, pytorch_loader), data_config)
