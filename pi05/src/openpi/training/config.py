import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.robocasa_policy as robocasa_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    assets_dir: str | None = None
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    repo_id: str | None = None
    asset_id: str | None = None
    norm_stats: dict[str, _transforms.NormStats] | None = None
    repack_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    data_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    model_transforms: _transforms.Group = dataclasses.field(
        default_factory=_transforms.Group
    )
    use_quantile_norm: bool = False
    action_sequence_keys: Sequence[str] = ("actions",)
    prompt_from_task: bool = False


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        pass


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ]
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        pass

    def create_base_config(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(
                epath.Path(self.assets.assets_dir or assets_dirs), asset_id
            ),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotRoboCasaDataConfig(DataConfigFactory):
    exclude_base_movement: bool = False

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        base_config = self.create_base_config(assets_dirs, model_config)
        constant_action_dims = {}
        output_values = {}
        if base_config.norm_stats is not None:
            constant_action_dims, output_values = (
                robocasa_policy.detect_constant_action_dims(base_config.norm_stats)
            )
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.base_0_rgb",
                        "observation/image_right": "observation.images.base_1_rgb",
                        "observation/wrist_image": "observation.images.wrist_0_rgb",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[
                robocasa_policy.RoboCasaInputs(
                    model_type=model_config.model_type,
                    constant_action_dims=constant_action_dims,
                    exclude_base_movement=self.exclude_base_movement,
                )
            ],
            outputs=[
                robocasa_policy.RoboCasaOutputs(
                    output_values=output_values,
                    exclude_base_movement=self.exclude_base_movement,
                )
            ],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            base_config,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class CIGConfig:
    enabled: bool = False
    lambda_weight: float = 0.1
    tau_c: float = 1.0
    num_negatives: int = 4
    instruction_pool_path: str | None = None
    cig_floor: float = 0.5
    hard_negatives: bool = True
    hard_neg_same_ratio: float = 0.5
    hard_neg_cross_ratio: float = 0.25


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    name: tyro.conf.Suppress[str]
    project_name: str = "robocasa-kitchen"
    exp_name: str = tyro.MISSING
    model: _model.BaseModelConfig = dataclasses.field(
        default_factory=pi0_config.Pi0Config
    )
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(
        default_factory=weight_loaders.NoOpWeightLoader
    )
    pytorch_weight_path: str | None = None
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(
        default_factory=_optimizer.CosineDecaySchedule
    )
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(
        default_factory=_optimizer.AdamW
    )
    ema_decay: float | None = 0.99
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(
        default_factory=nnx.Nothing
    )
    data: DataConfigFactory = dataclasses.field(
        default_factory=LeRobotRoboCasaDataConfig
    )
    assets_base_dir: str = "./assets"
    checkpoint_base_dir: str = "./checkpoints"
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 2
    num_train_steps: int = 30000
    log_interval: int = 100
    save_interval: int = 1000
    keep_period: int | None = 5000
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    policy_metadata: dict[str, Any] | None = None
    fsdp_devices: int = 1
    gradient_accumulation_steps: int = 1
    cig: CIGConfig = dataclasses.field(default_factory=CIGConfig)

    @property
    def assets_dirs(self) -> pathlib.Path:
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (
            pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name
        ).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli(
        {k: (k, v) for k, v in _CONFIGS_DICT.items()}
    )


def get_config(config_name: str) -> TrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(
            config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0
        )
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")
    return _CONFIGS_DICT[config_name]


def _release_config(name, cig):
    return TrainConfig(
        name=name,
        exp_name=name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10),
        data=LeRobotRoboCasaDataConfig(
            repo_id="robocasa-kitchen",
            assets=AssetsConfig(asset_id="robocasa-kitchen"),
            base_config=DataConfig(
                prompt_from_task=True, action_sequence_keys=("action",)
            ),
            exclude_base_movement=True,
        ),
        batch_size=128 if cig else 64,
        ema_decay=None if cig else 0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        num_train_steps=5000 if cig else 15000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=5e-05, decay_steps=100000, decay_lr=5e-05
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        cig=CIGConfig(
            enabled=cig,
            lambda_weight=0.1,
            tau_c=1.0,
            num_negatives=4,
            cig_floor=0.5,
            hard_negatives=True,
            hard_neg_same_ratio=0.5,
            hard_neg_cross_ratio=0.25,
            instruction_pool_path="../data/instruction_pool.json",
        ),
        save_interval=1000,
        wandb_enabled=False,
    )


_CONFIGS = [_release_config("pi05-baseline", False), _release_config("pi05-cig", True)]
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}
