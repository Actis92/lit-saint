from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from omegaconf import DictConfig


@dataclass
class PlaceHolders:
    place_1 : float

@dataclass
class CutMixConfig:
    noise_lambda: float


@dataclass
class MixUpConfig:
    lam: float


@dataclass
class ConstrastiveSimConfig:
    weight: float


@dataclass
class ConstrastiveConfig:
    projhead_style: str
    nce_temp: float
    lam: float


@dataclass
class DenoisingConfig:
    weight_cross_entropy: float
    weight_mse: float


@dataclass
class AugmentationConfig(DictConfig):
    cutmix: Optional[CutMixConfig]
    mixup: Optional[MixUpConfig]


@dataclass
class PreTrainTaskConfig(DictConfig):
    contrastive: Optional[ConstrastiveConfig]
    contrastive_sim: Optional[ConstrastiveSimConfig]
    denoising: Optional[DenoisingConfig]


@dataclass
class PreTrainConfig:
    place: PlaceHolders
    aug: AugmentationConfig
    task: PreTrainTaskConfig


@dataclass
class SaintConfig:
    pretrain: PreTrainConfig
    place: PlaceHolders


defaults = [
]
@dataclass
class TransformConfig:
    __class_fullname__: str
    bbox_params: Dict
    transforms: List[Any] = field(default_factory=lambda: [])

@dataclass
class TransformsConfig:
    train: Dict = field(default_factory=lambda: {"transform":TransformConfig})
    val: Dict = field(default_factory=lambda: {"transform":TransformConfig})
    predict: Dict = field(default_factory=lambda: {"transform":TransformConfig})
    test: Dict = field(default_factory=lambda: {"transform": TransformConfig})

@dataclass
class Config:
    transforms: TransformsConfig


