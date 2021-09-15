from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig


@dataclass
class PlaceHolders:
    place_1: float

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
class NetworkConfig:
    embedding_size: int
    depth: int
    heads: int
    attn_dropout: float
    ff_dropout: float
    attention_type: str


@dataclass
class SaintConfig:
    network: NetworkConfig
    pretrain: PreTrainConfig
    place: PlaceHolders



