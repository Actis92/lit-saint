from dataclasses import dataclass
from typing import Optional

from omegaconf import DictConfig


@dataclass
class CutMixConfig:
    """Define the parameters for CutMix augmentation"""
    lam: float  #: probability original values will be updated


@dataclass
class MixUpConfig:
    """Define the parameters for CutMix augmentation"""
    lam: float  #: weight used for the linear combination


@dataclass
class ConstrastiveSimConfig:
    """Define the parameters for ContrastiveSim pretraining task"""
    weight: float  #: weight of the loss for this pretraining task


@dataclass
class ConstrastiveConfig:
    """Define the parameters for Contrastive pretraining task"""
    projhead_style: str  #: can be same or different and it is used to project embeddings
    nce_temp: float  #: temperature used for the logits
    weight: float  #: weight of the loss for this pretraining task


@dataclass
class DenoisingConfig:
    """Define the parameters for Denoising pretraining task"""
    weight_cross_entropy: float  #: weight reconstruction loss for categorical features
    weight_mse: float  #: weight reconstruction loss for continuous features


@dataclass
class AugmentationConfig(DictConfig):
    """Define the parameters used for the augmentations"""
    cutmix: Optional[CutMixConfig] = None
    mixup: Optional[MixUpConfig] = None


@dataclass
class PreTrainTaskConfig(DictConfig):
    """Define the parameters used for pretraining tasks"""
    contrastive: Optional[ConstrastiveConfig] = None
    contrastive_sim: Optional[ConstrastiveSimConfig] = None
    denoising: Optional[DenoisingConfig] = None


@dataclass
class PreTrainConfig:
    """Define parameters for the steps used during the pretraining"""
    aug: AugmentationConfig
    task: PreTrainTaskConfig


@dataclass
class NetworkConfig:
    """Define the neural network parameters"""
    embedding_size: int  #: dimenstion of computed embeddings
    depth: int  #: number of attention blocks used in the transformer
    heads: int  #: number of heads used in the transformer
    ff_dropout: float  #: probability dropout in the feed forward layers
    attention_type: str  #: type of attention can be the self attention or intersample attention


@dataclass
class SaintConfig:
    """Define all the parameters used in SAINT"""
    network: NetworkConfig
    pretrain: PreTrainConfig
