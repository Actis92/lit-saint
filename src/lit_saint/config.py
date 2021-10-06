import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CutMixConfig:
    """Define the parameters for CutMix augmentation"""
    lam: float = 0.1  #: probability original values will be updated


@dataclass
class MixUpConfig:
    """Define the parameters for CutMix augmentation"""
    lam: float = 0.1  #: weight used for the linear combination


@dataclass
class ConstrastiveSimConfig:
    """Define the parameters for ContrastiveSim pretraining task"""
    weight: float = 0.1  #: weight of the loss for this pretraining task


@dataclass
class ConstrastiveConfig:
    """Define the parameters for Contrastive pretraining task"""
    projhead_style: str = 'same'  #: can be same or different and it is used to project embeddings
    nce_temp: float = 0.5  #: temperature used for the logits
    weight: float = 0.1  #: weight of the loss for this pretraining task


@dataclass
class DenoisingConfig:
    """Define the parameters for Denoising pretraining task"""
    weight_cross_entropy: float = 0.5  #: weight reconstruction loss for categorical features
    weight_mse: float = 0.5  #: weight reconstruction loss for continuous features


@dataclass
class AugmentationConfig:
    """Define the parameters used for the augmentations"""
    cutmix: Optional[CutMixConfig] = CutMixConfig()
    mixup: Optional[MixUpConfig] = MixUpConfig()


@dataclass
class PreTrainTaskConfig:
    """Define the parameters used for pretraining tasks"""
    contrastive: Optional[ConstrastiveConfig] = None
    contrastive_sim: Optional[ConstrastiveSimConfig] = ConstrastiveSimConfig()
    denoising: Optional[DenoisingConfig] = DenoisingConfig()


@dataclass
class PreTrainConfig:
    """Define parameters for the steps used during the pretraining"""
    aug: AugmentationConfig = AugmentationConfig()
    task: PreTrainTaskConfig = PreTrainTaskConfig()


@dataclass
class NetworkConfig:
    """Define the neural network parameters"""
    num_workers: int = os.cpu_count()  #: number of cores to use
    embedding_size: int = 10  #: dimension of computed embeddings
    depth: int = 3  #: number of attention blocks used in the transformer
    heads: int = 1  #: number of heads used in the transformer
    ff_dropout: float = .0  #: probability dropout in the feed forward layers
    attention_type: str = 'col'  #: type of attention can be the self attention or intersample attention
    learning_rate: float = 0.03  #: value used to specify the learning rate for the optimizer
    learning_rate_pretraining: float = 0.03  #: value used to specify the learning rate for the optimizer pretraining
    internal_dimension_output_layer: int = 20  #: internal dimension of the MLP that compute the output
    internal_dimension_embed_continuous: int = 100  #: internal dimension of the mlp used to project continuous columns


@dataclass
class SaintConfig:
    """Define all the parameters used in SAINT"""
    network: NetworkConfig = NetworkConfig()
    pretrain: PreTrainConfig = PreTrainConfig()
