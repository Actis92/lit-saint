import copy
from typing import List, Tuple, Optional, Callable, Dict

import torch
import torch.nn.functional as f
import torchmetrics
from einops import rearrange
from pytorch_lightning.core import LightningModule
from torch import nn, Tensor
import numpy as np
from torchmetrics import Metric

from lit_saint.config import SaintConfig
from lit_saint.modules import SimpleMLP, RowColTransformer, SepMLP
from lit_saint.augmentations import cutmix, mixup, get_random_index


class Saint(LightningModule):
    """Contains the definition of the network and how to execute the pretraining tasks, and how perform
    the training, validation and test steps

    :param categories: List with the number of unique values for each categorical column
    :param continuous: List of indices with continuous columns
    :param dim_target: if categorical represent number of classes of the target otherwise is 1
    :param config: configuration of the model
    :param metrics: Dictionary containing custom metrics to compute during training loop
    :param metrics_single_class: boolean flag, if True the metrics are computed separately for each class
    :param optimizer: custom optimizer to compute gradient of the network
    :param loss_fn: custom loss function to be optimized
    """
    def __init__(
            self,
            categories: List[int],
            continuous: List[int],
            dim_target: int,
            config: SaintConfig,
            metrics: Dict[str, Metric] = None,
            metrics_single_class: bool = True,
            optimizer: Callable = torch.optim.Adam,
            loss_fn: Callable = None,
    ):
        super().__init__()
        self.save_hyperparameters("categories", "continuous", "dim_target", "config")
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.mc_dropout = False
        self.config = config
        self.pretraining = False
        self.dim_target = dim_target
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_continuous = len(continuous) if len(continuous) > 0 else 1
        self.num_columns = self.num_continuous + self.num_categories
        # define offset in order to have unique value for each category
        cat_mask_offset = f.pad(torch.tensor(categories), (1, 0), value=0)
        self.cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
        self._define_network_components(categories)
        self.metrics_single_class = metrics_single_class
        self.metrics = metrics if metrics else {}
        self.train_metrics = self._define_metrics(self.metrics, metrics_single_class)
        self.val_metrics = self._define_metrics(self.metrics, metrics_single_class)

    def _define_metrics(self, metrics: Dict[str, Metric], metrics_single_class: bool) -> Dict[str, Metric]:
        """Define custom metrics computed during training loop

        :param metrics: Custom metrics to compute
        :param metrics_single_class: boolean flag, if True the metrics are computed separately for each class
        """
        metrics_step = {}
        if metrics_single_class:
            for key, value in metrics.items():
                for i in range(self.dim_target):
                    metrics_step[f"{key}_{i}"] = copy.deepcopy(value)
        else:
            for key, value in metrics.items():
                metrics_step[key] = copy.deepcopy(value)
        return metrics_step

    def _define_network_components(self, categories: List[int]) -> None:
        """Define the components for the neural network

        :param categories: Number of unique categories for each categorical column
        """
        self._define_embedding_layers()
        self._define_transformer()
        self._define_mlp(categories)
        self._define_projection_head()
        self.mlpfory = SimpleMLP(self.config.network.embedding_size * self.num_columns,
                                 self.config.train.internal_dimension_output_layer, self.dim_target,
                                 dropout=self.config.train.mlpfory_dropout)

    def _define_transformer(self) -> None:
        """Instantiate the type of Transformed that will be used in SAINT"""
        self.transformer = RowColTransformer(
            dim=self.config.network.embedding_size,
            nfeats=self.num_categories + self.num_continuous,
            depth=self.config.network.transformer.depth,
            heads=self.config.network.transformer.heads,
            dim_head=self.config.network.transformer.dim_head,
            ff_dropout=self.config.network.transformer.dropout,
            style=self.config.network.transformer.attention_type.value,
            scale_dim_internal_col=self.config.network.transformer.scale_dim_internal_col,
            scale_dim_internal_row=self.config.network.transformer.scale_dim_internal_row
        )

    def _define_embedding_layers(self) -> None:
        """Instatiate embedding layers"""
        # embed continuos variables using one different MLP for each column
        self.embedding_continuos = nn.ModuleList([SimpleMLP(1, self.config.network.internal_dimension_embed_continuous,
                                                            self.config.network.embedding_size,
                                                            dropout=self.config.network.dropout_embed_continuous)
                                                  for _ in range(self.num_continuous)])
        # embedding layer categorical columns
        self.embedding_categorical = nn.Embedding(self.num_unique_categories, self.config.network.embedding_size)

    def _define_mlp(self, categories: List[int]) -> None:
        """Define MLP used for the Denoising task"""
        self.mlp1 = SepMLP(dim=self.config.network.embedding_size, dim_out_for_each_feat=categories,
                           scale_dim_internal=self.config.pretrain.task.denoising.scale_dim_internal_sepmlp,
                           dropout=self.config.pretrain.task.denoising.dropout)
        self.mlp2 = SepMLP(dim=self.config.network.embedding_size,
                           dim_out_for_each_feat=list(np.ones(self.num_continuous).astype(int)),
                           scale_dim_internal=self.config.pretrain.task.denoising.scale_dim_internal_sepmlp,
                           dropout=self.config.pretrain.task.denoising.dropout)

    def _define_projection_head(self) -> None:
        """Define projection heads for contrastive learning"""
        self.pt_mlp = SimpleMLP(self.config.network.embedding_size * self.num_columns,
                                6 * self.config.network.embedding_size * self.num_columns // 5,
                                self.config.network.embedding_size * self.num_columns // 2,
                                dropout=self.config.pretrain.task.contrastive.dropout)
        self.pt_mlp2 = SimpleMLP(self.config.network.embedding_size * self.num_columns,
                                 6 * self.config.network.embedding_size * self.num_columns // 5,
                                 self.config.network.embedding_size * self.num_columns // 2,
                                 dropout=self.config.pretrain.task.contrastive.dropout)

    def _embed_data(self, x_categ: Tensor, x_cont: Tensor) -> Tuple[Tensor, Tensor]:
        """Converts categorical and continuous values in embeddings

        :param x_categ: contains the values for the categorical features
        :param x_cont: contains the values for the continuous features
        """
        x_categ = x_categ + self.cat_mask_offset.type_as(x_categ)
        x_categ_enc = self.embedding_categorical(x_categ)
        n1, n2 = x_cont.shape
        _, n3 = x_categ.shape
        x_cont_enc = torch.empty(n1, n2, self.config.network.embedding_size).to(x_categ_enc.device)
        for i in range(self.num_continuous):
            x_cont_enc[:, i, :] = self.embedding_continuos[i](x_cont[:, i])

        return x_categ_enc, x_cont_enc

    def _embeddings_contrastive(self, embed_categ: Tensor, embed_cont: Tensor, embed_categ_noised: Tensor,
                                embed_cont_noised: Tensor, projhead_style) -> Tuple[Tensor, Tensor]:
        """Given embeddings original and noised version of the categorical and continuous features are
        transformed using attention, normalized and eventually projected using MLP

        :param embed_categ: embeddings categorical features
        :param embed_cont: embeddings continuous features
        :param embed_categ_noised: embedding categorical features after the augmentation
        :param embed_cont_noised: embedding continuous features after the augmentation
        :param projhead_style: how project the values, can be same if use same MLP or different if use one MLP for
        the original values and another for the augmented ones
        """
        embed_tranformed = self.transformer(embed_categ, embed_cont)
        embed_transformed_noised = self.transformer(embed_categ_noised, embed_cont_noised)
        embed_tranformed = (embed_tranformed / embed_tranformed.norm(dim=-1, keepdim=True)).flatten(1, 2)
        embed_transformed_noised = (embed_transformed_noised /
                                    embed_transformed_noised.norm(dim=-1, keepdim=True)).flatten(1, 2)
        if projhead_style.value == 'different':
            embed_tranformed = self.pt_mlp(embed_tranformed)
            embed_transformed_noised = self.pt_mlp2(embed_transformed_noised)
        elif projhead_style.value == 'same':
            embed_tranformed = self.pt_mlp(embed_tranformed)
            embed_transformed_noised = self.pt_mlp(embed_transformed_noised)
        else:
            print('Not using projection head')
        return embed_tranformed, embed_transformed_noised

    def _pretraining_augmentation(self, x_categ: Tensor, x_cont: Tensor, embed_categ: Tensor,
                                  embed_cont: Tensor) -> Tuple[Tensor, Tensor]:
        """ It applies pretraining task, can be cutmix on the original values, and mixup that is applied
        instead on the embedded values

        :param x_categ: values of categorical features
        :param x_cont: values of continuous features
        :param embed_categ: embeddings categorical features
        :param embed_cont: embeddings continuous features
        """
        if self.config.pretrain.aug.cutmix:
            random_index = get_random_index(x_categ)
            x_categ_noised = cutmix(x_categ, random_index, self.config.pretrain.aug.cutmix.lam)
            x_cont_noised = cutmix(x_cont, random_index, self.config.pretrain.aug.cutmix.lam)
            embed_categ_noised, embed_cont_noised = self._embed_data(x_categ_noised, x_cont_noised)
        else:
            # if not apply cutmix the noised embeddings are equal to the original embeddings
            embed_categ_noised, embed_cont_noised = embed_categ, embed_cont

        if self.config.pretrain.aug.mixup:
            random_index = get_random_index(embed_categ_noised)
            embed_categ_noised = mixup(embed_categ_noised, random_index, lam=self.config.pretrain.aug.mixup.lam)
            embed_cont_noised = mixup(embed_cont_noised, random_index, lam=self.config.pretrain.aug.mixup.lam)
        return embed_categ_noised, embed_cont_noised

    def _pretraining_denoising(self, x_categ: Tensor, x_cont: Tensor, embed_categ_noised: Tensor,
                               embed_cont_noised: Tensor) -> Tensor:
        """Given the embeddings after the application of augmentation task it tries to reconstruct
        the original values, and compute the loss separating the continuous values, using MSE loss,
        from the categorical where is used the cross entropy loss. Then return the combination of the loss
        functions

        :param x_categ: values of categorical features
        :param x_cont: values of continuous features
        :param embed_categ_noised: embeddings categorical features after the augmentation
        :param embed_cont_noised: embeddings continuous features after the augmentation
        """
        embed_transformed_noised = self.transformer(embed_categ_noised, embed_cont_noised)
        output_categorical = self.mlp1(embed_transformed_noised[:, :self.num_categories, :])
        output_continuos = self.mlp2(embed_transformed_noised[:, self.num_categories:, :])
        output_continuos = torch.cat(output_continuos, dim=1)
        loss_continuos_columns = f.mse_loss(output_continuos, x_cont)
        loss_categorical_columns = Tensor([0]).to(embed_transformed_noised.device)
        # for each categorical column we compute the cross_entropy where in x_categ
        # there is the index of the categorical value obtained using the Label Encoding
        for j in range(self.num_categories - 1):
            loss_categorical_columns += f.cross_entropy(output_categorical[j], x_categ[:, j])
        return torch.mul(self.config.pretrain.task.denoising.weight_cross_entropy, loss_categorical_columns) +\
            torch.mul(self.config.pretrain.task.denoising.weight_mse, loss_continuos_columns)

    def _pretraining_contrastive(self, embed_categ: Tensor, embed_cont: Tensor, embed_categ_noised: Tensor,
                                 embed_cont_noised: Tensor) -> Tensor:
        """Using the contrastive learning the objective is that the embeddings of the original values,
        and the embeddings of the augmented ones must be similar

        :param embed_categ: embeddings original categorical features
        :param embed_cont: embeddings original continuous features
        :param embed_categ_noised: embeddings categorical features after the augmentation
        :param embed_cont_noised: embeddings continuous features after the augmentation
        """
        if self.config.pretrain.task.contrastive.constrastive_type.value == 'standard':
            embed_tranformed, embed_transformed_noised = self._embeddings_contrastive(
                embed_categ, embed_cont, embed_categ_noised, embed_cont_noised,
                self.config.pretrain.task.contrastive.projhead_style)
            # @ is the matrix multiplication operator
            logits_1 = embed_tranformed @ embed_transformed_noised.t() / self.config.pretrain.task.contrastive.nce_temp
            logits_2 = embed_transformed_noised @ embed_tranformed.t() / self.config.pretrain.task.contrastive.nce_temp
            # targets it's a list of numbers from zero to size of the tensor, in order to make that
            # z0 and z0' are similar, but z0 is different from the other indexes
            targets = torch.arange(logits_1.size(0))
            loss_1 = f.cross_entropy(logits_1, targets)
            loss_2 = f.cross_entropy(logits_2, targets)
            return self.config.pretrain.task.contrastive.weight * (loss_1 + loss_2) / 2
        elif self.config.pretrain.task.contrastive.constrastive_type.value == 'simsiam':
            # it apply the concept of simsiam we want that the embedding minimize the cosine similarity
            # the idea is that we want on the diagonal all 1, it means they are equal /because normalized)
            embed_tranformed, embed_transformed_noised = self._embeddings_contrastive(
                embed_categ, embed_cont, embed_categ_noised, embed_cont_noised,
                self.config.pretrain.task.contrastive.projhead_style)
            return - self.config.pretrain.task.contrastive.weight * \
                f.cosine_similarity(embed_tranformed, embed_transformed_noised).add_(-1).sum()

    def configure_optimizers(self):
        if self.pretraining:
            lr = self.config.pretrain.optimizer.learning_rate
            other_params = self.config.pretrain.optimizer.other_params
        else:
            lr = self.config.train.optimizer.learning_rate
            other_params = self.config.pretrain.optimizer.other_params

        optimizer = self.optimizer(self.parameters(), lr=lr, **other_params)
        return optimizer

    def forward(self, x_categ: Tensor, x_cont: Tensor) -> Tensor:
        x_categ_enc, x_cont_enc = self._embed_data(x_categ, x_cont)
        reps = self.transformer(x_categ_enc, x_cont_enc)
        reps = rearrange(reps, 'b h n -> b (h n)')
        y_outs = self.mlpfory(reps)
        return y_outs

    def pretraining_step(self, x_categ: Tensor, x_cont: Tensor) -> Tensor:
        """Defines all the step that must be sued in the pretraing step and return the computed loss

        :param x_categ: values of categorical features
        :param x_cont: values of continuous features
        """
        embed_categ, embed_cont = self._embed_data(x_categ, x_cont)
        loss = Tensor([0]).to(embed_categ.device)
        embed_categ_noised, embed_cont_noised = self._pretraining_augmentation(x_categ, x_cont,
                                                                               embed_categ, embed_cont)
        loss += self._pretraining_contrastive(embed_categ, embed_cont,
                                              embed_categ_noised, embed_cont_noised)
        if self.config.pretrain.task.denoising:
            loss += self._pretraining_denoising(x_categ, x_cont, embed_categ_noised, embed_cont_noised)
        return loss

    def shared_step(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """It define the commons step executed during training, validation and test

        :param batch: Contains a batch of data
        """
        x_categ, x_cont, target = batch
        if self.pretraining:
            return self.pretraining_step(x_categ, x_cont), torch.Tensor(), torch.Tensor()
        else:
            y_pred = self(x_categ, x_cont)
            if self.loss_fn:
                return self.loss_fn(y_pred, target), y_pred, target
            else:
                if self.dim_target > 1:
                    loss = self._classification_loss(y_pred, target)
                    return loss, y_pred, target
                else:
                    return self._regression_loss(y_pred, target), y_pred, target

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, y_pred, target = self.shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if not self.pretraining:
            self.compute_metrics(metrics=self.train_metrics, y_pred=y_pred, target=target)
            if len(self.train_metrics) > 0:
                self.log("train_metrics", self.train_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def compute_metrics(self, metrics: Dict[str, Metric], y_pred: Tensor, target: Tensor) -> None:
        """ Compute custom metrics during training loop

        :param metrics: Metrics to compute
        :param y_pred: predicted value
        :param target: true value
        """
        pred = nn.Softmax(dim=-1)(y_pred) if self.dim_target > 1 else y_pred.detach()
        for key, value in metrics.items():
            if self.metrics_single_class:
                index_class_metric = int(key.split("_")[-1])
                target_single_class = target[target == index_class_metric]
                if target_single_class.shape[0] > 0:
                    value(pred[target == index_class_metric], target[target == index_class_metric])
            else:
                value(pred, target)

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, y_pred, target = self.shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if not self.pretraining:
            self.compute_metrics(metrics=self.val_metrics, y_pred=y_pred, target=target)
            if len(self.val_metrics) > 0:
                self.log("val_metrics", self.val_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int) -> Tensor:
        loss, _, _ = self.shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int,
                     dataloader_idx: Optional[int] = None) -> Tensor:
        if self.mc_dropout:
            for m in self.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
        x_categ, x_cont, _ = batch
        y_pred = self(x_categ, x_cont)
        if self.dim_target > 1:
            return nn.Softmax(dim=-1)(y_pred)
        else:
            return y_pred

    @staticmethod
    def _classification_loss(y_pred: Tensor, target: Tensor) -> Tensor:
        """Loss function used in case of a classification problem

        :param y_pred: Values predicted
        :param target: Values to predict
        """
        return f.cross_entropy(y_pred, target)

    @staticmethod
    def _regression_loss(y_pred: Tensor, target: Tensor) -> Tensor:
        """Loss function used in case of regression problem

        :param y_pred: Values predicted
        :param target: Values to predict
        """
        return f.mse_loss(y_pred, target)

    def set_pretraining(self, pretraining: bool):
        self.pretraining = pretraining

    def set_mcdropout(self, mc_dropout: bool):
        self.mc_dropout = mc_dropout
