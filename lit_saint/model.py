from typing import List, Tuple, Optional

import torch
import torch.nn.functional as f
from pytorch_lightning.core import LightningModule
from torch import nn, Tensor
import numpy as np

from lit_saint.config import SaintConfig
from lit_saint.modules import SimpleMLP, RowColTransformer, SepMLP
from lit_saint.augmentations import add_noise, mixup_data


class SAINT(LightningModule):
    """
    :param categories: List with the number of unique values for each categorical column
    :param num_continuos: number of continuos columns
    :param embedding_size: embedding dimension
    :param attentiontype: type of attention that can be used. possible values: col, row, colrow
    """
    def __init__(
            self,
            categories: List[int],
            num_continuous: int,
            config: SaintConfig,
            pretraining: bool = False,
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'
        # categories related calculations
        self.config = config
        self.pretraining = pretraining
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)
        self.num_continuous = num_continuous
        self.num_columns = self.num_continuous + self.num_categories
        # define offset in order to have unique value for each category
        cat_mask_offset = f.pad(torch.tensor(categories), (1, 0), value=0)
        self.cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]
        self._define_network_components(categories)

    def _define_network_components(self, categories):
        self._define_embedding_layers()
        self._define_transformer()
        self._define_mlp(categories)
        self._define_projection_head()
        self.mlpfory = SimpleMLP(self.config.network.embedding_size, 1000, 2)

    def _define_transformer(self):
        """Instatiate the type of Transformed that will be used in SAINT"""
        self.transformer = RowColTransformer(
            dim=self.config.network.embedding_size,
            nfeats=self.num_categories + self.num_continuous,
            depth=self.config.network.depth,
            heads=self.config.network.heads,
            ff_dropout=self.config.network.ff_dropout,
            style=self.config.network.attention_type
        )

    def _define_embedding_layers(self) -> None:
        """Instatiate embedding layers"""
        # embed continuos variables using one different MLP for each column
        self.embedding_continuos = nn.ModuleList([SimpleMLP(1, 100, self.config.network.embedding_size)
                                                  for _ in range(self.num_continuous)])
        # embedding layer categorical columns
        self.embedding_categorical = nn.Embedding(self.num_unique_categories, self.config.network.embedding_size)

    def _define_mlp(self, categories: List[int]) -> None:
        """Define MLP used for the inference"""
        self.mlp1 = SepMLP(dim=self.config.network.embedding_size, dim_out_for_each_feat=categories)
        self.mlp2 = SepMLP(dim=self.config.network.embedding_size,
                           dim_out_for_each_feat=list(np.ones(self.num_continuous).astype(int)))

    def _define_projection_head(self) -> None:
        """Define projection heads for contrastive learning"""
        self.pt_mlp = SimpleMLP(self.config.network.embedding_size * self.num_columns,
                                  6 * self.config.network.embedding_size * self.num_columns // 5,
                                  self.config.network.embedding_size * self.num_columns // 2)
        self.pt_mlp2 = SimpleMLP(self.config.network.embedding_size * self.num_columns,
                                  6 * self.config.network.embedding_size * self.num_columns // 5,
                                  self.config.network.embedding_size * self.num_columns // 2)

    def _embed_data(self, x_categ: Tensor, x_cont: Tensor) -> Tuple[Tensor, Tensor]:
        """Converts categorical and continuos values in embeddings"""
        x_categ = x_categ + self.cat_mask_offset.type_as(x_categ)
        # mask the target
        x_categ[:, -1] = torch.zeros(x_categ.shape[0])
        x_categ_enc = self.embedding_categorical(x_categ)
        n1, n2 = x_cont.shape
        _, n3 = x_categ.shape
        x_cont_enc = torch.empty(n1, n2, self.config.network.embedding_size)
        for i in range(self.num_continuous):
            x_cont_enc[:, i, :] = self.embedding_continuos[i](x_cont[:, i])
        x_cont_enc = x_cont_enc

        return x_categ_enc, x_cont_enc

    def _contrastive(self, embed_categ: Tensor, embed_cont: Tensor, embed_categ_noised: Tensor,
                     embed_cont_noised: Tensor, projhead_style: str = "different") -> Tuple[Tensor, Tensor]:
        embed_tranformed = self.transformer(embed_categ, embed_cont)
        embed_transformed_noised = self.transformer(embed_categ_noised, embed_cont_noised)
        embed_tranformed = (embed_tranformed / embed_tranformed.norm(dim=-1, keepdim=True)).flatten(1, 2)
        embed_transformed_noised = (embed_transformed_noised / embed_transformed_noised.norm(dim=-1, keepdim=True)).flatten(1, 2)
        if projhead_style == 'different':
            embed_tranformed = self.pt_mlp(embed_tranformed)
            embed_transformed_noised = self.pt_mlp2(embed_transformed_noised)
        elif projhead_style == 'same':
            embed_tranformed = self.pt_mlp(embed_tranformed)
            embed_transformed_noised = self.pt_mlp(embed_transformed_noised)
        else:
            print('Not using projection head')
        return embed_tranformed, embed_transformed_noised

    def _pretraining_augmentation(self, x_categ: Tensor, x_cont: Tensor, embed_categ: Tensor,
                                  embed_cont: Tensor) -> Tuple[Tensor, Tensor]:
        if self.config.pretrain.aug.get('cutmix'):
            x_categ_noised, x_cont_noised = add_noise(x_categ, x_cont, self.config.pretrain.aug.cutmix.noise_lambda)
            embed_categ_noised, embed_cont_noised = self._embed_data(x_categ_noised, x_cont_noised)
        else:
            embed_categ_noised, embed_cont_noised = embed_categ, embed_cont

        if self.config.pretrain.aug.get("mixup"):
            embed_categ_noised, embed_cont_noised = mixup_data(embed_categ_noised, embed_cont_noised,
                                                               lam=self.config.pretrain.aug.mixup.lam)
        return embed_categ_noised, embed_cont_noised

    def _pretraining_denoising(self, x_categ: Tensor, x_cont: Tensor, embed_categ_noised: Tensor,
                               embed_cont_noised: Tensor) -> Tensor:
        embed_transformed_noised = self.transformer(embed_categ_noised, embed_cont_noised)
        output_categorical = self.mlp1(embed_transformed_noised[:, :self.num_categories, :])
        output_continuos = self.mlp2(embed_transformed_noised[:, self.num_categories:, :])
        output_continuos = torch.cat(output_continuos, dim=1)
        loss_continuos_columns = f.mse_loss(output_continuos, x_cont)
        loss_categorical_columns = Tensor([0])
        # for each categorical column we compute the cross_entropy where in x_categ
        # there is the index of the categorical value obtained using the Label Encoding
        for j in range(self.num_categories - 1):
            loss_categorical_columns += f.cross_entropy(output_categorical[j], x_categ[:, j])
        return torch.mul(self.config.pretrain.task.denoising.weight_cross_entropy, loss_categorical_columns) +\
               torch.mul(self.config.pretrain.task.denoising.weight_mse, loss_continuos_columns)

    def _pretraining_contrastive(self, embed_categ: Tensor, embed_cont: Tensor, embed_categ_noised: Tensor,
                                 embed_cont_noised: Tensor) -> Tensor:
        if self.config.pretrain.task.get("contrastive"):
            embed_tranformed, embed_transformed_noised = self._contrastive(embed_categ, embed_cont,
                                                               embed_categ_noised, embed_cont_noised,
                                                               self.config.pretrain.task.contrastive.projhead_style)
            # @ is the matrix multiplication operator
            logits_1 = embed_tranformed @ embed_transformed_noised.t() / self.config.pretrain.task.contrastive.nce_temp
            logits_2 = embed_transformed_noised @ embed_tranformed.t() / self.config.pretrain.task.contrastive.nce_temp
            # targets it's a list of numbers from zero to size of the tensor, in order to make that
            # z0 and z0' are similar, but z0 is different from the other indexes
            targets = torch.arange(logits_1.size(0))
            loss_1 = f.cross_entropy(logits_1, targets)
            loss_2 = f.cross_entropy(logits_2, targets)
            return self.config.pretrain.task.contrastive.lam * (loss_1 + loss_2) / 2
        elif self.config.pretrain.task.get("contrastive_sim"):
            # it apply the concept of simsiam we want that the embedding minimize the cosine similarity
            # the idea is that we want on the diagonal all 1, it means they are equal /because normalized)
            embed_tranformed, embed_transformed_noised = self._contrastive(embed_categ, embed_cont,
                                                               embed_categ_noised, embed_cont_noised)
            c1 = embed_tranformed @ embed_transformed_noised.t()
            return self.config.pretrain.task.contrastive_sim.weight * torch.diagonal(-1 * c1).add_(1).pow_(2).sum()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)

    def forward(self, x_categ: Tensor, x_cont: Tensor) -> Tensor:
        x_categ_enc, x_cont_enc = self._embed_data(x_categ, x_cont)
        reps = self.transformer(x_categ_enc, x_cont_enc)
        # select only the representations corresponding to y and apply mlp on it
        # in the next step to get the predictions.
        y_reps = reps[:, self.num_categories - 1, :]
        y_outs = self.mlpfory(y_reps)
        return y_outs

    def pretraining_step(self, x_categ: Tensor, x_cont: Tensor) -> Tensor:
        loss = Tensor([0])
        embed_categ, embed_cont = self._embed_data(x_categ, x_cont)
        embed_categ_noised, embed_cont_noised = self._pretraining_augmentation(x_categ, x_cont,
                                                                               embed_categ, embed_cont)
        loss += self._pretraining_contrastive(embed_categ, embed_cont,
                                              embed_categ_noised, embed_cont_noised)
        if self.config.pretrain.task.get("denoising"):
            loss += self._pretraining_denoising(x_categ, x_cont, embed_categ_noised, embed_cont_noised)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x_categ, x_cont = batch
        loss = Tensor([0])
        if self.pretraining:
            loss += self.pretraining_step(x_categ, x_cont)
        else:
            y_pred = self(x_categ, x_cont)
            loss += f.cross_entropy(y_pred, x_categ[:, self.num_categories - 1])
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_categ, x_cont = batch
        loss = Tensor([0])
        if self.pretraining:
            loss += self.pretraining_step(x_categ, x_cont)
        else:
            y_pred = self(x_categ, x_cont)
            loss += f.cross_entropy(y_pred, x_categ[:, self.num_categories - 1])
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x_categ, x_cont = batch
        loss = Tensor([0])
        if self.pretraining:
            loss += self.pretraining_step(x_categ, x_cont)
        else:
            y_pred = self(x_categ, x_cont)
            loss += f.cross_entropy(y_pred, x_categ[:, self.num_categories - 1])
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def predict_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: Optional[int] = None) -> Tensor:
        x_categ, x_cont = batch
        y_pred = self(x_categ, x_cont)
        return y_pred.argmax(1)

