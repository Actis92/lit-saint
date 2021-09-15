from typing import List

import torch
import torch.nn.functional as f
from pytorch_lightning.core import LightningModule
from torch import nn
import numpy as np

from lit_saint.config import SaintConfig
from lit_saint.modules import SimpleMLP, Transformer, RowColTransformer, SepMLP
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
        self.total_tokens = self.num_unique_categories
        self.num_continuous = num_continuous
        nfeats = self.num_categories + num_continuous
        self._define_masking()
        self._define_transformer(nfeats)
        self._define_mlp(categories)
        self._projection_head()
        self.mlpfory = SimpleMLP(self.config.network.embedding_size, 1000, 2)

    def _define_transformer(self, nfeats):
        if self.config.network.attention_type == 'col':
            self.transformer = Transformer(
                dim=self.config.network.embedding_size,
                depth=self.config.network.depth,
                heads=self.config.network.heads,
                attn_dropout=self.config.network.attn_dropout,
                ff_dropout=self.config.network.ff_dropout
            )
        elif self.config.network.attention_type in ['row', 'colrow']:
            self.transformer = RowColTransformer(
                num_tokens=self.total_tokens,
                dim=self.config.network.embedding_size,
                nfeats=nfeats,
                depth=self.config.network.depth,
                heads=self.config.network.heads,
                attn_dropout=self.config.network.attn_dropout,
                ff_dropout=self.config.network.ff_dropout,
                style=self.config.network.attention_type
            )

    def _define_masking(self):
        self.simple_MLP = nn.ModuleList([SimpleMLP(1, 100, self.config.network.embedding_size)
                                         for _ in range(self.num_continuous)])
        self.embeds = nn.Embedding(self.total_tokens, self.config.network.embedding_size)
        self.mask_embeds_cat = nn.Embedding(self.num_categories * 2, self.config.network.embedding_size)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous * 2, self.config.network.embedding_size)
        cat_mask_offset = f.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value=0)
        self.cat_mask_offset = cat_mask_offset.cumsum(dim=-1)[:-1]

        con_mask_offset = f.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value=0)
        self.con_mask_offset = con_mask_offset.cumsum(dim=-1)[:-1]

    def _define_mlp(self, categories):
        self.mlp1 = SepMLP(dim=self.config.network.embedding_size, len_feats=self.num_categories, categories=categories)
        self.mlp2 = SepMLP(dim=self.config.network.embedding_size, len_feats=self.num_continuous,
                           categories=np.ones(self.num_continuous).astype(int))

    def _embed_data(self, x_categ, x_cont):
        x_categ = x_categ + self.cat_mask_offset.type_as(x_categ)
        x_categ_enc = self.embeds(x_categ)
        n1, n2 = x_cont.shape
        _, n3 = x_categ.shape
        x_cont_enc = torch.empty(n1, n2, self.config.network.embedding_size)
        for i in range(self.num_continuous):
            x_cont_enc[:, i, :] = self.simple_MLP[i](x_cont[:, i])
        x_cont_enc = x_cont_enc

        return x_categ_enc, x_cont_enc

    def _projection_head(self):
        self.pt_mlp = SimpleMLP(self.config.network.embedding_size * (self.num_continuous + self.num_categories),
                                  6 * self.config.network.embedding_size * (self.num_continuous + self.num_categories) // 5,
                                  self.config.network.embedding_size * (self.num_continuous + self.num_categories) // 2)
        self.pt_mlp2 = SimpleMLP(self.config.network.embedding_size * (self.num_continuous + self.num_categories),
                                  6 * self.config.network.embedding_size * (self.num_continuous + self.num_categories) // 5,
                                  self.config.network.embedding_size * (self.num_continuous + self.num_categories) // 2)

    def forward(self, x_categ, x_cont):
        x = self.transformer(x_categ, x_cont)
        cat_outs = self.mlp1(x[:, :self.num_categories, :])
        con_outs = self.mlp2(x[:, self.num_categories:, :])
        return cat_outs, con_outs

    def training_step(self, batch, batch_idx):
        x_categ, x_cont = batch
        if self.pretraining:
            loss = 0
            embed_categ, embed_cont = self._embed_data(x_categ, x_cont)
            embed_categ_noised, embed_cont_noised = self._pretraining_augmentation(x_categ, x_cont,
                                                                                   embed_categ, embed_cont)
            loss += self._pretraining_contrastive(embed_categ, embed_cont,
                                                  embed_categ_noised, embed_cont_noised)
            if self.config.pretrain.task.get("denoising"):
                loss += self._pretraining_denoising(x_categ, x_cont, embed_categ_noised, embed_cont_noised)
            self.log("loss", loss)
            return loss
        else:
            x_categ_enc, x_cont_enc = self._embed_data(x_categ, x_cont)
            reps = self.transformer(x_categ_enc, x_cont_enc)
            # select only the representations corresponding to y and apply mlp on it in the next step to get the predictions.
            y_reps = reps[:, self.num_categories - 1, :]
            y_outs = self.mlpfory(y_reps)
            return f.cross_entropy(y_outs, x_categ[:, self.num_categories - 1])

    def _contrastive(self, embed_categ, embed_cont, embed_categ_noised, embed_cont_noised, projhead_style="different"):
        aug_features_1 = self.transformer(embed_categ, embed_cont)
        aug_features_2 = self.transformer(embed_categ_noised, embed_cont_noised)
        aug_features_1 = (aug_features_1 / aug_features_1.norm(dim=-1, keepdim=True)).flatten(1, 2)
        aug_features_2 = (aug_features_2 / aug_features_2.norm(dim=-1, keepdim=True)).flatten(1, 2)
        if projhead_style == 'different':
            aug_features_1 = self.pt_mlp(aug_features_1)
            aug_features_2 = self.pt_mlp2(aug_features_2)
        elif projhead_style == 'same':
            aug_features_1 = self.pt_mlp(aug_features_1)
            aug_features_2 = self.pt_mlp(aug_features_2)
        else:
            print('Not using projection head')
        return aug_features_1, aug_features_2

    def _pretraining_augmentation(self, x_categ, x_cont, embed_categ, embed_cont):
        if self.config.pretrain.aug.get('cutmix'):
            x_categ_noised, x_cont_noised = add_noise(x_categ, x_cont, self.config.pretrain.aug.cutmix.noise_lambda)
            embed_categ_noised, embed_cont_noised = self._embed_data(x_categ_noised, x_cont_noised)
        else:
            embed_categ_noised, embed_cont_noised = embed_categ, embed_cont

        if self.config.pretrain.aug.get("mixup"):
            embed_categ_noised, embed_cont_noised = mixup_data(embed_categ_noised, embed_cont_noised,
                                                               lam=self.config.pretrain.aug.mixup.lam)
        return embed_categ_noised, embed_cont_noised

    def _pretraining_denoising(self, x_categ, x_cont, embed_categ_noised, embed_cont_noised):
        cat_outs, con_outs = self(embed_categ_noised, embed_cont_noised)
        con_outs = torch.cat(con_outs, dim=1)
        loss_continuos_columns = f.mse_loss(con_outs, x_cont)
        loss_categorical_columns = 0
        for j in range(self.num_categories - 1):
            loss_categorical_columns += f.cross_entropy(cat_outs[j], x_categ[:, j])
        return self.config.pretrain.task.denoising.weight_cross_entropy * loss_categorical_columns + \
                self.config.pretrain.task.denoising.weight_mse * loss_continuos_columns

    def _pretraining_contrastive(self, embed_categ, embed_cont, embed_categ_noised, embed_cont_noised):
        if self.config.pretrain.task.get("contrastive"):
            aug_features_1, aug_features_2 = self._contrastive(embed_categ, embed_cont,
                                                               embed_categ_noised, embed_cont_noised,
                                                               self.config.pretrain.task.contrastive.projhead_style)
            # @ is the matrix multiplication operator
            logits_per_aug1 = aug_features_1 @ aug_features_2.t() / self.config.pretrain.task.contrastive.nce_temp
            logits_per_aug2 = aug_features_2 @ aug_features_1.t() / self.config.pretrain.task.contrastive.nce_temp
            targets = torch.arange(logits_per_aug1.size(0))
            loss_1 = f.cross_entropy(logits_per_aug1, targets)
            loss_2 = f.cross_entropy(logits_per_aug2, targets)
            return self.config.pretrain.task.contrastive.lam * (loss_1 + loss_2) / 2
        elif self.config.pretrain.task.get("contrastive_sim"):
            aug_features_1, aug_features_2 = self._contrastive(embed_categ, embed_cont,
                                                               embed_categ_noised, embed_cont_noised)
            c1 = aug_features_1 @ aug_features_2.t()
            return self.config.pretrain.task.contrastive_sim.weight * torch.diagonal(-1 * c1).add_(1).pow_(2).sum()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.02)
