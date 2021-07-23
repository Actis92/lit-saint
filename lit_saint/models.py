from typing import Any

import torch
from einops import rearrange
from pytorch_lightning.core import LightningModule
from torch import nn


class Residual(LightningModule):
    def __init__(self, fn, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.fn = fn

    def forward(self, x, *args, **kwargs) -> Any:
        return self.fn(x, *args, **kwargs) + x


class PreNorm(LightningModule):
    def __init__(self, dim, fn, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.dim = dim
        self.fn = fn

    def forward(self, x, *args, **kwargs) -> Any:
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(LightningModule):
    def __init__(self, dim, mult=4, dropout=0., *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, *args, **kwargs) -> Any:
        return self.net(x)


class RowColTransformer(LightningModule):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, attn_dropout, ff_dropout,
                 style='col', *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                                                dropout=attn_dropout))),
                    PreNorm(dim, Residual(FeedForward(dim, dropout=ff_dropout))),
                    PreNorm(dim*nfeats, Residual(nn.MultiheadAttention(embed_dim=dim, num_heads=heads,
                                                                       dropout=attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout=ff_dropout))),
                ]))
            else:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(nn.MultiheadAttention(embed_dim=dim*nfeats, num_heads=heads,
                                                                       dropout=attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout=ff_dropout))),
                ]))

    def forward(self, x, x_cont=None, mask=None):
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2 in self.layers:
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn2(x)
                x = ff2(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        return x

