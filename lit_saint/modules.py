from typing import Any, List

import torch
import torch.nn.functional as f
from einops import rearrange
from torch import nn, einsum, Tensor


class Residual(nn.Module):
    """Define a residual block given a Pytorch Module, used for the skip connection"""
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    """Apply Layer Normalization before a Module"""
    def __init__(self, dim, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        return self.fn(self.norm(x), *args, **kwargs)


class GEGLU(nn.Module):
    """Gated GELU, it splits a tensor in two slices based on the last dimension, and then multiply the
       first half and the gelu of the second half
    """
    def forward(self, x: Tensor) -> Tensor:
        x, gates = x.chunk(2, dim=-1)
        return x * f.gelu(gates)


class Attention(nn.Module):
    """Module that implements self attention
    :param dim: dimension embedding used in input
    :param heads: number of heads for the Multi Head Attention
    :param dim_head: output dimension given an head
    """
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        # create query, key value from x
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        # split query, key, value for each head adding a new dimension in the tensors
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        # product matrix between q and k transposed
        logits = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = logits.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        # concatenate result obtained by different heads
        out = rearrange(out, 'b h n d -> b n (h d)', h=self.heads)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    """Module use to define RowColTransformer
    :param dim: dimension of embedding in input to the module
    :param nfeats:
    :param depth: depth of the network, this imply how many times is applied the attention
    :param heads: number of heads for the Multi Head Attention
    :param ff_dropout: probability used in the dropout layer of the feed forward layers applied after the attention
    :param style: can be col, row, or colrow
    """
    def __init__(self, dim: int, nfeats: int, depth: int, heads: int, ff_dropout: float, style: str = 'col'):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.style = style
        for _ in range(depth):
            if "col" in self.style:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=64))),
                    PreNorm(dim, Residual(SimpleMLP(dim, dim*4, dim, GEGLU(), dropout=ff_dropout)))
                ]))
            if "row" in self.style:
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim, heads=heads, dim_head=64))),
                    PreNorm(dim*nfeats, Residual(SimpleMLP(dim*nfeats, dim*nfeats*4, dim*nfeats, GEGLU(),
                                                           dropout=ff_dropout))),
                ]))

    def forward(self, x: Tensor, x_cont: Tensor) -> Tensor:
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
        elif self.style == 'row':
            for attn1, ff1 in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x = attn1(x)
                x = ff1(x)
                x = rearrange(x, '1 b (n d) -> b n d', n=n)
        else:
            for attn1, ff1 in self.layers:
                x = attn1(x)
                x = ff1(x)
        return x


class SimpleMLP(nn.Module):
    """Module that implements a MLP that consists of 2 linear layers with an activation layer
    and dropout in between them. In case of GEGLU activation function, there is a multiplier that
    allow to divide the tensor in 2 parts
    :param dim: dimension of embedding in input to the module
    :param dim_internal: dimension after the first linear layer
    :param dim_out: output dimension
    :param activation_module: module with the activation function to use between the 2 linear layers
    :param dropout: probability used in the dropout layer
    """
    def __init__(self, dim: int, dim_internal: int, dim_out: int,
                 activation_module: nn.Module = nn.ReLU(), dropout: float = 0.):
        super().__init__()
        mult = 1
        if activation_module._get_name() == 'GEGLU':
            # need to increase size of tensor due to how it's implemented GEGLU
            mult = 2
        self.layers = nn.Sequential(
            nn.Linear(dim, dim_internal * mult),
            activation_module,
            nn.Dropout(dropout),
            nn.Linear(dim_internal, dim_out)
        )

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class SepMLP(nn.Module):
    """Module that implements a separable MLP, this means that for each feature is used a different SimpleMLP
    :param dim: dimension of embedding in input to the module
    :param dim_out_for_each_feat: output dimension for each SimpleMLP
    """
    def __init__(self, dim: int, dim_out_for_each_feat: List[int]):
        super().__init__()
        self.len_feats = len(dim_out_for_each_feat)
        self.layers = nn.ModuleList([])
        for i in range(self.len_feats):
            self.layers.append(SimpleMLP(dim=dim, dim_internal=5 * dim, dim_out=dim_out_for_each_feat[i]))

    def forward(self, x: Tensor) -> List[Tensor]:
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

