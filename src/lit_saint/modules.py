from typing import List

import torch
import torch.nn.functional as f
from einops import rearrange
from torch import nn, einsum, Tensor


class Residual(nn.Module):
    """Define a residual block given a Pytorch Module, used for the skip connection"""
    def __init__(self, fn: nn.Module):
        """
        :param fn: pytorch Module that will be used in order to create the skip connection
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs) -> Tensor:
        return self.fn(x, *args, **kwargs) + x


class PreNorm(nn.Module):
    """Apply Layer Normalization before a Module"""
    def __init__(self, dim, fn: nn.Module):
        """
        :param dim: dimension of embedding in input to the module
        :param fn: pytorch Module that will be applied after the normalization layer
        """
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
    """Module that implements self attention"""
    def __init__(self, dim: int, heads: int, dim_head: int):
        """
        :param dim: dimension embedding used in input
        :param heads: number of heads for the Multi Head Attention
        :param dim_head: output dimension given an head
        """
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
    """Module use to define RowColTransformer"""
    def __init__(self, dim: int, nfeats: int, depth: int, heads: int, dim_head: int,
                 ff_dropout: float, scale_dim_internal_col: float,
                 scale_dim_internal_row: float, style: str):
        """
        :param dim: dimension of embedding in input to the module
        :param nfeats: total number of features, needed for the intersample attention
        :param depth: depth of the network, this imply how many times is applied the attention
        :param heads: number of heads for the Multi Head Attention
        :param dim_head: dimension of embedding used in each head
        :param ff_dropout: probability used in the dropout layer of the feed forward layers applied after the attention
        :param scale_dim_internal_col: scale factor input dimension in order to obtain the output dimension of the
        first linear layer in case of style col
        :param scale_dim_internal_row: scale factor input dimension in order to obtain the output dimension of the
        first linear layer in case of style row
        :param style: can be col, row, or colrow
        """
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([]) for _ in range(depth)])
        self.style = style
        for i in range(depth):
            if "col" in self.style:
                self.layers[i].extend(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=dim_head))),
                    PreNorm(dim, Residual(SimpleMLP(dim, int(dim * scale_dim_internal_col),
                                                    dim, GEGLU(), dropout=ff_dropout)))
                ]))
            if "row" in self.style:
                self.layers[i].extend(nn.ModuleList([
                    PreNorm(dim * nfeats, Residual(Attention(dim * nfeats, heads=heads, dim_head=dim_head))),
                    PreNorm(dim * nfeats, Residual(SimpleMLP(dim * nfeats,
                                                             int(dim * nfeats * scale_dim_internal_row),
                                                             dim * nfeats, GEGLU(), dropout=ff_dropout))),
                ]))

    @staticmethod
    def forward_col(x: Tensor, attn: nn.Module, ff: nn.Module) -> Tensor:
        x = attn(x)
        x = ff(x)
        return x

    @staticmethod
    def forward_row(x: Tensor, attn: nn.Module, ff: nn.Module) -> Tensor:
        n = x.shape[1]
        # doing this arrange we have a batch of dimension 1 so we make attention between samples
        x = rearrange(x, 'b n d -> 1 b (n d)')
        x = attn(x)
        x = ff(x)
        x = rearrange(x, '1 b (n d) -> b n d', n=n)
        return x

    def forward(self, x: Tensor, x_cont: Tensor) -> Tensor:
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        if self.style == 'colrow':
            for attn_col, ff_col, attn_row, ff_row in self.layers:
                x = self.forward_col(x, attn_col, ff_col)
                x = self.forward_row(x, attn_row, ff_row)
        elif self.style == 'row':
            for attn, ff in self.layers:
                x = self.forward_row(x, attn, ff)
        else:
            for attn, ff in self.layers:
                x = self.forward_col(x, attn, ff)
        return x


class SimpleMLP(nn.Module):
    """Module that implements a MLP that consists of 2 linear layers with an activation layer
    and dropout in between them. In case of GEGLU activation function, there is a multiplier that
    allow to divide the tensor in 2 parts
    """
    def __init__(self, dim: int, dim_internal: int, dim_out: int,
                 activation_module: nn.Module = nn.ReLU(), dropout: float = 0.):
        """
        :param dim: dimension of embedding in input to the module
        :param dim_internal: dimension after the first linear layer
        :param dim_out: output dimension
        :param activation_module: module with the activation function to use between the 2 linear layers
        :param dropout: probability used in the dropout layer
        """
        super().__init__()
        mult = 1
        if activation_module._get_name() == 'GEGLU':
            # need to increase size of tensor due to how it's implemented GEGLU
            mult = 2
        self.layers = nn.Sequential(
            nn.Linear(dim, dim_internal * mult),
            activation_module,
            nn.Dropout(dropout),
            nn.Linear(dim_internal, dim_out),
        )

    def forward(self, x: Tensor) -> Tensor:
        if len(x.shape) == 1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class SepMLP(nn.Module):
    """Module that implements a separable MLP, this means that for each feature is used a different SimpleMLP
    """
    def __init__(self, dim: int, dim_out_for_each_feat: List[int], scale_dim_internal: float, dropout: float = .0):
        """
        :param dim: dimension of embedding in input to the module
        :param dim_out_for_each_feat: output dimension for each SimpleMLP
        :param scale_dim_internal: scale factor input dimension in order to obtain the output dimension of the
        first linear layer
        :param dropout: probability used in the dropout layer
        """
        super().__init__()
        self.len_feats = len(dim_out_for_each_feat)
        self.layers = nn.ModuleList([])
        for i in range(self.len_feats):
            self.layers.append(SimpleMLP(dim=dim, dim_internal=int(dim * scale_dim_internal),
                                         dim_out=dim_out_for_each_feat[i],
                                         dropout=dropout))

    def forward(self, x: Tensor) -> List[Tensor]:
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred
