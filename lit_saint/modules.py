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
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 16,
        dropout: float = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class RowColTransformer(nn.Module):
    def __init__(self, num_tokens: int, dim: int, nfeats: int, depth: int, heads: int,
                 attn_dropout: float, ff_dropout: float, style: str = 'col'):
        super().__init__()
        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed = nn.Embedding(nfeats, dim)
        self.style = style
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=64, dropout=attn_dropout))),
                    PreNorm(dim, Residual(SimpleMLP(dim, dim*4, dim, GEGLU(), dropout=ff_dropout)))
                ]))
            self.layers.append(nn.ModuleList([
                PreNorm(dim*nfeats, Residual(Attention(dim, heads=heads, dim_head=64, dropout=attn_dropout))),
                PreNorm(dim*nfeats, Residual(SimpleMLP(dim*nfeats, dim*nfeats*4, dim*nfeats, GEGLU(), dropout=ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
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


# transformer
class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, attn_dropout: float, ff_dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads=heads, dim_head=16, dropout=attn_dropout))),
                PreNorm(dim, Residual(SimpleMLP(dim, dim*4, dim, GEGLU(), dropout=ff_dropout))),
            ]))

    def forward(self, x: Tensor, x_cont: Tensor = None) -> Tensor:
        if x_cont is not None:
            x = torch.cat((x, x_cont), dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class SimpleMLP(nn.Module):
    def __init__(self, dim_in: int, dim_internal: int, dim_out: int,
                 activation_module: nn.Module = nn.ReLU(), dropout: float = 0.):
        super().__init__()
        mult = 1
        if activation_module._get_name() == 'GEGLU':
            mult = 2
        self.layers = nn.Sequential(
            nn.Linear(dim_in, dim_internal * mult),
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
    def __init__(self, dim: int, dim_out_for_each_feat: List[int]):
        super().__init__()
        self.len_feats = len(dim_out_for_each_feat)
        self.layers = nn.ModuleList([])
        for i in range(self.len_feats):
            self.layers.append(SimpleMLP(dim_in=dim, dim_internal=5 * dim, dim_out=dim_out_for_each_feat[i]))

    def forward(self, x: Tensor) -> List[Tensor]:
        y_pred = list([])
        for i in range(self.len_feats):
            x_i = x[:, i, :]
            pred = self.layers[i](x_i)
            y_pred.append(pred)
        return y_pred

