from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, embed_dim: int, in_chans: int = 3) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size {img_size} is not divisible by patch_size {patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-2:] != (self.img_size, self.img_size):
            raise ValueError(
                f"expected image {(self.img_size, self.img_size)}, got {tuple(x.shape[-2:])}"
            )
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float, drop: float) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(F.gelu(self.fc1(x)))
        return self.drop(self.fc2(x))


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, drop: float) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(drop)
        self.last_bmm_macs = 0

    def forward(self, x: Tensor, active: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        # Inactive tokens stay in the tensor so the batch stays rectangular,
        # but they are not keys. The class token is always active.
        scores = scores.masked_fill(~active[:, None, None, :], torch.finfo(scores.dtype).min)
        weights = self.drop(scores.softmax(dim=-1))
        mixed = (weights @ value).transpose(1, 2).reshape(batch, tokens, channels)
        self.last_bmm_macs += 2 * batch * self.num_heads * tokens * tokens * self.head_dim
        return self.drop(self.proj(mixed))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        drop: float,
        drop_path: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, active: Tensor | None = None) -> Tensor:
        if active is None:
            active = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        updated = x + self.drop_path(self.attn(self.norm1(x), active))
        updated = updated + self.drop_path(self.mlp(self.norm2(updated)))
        return torch.where(active.unsqueeze(-1), updated, x)
