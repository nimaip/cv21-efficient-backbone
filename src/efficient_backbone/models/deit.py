from __future__ import annotations

import torch
from torch import Tensor, nn

from efficient_backbone.models.layers import PatchEmbed, TransformerBlock, init_vit_weights
from efficient_backbone.outputs import ForwardResult


class DeiT(nn.Module):
    """ViT trained as a non-distilled DeiT.

    Official DeiT-Tiny is a 224px, patch-16 network (192 dim, 12 layers, 3 heads)
    with a distillation token. CIFAR images are 32px, so this uses patch size 4.
    There is no distillation token: Edge-RecViT compares against the non-distilled
    DeiT recipe, and this baseline does the same.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 12,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        drop_path: float = 0.1,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.depth = depth
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, tokens, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, rate)
                for rate in path_rates
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(init_vit_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> ForwardResult:
        tokens = self.patch_embed(x)
        batch = tokens.shape[0]
        cls = self.cls_token.expand(batch, -1, -1)
        hidden = self.pos_drop(torch.cat([cls, tokens], dim=1) + self.pos_embed)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.norm(hidden)
        return ForwardResult(logits=self.head(hidden[:, 0]))
