from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficient_backbone.models.layers import PatchEmbed, TransformerBlock, init_vit_weights
from efficient_backbone.outputs import ForwardResult


class EdgeAwareRanker(nn.Module):
    """Two-layer MLP from the Edge-RecViT paper, without biases.

    H = GELU(W1 X), Z = W2 H, with W1 shaped (2C, C) and W2 shaped (L, 2C).
    The class token is not an input. Depth is the argmax of softmax(Z).
    """

    def __init__(self, dim: int, max_depth: int) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(2 * dim, dim))
        self.w2 = nn.Parameter(torch.empty(max_depth, 2 * dim))
        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)

    def forward(self, tokens: Tensor) -> Tensor:
        hidden = F.gelu(F.linear(tokens, self.w1))
        return F.linear(hidden, self.w2)


class EdgeRecViT(nn.Module):
    """Head, shared recursive block, and tail, with an edge-aware depth ranker.

    This follows Li et al., CVPR 2026, at CIFAR resolution. A few choices are
    made explicit because the paper leaves them underspecified:

    * Training here is from scratch. The paper fine-tunes an ImageNet DeiT.
    * Depth L runs as: layer 1 = head, layers 2..L-1 = one shared block,
      layer L = tail. The paper's recurrence writes the tail as a function of
      y^10, which skips a state; the prose (head, ten shared steps, tail) is
      what this module implements. For L=12 that is ten shared applications.
    * The class token always has depth L and is not ranked.
    * Tokens with depth < L never enter the tail.
    * Exited tokens are masked out of attention. The tensor stays dense, so a
      training step does not realize the FLOP cut; mean depth is the savings
      statistic.
    * lambda_stb and lambda_du are not reported in the paper. The config owns
      them. STB is the mean squared log-sum-exp of the depth logits. DU is
      sum_l E_l A_l, with A detached, and it is minimized. Minimizing that dot
      product penalizes a peaked depth histogram. The sentence that describes
      a DU value near 1 as the desired state does not match a term that is
      added to the loss, so the code follows the formula.
    * Confidence-weighted CE multiplies the classification logits by the mean
      token confidence (the max depth probability), as written.
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
        use_ranker: bool = True,
        lambda_stb: float = 0.01,
        lambda_du: float = 1.0,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be at least 2 so the tail is a distinct layer")
        self.img_size = img_size
        self.depth = depth
        self.use_ranker = use_ranker
        self.lambda_stb = lambda_stb
        self.lambda_du = lambda_du
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        sequence = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        self.ranker = EdgeAwareRanker(embed_dim, depth)
        self.head = TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, 0.0)
        self.shared = TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, drop_path * 0.5)
        self.tail = TransformerBlock(embed_dim, num_heads, mlp_ratio, drop, drop_path)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self.apply(init_vit_weights)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # apply() also walks the ranker, which has no Linear child. Re-init so
        # the depth logits are not left at a tied constant (argmax would be 0).
        nn.init.trunc_normal_(self.ranker.w1, std=0.02)
        nn.init.trunc_normal_(self.ranker.w2, std=0.02)

    def forward(self, x: Tensor) -> ForwardResult:
        tokens = self.patch_embed(x)
        batch, patch_count, _ = tokens.shape
        cls = self.cls_token.expand(batch, -1, -1)
        hidden = self.pos_drop(torch.cat([cls, tokens], dim=1) + self.pos_embed)

        regularizers: dict[str, Tensor] = {}
        loss_logits: Tensor | None = None
        if self.use_ranker:
            depth_logits = self.ranker(hidden[:, 1:])
            probs = depth_logits.softmax(dim=-1)
            token_depth = probs.argmax(dim=-1) + 1
            confidence = probs.amax(dim=-1)
            logsumexp = torch.logsumexp(depth_logits, dim=-1)
            stability = (logsumexp ** 2).mean()
            expected = probs.mean(dim=(0, 1))
            assigned = torch.zeros(self.depth, device=hidden.device, dtype=hidden.dtype)
            flat_depth = (token_depth - 1).reshape(-1)
            assigned.scatter_add_(
                0,
                flat_depth,
                torch.ones(flat_depth.numel(), device=hidden.device, dtype=hidden.dtype),
            )
            assigned = assigned / flat_depth.numel()
            uniformity = torch.sum(expected * assigned.detach())
            regularizers = {
                "stb": stability * self.lambda_stb,
                "du": uniformity * self.lambda_du,
            }
            mean_confidence = confidence.mean(dim=1)
        else:
            token_depth = torch.full(
                (batch, patch_count),
                self.depth,
                device=hidden.device,
                dtype=torch.long,
            )
            mean_confidence = None
            assigned = None

        depth = torch.cat(
            [
                torch.full((batch, 1), self.depth, device=hidden.device, dtype=torch.long),
                token_depth,
            ],
            dim=1,
        )
        everyone = torch.ones(batch, depth.shape[1], dtype=torch.bool, device=hidden.device)
        hidden = self.head(hidden, everyone)
        for level in range(2, self.depth):
            hidden = self.shared(hidden, depth >= level)
        hidden = self.tail(hidden, depth >= self.depth)
        hidden = self.norm(hidden)
        logits = self.classifier(hidden[:, 0])
        if mean_confidence is not None:
            loss_logits = logits * mean_confidence.unsqueeze(1)

        debug: dict = {
            "depth": depth.detach(),
            "mean_depth": token_depth.float().mean().detach(),
        }
        if assigned is not None:
            debug["depth_hist"] = assigned.detach()
            debug["stability"] = stability.detach()
            debug["uniformity"] = uniformity.detach()
            debug["confidence"] = mean_confidence.mean().detach()
        return ForwardResult(
            logits=logits,
            regularizers=regularizers,
            loss_logits=loss_logits,
            debug=debug,
        )
