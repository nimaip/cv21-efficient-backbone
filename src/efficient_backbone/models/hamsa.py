from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from efficient_backbone.models.layers import DropPath, Mlp, PatchEmbed, init_vit_weights
from efficient_backbone.outputs import ForwardResult


def fft_circular_conv(sequence: Tensor, kernel: Tensor) -> Tensor:
    """Circular convolution along tokens via the convolution theorem.

    sequence: (B, L, C), kernel: (C, L). PyTorch's FFT is unnormalized and
    IFFT divides by L, which is the pairing that matches circular convolution.
    A complex kernel yields a complex result; the real part is returned so the
    residual stream stays real.
    """
    spectrum = torch.fft.fft(sequence, dim=1)
    kernel_spectrum = torch.fft.fft(kernel, dim=-1)
    return torch.fft.ifft(spectrum * kernel_spectrum.transpose(0, 1), dim=1).real


class SpectralAdaptiveStateSpace(nn.Module):
    """SASS: complex kernel, SpectralPulseNet, and SAGU, with no scan.

    Order follows the paper's composition Kernel ∘ PulseNet ∘ SAGU, applied to
    the token-sequence FFT. Raster order of patches is the axis of that
    convolution. There is no multi-direction scan and no (A, B, C) discretization.

    The paper writes one complex kernel of length L in the 1D derivation, and
    L×L complex maps inside SAGU. Here the kernel is per channel, which is the
    usual vision-SSM extension, and the L×L maps are shared across channels.
    Gate matrices are real: the paper applies sigmoid to a real magnitude and
    says the gate itself is real. W1, the linear branch, stays complex.
    """

    def __init__(self, channels: int, seq_len: int, sigma: float = 0.02) -> None:
        super().__init__()
        self.channels = channels
        self.seq_len = seq_len
        self.psi_re = nn.Parameter(torch.randn(channels, seq_len) * sigma)
        self.psi_im = nn.Parameter(torch.randn(channels, seq_len) * sigma)
        self.w1_re = nn.Parameter(torch.empty(seq_len, seq_len))
        self.w1_im = nn.Parameter(torch.empty(seq_len, seq_len))
        self.w2 = nn.Parameter(torch.empty(seq_len, seq_len))
        self.wg = nn.Parameter(torch.empty(seq_len, seq_len))
        self.bg = nn.Parameter(torch.zeros(seq_len))
        nn.init.xavier_uniform_(self.w1_re)
        nn.init.normal_(self.w1_im, std=sigma)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.wg)
        self.last_fft_macs = 0

    def forward(self, sequence: Tensor) -> Tensor:
        batch, length, channels = sequence.shape
        if length != self.seq_len or channels != self.channels:
            raise ValueError(
                f"expected sequence {(self.seq_len, self.channels)}, got {(length, channels)}"
            )
        spectrum = torch.fft.fft(sequence, dim=1).permute(0, 2, 1)
        linear = torch.complex(self.w1_re, self.w1_im)
        mixed = spectrum @ linear
        sagu_gate = torch.sigmoid(spectrum.abs() @ self.w2)
        spectrum = mixed * sagu_gate
        pulse_gate = torch.sigmoid(spectrum.abs() @ self.wg + self.bg)
        spectrum = spectrum * pulse_gate
        kernel = torch.complex(self.psi_re, self.psi_im)
        kernel_spectrum = torch.fft.fft(kernel, dim=-1)
        spectrum = spectrum * kernel_spectrum
        restored = torch.fft.ifft(spectrum.permute(0, 2, 1), dim=1).real
        # Engineering estimate only: about 5 L log2(L) flops per real FFT.
        self.last_fft_macs += int(3 * 5 * length * math.log2(length) * channels * batch)
        return restored


class HamsaBlock(nn.Module):
    """Pre-norm HSS residual plus an FFN residual.

    HSS projects the token sequence into a state path and a gate path, runs
    SASS on the state path, and multiplies the paths back together (a GLU).
    The paper writes this block as HSS + FFN. Pre-norm is the residual form
    used here so a deep stack can train from scratch.
    """

    def __init__(
        self,
        dim: int,
        seq_len: int,
        mlp_ratio: float,
        expand: int,
        drop: float,
        drop_path: float,
        sigma: float,
    ) -> None:
        super().__init__()
        hidden = dim * expand
        self.norm1 = nn.LayerNorm(dim)
        self.proj_u = nn.Linear(dim, hidden)
        self.proj_v = nn.Linear(dim, hidden)
        self.sass = SpectralAdaptiveStateSpace(hidden, seq_len, sigma)
        self.proj_y = nn.Linear(hidden, hidden)
        self.proj_o = nn.Linear(hidden, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio, drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        state_in = self.norm1(x)
        state = F.gelu(self.proj_u(state_in))
        gate = F.gelu(self.proj_v(state_in))
        mixed = self.proj_y(self.sass(state)) * gate
        x = x + self.drop_path(self.proj_o(mixed))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Hamsa(nn.Module):
    """Scanning-free spectral state-space classifier for a fixed image size.

    Patches are embedded, ordered in raster order, mixed by stacked HAMSA
    blocks, and pooled with a mean over tokens. There is no class token.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 192,
        depth: int = 8,
        mlp_ratio: float = 4.0,
        expand: int = 1,
        drop: float = 0.0,
        drop_path: float = 0.05,
        sigma: float = 0.02,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.depth = depth
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        seq_len = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        path_rates = torch.linspace(0, drop_path, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                HamsaBlock(embed_dim, seq_len, mlp_ratio, expand, drop, rate, sigma)
                for rate in path_rates
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.apply(init_vit_weights)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # apply() resets Linear layers and would also leave SASS kernels as the
        # constructor set them, which is what we want. Re-apply sigma init in
        # case a future init_vit_weights starts touching every parameter.
        for block in self.blocks:
            nn.init.normal_(block.sass.psi_re, std=sigma)
            nn.init.normal_(block.sass.psi_im, std=sigma)

    def forward(self, x: Tensor) -> ForwardResult:
        hidden = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            hidden = block(hidden)
        pooled = self.norm(hidden).mean(dim=1)
        return ForwardResult(logits=self.head(pooled))
