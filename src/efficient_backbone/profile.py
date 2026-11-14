from __future__ import annotations

import argparse

import torch
from torch import nn

from efficient_backbone.models import build_model
from efficient_backbone.utils import count_parameters


def measure(model: nn.Module, img_size: int = 32, batch: int = 1) -> dict[str, float]:
    """Count parameters and the MACs this forward actually executed.

    Linear and conv MACs come from hooks. Attention adds the two token-token
    matmuls. HAMSA adds a rough FFT estimate (5 L log2 L per transform).
    Edge-RecViT still runs dense masked attention, so this number is the dense
    cost. `active_depth_fraction` is the share of the nominal depth that patch
    tokens used; multiply by it for a proportional savings estimate.
    """
    for module in model.modules():
        if hasattr(module, "last_bmm_macs"):
            module.last_bmm_macs = 0
        if hasattr(module, "last_fft_macs"):
            module.last_fft_macs = 0
    macs = {"linear": 0, "conv": 0}

    def linear_hook(mod, inputs, output):
        del output
        incoming = inputs[0]
        rows = incoming.reshape(-1, incoming.shape[-1]).shape[0]
        macs["linear"] += rows * mod.in_features * mod.out_features

    def conv_hook(mod, inputs, output):
        del inputs
        spatial = output.shape[0] * output.shape[2] * output.shape[3]
        kernel = mod.kernel_size[0] * mod.kernel_size[1]
        macs["conv"] += spatial * (mod.in_channels // mod.groups) * mod.out_channels * kernel

    hooks = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
    model.eval()
    image = torch.zeros(batch, 3, img_size, img_size)
    result = model(image)
    for hook in hooks:
        hook.remove()
    bmm = sum(getattr(module, "last_bmm_macs", 0) for module in model.modules())
    fft = sum(getattr(module, "last_fft_macs", 0) for module in model.modules())
    total = macs["linear"] + macs["conv"] + bmm + fft
    mean_depth = None
    if result.debug and "mean_depth" in result.debug:
        mean_depth = float(result.debug["mean_depth"])
    nominal = getattr(model, "depth", None)
    fraction = None
    if mean_depth is not None and nominal:
        fraction = mean_depth / float(nominal)
    return {
        "params": float(count_parameters(model)),
        "macs": float(total),
        "gflops": total / 1e9,
        "mean_depth": -1.0 if mean_depth is None else mean_depth,
        "active_depth_fraction": -1.0 if fraction is None else fraction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter and MAC counts at 32px.")
    parser.add_argument("--img-size", type=int, default=32)
    args = parser.parse_args()
    specs = [
        {"name": "resnet18_cifar"},
        {"name": "deit_tiny", "img_size": args.img_size},
        {"name": "edge_recvit_tiny", "img_size": args.img_size},
        {"name": "hamsa_tiny", "img_size": args.img_size},
    ]
    print(f"{'model':<20} {'params_m':>10} {'macs_g':>10} {'mean_depth':>12} {'depth_frac':>12}")
    for spec in specs:
        model = build_model(spec, num_classes=100)
        stats = measure(model, img_size=args.img_size)
        depth = "" if stats["mean_depth"] < 0 else f"{stats['mean_depth']:.2f}"
        frac = "" if stats["active_depth_fraction"] < 0 else f"{stats['active_depth_fraction']:.2f}"
        print(
            f"{spec['name']:<20} {stats['params'] / 1e6:10.3f} "
            f"{stats['gflops']:10.3f} {depth:>12} {frac:>12}"
        )


if __name__ == "__main__":
    main()
