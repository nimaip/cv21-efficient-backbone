from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from efficient_backbone.models import build_model
from efficient_backbone.utils import count_parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one image through a backbone and, for Edge-RecViT, save the depth map."
    )
    parser.add_argument("--model", default="edge_recvit_tiny")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("outputs/depth.png"))
    parser.add_argument("--img-size", type=int, default=32)
    args = parser.parse_args()
    cfg = {"name": args.model}
    if args.model != "resnet18_cifar":
        cfg["img_size"] = args.img_size
    model = build_model(cfg, num_classes=100)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    model.eval()
    image = torch.rand(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        result = model(image)
    print(f"model={args.model} params={count_parameters(model)} logits={tuple(result.logits.shape)}")
    depth = result.debug.get("depth") if result.debug else None
    if depth is None:
        print("This model has no token-depth map.")
        return
    grid = int(model.grid_size)
    token_depth = depth[0, 1:].float().reshape(1, 1, grid, grid)
    scale = max(1, int(model.depth) - 1)
    normalized = (token_depth - 1.0) / scale
    up = F.interpolate(normalized, size=256, mode="nearest")
    rgb = torch.cat([up, torch.zeros_like(up), 1.0 - up], dim=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_image(rgb, args.out)
    mean_depth = float(result.debug["mean_depth"])
    print(f"mean_patch_depth={mean_depth:.2f} saved={args.out}")


if __name__ == "__main__":
    main()
