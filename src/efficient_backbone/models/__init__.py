from __future__ import annotations

from torch import nn

from efficient_backbone.models.deit import DeiT
from efficient_backbone.models.edge_recvit import EdgeRecViT
from efficient_backbone.models.hamsa import Hamsa
from efficient_backbone.models.resnet import ResNet18CIFAR

BUILDERS = {
    "resnet18_cifar": lambda num_classes, **kwargs: ResNet18CIFAR(num_classes=num_classes, **kwargs),
    "deit_tiny": lambda num_classes, **kwargs: DeiT(num_classes=num_classes, **kwargs),
    "edge_recvit_tiny": lambda num_classes, **kwargs: EdgeRecViT(num_classes=num_classes, **kwargs),
    "hamsa_tiny": lambda num_classes, **kwargs: Hamsa(num_classes=num_classes, **kwargs),
}


def build_model(model_cfg: dict, num_classes: int) -> nn.Module:
    cfg = dict(model_cfg)
    name = cfg.pop("name")
    try:
        builder = BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(BUILDERS))
        raise KeyError(f"Unknown model '{name}'. Known models: {known}") from exc
    return builder(num_classes=num_classes, **cfg)
