from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from efficient_backbone.outputs import ForwardResult


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down: nn.Module | None = None
        if stride != 1 or in_planes != planes:
            self.down = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = x if self.down is None else self.down(x)
        hidden = F.relu(self.bn1(self.conv1(x)), inplace=False)
        hidden = self.bn2(self.conv2(hidden))
        return F.relu(hidden + residual, inplace=False)


class ResNet18CIFAR(nn.Module):
    """ResNet-18 with the CIFAR stem: 3x3 stride-1 convolution and no max-pool.

    The ImageNet stem (7x7 stride-2 plus max-pool) throws away the 32x32 input
    before the residual stages. This is the usual CIFAR classification baseline.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        self._init_weights()

    @staticmethod
    def _make_layer(in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(in_planes, planes, stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(planes, planes, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> ForwardResult:
        x = F.relu(self.bn1(self.conv1(x)), inplace=False)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return ForwardResult(logits=self.fc(x))
