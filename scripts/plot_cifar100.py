#!/usr/bin/env python3
"""Draw the epoch-149 CIFAR-100 accuracies as an SVG bar chart.

Reads results/cifar100.json. No third-party plotting library.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "cifar100.json"
OUTPUT = ROOT / "results" / "cifar100_val_acc.svg"

COLORS = {
    "resnet18_cifar": "#1f4e79",
    "hamsa_tiny": "#7030a0",
    "deit_tiny": "#c45911",
    "edge_recvit_tiny": "#548235",
}
LABELS = {
    "resnet18_cifar": "ResNet-18",
    "hamsa_tiny": "HAMSA",
    "deit_tiny": "DeiT-Tiny",
    "edge_recvit_tiny": "Edge-RecViT",
}


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    runs = sorted(payload["runs"], key=lambda row: row["final"]["val_acc"], reverse=True)
    width, height = 640, 380
    left, right, top, bottom = 64, 24, 48, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    slot = plot_w / len(runs)
    bar_w = slot * 0.62
    ceiling = 100.0

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="320" y="28" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#222222">CIFAR-100 top-1 at epoch 149</text>',
    ]
    for tick in (0, 25, 50, 75, 100):
        y = top + plot_h * (1 - tick / ceiling)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        parts.append(
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#666666">{tick}</text>'
        )
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#222222"/>')
    for index, run in enumerate(runs):
        acc = float(run["final"]["val_acc"])
        bar_h = plot_h * (acc / ceiling)
        x = left + index * slot + (slot - bar_w) / 2
        y = top + plot_h - bar_h
        color = COLORS.get(run["model"], "#444444")
        label = LABELS.get(run["model"], run["model"])
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}"/>')
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222222">{acc:.2f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 22}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222222">{label}</text>'
        )
        millions = run["params"] / 1e6
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{top + plot_h + 40}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#666666">{millions:.2f}M params</text>'
        )
    parts.append("</svg>")
    OUTPUT.write_text("\n".join(parts) + "\n")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
