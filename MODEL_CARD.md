# Model card

Four small image classifiers trained from scratch on CIFAR-100. The comparison is accuracy against parameter count and dense MACs. Numbers are the epoch-149 line of each training log, stored in `results/cifar100.json`.

## Models

| Name | What it is | Parameters | Dense MACs | CIFAR-100 top-1 |
|---|---|---:|---:|---:|
| `resnet18_cifar` | ResNet-18 with a CIFAR stem (3×3, stride 1, no max-pool) | 11.22M | 0.56G | 79.07% |
| `hamsa_tiny` | Scanning-free spectral state-space model, depth 8, dim 192 | 3.93M | 0.24G | 69.24% |
| `deit_tiny` | Non-distilled DeiT, patch 4, dim 192, depth 12, 3 heads | 5.38M | 0.37G | 63.31% |
| `edge_recvit_tiny` | Edge-RecViT: head, one shared block, tail, plus a token-depth ranker | 1.45M | 0.37G | 55.63% |

HAMSA is the only new model that beat DeiT-Tiny on this setup. It did so with fewer parameters and fewer MACs. ResNet-18 remains highest on raw accuracy. Edge-RecViT is the smallest network and finished below DeiT-Tiny. Its MAC count matches DeiT because tokens that exit early are masked inside a dense attention call.

Edge-RecViT's final mean patch depth was 6.54 out of 12, with mass in every depth bin. The ranker did not collapse onto one depth.

## Training data

CIFAR-100, 32×32, 100 classes, 50,000 train images and 10,000 test images. The logged validation accuracy is top-1 on the test split. Augmentation is random crop, horizontal flip, RandAugment, and random erasing. There is no mixup or cutmix. Seed is 0. One run per model.

## Training recipe

150 epochs, batch 128. ResNet uses SGD (lr 0.1, momentum 0.9, weight decay 5e-4, label smoothing 0.1). The other three use AdamW (lr 1e-3, weight decay 0.05, 5 warmup epochs, gradient clip 1.0). Edge-RecViT uses label smoothing 0, a stability weight of 0.01, and a depth-uniformity weight of 1.0.

Hardware was one Tesla V100 (16 GB) on PACE ICE, account `coc`, QoS `coc-ice`, partition `coc-gpu`. PyTorch was `2.14.1+cu126`. Each successful job finished in about 45–50 minutes.

| Model | Job | Checkpoint directory |
|---|---|---|
| ResNet-18 | 7094046 | `resnet18_cifar-20261115-172818` |
| DeiT-Tiny | 7095305 | `deit_tiny-20261115-175700` |
| Edge-RecViT | 7095306 | `edge_recvit_tiny-20261115-175700` |
| HAMSA | 7096004 | `hamsa_tiny-20261115-183919` |

Those directories live on ICE under `/home/hice1/nponna3/scratch/cv21/checkpoints/p01/`. The checkpoints themselves are not in git.

## Intended use

A from-scratch CIFAR-100 comparison of two 2026 efficiency ideas against a small DeiT and a CIFAR ResNet-18. This is not an ImageNet model and it is not a drop-in replacement for the published Edge-RecViT or HAMSA checkpoints. Those papers fine-tune or train at ImageNet resolution.

## Demo

An untrained Edge-RecViT depth picture (red is deeper):

```bash
PYTHONPATH=src .venv/bin/python -m efficient_backbone.demo \
  --model edge_recvit_tiny --out outputs/depth.png
```

The trained depth map needs the Edge-RecViT checkpoint from the directory above:

```bash
PYTHONPATH=src .venv/bin/python -m efficient_backbone.demo \
  --checkpoint /path/to/edge_recvit_tiny-20261115-175700/best.pt \
  --out outputs/depth.png
```

The bar chart of the four scores is `results/cifar100_val_acc.svg`. Regenerate it with `python scripts/plot_cifar100.py`.

## Limits

* One seed. The gaps are large enough to rank the four runs, and too small a sample to claim a stable margin.
* `best.pt` keeps the best validation epoch. The table quotes epoch 149, which is the last logged line, not a separately reloaded `best.pt` evaluation.
* Edge-RecViT does not spend fewer MACs when tokens exit. A sparse attention kernel would be required before the depth map becomes a compute saving.
* HAMSA job 7095307 crashed under autocast because a complex half-precision activation met a complex float32 weight. Job 7096004 ran after that spectral block was pinned to float32. The checkpoint's recorded git commit can still read `3f3e87e`, which is the commit that was synced before that fix.
* Published Edge-RecViT CIFAR numbers fine-tune an ImageNet DeiT at 224×224. Published HAMSA numbers are ImageNet-1K. Do not line these CIFAR-100 scores up against those tables.
