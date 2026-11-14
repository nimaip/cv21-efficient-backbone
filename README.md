# cv21-efficient-backbone

Small image-classification backbones for a from-scratch CIFAR-100 comparison. The baselines are a CIFAR ResNet-18 and a non-distilled DeiT-Tiny. The two methods are a recursive edge-aware Vision Transformer (Edge-RecViT) and a scanning-free spectral state-space model (HAMSA).

This folder is its own git repo. It is not part of a published umbrella project.

## Idea

A Vision Transformer splits an image into patches and mixes every patch with every other patch in every layer. That global mix is accurate and expensive: attention grows with the square of the token count, and every layer owns its own weights.

Two 2026 papers cut that cost in different ways:

* **Edge-RecViT** (Li et al., CVPR 2026) keeps the transformer. An MLP ranker, run before the transformer, assigns each patch a depth. Structurally rich tokens are meant to be refined longer; simple tokens leave early. The class token always runs the full depth. The network itself is only three transformer blocks — a head, one shared middle block, and a tail — and the middle block is reapplied until a token's depth is used up. Parameter count stays near three layers. Compute is supposed to follow the depth map.
* **HAMSA** (Patro and Agneeswaran, CVPR 2026 Findings) leaves attention. Vision state-space models such as Vim and VMamba turn a grid into a sequence by scanning it (rows, columns, sometimes several directions) because a classic state-space model reads in order. HAMSA uses the fact that a state-space update is a convolution: `y = K ∗ u` is a product in the Fourier domain, `y = F⁻¹(F(u) ⊙ F(K))`, which an FFT computes in `O(L log L)`. The kernel is one complex vector per channel, initialized from a small Gaussian, with no `(A, B, C)` matrices and no zero-order-hold discretization. SpectralPulseNet and a Spectral Adaptive Gating Unit then reweight frequency bins from the input magnitude.

The training target is to beat the DeiT-Tiny and ResNet-18 baselines on CIFAR-100 top-1 against parameter count and MACs. No run has been trained yet.

Papers:

* [Edge-RecViT](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.html)
* [HAMSA](https://arxiv.org/abs/2604.14724)
* [DeiT](https://arxiv.org/abs/2012.12877) (non-distilled recipe; CIFAR uses patch size 4)
* ResNet-18 with a CIFAR stem (He et al., 2016)

## What this code does

| Model | Config | Params | Dense MACs | FLOPs (2×MAC) |
|---|---|---:|---:|---:|
| `resnet18_cifar` | `configs/cifar100_resnet18.yaml` | 11.22M | 0.56G | 1.11G |
| `deit_tiny` | `configs/cifar100_deit_tiny.yaml` | 5.38M | 0.37G | 0.73G |
| `edge_recvit_tiny` | `configs/cifar100_edge_recvit.yaml` | 1.45M | 0.37G | 0.73G |
| `hamsa_tiny` | `configs/cifar100_hamsa.yaml` | 3.93M | 0.24G | 0.47G |

Counts are from one untrained 32×32 forward (`python -m efficient_backbone.profile`). MACs are hook counts on conv and linear layers, plus the two attention matmuls. HAMSA also adds a rough FFT estimate. FLOPs use the papers' convention of 1 MAC = 2 FLOPs. CIFAR-100 top-1 is empty until the first real training job.

Edge-RecViT's parameter count is the part the shared block actually delivers (about 27% of DeiT-Tiny here, in line with a 3-layer stack against a 12-layer DeiT). Its MAC count matches DeiT because exited tokens are masked inside a dense attention call. The training log records `mean_depth` and `depth_hist`. Those are the quantities a later sparse kernel would turn into a real FLOP cut. At initialization the depth map is noise.

## Scope of the implementation

These are small models aimed at one GPU, trained from scratch on CIFAR-100. The published Edge-RecViT CIFAR numbers fine-tune an ImageNet DeiT at 224×224. The published HAMSA numbers are ImageNet-1K.

Choices the papers do not fully specify, and what this repo does:

* Edge-RecViT depth `L` is head, then the shared block at levels `2 .. L-1`, then the tail at level `L`. For `L = 12` that is ten shared applications. The paper's formula writes the tail as a function of an earlier state; the code follows the head / ten shared steps / tail description.
* The ranker is the paper's bias-free two-layer MLP on patch tokens. It does not take an extra edge detector. "Edge-aware" is the behavior the losses are supposed to learn.
* `lambda_stb = 0.01` and `lambda_du = 1.0` are repo defaults. The paper writes the losses and does not give the weights. STB is the mean squared log-sum-exp of the depth logits. DU is `sum_l E_l A_l` with the hard histogram detached, and it is minimized, which penalizes a peaked depth histogram.
* Classification for Edge-RecViT multiplies the logits by the average token confidence (the maximum depth probability) before cross-entropy. Label smoothing is off for that model. There is no mixup or cutmix for any model.
* HAMSA uses a per-channel complex kernel. The 1D writeup describes a single kernel of length `L`. The `L×L` SAGU and SpectralPulseNet maps are shared across channels. Sigmoid gates are real. The linear SAGU branch is complex. The inverse FFT's real part is the block output, since the residual stream is real.
* HAMSA blocks are pre-norm residuals with a GELU feed-forward network. The sequence is the raster order of patches. Mixing is a circular convolution along that order. There is no class token; the head reads the mean token.
* The same crop, flip, RandAugment, and random-erasing pipeline is used for every real CIFAR run.

## Layout

```
configs/                  one YAML per run
scripts/train.sh          Mac entry
scripts/train_cifar.sbatch
scripts/sync_to_ice.sh
src/efficient_backbone/   models, data, train, profile, demo
tests/test_models.py
```

Checkpoints, CIFAR, logs, and `outputs/` stay out of git. On ICE they go under `~/scratch/cv21/`.

## Mac setup

System Python on this machine is 3.14. The checked environment is CPython 3.12:

```bash
cd p01-efficient-backbone
uv python install 3.12
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Smoke train (two synthetic batches, no CIFAR download):

```bash
scripts/train.sh --config configs/cifar100_deit_tiny.yaml --smoke
```

Depth-map picture for an untrained Edge-RecViT (red means deeper). This becomes meaningful after a checkpoint exists:

```bash
PYTHONPATH=src .venv/bin/python -m efficient_backbone.demo \
  --model edge_recvit_tiny --out outputs/depth.png
# after a run:
PYTHONPATH=src .venv/bin/python -m efficient_backbone.demo \
  --checkpoint checkpoints/<run>/best.pt --out outputs/depth.png
```

A real CIFAR-100 training job downloads the dataset into `./data` on first use:

```bash
scripts/train.sh --config configs/cifar100_edge_recvit.yaml
```

Each run writes `checkpoints/<model>-<timestamp>/{config.yaml,metrics.jsonl,last.pt,best.pt}` and records the git commit, hostname, and torch version inside the checkpoint.

## ICE

On 2026-11-14, `cv21` was Python 3.11.18 and `import torch` failed. Install PyTorch into that env before submitting a job. The cluster also has `module load pytorch/25.10`, which is an Apptainer image with its own Python, separate from `cv21`. This job script uses `cv21`. From a login node, on the GT VPN:

```bash
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate cv21
python -c "import torch"
# if that fails, install a CUDA build that matches the cluster modules, then re-check
```

Sync and train:

```bash
scripts/sync_to_ice.sh
ssh nponna3@login-ice.pace.gatech.edu
cd ~/scratch/cv21/p01-efficient-backbone
sbatch scripts/train_cifar.sbatch configs/cifar100_resnet18.yaml
sbatch scripts/train_cifar.sbatch configs/cifar100_deit_tiny.yaml
sbatch scripts/train_cifar.sbatch configs/cifar100_edge_recvit.yaml
sbatch scripts/train_cifar.sbatch configs/cifar100_hamsa.yaml
```

Jobs request one V100, 8 CPUs, 32 GB, and 8 hours on account `coc` / QoS `coc-ice` / partition `coc-gpu`. Logs land in `~/scratch/cv21/logs/`. Checkpoints land in `~/scratch/cv21/checkpoints/p01/`. CIFAR lands in `~/scratch/cv21/datasets/`.

Pull logs back with:

```bash
rsync -a --progress \
  nponna3@login-ice.pace.gatech.edu:~/scratch/cv21/logs/ \
  ./logs-ice/
```

## Metrics

`metrics.jsonl` records epoch, learning rate, train loss, train accuracy, validation loss, and validation accuracy. Edge-RecViT rows also record mean patch depth and the depth histogram. Validation accuracy is the model-selection metric (`best.pt`).

## Known limits

* No trained accuracy yet. The table above is size and a single forward, not a leaderboard.
* Edge-RecViT attention is masked but dense, so the wall-clock and the hook MAC count do not shrink when tokens exit.
* HAMSA's `L×L` spectral maps are cheap at CIFAR (`L = 64`) and would dominate if this block were dropped onto a long token sequence without sharing them.
* Depth decisions use `argmax`, which does not carry a gradient. The ranker learns through the stability loss, the uniformity loss, and the confidence weight on the logits.
