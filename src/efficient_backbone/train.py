from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
import yaml

from efficient_backbone.data import build_loaders
from efficient_backbone.models import build_model
from efficient_backbone.outputs import ForwardResult
from efficient_backbone.utils import (
    PROJECT_ROOT,
    count_parameters,
    git_commit,
    param_groups,
    select_device,
    set_seed,
)


def load_config(path: Path) -> dict:
    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return cfg


def learning_rate(epoch: int, epochs: int, warmup: int, base_lr: float) -> float:
    if warmup > 0 and epoch < warmup:
        return base_lr * float(epoch + 1) / float(warmup)
    progress = (epoch - warmup) / max(1, epochs - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _move(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device):
    images, targets = batch
    return images.to(device, non_blocking=True), targets.to(device, non_blocking=True)


def _forward_loss(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor, smoothing: float):
    result = model(images)
    if not isinstance(result, ForwardResult):
        raise TypeError("models must return ForwardResult")
    return result, result.loss(targets, label_smoothing=smoothing)


def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    limit: int | None,
) -> dict[str, float]:
    model.eval()
    correct = 0
    seen = 0
    loss_sum = 0.0
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if limit is not None and step >= limit:
                break
            images, targets = _move(batch, device)
            result, loss = _forward_loss(model, images, targets, smoothing=0.0)
            correct += int((result.logits.argmax(dim=-1) == targets).sum().item())
            seen += int(targets.shape[0])
            loss_sum += float(loss.item()) * int(targets.shape[0])
    if seen == 0:
        raise RuntimeError("evaluation loader produced no batches")
    return {"val_loss": loss_sum / seen, "val_acc": 100.0 * correct / seen}


def run_training(cfg: dict, *, smoke: bool = False, device_name: str = "auto") -> dict:
    if smoke:
        cfg = dict(cfg)
        cfg["data"] = dict(cfg["data"])
        cfg["train"] = dict(cfg["train"])
        cfg["data"]["name"] = "synthetic"
        cfg["data"]["synthetic_size"] = int(cfg["data"].get("synthetic_size", 32))
        cfg["train"]["epochs"] = 1
        cfg["train"]["batch_size"] = min(8, int(cfg["train"]["batch_size"]))
        cfg["train"]["num_workers"] = 0
        cfg["train"]["limit_batches"] = int(cfg["train"].get("limit_batches", 2))
        cfg["train"]["limit_eval_batches"] = 1
        cfg["train"]["amp"] = False

    set_seed(int(cfg.get("seed", 0)))
    device = select_device(device_name if not smoke else "cpu")
    train_loader, eval_loader, num_classes = build_loaders(cfg, smoke=smoke)
    model = build_model(cfg["model"], num_classes=num_classes).to(device)
    train_cfg = cfg["train"]
    groups = param_groups(model, float(train_cfg["weight_decay"]))
    optimizer_name = str(train_cfg["optimizer"]).lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=float(train_cfg["lr"]))
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            groups,
            lr=float(train_cfg["lr"]),
            momentum=float(train_cfg.get("momentum", 0.9)),
        )
    else:
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'")

    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    epochs = int(train_cfg["epochs"])
    warmup = int(train_cfg.get("warmup_epochs", 0))
    base_lr = float(train_cfg["lr"])
    smoothing = float(train_cfg.get("label_smoothing", 0.0))
    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    train_limit = train_cfg.get("limit_batches")
    eval_limit = train_cfg.get("limit_eval_batches")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    model_name = cfg["model"]["name"]
    root = Path(os.environ.get("CV21_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
    run_dir = root / f"{model_name}-{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics_path = run_dir / "metrics.jsonl"

    best_acc = -1.0
    last_metrics: dict = {}
    print(
        f"device={device.type} model={model_name} params={count_parameters(model)} "
        f"run_dir={run_dir}"
    )
    for epoch in range(epochs):
        lr = learning_rate(epoch, epochs, warmup, base_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        loss_total = 0.0
        correct = 0
        seen = 0
        depth_sum = 0.0
        depth_batches = 0
        hist_sum = None
        for step, batch in enumerate(train_loader):
            if train_limit is not None and step >= int(train_limit):
                break
            images, targets = _move(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                result, loss = _forward_loss(model, images, targets, smoothing)
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} step {step}: {loss_value}")
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            batch_size = int(targets.shape[0])
            loss_total += loss_value * batch_size
            correct += int((result.logits.argmax(dim=-1) == targets).sum().item())
            seen += batch_size
            if "mean_depth" in result.debug:
                depth_sum += float(result.debug["mean_depth"])
                depth_batches += 1
            if "depth_hist" in result.debug:
                hist = result.debug["depth_hist"].detach().float().cpu()
                hist_sum = hist if hist_sum is None else hist_sum + hist
        if seen == 0:
            raise RuntimeError("training loader produced no batches; lower batch_size or drop_last")
        val = evaluate(model, eval_loader, device, None if eval_limit is None else int(eval_limit))
        record = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": loss_total / seen,
            "train_acc": 100.0 * correct / seen,
            **val,
        }
        if depth_batches:
            record["mean_depth"] = depth_sum / depth_batches
        if hist_sum is not None:
            record["depth_hist"] = (hist_sum / depth_batches).tolist()
        last_metrics = record
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        extra = ""
        if "mean_depth" in record:
            extra = f" mean_depth={record['mean_depth']:.2f}"
        print(
            f"epoch={epoch} lr={lr:.6f} train_loss={record['train_loss']:.4f} "
            f"train_acc={record['train_acc']:.2f} val_loss={record['val_loss']:.4f} "
            f"val_acc={record['val_acc']:.2f}{extra}"
        )
        state = {
            "model": model.state_dict(),
            "epoch": epoch,
            "metrics": record,
            "config": cfg,
            "git_commit": git_commit(),
            "hostname": platform.node(),
            "torch": torch.__version__,
            "params": count_parameters(model),
        }
        torch.save(state, run_dir / "last.pt")
        if record["val_acc"] >= best_acc:
            best_acc = record["val_acc"]
            torch.save(state, run_dir / "best.pt")
    return {"run_dir": str(run_dir), "best_acc": best_acc, **last_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CIFAR backbone.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Two synthetic batches on CPU.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if not args.smoke and not cfg["data"].get("root"):
        cfg["data"]["root"] = os.environ.get("CV21_DATA", str(PROJECT_ROOT / "data"))
    run_training(cfg, smoke=args.smoke, device_name=args.device)


if __name__ == "__main__":
    main()
