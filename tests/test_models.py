from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch

from efficient_backbone.models import build_model
from efficient_backbone.models.hamsa import fft_circular_conv
from efficient_backbone.train import load_config, run_training
from efficient_backbone.utils import count_parameters

ROOT = Path(__file__).resolve().parents[1]


class ModelTests(unittest.TestCase):
    def test_circular_conv_matches_direct_sum(self) -> None:
        sequence = torch.randn(2, 16, 4)
        kernel = torch.randn(4, 16)
        got = fft_circular_conv(sequence, kernel)
        expected = torch.zeros_like(sequence)
        for shift in range(16):
            expected = expected + torch.roll(sequence, shifts=shift, dims=1) * kernel[:, shift].view(1, 1, -1)
        self.assertTrue(torch.allclose(got, expected, atol=1e-4, rtol=1e-4))

    def test_builders_forward_and_backward(self) -> None:
        specs = [
            {"name": "resnet18_cifar"},
            {"name": "deit_tiny", "img_size": 32, "depth": 2},
            {"name": "edge_recvit_tiny", "img_size": 32, "depth": 4, "lambda_stb": 0.01, "lambda_du": 1.0},
            {"name": "hamsa_tiny", "img_size": 32, "depth": 2},
        ]
        images = torch.rand(2, 3, 32, 32)
        targets = torch.randint(0, 100, (2,))
        for spec in specs:
            model = build_model(spec, num_classes=100)
            model.train()
            result = model(images)
            self.assertEqual(tuple(result.logits.shape), (2, 100))
            self.assertTrue(torch.isfinite(result.logits).all())
            loss = result.loss(targets)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            grad_norm = sum(
                float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None
            )
            self.assertGreater(grad_norm, 0.0)

    def test_edge_recvit_is_smaller_than_deit_and_ranks_tokens(self) -> None:
        deit = build_model({"name": "deit_tiny"}, num_classes=100)
        edge = build_model({"name": "edge_recvit_tiny"}, num_classes=100)
        self.assertLess(count_parameters(edge), count_parameters(deit) * 0.6)
        self.assertIsNot(edge.head, edge.shared)
        self.assertIsNot(edge.tail, edge.shared)
        edge.eval()
        result = edge(torch.rand(2, 3, 32, 32))
        depth = result.debug["depth"]
        self.assertTrue(torch.equal(depth[:, 0], torch.full((2,), edge.depth)))
        self.assertGreaterEqual(int(depth.min()), 1)
        self.assertLessEqual(int(depth.max()), edge.depth)
        self.assertIn("stb", result.regularizers)
        self.assertIn("du", result.regularizers)

    def test_edge_ranker_receives_gradients(self) -> None:
        model = build_model({"name": "edge_recvit_tiny", "depth": 4}, num_classes=100)
        model.train()
        result = model(torch.rand(4, 3, 32, 32))
        result.loss(torch.randint(0, 100, (4,))).backward()
        self.assertIsNotNone(model.ranker.w1.grad)
        self.assertGreater(float(model.ranker.w1.grad.abs().sum()), 0.0)

    def test_fixed_depth_skips_regularizers(self) -> None:
        model = build_model(
            {"name": "edge_recvit_tiny", "depth": 4, "use_ranker": False},
            num_classes=100,
        )
        result = model(torch.rand(1, 3, 32, 32))
        self.assertEqual(result.regularizers, {})
        self.assertTrue(torch.equal(result.debug["depth"], torch.full_like(result.debug["depth"], 4)))

    def test_resnet_uses_cifar_stem(self) -> None:
        model = build_model({"name": "resnet18_cifar"}, num_classes=100)
        self.assertEqual(model.conv1.kernel_size, (3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertGreater(count_parameters(model), 11_000_000)
        self.assertLess(count_parameters(model), 12_000_000)

    def test_smoke_training_writes_checkpoint(self) -> None:
        cfg = load_config(ROOT / "configs" / "cifar100_edge_recvit.yaml")
        cfg["model"]["depth"] = 4
        cfg["train"]["limit_batches"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["CV21_CHECKPOINT_DIR"] = tmp
            try:
                summary = run_training(cfg, smoke=True, device_name="cpu")
            finally:
                os.environ.pop("CV21_CHECKPOINT_DIR", None)
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "last.pt").is_file())
            self.assertTrue((run_dir / "metrics.jsonl").is_file())
            self.assertTrue(math_finite(summary["train_loss"]))


def math_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


if __name__ == "__main__":
    unittest.main()
