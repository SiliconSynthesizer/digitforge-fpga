#!/usr/bin/env python3
"""Train the exact FPGA-friendly MNIST CNN and export integer RTL artifacts.

The exported fixed-point rule is:
  accumulator = bias_q + sum(activation_q * weight_q)
  activation_out = saturate_uint8(max(accumulator, 0) >> SHIFT)

Weights are INT8. Biases and accumulators are INT32. Pooling is exact UINT8 max.
The FC layer keeps signed INT32 logits and the hardware performs argmax only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_DIR = ROOT / "weights"
IMAGES_DIR = ROOT / "test_images"
RTL_DIR = ROOT / "rtl"
DATA_DIR = ROOT / "data"


class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1, bias=True)
        self.fc = nn.Linear(16 * 7 * 7, 10, bias=True)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)  # NCHW -> channel-major CHW order per sample
        return self.fc(x)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def signed_hex(x: int, bits: int) -> str:
    mask = (1 << bits) - 1
    return f"{x & mask:0{bits // 4}X}"


def export_hex(path: Path, values: np.ndarray, bits: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = np.asarray(values).reshape(-1)
    with path.open("w") as f:
        for x in flat:
            f.write(signed_hex(int(x), bits) + "\n")


def quantize_weight(w: torch.Tensor):
    max_abs = float(w.detach().abs().max().cpu())
    if max_abs == 0.0:
        scale = 1.0
    else:
        scale = max_abs / 127.0
    q = torch.round(w.detach().cpu() / scale).clamp(-127, 127).to(torch.int64)
    return q, scale


def quantize_bias(b: torch.Tensor, input_scale: float, weight_scale: float):
    scale = input_scale * weight_scale
    q = torch.round(b.detach().cpu() / scale).clamp(-(2**31), 2**31 - 1).to(torch.int64)
    return q


def choose_shift(relu_acc: torch.Tensor, percentile: float = 0.999):
    vals = relu_acc.detach().reshape(-1)
    if vals.numel() > 500_000:
        # Deterministic evenly-spaced sample keeps calibration fast and repeatable.
        idx = torch.linspace(0, vals.numel() - 1, steps=500_000).long()
        vals = vals[idx]
    target = float(torch.quantile(vals.double(), percentile).item())
    if target <= 255.0:
        return 0
    return max(0, int(math.ceil(math.log2(target / 255.0))))


def requant_relu(acc: torch.Tensor, shift: int):
    # acc is float64 but contains integer-valued sums exactly in this range.
    q = torch.floor(torch.clamp(acc, min=0.0) / float(1 << shift))
    return torch.clamp(q, 0, 255)


def qconv(xq, wq, bq, shift):
    acc = F.conv2d(xq.double(), wq.double(), bq.double(), stride=1, padding=1)
    return requant_relu(acc, shift), acc


def build_quantized_artifacts(model: MNISTNet, calib_images_u8: torch.Tensor):
    input_scale = 1.0 / 255.0

    c1w_q, c1w_scale = quantize_weight(model.conv1.weight)
    c1b_q = quantize_bias(model.conv1.bias, input_scale, c1w_scale)
    _, c1_acc = qconv(calib_images_u8, c1w_q, c1b_q, 0)
    c1_shift = choose_shift(torch.clamp(c1_acc, min=0.0))
    c1_q = requant_relu(c1_acc, c1_shift)
    p1_q = F.max_pool2d(c1_q, 2)
    c1_out_scale = input_scale * c1w_scale * (1 << c1_shift)

    c2w_q, c2w_scale = quantize_weight(model.conv2.weight)
    c2b_q = quantize_bias(model.conv2.bias, c1_out_scale, c2w_scale)
    _, c2_acc = qconv(p1_q, c2w_q, c2b_q, 0)
    c2_shift = choose_shift(torch.clamp(c2_acc, min=0.0))
    c2_q = requant_relu(c2_acc, c2_shift)
    p2_q = F.max_pool2d(c2_q, 2)
    c2_out_scale = c1_out_scale * c2w_scale * (1 << c2_shift)

    fcw_q, fcw_scale = quantize_weight(model.fc.weight)
    fcb_q = quantize_bias(model.fc.bias, c2_out_scale, fcw_scale)

    return {
        "conv1_weights": c1w_q,
        "conv1_biases": c1b_q,
        "conv2_weights": c2w_q,
        "conv2_biases": c2b_q,
        "fc_weights": fcw_q,
        "fc_biases": fcb_q,
        "conv1_shift": c1_shift,
        "conv2_shift": c2_shift,
        "input_scale": input_scale,
        "conv1_weight_scale": c1w_scale,
        "conv1_output_scale": c1_out_scale,
        "conv2_weight_scale": c2w_scale,
        "conv2_output_scale": c2_out_scale,
        "fc_weight_scale": fcw_scale,
    }


def quantized_forward(images_u8: torch.Tensor, q):
    c1_q, _ = qconv(images_u8, q["conv1_weights"], q["conv1_biases"], q["conv1_shift"])
    p1_q = F.max_pool2d(c1_q, 2)
    c2_q, _ = qconv(p1_q, q["conv2_weights"], q["conv2_biases"], q["conv2_shift"])
    p2_q = F.max_pool2d(c2_q, 2)
    flat = p2_q.reshape(p2_q.shape[0], -1).double()
    logits = flat @ q["fc_weights"].double().t() + q["fc_biases"].double()
    return logits, (c1_q, p1_q, c2_q, p2_q)


def evaluate_float(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += int((pred == y).sum())
            total += y.numel()
    return correct / total


def evaluate_quantized(test_ds, q, max_samples: int, batch_size: int = 128):
    total = min(len(test_ds), max_samples if max_samples > 0 else len(test_ds))
    correct = 0
    for start in range(0, total, batch_size):
        stop = min(total, start + batch_size)
        imgs = test_ds.data[start:stop].unsqueeze(1).double()
        labels = test_ds.targets[start:stop]
        logits, _ = quantized_forward(imgs, q)
        correct += int((logits.argmax(1).cpu() == labels).sum())
    return correct / total


def export_test_images(test_ds, q):
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    found = {}
    batch = 128
    for start in range(0, len(test_ds), batch):
        stop = min(len(test_ds), start + batch)
        imgs = test_ds.data[start:stop].unsqueeze(1).double()
        labels = test_ds.targets[start:stop]
        logits, _ = quantized_forward(imgs, q)
        preds = logits.argmax(1).cpu()
        for i in range(stop - start):
            label = int(labels[i])
            if label not in found and int(preds[i]) == label:
                found[label] = start + i
        if len(found) == 10:
            break

    if len(found) != 10:
        raise RuntimeError(f"Could not find one correctly quantized test sample for each digit; found {sorted(found)}")

    lines = []
    for digit in range(10):
        idx = found[digit]
        img = test_ds.data[idx].numpy().reshape(-1)
        path = IMAGES_DIR / f"digit_{digit}.hex"
        with path.open("w") as f:
            for px in img:
                f.write(f"{int(px):02X}\n")
        lines.append(f"digit_{digit}.hex {digit} dataset_index={idx}\n")
    (IMAGES_DIR / "labels.txt").write_text("".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibration-samples", type=int, default=1024)
    ap.add_argument("--quant-eval-samples", type=int, default=10000, help="0 means all test samples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    seed_everything(args.seed)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    device = torch.device(args.device)
    model = MNISTNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on {device} ...")
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * y.numel()
        acc = evaluate_float(model, test_loader, device)
        print(f"epoch {epoch+1}/{args.epochs}: loss={running/len(train_ds):.5f} test_acc={acc*100:.2f}%")

    # Calibration images use exact original 0..255 integers, not normalized floats.
    ncal = min(args.calibration_samples, len(train_ds))
    calib = train_ds.data[:ncal].unsqueeze(1).double()
    q = build_quantized_artifacts(model.cpu(), calib)

    export_hex(WEIGHTS_DIR / "conv1_weights.hex", q["conv1_weights"].numpy(), 8)
    export_hex(WEIGHTS_DIR / "conv1_biases.hex", q["conv1_biases"].numpy(), 32)
    export_hex(WEIGHTS_DIR / "conv2_weights.hex", q["conv2_weights"].numpy(), 8)
    export_hex(WEIGHTS_DIR / "conv2_biases.hex", q["conv2_biases"].numpy(), 32)
    export_hex(WEIGHTS_DIR / "fc_weights.hex", q["fc_weights"].numpy(), 8)
    export_hex(WEIGHTS_DIR / "fc_biases.hex", q["fc_biases"].numpy(), 32)

    meta = {
        "conv1_shift": int(q["conv1_shift"]),
        "conv2_shift": int(q["conv2_shift"]),
        "input_scale": float(q["input_scale"]),
        "conv1_weight_scale": float(q["conv1_weight_scale"]),
        "conv1_output_scale": float(q["conv1_output_scale"]),
        "conv2_weight_scale": float(q["conv2_weight_scale"]),
        "conv2_output_scale": float(q["conv2_output_scale"]),
        "fc_weight_scale": float(q["fc_weight_scale"]),
        "weight_order": "out_channel,input_channel,kernel_row,kernel_col; FC=out_neuron,input_feature",
        "feature_order": "channel,row,column (CHW)",
    }
    (WEIGHTS_DIR / "quantization.json").write_text(json.dumps(meta, indent=2) + "\n")
    (RTL_DIR / "quant_params.vh").write_text(
        "// Auto-generated by python/train_export_mnist.py\n"
        "`ifndef MNIST_CNN_QUANT_PARAMS_VH\n"
        "`define MNIST_CNN_QUANT_PARAMS_VH\n"
        f"`define CONV1_SHIFT {q['conv1_shift']}\n"
        f"`define CONV2_SHIFT {q['conv2_shift']}\n"
        "`endif\n"
    )

    qacc = evaluate_quantized(test_ds, q, args.quant_eval_samples)
    print(f"Quantized integer accuracy: {qacc*100:.2f}%")
    print(f"Conv1 shift={q['conv1_shift']}  Conv2 shift={q['conv2_shift']}")

    export_test_images(test_ds, q)
    torch.save(model.state_dict(), WEIGHTS_DIR / "float_model_state.pt")
    print("Exported trained weights, biases, quantization metadata, RTL shift include, and 10 real MNIST test images.")


if __name__ == "__main__":
    main()
