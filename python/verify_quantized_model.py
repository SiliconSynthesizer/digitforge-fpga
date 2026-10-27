#!/usr/bin/env python3
"""Bit-accurate integer reference model for the RTL data path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
IMAGES = ROOT / "test_images"


def read_signed_hex(path: Path, bits: int) -> np.ndarray:
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    vals = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        x = int(line, 16) & mask
        if x & sign:
            x -= 1 << bits
        vals.append(x)
    return np.asarray(vals, dtype=np.int64)


def read_image(path: Path) -> np.ndarray:
    vals = [int(x.strip(), 16) for x in path.read_text().splitlines() if x.strip()]
    if len(vals) != 784:
        raise ValueError(f"{path}: expected 784 pixels, got {len(vals)}")
    return np.asarray(vals, dtype=np.int64).reshape(28, 28)


def conv_relu_shift(x: np.ndarray, w: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    # x: [C,H,W], w: [O,C,3,3], b: [O]
    c, h, wid = x.shape
    out_ch = w.shape[0]
    out = np.zeros((out_ch, h, wid), dtype=np.int64)
    for oc in range(out_ch):
        for r in range(h):
            for col in range(wid):
                acc = int(b[oc])
                for ic in range(c):
                    for kr in range(3):
                        rr = r + kr - 1
                        if rr < 0 or rr >= h:
                            continue
                        for kc in range(3):
                            cc = col + kc - 1
                            if cc < 0 or cc >= wid:
                                continue
                            acc += int(x[ic, rr, cc]) * int(w[oc, ic, kr, kc])
                if acc <= 0:
                    q = 0
                else:
                    q = acc >> shift
                    if q > 255:
                        q = 255
                out[oc, r, col] = q
    return out


def maxpool2(x: np.ndarray) -> np.ndarray:
    c, h, w = x.shape
    out = np.zeros((c, h // 2, w // 2), dtype=np.int64)
    for ch in range(c):
        for r in range(h // 2):
            for col in range(w // 2):
                out[ch, r, col] = np.max(x[ch, r*2:r*2+2, col*2:col*2+2])
    return out


def infer(image: np.ndarray, params: dict, debug: bool = False):
    c1w = read_signed_hex(WEIGHTS / "conv1_weights.hex", 8).reshape(8, 1, 3, 3)
    c1b = read_signed_hex(WEIGHTS / "conv1_biases.hex", 32)
    c2w = read_signed_hex(WEIGHTS / "conv2_weights.hex", 8).reshape(16, 8, 3, 3)
    c2b = read_signed_hex(WEIGHTS / "conv2_biases.hex", 32)
    fcw = read_signed_hex(WEIGHTS / "fc_weights.hex", 8).reshape(10, 784)
    fcb = read_signed_hex(WEIGHTS / "fc_biases.hex", 32)

    x = image.reshape(1, 28, 28)
    conv1 = conv_relu_shift(x, c1w, c1b, int(params["conv1_shift"]))
    pool1 = maxpool2(conv1)
    conv2 = conv_relu_shift(pool1, c2w, c2b, int(params["conv2_shift"]))
    pool2 = maxpool2(conv2)
    flat = pool2.reshape(-1)  # CHW order, exactly like PyTorch NCHW flatten and RTL

    scores = np.empty(10, dtype=np.int64)
    for j in range(10):
        scores[j] = int(fcb[j]) + int(np.dot(flat, fcw[j]))
    pred = int(np.argmax(scores))

    if debug:
        print("Conv1 first 10:", conv1.reshape(-1)[:10].tolist())
        print("Pool1 first 10:", pool1.reshape(-1)[:10].tolist())
        print("Conv2 first 10:", conv2.reshape(-1)[:10].tolist())
        print("Pool2 first 10:", pool2.reshape(-1)[:10].tolist())
        print("Scores:", scores.tolist())
        print("Prediction:", pred)
    return pred, scores, (conv1, pool1, conv2, pool2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, help="One 784-line HEX image to verify")
    ap.add_argument("--all", action="store_true", help="Verify digit_0.hex ... digit_9.hex")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    params_path = WEIGHTS / "quantization.json"
    if not params_path.exists():
        raise SystemExit("Missing trained artifacts. Run: python python/train_export_mnist.py")
    params = json.loads(params_path.read_text())

    if args.image:
        pred, scores, _ = infer(read_image(args.image), params, args.debug)
        print(f"{args.image.name}: predicted {pred}")
        print("scores:", " ".join(str(int(x)) for x in scores))
        return

    paths = [IMAGES / f"digit_{d}.hex" for d in range(10)] if args.all or not args.image else [args.image]
    passed = 0
    for expected, path in enumerate(paths):
        pred, scores, _ = infer(read_image(path), params, args.debug)
        ok = pred == expected
        print(f"Digit {expected}: predicted={pred} {'PASS' if ok else 'FAIL'}")
        if ok:
            passed += 1
    print(f"Passed: {passed}/{len(paths)}")
    if passed != len(paths):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
