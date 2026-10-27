import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "test_images"
RESULT_DIR = Path(__file__).resolve().parent / "blur_results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def to_signed32(raw: int) -> int:
    raw &= 0xFFFFFFFF
    return raw - (1 << 32) if raw & 0x80000000 else raw


def load_hex_image(path: Path) -> np.ndarray:
    pixels = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    if len(pixels) != 784:
        raise ValueError(f"{path}: expected 784 pixels, got {len(pixels)}")
    return np.asarray(pixels, dtype=np.uint8).reshape(28, 28)


def box_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    """Integer-friendly NxN box blur with edge padding."""
    if kernel_size <= 1:
        return image.copy()
    if kernel_size % 2 == 0:
        raise ValueError("BLUR kernel size must be odd, e.g. 1, 3, 5, 7")

    pad = kernel_size // 2
    padded = np.pad(image.astype(np.uint32), pad, mode="edge")
    out = np.empty_like(image, dtype=np.uint8)
    area = kernel_size * kernel_size

    for r in range(28):
        for c in range(28):
            total = int(padded[r:r + kernel_size, c:c + kernel_size].sum())
            out[r, c] = min(255, (total + area // 2) // area)
    return out


def save_prediction_plot(image: np.ndarray, expected: int, predicted: int, scores, kernel_size: int):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap="gray", vmin=0, vmax=255)
    ax1.set_title(
        f"Input image (blur {kernel_size}x{kernel_size})\n"
        f"Expected: {expected}   Predicted: {predicted}"
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    classes = np.arange(10)
    ax2.bar(classes, scores)
    ax2.set_xticks(classes)
    ax2.set_xlabel("Digit class")
    ax2.set_ylabel("RTL score / logit")
    ax2.set_title("10 FPGA classification scores")
    ax2.axvline(predicted, linestyle="--", linewidth=1)

    fig.tight_layout()
    outfile = RESULT_DIR / f"digit_{expected}_blur_{kernel_size}x{kernel_size}_pred_{predicted}.png"
    fig.savefig(outfile, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return outfile


async def reset_dut(dut):
    dut.rst.value = 1
    dut.pixel_valid.value = 0
    dut.pixel_addr.value = 0
    dut.pixel_data.value = 0
    dut.start.value = 0
    for _ in range(5):
        await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)


async def load_image(dut, image_2d: np.ndarray):
    while int(dut.busy.value):
        await RisingEdge(dut.clk)

    flat = image_2d.reshape(-1)
    for addr, pixel in enumerate(flat):
        dut.pixel_addr.value = addr
        dut.pixel_data.value = int(pixel)
        dut.pixel_valid.value = 1
        await RisingEdge(dut.clk)

    dut.pixel_valid.value = 0
    await RisingEdge(dut.clk)


async def run_inference(dut, timeout_cycles=900_000):
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0

    saw_busy = False
    for _ in range(timeout_cycles):
        await RisingEdge(dut.clk)
        if int(dut.busy.value):
            saw_busy = True
        if int(dut.done.value):
            break
    else:
        raise TimeoutError("CNN inference timeout")

    assert saw_busy, "busy never asserted"
    prediction = int(dut.predicted_digit.value)
    scores = [to_signed32(int(getattr(dut, f"score{i}").value)) for i in range(10)]
    return prediction, scores


def parse_blur_sizes():
    raw = os.getenv("BLUR_SIZES", "1,3,5")
    sizes = [int(x.strip()) for x in raw.split(",") if x.strip()]
    for k in sizes:
        if k < 1 or k % 2 == 0:
            raise ValueError("BLUR_SIZES must contain positive odd integers, e.g. 1,3,5")
    return sizes


@cocotb.test()
async def test_mnist_with_blur_and_plots(dut):
    """
    Tests digit_0.hex ... digit_9.hex at several blur strengths.

    Default BLUR_SIZES=1,3,5 where 1 means the original image.
    Plots and a CSV summary are written into simulation/blur_results/.

    By default this is a robustness experiment, so an incorrect prediction is
    reported rather than failing the whole simulation. Set REQUIRE_BLUR_CORRECT=1
    if every tested blurred image must classify correctly.
    """
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    blur_sizes = parse_blur_sizes()
    require_all = os.getenv("REQUIRE_BLUR_CORRECT", "0") == "1"
    rows = []
    total = 0
    correct = 0

    dut._log.info("Testing blur sizes: %s", blur_sizes)

    for kernel_size in blur_sizes:
        level_correct = 0
        for expected in range(10):
            path = IMAGE_DIR / f"digit_{expected}.hex"
            assert path.exists(), f"Missing {path}. Run ../python/train_export_mnist.py first."

            original = load_hex_image(path)
            image = box_blur(original, kernel_size)

            await load_image(dut, image)
            prediction, scores = await run_inference(dut)
            is_correct = prediction == expected

            total += 1
            correct += int(is_correct)
            level_correct += int(is_correct)

            plot_path = save_prediction_plot(
                image=image,
                expected=expected,
                predicted=prediction,
                scores=scores,
                kernel_size=kernel_size,
            )

            rows.append({
                "digit": expected,
                "blur_kernel": kernel_size,
                "predicted": prediction,
                "correct": int(is_correct),
                **{f"score{i}": scores[i] for i in range(10)},
                "plot": str(plot_path),
            })

            dut._log.info(
                "Digit %d | blur %dx%d | predicted %d | %s | plot=%s",
                expected,
                kernel_size,
                kernel_size,
                prediction,
                "PASS" if is_correct else "FAIL",
                plot_path.name,
            )

            # done is specified as a one-cycle pulse.
            await RisingEdge(dut.clk)
            assert int(dut.done.value) == 0, "done should pulse for one clock only"

            if require_all:
                assert is_correct, (
                    f"Digit {expected} misclassified as {prediction} "
                    f"with {kernel_size}x{kernel_size} blur"
                )

        dut._log.info(
            "Blur %dx%d accuracy on these 10 images: %d/10 = %.1f%%",
            kernel_size,
            kernel_size,
            level_correct,
            100.0 * level_correct / 10,
        )

    csv_path = RESULT_DIR / "blur_summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    dut._log.info("================================================")
    dut._log.info("Blur robustness result: %d/%d correct = %.1f%%", correct, total, 100.0 * correct / total)
    dut._log.info("Plots: %s", RESULT_DIR)
    dut._log.info("CSV  : %s", csv_path)
    dut._log.info("================================================")
