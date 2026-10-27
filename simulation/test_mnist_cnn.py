import os
from pathlib import Path
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "test_images"


def to_signed32(raw: int) -> int:
    raw &= 0xFFFFFFFF
    return raw - (1 << 32) if raw & 0x80000000 else raw


def load_hex_image(path: Path):
    pixels = [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]
    if len(pixels) != 784:
        raise ValueError(f"{path}: expected 784 pixels, got {len(pixels)}")
    return pixels


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


async def load_image(dut, image):
    while int(dut.busy.value):
        await RisingEdge(dut.clk)
    for addr, pixel in enumerate(image):
        dut.pixel_addr.value = addr
        dut.pixel_data.value = pixel
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


def try_dump_intermediates(dut):
    if os.getenv("DUMP_INTERMEDIATES", "0") != "1":
        return
    for name in ["u_conv1_mem", "u_pool1_mem", "u_conv2_mem", "u_pool2_mem"]:
        try:
            mem = getattr(getattr(dut, name), "mem")
            vals = [int(mem[i].value) for i in range(10)]
            dut._log.info("%s first 10: %s", name, vals)
        except Exception as exc:
            dut._log.info("%s memory not exposed by this simulator (%s)", name, exc)


@cocotb.test()
async def test_ten_mnist_digits(dut):
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    await reset_dut(dut)

    passed = 0
    for expected in range(10):
        path = IMAGE_DIR / f"digit_{expected}.hex"
        assert path.exists(), f"Missing {path}. Run ../python/train_export_mnist.py first."
        image = load_hex_image(path)
        await load_image(dut, image)
        prediction, scores = await run_inference(dut)

        dut._log.info("================================================")
        dut._log.info("Image           : %s", path.name)
        dut._log.info("Expected digit  : %d", expected)
        dut._log.info("Predicted digit : %d", prediction)
        for i, s in enumerate(scores):
            dut._log.info("Score[%d] = %d", i, s)

        if prediction == expected:
            passed += 1
            dut._log.info("RESULT: PASS")
        else:
            dut._log.error("RESULT: FAIL")
        dut._log.info("================================================")
        try_dump_intermediates(dut)

        # done must be a one-cycle pulse.
        await RisingEdge(dut.clk)
        assert int(dut.done.value) == 0, "done should pulse for one clock only"

    dut._log.info("Passed: %d", passed)
    dut._log.info("Failed: %d", 10 - passed)
    assert passed == 10
