# DigitForge FPGA

**A fixed-point handwritten-digit CNN accelerator, from PyTorch training to Verilog RTL.**

DigitForge FPGA connects model training, INT8 parameter export, a NumPy integer reference model, and Cocotb hardware verification in one compact MNIST workflow. Its sequential compute engines emphasize resource reuse and readable RTL.

## Highlights

- Two convolution/ReLU/max-pooling stages and a ten-class fully connected classifier.
- UINT8 activations, INT8 weights, INT32 biases and scores, with calibrated shifts and saturation.
- External pixel loading and repeated inference without resynthesis.
- Python training/export and integer reference verification.
- Cocotb digit-classification tests and configurable blur experiments with plots and CSV results.
- Portable Verilog-2001 RTL with synchronous RAM; board integration and timing constraints are supplied by the user.

## Technologies

Verilog-2001 · Python · PyTorch / torchvision · NumPy · Cocotb · Icarus Verilog · Verilator · Matplotlib · GNU Make. Vivado may be used for FPGA synthesis.

## Quick start (Ubuntu)

```bash
git clone git@github-silicon:SiliconSynthesizer/digitforge-fpga.git
cd digitforge-fpga
sudo apt update
sudo apt install -y python3-venv make iverilog
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python python/train_export_mnist.py --epochs 3 --device cpu
python python/verify_quantized_model.py --all --debug
cd simulation
make SIM=icarus
BLUR_SIZES=1,3,5,7 make blur SIM=icarus
```

The clone command uses the owner's `github-silicon` SSH alias; other authorized collaborators can use `git@github.com:SiliconSynthesizer/digitforge-fpga.git`.
The pinned CPU PyTorch packages match `run_all.sh`; use a Python version supported by those releases (for example Python 3.10 or 3.11). Training requires internet access to download MNIST. Trained weights and test images are generated locally and are not bundled. See [VALIDATION.md](VALIDATION.md) for the checks performed; no device utilization, timing, or accuracy results are claimed without measurement.

After installing system prerequisites, `bash run_all.sh` runs environment setup, training/export, reference verification, and both simulation flows.

## Design overview

This project implements a small handwritten-digit CNN accelerator whose **input image is loaded externally at run time**. CNN model parameters are initialized from trained HEX files, but the 28×28 test image is never hard-coded into the synthesizable top module.

## 1. Architecture

```text
External 28x28 UINT8 image
        |
        | pixel_addr / pixel_data / pixel_valid
        v
Image RAM (784 x 8)
        |
        v
Conv1 3x3, 1->8, pad=1
        |
      ReLU + fixed-point shift + UINT8 saturation
        |
MaxPool 2x2
        |
Conv2 3x3, 8->16, pad=1
        |
      ReLU + fixed-point shift + UINT8 saturation
        |
MaxPool 2x2
        |
7x7x16 = 784 features (CHW order)
        |
Fully connected 784->10
        |
10 signed INT32 logits
        |
Argmax (no Softmax)
        v
Predicted digit 0..9
```

Shape check:

```text
28x28x1 -> 28x28x8 -> 14x14x8 -> 14x14x16 -> 7x7x16 -> 784 -> 10
```

## 2. Why this implementation is FPGA-friendly

The convolution and fully connected engines are **sequential**. They reuse a single multiply operation instead of instantiating every multiplication in parallel. This favors low resource usage and easy verification over maximum throughput.

More parallel MACs can be added later for speed, at the cost of DSPs, BRAM bandwidth, routing, and controller complexity.

All arithmetic in RTL is integer/fixed point. There is no `real`, floating point, delay statement, or software-only arithmetic in the data path.

## 3. Top-level image interface

```verilog
pixel_valid
pixel_data[7:0]
pixel_addr[9:0]
start
busy
done
predicted_digit[3:0]
score0 ... score9
```

A host writes one pixel per clock while `busy=0`:

```text
address = row*28 + column
0 .. 783
```

After 784 write cycles, pulse `start` for one clock. The controller does not accept a start until 784 pixel writes have been observed. Host writes are ignored while the accelerator is busy.

`done` is a **one-clock pulse**. Scores and `predicted_digit` remain valid afterward until the next inference overwrites them.

Multiple images can be classified without reset or resynthesis:

```text
load 784 pixels -> start -> wait done -> read result
load next 784 pixels -> start -> wait done -> read result
...
```

## 4. Memory layouts

All intermediate feature maps use channel-major CHW order:

```text
address = channel*(HEIGHT*WIDTH) + row*WIDTH + column
```

Therefore Pool2 flatten is exactly:

```text
feature_index = channel*49 + row*7 + column
```

This matches PyTorch NCHW flattening.

Buffer sizes:

| Buffer | Shape | Values | Address width |
|---|---:|---:|---:|
| Image | 28x28x1 | 784 | 10 |
| Conv1 | 28x28x8 | 6272 | 13 |
| Pool1 | 14x14x8 | 1568 | 11 |
| Conv2 | 14x14x16 | 3136 | 12 |
| Pool2 | 7x7x16 | 784 | 10 |

The RAM module has a synchronous one-cycle read. This style is intentionally compatible with FPGA block/distributed RAM inference, subject to normal Vivado inference decisions.

## 5. Weight organization and parameter counts

| Layer | Weight shape | Weight count | Bias count |
|---|---:|---:|---:|
| Conv1 | 8x1x3x3 | 72 | 8 |
| Conv2 | 16x8x3x3 | 1152 | 16 |
| FC | 10x784 | 7840 | 10 |

Conv weights are linearized in this order:

```text
for output_channel
  for input_channel
    for kernel_row
      for kernel_column
```

For Conv2:

```text
weight_address = (((out_ch*8)+in_ch)*3+kernel_row)*3+kernel_col
```

FC order:

```text
weight_address = output_neuron*784 + input_feature
```

PyTorch tensors are flattened in those exact native row-major orders by the exporter.

## 6. Quantization

Input pixels are exact UINT8 MNIST values 0..255. Weights are symmetric signed INT8. Biases are signed INT32 in accumulator units.

For a convolution:

```text
acc = bias_q + sum(UINT8_activation * INT8_weight)
relu_acc = max(acc, 0)
activation_q = saturate_uint8(relu_acc >> SHIFT)
```

The Python exporter derives the INT8 weight scales and calibrated power-of-two output shifts, then writes the selected shifts to `rtl/quant_params.vh`. It also writes `weights/quantization.json` for the Python golden model.

No overflowing value is silently truncated when converting a convolution result to UINT8; values above 255 saturate to 255.

An INT8 HEX byte such as `FE` is loaded into `reg signed [7:0]` and therefore represents -2, not 254.

The FC stage does not requantize its logits because only their ordering is required. It accumulates signed INT32 class scores and Argmax selects the largest.

## 7. Padding and pooling

Convolution is 3x3, stride 1, padding 1. Out-of-range source coordinates contribute zero and no invalid RAM address is generated.

Max-pooling reads each 2x2 window sequentially and stores the maximum.

## 8. Project files

```text
digitforge-fpga/
├── rtl/
│   ├── mnist_cnn_top.v
│   ├── conv_engine.v
│   ├── pool_engine.v
│   ├── fc_engine.v
│   ├── argmax.v
│   ├── sync_ram_1r1w.v
│   └── quant_params.vh
├── weights/
│   └── README.md               # generated HEX files appear here
├── test_images/
│   └── README.md               # digit_0.hex ... digit_9.hex appear here
├── python/
│   ├── train_export_mnist.py
│   └── verify_quantized_model.py
├── simulation/
│   ├── test_mnist_cnn.py
│   ├── Makefile
│   └── requirements.txt
└── README.md
```

The controller is intentionally integrated into `mnist_cnn_top.v` so the number of RTL files stays small.

## 9. Train and export real MNIST parameters

Install dependencies:

```bash
python3 -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -r requirements.txt
```

Train and export:

```bash
python3 python/train_export_mnist.py --epochs 3
```

The script downloads MNIST using torchvision, trains the exact 8/16-channel CNN, quantizes it, evaluates the integer model, then creates:

```text
weights/conv1_weights.hex
weights/conv1_biases.hex
weights/conv2_weights.hex
weights/conv2_biases.hex
weights/fc_weights.hex
weights/fc_biases.hex
weights/quantization.json
weights/float_model_state.pt
rtl/quant_params.vh
test_images/digit_0.hex ... digit_9.hex
test_images/labels.txt
```

The project does **not** include fake/random weight data as a substitute for training.

## 10. Verify the quantized Python golden model

```bash
python3 python/verify_quantized_model.py --all --debug
```

It uses integer arithmetic in the same order as RTL and prints the first ten values of Conv1, Pool1, Conv2, Pool2, all FC scores, and the prediction.

## 11. Cocotb simulation

Install Icarus Verilog and Cocotb, then:

```bash
cd simulation
make train            # only if artifacts have not been generated yet
make verify           # Python integer golden model
make SIM=icarus       # Cocotb + Icarus
```

Verilator can be selected when installed:

```bash
make SIM=verilator
```

The Cocotb test performs, for each digit 0..9:

```text
reset once
write 784 pixels
pulse start
observe busy
wait for done
read predicted_digit + ten scores
check expected class
repeat without resetting the RTL
```

The default timeout is 900,000 clocks per inference. The low-resource sequential architecture is intentionally much slower than a highly parallel CNN.

## 12. Waveforms

Try:

```bash
make SIM=icarus WAVES=1
```

Cocotb/simulator versions may produce `dump.vcd` or `dump.fst`. Open the generated waveform with GTKWave, for example:

```bash
gtkwave dump.vcd
```

Useful signals include `clk`, `rst`, `pixel_valid`, `pixel_addr`, `start`, `busy`, `done`, `debug_state`, `debug_mac_acc`, stage start/done signals, RAM addresses, scores, and `predicted_digit`.

Set:

```bash
DUMP_INTERMEDIATES=1 make SIM=icarus
```

and the Cocotb test will also attempt to read the first ten entries of each intermediate RAM. Whether unpacked memories are exposed hierarchically depends on the simulator.

## 13. Vivado synthesis

1. Run the training/export script first.
2. Create a Vivado RTL project and add every `.v` file in `rtl/` plus `rtl/quant_params.vh`.
3. Add the six generated HEX files to the Vivado project as memory initialization/support files.
4. The engines call `$readmemh()` with the bare filenames such as `conv1_weights.hex`. Keep those files in Vivado's accessible synthesis/simulation working set.
5. Set `mnist_cnn_top` as top.
6. Add clock constraints for the intended board clock, for example 100 MHz.
7. Run synthesis/implementation and inspect BRAM/DSP inference and timing.

The image RAM is **not** initialized from a test image. In hardware, replace the simple pixel loading signals later with AXI4-Lite, AXI4-Stream, DMA, UART, SPI, PCIe, or a processor interface without changing the CNN memory/data organization.

## 14. Practical resource/latency trade-off

This reference design emphasizes correctness, low resource use, and readability. The main acceleration path is to process multiple taps or output channels in parallel and bank the memories accordingly. Do that only after this one-MAC baseline is passing bit-for-bit against the Python model.


## Blur robustness test + prediction plots

After training/exporting the weights and the ten `test_images/digit_*.hex` files, you can test the exact same RTL with blurred input images. The blur is created in Python and then sent to the FPGA one pixel at a time through `pixel_addr`, `pixel_data`, and `pixel_valid`; therefore the accelerator RTL is not modified.

Install the plotting dependency inside the virtual environment:

```bash
source venv/bin/activate
pip install numpy cocotb matplotlib
```

From `simulation/`, run the default original + 3x3 + 5x5 blur experiment:

```bash
cd simulation
make blur SIM=icarus
```

Equivalent direct Cocotb command:

```bash
make MODULE=test_blur_mnist SIM=icarus
```

To choose blur strengths yourself, use positive odd kernel sizes:

```bash
BLUR_SIZES=1,3,5,7 make blur SIM=icarus
```

`1` means no blur (the original image). `3`, `5`, and `7` mean 3x3, 5x5, and 7x7 box blur respectively.

The test produces:

```text
simulation/blur_results/
├── digit_0_blur_1x1_pred_0.png
├── digit_0_blur_3x3_pred_0.png
├── ...
└── blur_summary.csv
```

Each PNG contains the actual 28x28 image that was sent to the RTL, the expected digit, the FPGA-predicted digit, and a bar chart of all ten signed RTL class scores.

By default, blurred misclassifications are reported as `FAIL` for that image but do not stop the complete robustness experiment. This is intentional because strong blur can legitimately lower model accuracy. To make any wrong blurred prediction fail the Cocotb test immediately, run:

```bash
REQUIRE_BLUR_CORRECT=1 BLUR_SIZES=1,3 make blur SIM=icarus
```

## 15. How to run the complete project

### First-time setup

Install the required Ubuntu tools:

```bash
sudo apt update
sudo apt install -y python3-venv iverilog gtkwave
```

Go to the project directory and create a Python virtual environment:

```bash
cd digitforge-fpga
python3 -m venv venv
source venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

For this MNIST project a CPU-only PyTorch installation is sufficient and avoids downloading unnecessary CUDA/Triton packages:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numpy cocotb matplotlib
```

### Step 1 — Train the CNN and export FPGA files

Run this from the project root:

```bash
python python/train_export_mnist.py --epochs 3
```

This trains the MNIST CNN and generates the required quantized weight/bias HEX files and MNIST test-image HEX files.

### Step 2 — Verify the quantized Python model

```bash
python python/verify_quantized_model.py --all --debug
```

Use this step to confirm that the integer/fixed-point model is working before RTL simulation.

### Step 3 — Run the normal Cocotb RTL test

```bash
cd simulation
make clean
make SIM=icarus
```

The Cocotb test loads each 28x28 image into the RTL through `pixel_addr`, `pixel_data`, and `pixel_valid`, starts inference, waits for `done`, and checks `predicted_digit` and the ten output scores.

### Step 4 — Test original and blurred images

From the `simulation` directory:

```bash
make clean
BLUR_SIZES=1,3,5,7 make blur SIM=icarus
```

The blur values mean:

```text
1x1 = original image
3x3 = light blur
5x5 = stronger blur
7x7 = still stronger blur
```

The blur experiment writes its results to:

```text
simulation/blur_results/
```

The folder contains PNG plots and `blur_summary.csv`. Each PNG shows the actual image sent to the FPGA, expected digit, FPGA-predicted digit, and all ten classification scores.

Open the result folder on Ubuntu with:

```bash
xdg-open blur_results
```

### Run the main flow together

After the environment has already been created, the main commands are:

```bash
cd digitforge-fpga
source venv/bin/activate
python python/train_export_mnist.py --epochs 3
python python/verify_quantized_model.py --all --debug
cd simulation
make clean
make SIM=icarus
BLUR_SIZES=1,3,5,7 make blur SIM=icarus
```

For later sessions, you do not need to recreate the virtual environment. Activate it with:

```bash
cd digitforge-fpga
source venv/bin/activate
```
