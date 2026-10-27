# Validation status

## Portfolio preparation — 2026-10-27

- All four Python source/test files passed AST syntax parsing.
- The complete RTL compiled with Icarus Verilog using `iverilog -g2001 -Wall -I rtl -s mnist_cnn_top -o /tmp/digitforge-fpga.vvp rtl/*.v`.
- `bash -n run_all.sh` passed.
- A local pattern scan found no private keys, known GitHub/AWS token formats, quoted credential assignments, or credential-bearing HTTP URLs in the 21 project files. Such scanning is a precaution, not a guarantee.
- Training, inference accuracy, full Cocotb simulation, and FPGA synthesis/timing were not run during this upload. Trained model artifacts are absent.

## Earlier packaging record

The following records describe checks and environment limitations reported during the original packaging, rather than the current environment.


Checks performed while packaging this project:

- Python syntax compilation passed for:
  - `python/train_export_mnist.py`
  - `python/verify_quantized_model.py`
  - `simulation/test_mnist_cnn.py`
- A randomized arithmetic cross-check confirmed that the NumPy golden-model implementation of Conv -> ReLU -> shift -> saturation -> MaxPool matches the PyTorch integer-export math.
- Required source-file presence and basic RTL structural checks passed.

Checks not performed in the packaging environment:

- MNIST training/export was not executed because the dataset is not present locally and the Python runtime has no network access.
- Icarus Verilog and Verilator are not installed in the packaging environment, so RTL compilation/Cocotb simulation was not executed here.

The project deliberately does not include fabricated/random model parameters. Run `python3 python/train_export_mnist.py` on a machine with MNIST download access to generate the trained HEX files and real MNIST test images, then run the Python golden model and Cocotb simulation.
