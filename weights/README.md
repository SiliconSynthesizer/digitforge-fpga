# Generated trained model artifacts

Run:

```bash
python3 python/train_export_mnist.py
```

The script creates the six required trained parameter HEX files:

- `conv1_weights.hex` — 72 signed INT8 values
- `conv1_biases.hex` — 8 signed INT32 values
- `conv2_weights.hex` — 1152 signed INT8 values
- `conv2_biases.hex` — 16 signed INT32 values
- `fc_weights.hex` — 7840 signed INT8 values
- `fc_biases.hex` — 10 signed INT32 values

It also creates `quantization.json` and saves the float model state for reproducibility.

This repository intentionally does not substitute random or fabricated weights. The exporter trains MNIST and writes the real trained values.
