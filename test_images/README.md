# Generated MNIST verification images

Run `python3 python/train_export_mnist.py`.

The script selects one real MNIST test-set image per class that the exported quantized model classifies correctly and creates:

`digit_0.hex` through `digit_9.hex`.

Each file contains exactly 784 lines, one 8-bit hexadecimal pixel per line, in row-major image order (`address = row*28 + column`).
