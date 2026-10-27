// This file is overwritten by python/train_export_mnist.py after calibration.
// The defaults only let the RTL elaborate before trained artifacts are generated.
`ifndef MNIST_CNN_QUANT_PARAMS_VH
`define MNIST_CNN_QUANT_PARAMS_VH
`define CONV1_SHIFT 8
`define CONV2_SHIFT 8
`endif
