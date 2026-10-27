.DEFAULT_GOAL := sim

TOPLEVEL_LANG = verilog
SIM ?= icarus

VERILOG_SOURCES = \
	$(PWD)/../rtl/sync_ram_1r1w.v \
	$(PWD)/../rtl/conv_engine.v \
	$(PWD)/../rtl/pool_engine.v \
	$(PWD)/../rtl/fc_engine.v \
	$(PWD)/../rtl/argmax.v \
	$(PWD)/../rtl/mnist_cnn_top.v

TOPLEVEL = mnist_cnn_top
MODULE = test_mnist_cnn

COMPILE_ARGS += -I$(PWD)/../rtl

# Cocotb simulator makefiles provide WAVES=1 support where the simulator supports it.
# Depending on cocotb/simulator version, the generated waveform may be dump.vcd or dump.fst.

.PHONY: check_artifacts link_artifacts train verify clean_links blur

check_artifacts:
	@test -f ../weights/conv1_weights.hex || (echo "Missing trained weights. Run: make train"; exit 1)
	@test -f ../weights/conv1_biases.hex || (echo "Missing trained weights. Run: make train"; exit 1)
	@test -f ../weights/conv2_weights.hex || (echo "Missing trained weights. Run: make train"; exit 1)
	@test -f ../weights/conv2_biases.hex || (echo "Missing trained weights. Run: make train"; exit 1)
	@test -f ../weights/fc_weights.hex || (echo "Missing trained weights. Run: make train"; exit 1)
	@test -f ../weights/fc_biases.hex || (echo "Missing trained weights. Run: make train"; exit 1)
	@test -f ../test_images/digit_0.hex || (echo "Missing test images. Run: make train"; exit 1)

link_artifacts: check_artifacts
	@ln -sf ../weights/conv1_weights.hex conv1_weights.hex
	@ln -sf ../weights/conv1_biases.hex conv1_biases.hex
	@ln -sf ../weights/conv2_weights.hex conv2_weights.hex
	@ln -sf ../weights/conv2_biases.hex conv2_biases.hex
	@ln -sf ../weights/fc_weights.hex fc_weights.hex
	@ln -sf ../weights/fc_biases.hex fc_biases.hex

train:
	cd .. && python3 python/train_export_mnist.py

verify: check_artifacts
	cd .. && python3 python/verify_quantized_model.py --all --debug

clean_links:
	rm -f conv1_weights.hex conv1_biases.hex conv2_weights.hex conv2_biases.hex fc_weights.hex fc_biases.hex

blur: link_artifacts
	$(MAKE) MODULE=test_blur_mnist SIM=$(SIM) sim

sim: link_artifacts

include $(shell cocotb-config --makefiles)/Makefile.sim
