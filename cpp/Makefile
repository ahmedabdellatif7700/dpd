# The dpd stage's C++/CUDA build on the host (the image builds the same with docker/Dockerfile).
# From the user's C++ template: CMake + Ninja, Debug/Release, binaries in bin/.

.DEFAULT_GOAL := build

BUILD_TYPE ?= Release
BUILD_DIR  ?= build
BIN_DIR    ?= bin
JOBS       ?= 4
CUDA_ARCH  ?= 87
NVCC       ?= /usr/local/cuda/bin/nvcc

.PHONY: build run test debug clean

build:
	@echo
	@echo "Building the dpd stage ($(BUILD_TYPE), sm_$(CUDA_ARCH))..."
	@cmake -S . -B $(BUILD_DIR) -G Ninja \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DCMAKE_CUDA_COMPILER=$(NVCC)
	@ninja -C $(BUILD_DIR) -j$(JOBS)

run: build
	@echo
	@echo "Running the GPU self-test..."
	@./$(BIN_DIR)/gpu_check

test: build
	@ctest --test-dir $(BUILD_DIR) --output-on-failure

debug:
	@$(MAKE) build BUILD_TYPE=Debug
	@gdb --args ./$(BIN_DIR)/gpu_check

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR) $(BIN_DIR)
