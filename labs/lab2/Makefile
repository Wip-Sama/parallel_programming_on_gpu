# ==============================================================================
# Makefile for Lab 2: Color-to-Gray Filter
# Course: Massively Parallel Programming on GPUs
# Teacher: Donato D'Ambrosio
# Department of Mathematics and Computer Science, University of Calabria
# 
# Targets:
#   make              - Compiles lab2 using auto-detected or specified architecture
#   make gtx980       - Compiles specifically for NVIDIA GTX 980 (Maxwell, sm_52, CUDA 11.4)
#   make rtx4080      - Compiles specifically for NVIDIA RTX 4080 (Ada Lovelace, sm_89, CUDA 12/13)
#   make universal    - Compiles with fatbin (sm_52 + compute_52 PTX) for cross-compatibility
#   make run          - Runs test execution (both CPU and GPU 16x16)
#   make verify       - Runs execution and compares CPU & GPU MD5 checksums
#   make bench        - Runs automated Python multi-run benchmark suite (4 runs each)
#   make clean        - Removes compiled binaries, temporary data, and output images
# ==============================================================================

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(shell test -x $(CUDA_HOME)/bin/nvcc && echo $(CUDA_HOME)/bin/nvcc || which nvcc 2>/dev/null)

# Base compiler flags:
# - -Xcompiler -fopenmp : Enables OpenMP for the reference CPU implementation
# - --fmad=false        : Prevents FMA contraction so CPU and GPU produce 100% bit-exact outputs (md5sum match)
# - -I. -I..            : Includes local directory for stb_image.h and stb_image_write.h
NVCC_FLAGS ?= -O3 -std=c++11 -lineinfo --fmad=false -Xcompiler -fopenmp -I. -I.. -I../common -I../common/include -I../../common/include -diag-suppress 550

# Detect architecture if not specified
# GTX 980 uses sm_52; RTX 4080 uses sm_89
ARCH ?= auto

ifeq ($(ARCH),sm_52)
    GENCODE := -gencode arch=compute_52,code=sm_52
else ifeq ($(ARCH),sm_89)
    GENCODE := -gencode arch=compute_89,code=sm_89
else ifeq ($(ARCH),universal)
    GENCODE := -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
else ifeq ($(ARCH),auto)
    # Check if nvcc supports sm_89 (CUDA >= 11.8)
    CUDA_VER_MAJOR := $(shell $(NVCC) --version 2>/dev/null | grep -o 'release [0-9]*' | awk '{print $$2}')
    ifeq ($(shell test $(CUDA_VER_MAJOR) -ge 12 2>/dev/null && echo yes),yes)
        GENCODE := -arch=sm_89
    else ifeq ($(shell test $(CUDA_VER_MAJOR) -eq 13 2>/dev/null && echo yes),yes)
        GENCODE := -arch=sm_89
    else
        GENCODE := -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
    endif
endif

TARGETS := lab2

.PHONY: all clean run verify bench plots gtx980 rtx4080 universal help

all: $(TARGETS)

lab2: lab2.cu common.h
	@echo "Compiling $< -> $@ ($(GENCODE))..."
	$(NVCC) $(NVCC_FLAGS) $(GENCODE) $< -o $@

gtx980:
	$(MAKE) ARCH=sm_52

rtx4080:
	$(MAKE) ARCH=sm_89 NVCC=/usr/local/cuda-13.0/bin/nvcc

universal:
	$(MAKE) ARCH=universal NVCC=/usr/bin/nvcc

run: lab2
	./lab2 both 16

verify: lab2
	@echo "Running execution..."
	./lab2 both 16
	@echo "Running Python bit-exact correctness verification..."
	python3 scripts/verify_outputs.py output_cpu.png output_gpu.png

plots:
	python3 scripts/generate_plots.py

bench: lab2
	python3 scripts/average_benchmarks.py

clean:
	rm -f $(TARGETS) *.o *.prof output*.png

help:
	@echo "Available commands:"
	@echo "  make              - Compile lab2 for current environment"
	@echo "  make gtx980       - Compile for GTX 980 (sm_52, Maxwell, CUDA 11.4)"
	@echo "  make rtx4080      - Compile for RTX 4080 (sm_89, Ada, CUDA 13.0)"
	@echo "  make universal    - Compile with sm_52 + PTX to run on both GPUs"
	@echo "  make run          - Run single test configuration (both CPU and GPU 16x16)"
	@echo "  make verify       - Verify CPU and GPU outputs bit-for-bit with md5sum"
	@echo "  make bench        - Run automated Python 4-run benchmark suite"
	@echo "  make clean        - Remove binaries and output images"
