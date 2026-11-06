# ==============================================================================
# Makefile for Lab 3: Naive and Tiled Matrix Multiplication
# Course: Massively Parallel Programming on GPUs
# Teacher: Donato D'Ambrosio
# Department of Mathematics and Computer Science, University of Calabria
# 
# Targets:
#   make         - Compiles lab3 using auto-detected or specified architecture
#   make run     - Runs single test execution (M=2048, K=1024, N=512, Block=16x16)
#   make bench   - Runs full automated multi-run benchmark suite (1 warm-up + 4 repetitions)
#   make plots   - Generates publication-quality analytical bar charts in reports/
#   make gtx980  - Compiles specifically for NVIDIA GTX 980 (Maxwell, sm_52, CUDA 11.4)
#   make rtx4080 - Compiles specifically for NVIDIA RTX 4080 (Ada Lovelace, sm_89, CUDA 12/13)
#   make clean   - Removes compiled binaries, temporary data, and generated reports
# ==============================================================================

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(shell test -x $(CUDA_HOME)/bin/nvcc && echo $(CUDA_HOME)/bin/nvcc || which nvcc 2>/dev/null)

# Base compiler flags
# - -Xcompiler -fopenmp : Enables OpenMP for CPU baseline implementations
# - -O3 -std=c++11      : Optimization level and standard selection
# - -I...               : Include paths for local headers and shared repository utilities
NVCC_FLAGS ?= -O3 -std=c++11 -lineinfo -Xcompiler -fopenmp -I. -I.. -I../common -I../common/include -I../../common/include

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
    CUDA_VER_MAJOR := $(shell $(NVCC) --version 2>/dev/null | grep -o 'release [0-9]*' | awk '{print $$2}')
    ifeq ($(shell test $(CUDA_VER_MAJOR) -ge 12 2>/dev/null && echo yes),yes)
        GENCODE := -arch=sm_89
    else ifeq ($(shell test $(CUDA_VER_MAJOR) -eq 13 2>/dev/null && echo yes),yes)
        GENCODE := -arch=sm_89
    else
        GENCODE := -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52
    endif
endif

TARGETS := lab3

.PHONY: all clean run bench plots gtx980 rtx4080 universal help

all: $(TARGETS)

lab3: lab3.cu common.h
	@echo "Compiling $< -> $@ ($(GENCODE))..."
	$(NVCC) $(NVCC_FLAGS) $(GENCODE) $< -o $@

run: lab3
	./lab3

bench: lab3
	python3 scripts/average_benchmarks.py

plots:
	python3 scripts/generate_plots.py

gtx980:
	$(MAKE) ARCH=sm_52

rtx4080:
	$(MAKE) ARCH=sm_89 NVCC=/usr/local/cuda-13.0/bin/nvcc

universal:
	$(MAKE) ARCH=universal NVCC=/usr/bin/nvcc

clean:
	rm -f $(TARGETS) *.o *.prof output*.png
	rm -rf reports/*.csv reports/*.md reports/*.png

help:
	@echo "Available commands:"
	@echo "  make         - Compile lab3 for current environment"
	@echo "  make run     - Run lab3 executable (all modes, default dimensions)"
	@echo "  make bench   - Run automated 4-run benchmark suite across CPU & GPU"
	@echo "  make plots   - Generate Cairo analytical bar plots in reports/"
	@echo "  make gtx980  - Compile for GTX 980 (sm_52, Maxwell, CUDA 11.4)"
	@echo "  make rtx4080 - Compile for RTX 4080 (sm_89, Ada, CUDA 13.0)"
	@echo "  make clean   - Remove binaries and temporary artifacts"
