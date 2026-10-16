# ==============================================================================
# Makefile for Lab 1: First CUDA Applications
# Course: Massively Parallel Programming on GPUs
# 
# Targets:
#   make              - Compiles lab1 using auto-detected or specified architecture
#   make gtx980       - Compiles specifically for NVIDIA GTX 980 (Maxwell, sm_52, CUDA 11.4)
#   make rtx4080      - Compiles specifically for NVIDIA RTX 4080 (Ada Lovelace, sm_89, CUDA 12/13)
#   make universal    - Compiles with fatbin (sm_52 + compute_52 PTX) for cross-compatibility
#   make run          - Runs all 4 benchmarks sequentially
#   make clean        - Removes compiled binaries and temporary benchmark data
# ==============================================================================

CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(shell test -x $(CUDA_HOME)/bin/nvcc && echo $(CUDA_HOME)/bin/nvcc || which nvcc 2>/dev/null)

# Base compiler flags
NVCC_FLAGS ?= -O3 -std=c++11 -lineinfo

# Detect default architecture if not provided
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

TARGETS := lab1
LEGACY_ALIASES := unified_monolithic unified_grid_stride standard_monolithic standard_grid_stride

.PHONY: all clean run gtx980 rtx4080 universal help

all: $(TARGETS)

lab1: lab1.cu common.h benchmark.h
	@echo "Compiling $< -> $@ ($(GENCODE))..."
	$(NVCC) $(NVCC_FLAGS) $(GENCODE) $< -o $@
	@ln -sf lab1 unified_monolithic
	@ln -sf lab1 unified_grid_stride
	@ln -sf lab1 standard_monolithic
	@ln -sf lab1 standard_grid_stride

gtx980:
	$(MAKE) ARCH=sm_52

rtx4080:
	$(MAKE) ARCH=sm_89 NVCC=/usr/local/cuda-13.0/bin/nvcc

universal:
	$(MAKE) ARCH=universal NVCC=/usr/bin/nvcc

run: all
	./lab1 all

clean:
	rm -f $(TARGETS) $(LEGACY_ALIASES) *.o *.prof

help:
	@echo "Available commands:"
	@echo "  make              - Compile lab1 for current environment"
	@echo "  make gtx980       - Compile for GTX 980 (sm_52, Maxwell, CUDA 11.4)"
	@echo "  make rtx4080      - Compile for RTX 4080 (sm_89, Ada, CUDA 13.0)"
	@echo "  make universal    - Compile with sm_52 + PTX to run on both GPUs"
	@echo "  make run          - Run full benchmark sweep with ./lab1 all"
	@echo "  make clean        - Remove binaries and build artifacts"
