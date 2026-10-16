# Lab 1: First CUDA Applications

**Course:** Massively Parallel Programming on GPUs  
**Instructor:** Donato D'Ambrosio  
**Department:** Mathematics and Computer Science, University of Calabria  

---

## 1. Overview & Implementations

This directory contains the complete source code, compilation system, and benchmark harness for **Lab 1**. Per the lab specifications, four distinct CUDA implementations of double-precision vector addition ($n = 2^{24}$) and matrix addition are provided, comparing execution models and memory management strategies against a CPU serial baseline:

| File | Memory Management | Kernel Execution Model | Description |
| :--- | :--- | :--- | :--- |
| **`unified_monolithic.cu`** | Unified Memory (`cudaMallocManaged`) | Monolithic (1 thread/element) | Managed memory with 1-to-1 grid mapping |
| **`unified_grid_stride.cu`** | Unified Memory (`cudaMallocManaged`) | Grid-Stride Loops | Managed memory with decoupled SM-scaled grid |
| **`standard_monolithic.cu`** | Standard Memory (`cudaMalloc` / `cudaMemcpy`) | Monolithic (1 thread/element) | Explicit DtoH copy and 1-to-1 grid mapping |
| **`standard_grid_stride.cu`** | Standard Memory (`cudaMalloc` / `cudaMemcpy`) | Grid-Stride Loops | Explicit DtoH copy and decoupled SM-scaled grid |

---

## 2. Key Architecture & Design Details

### A. Initialization & Computational Kernels
- **GPU-side Initialization:** As required by Section 3, all input buffers $A, B, C$ are initialized on the GPU via three calls to `vectorInit()` (or `matrixInit()`):
  - $A$ initialized to `1.0`
  - $B$ initialized to `2.0`
  - $C$ initialized to `0.0`
- **Computation:** A single call to `vectorAdd()` (or `matrixAdd()`) computes $C = A + B$.
- **Correctness Check:** The host CPU validates that every element $C[i] == 3.0$ within a floating-point tolerance of $10^{-9}$.

### B. Monolithic vs Grid-Stride Loop Execution
1. **Monolithic Kernels:**
   - Vector: Grid is explicitly sized as `(n + blockSize - 1) / blockSize` blocks.
   - Matrix: 2D grid covering $(nrows, ncols)$ using 2D block dimensions `(32, blockSize / 32)`. The 32-thread $X$-dimension ensures 100% coalesced memory access across contiguous columns.
2. **Grid-Stride Loop Kernels:**
   - Grid size is decoupled from the array size. Instead of launching millions of blocks, the grid is scaled to the number of Streaming Multiprocessors ($32 \times \text{numSMs}$ blocks).
   - Each thread iterates through the array with a stride of `gridDim.x * blockDim.x`.
   - Offers scalability, high instruction cache reuse, and out-of-bounds safety.

### C. Unified Memory vs Standard Memory on GTX 980 (Maxwell, CC 5.2)
- **Unified Memory:** Buffers are allocated with `cudaMallocManaged()`. On Maxwell architectures (GTX 980, Compute Capability 5.2), there is no hardware page-faulting support. The driver migrates pages upon kernel launch. `cudaDeviceSynchronize()` is required before the CPU accesses $C$ to avoid segmentation faults.
- **Standard Memory:** Buffers $d_A, d_B, d_C$ are allocated with `cudaMalloc()`. Results are transferred to host via `cudaMemcpy(h_C, d_C, ..., cudaMemcpyDeviceToHost)`.

### D. Matrix Addition Sizing Note
Section 4 states: `(nrows, ncols) = (n/2, n), being n the number of vector elements in the vector addition problem`.
- If $n = 2^{24}$ were literally substituted as dimensions $(2^{23}, 2^{24})$, three matrices of `double` would require over **1.1 Petabytes** of RAM, which exceeds the memory capacity of any system.
- To maintain the requested $(n_{matrix}/2, n_{matrix})$ aspect ratio while fitting safely in GPU VRAM (4 GB on GTX 980, 12 GB on RTX 4080), matrix addition defaults to **$nrows = 2048, ncols = 4096$** ($8,388,608$ doubles, 64 MB per matrix, 192 MB total).
- Custom matrix dimensions can also be supplied via command line arguments.

---

## 3. Building the Programs

A universal `Makefile` is included. It automatically detects your environment and supports specific target flags:

### Compiling on the Dual GTX 980 Machine (CUDA 11.4):
The NVIDIA GeForce GTX 980 uses the **Maxwell** architecture with **Compute Capability 5.2 (`sm_52`)**.
```bash
cd labs/lab1
make gtx980
```
*(Or simply run `make` on that machine; the Makefile defaults to `sm_52` on CUDA 11).*

### Compiling on the RTX 4080 Mobile Machine (CUDA 13.0 / 12.x):
The NVIDIA GeForce RTX 4080 Laptop GPU uses the **Ada Lovelace** architecture with **Compute Capability 8.9 (`sm_89`)**.
```bash
cd labs/lab1
make rtx4080
```

### Universal Build (sm_52 + compute_52 PTX):
Generates a fat binary compatible with GTX 980 that also JIT-compiles on RTX 4080:
```bash
make universal
```

---

## 4. Running Benchmarks & Comparisons

### Mode 1: Automated Comparison Sweep (All Block Sizes)
Running any of the executables without arguments executes:
1. High-resolution CPU serial baseline timing.
2. An automated sweep across all 6 block sizes: `32, 64, 128, 256, 512, 1024`.
3. High-resolution timing of $3\times$ Init, $1\times$ Add, and Memcpy (for Standard Memory).
4. Speedup calculations against CPU serial execution.
5. Correctness checks on CPU.

```bash
./unified_monolithic
./unified_grid_stride
./standard_monolithic
./standard_grid_stride
```

Or run all 4 in sequence with:
```bash
./run_comparisons.sh
```

### Mode 2: Single-Run Execution (Designed for `nvprof` and `ncu`)
Per the PDF assignment:
> *"Assess the CUDA implementations on one of the Nvidia GTX 980 GPU installed on the JPDM2 workstation using blockSize = 32, 64, 128, 256, 512 and 1024, by summing the elapsed times of the host-device memory copies, the three calls to matrixInit and the call to matrixAdd through nvprof."*

When given a block size argument, the programs execute **exactly one run** (ideal for profilers so nvprof logs exactly 3 calls to Init and 1 call to Add):

```bash
# Vector addition with block size 256:
./unified_monolithic vector 256
./standard_monolithic vector 256

# Matrix addition with block size 256:
./unified_monolithic matrix 256
./standard_monolithic matrix 256

# Selecting GPU on dual-GPU machines (e.g. GTX 980 #0 or #1):
./unified_monolithic --device 1 vector 256
```

---

## 5. Profiling Instructions

### On the Target Machine (2x GTX 980, Compute Capability 5.2, CUDA 11.4):
Use `nvprof` as requested in the assignment:

```bash
# Profile Vector Addition with block size 256 on device 0:
nvprof ./standard_monolithic vector 256

# Profile Matrix Addition with block size 256:
nvprof ./standard_monolithic matrix 256

# Running on the second GTX 980:
CUDA_VISIBLE_DEVICES=1 nvprof ./standard_monolithic vector 256
```

`nvprof` will output a table containing:
- `vectorInit` (Calls: 3)
- `vectorAdd` (Calls: 1)
- `[CUDA memcpy DtoH]` (Calls: 1, for standard memory)

Summing the execution times of these activities provides the total GPU elapsed time requested by the lab.

### On the Testing Machine (RTX 4080 Laptop, Compute Capability 8.9):
Because compute capability $\ge 8.0$ GPUs are not supported by the legacy `nvprof` tool, the programs include built-in CUDA Event timers (`cudaEventElapsedTime`) that report the exact millisecond breakdown for Init, Add, and Memcpy. You can also profile with `ncu` (NVIDIA Nsight Compute):
```bash
ncu --target-processes all ./standard_monolithic vector 256
```

