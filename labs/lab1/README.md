# Lab 1: First CUDA Applications

**Course:** Massively Parallel Programming on GPUs  
**Instructor:** Donato D'Ambrosio  
**Department:** Mathematics and Computer Science, University of Calabria  

---

## 1. Overview & Implementations

This directory contains the unified CUDA implementation, multi-architecture compilation system, and automated multi-run benchmark suite for **Lab 1**.

Rather than maintaining four separate files to vary two parameters, all four implementations and their corresponding kernels are unified into a single modular executable (`./lab1`) driven directly by command-line arguments:

```bash
./lab1 <u|s> <m|gs> <vector|matrix> [size] [options]
```

### The 4 Implementations

| Parameter 1 (`u`/`s`) | Parameter 2 (`m`/`gs`) | Operation (`vector`/`matrix`) | Kernel Names | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`u` (Unified)** | **`m` (Monolithic)** | `vector` / `matrix` | `vectorInit`, `vectorAdd`, `matrixInit`, `matrixAdd` | Managed memory (`cudaMallocManaged`) with 1-to-1 thread grid mapping |
| **`u` (Unified)** | **`gs` (Grid-Stride)** | `vector` / `matrix` | `vectorInitGSL`, `vectorAddGSL`, `matrixInitGSL`, `matrixAddGSL` | Managed memory with decoupled SM-scaled grid loops |
| **`s` (Standard)** | **`m` (Monolithic)** | `vector` / `matrix` | `vectorInit`, `vectorAdd`, `matrixInit`, `matrixAdd` | Explicit host/device allocations and `cudaMemcpy` (DtoH) transfer |
| **`s` (Standard)** | **`gs` (Grid-Stride)** | `vector` / `matrix` | `vectorInitGSL`, `vectorAddGSL`, `matrixInitGSL`, `matrixAddGSL` | Explicit memory with decoupled SM-scaled grid loops |

> [!NOTE]
> For backward compatibility with assignment scripts and profilers, symlinks (`unified_monolithic`, `unified_grid_stride`, `standard_monolithic`, `standard_grid_stride`) automatically point to `./lab1` and preset the memory and kernel models based on their invocation name.

---

## 2. Command Line Interface: `./lab1`

### Syntax
```bash
./lab1 <u|s> <m|gs> <vector|matrix> [size] [options]
```

### Positional Arguments
1. **Memory Model (`u` or `s`):**
   - `u` / `unified`: Unified Memory (`cudaMallocManaged`)
   - `s` / `standard`: Standard Memory (`cudaMalloc` + host `malloc` + `cudaMemcpy` DtoH)
   - `all`: Run all memory models
2. **Kernel Execution Model (`m` or `gs`):**
   - `m` / `monolithic`: Monolithic 1-to-1 thread mapping
   - `gs` / `grid_stride`: Grid-Stride Loops (`stride = gridDim * blockDim`, grid scaled to $32 \times \text{numSMs}$)
   - `all`: Run all kernel models
3. **Operation (`vector` or `matrix`):**
   - `vector` / `vec` / `v`: Double-precision vector addition ($n = 2^{24} = 16,777,216$ elements, 128 MB/vector)
   - `matrix` / `mat` / `m`: Double-precision matrix addition ($2048 \times 4096 = 8,388,608$ elements, 64 MB/matrix)
   - `all`: Run both operations
4. **`[size]` (Optional Block Size):**
   - `32`, `64`, `128`, `256`, `512`, `1024`: Run single benchmark with this block size
   - `all` (or omitted): Automatically sweep all 6 block sizes

### Options
- `-n <N>`, `--runs <N>`: Repeat benchmark $N$ times and average the timings directly in C++/CUDA (default: 1)
- `--device <id>`: Select CUDA device ID (default: 0)
- `--csv`: Output pure CSV format for automated benchmarking and scripting
- `--vec-n <N>`: Override default vector size
- `--mat-rows <R>`, `--mat-cols <C>`: Override default matrix dimensions
- `-h`, `--help`: Display usage and options

### Execution Examples
```bash
# 1. Single runs (ideal for profilers like nvprof or ncu):
./lab1 u m vector 256        # Unified, monolithic, vector addition, block size 256
./lab1 s gs matrix 512       # Standard memory, grid-stride loop, matrix addition, block size 512

# 2. In-engine multi-run averaging:
./lab1 s m vector 256 -n 5   # Run 5 times and report average times and speedup

# 3. Automated block size sweep:
./lab1 u gs vector           # Sweeps 32, 64, 128, 256, 512, 1024 for unified grid-stride vector
./lab1 s m matrix            # Sweeps all block sizes for standard monolithic matrix

# 4. Full comparison suite:
./lab1 all                   # Runs all 4 configurations for vector and matrix addition
```

---

## 3. Automated Multi-Run Averaging Suite (`average_benchmarks.py`)

A dedicated Python benchmark runner (`average_benchmarks.py`) is provided to run each configuration $N$ times, compute statistical metrics, print comparison tables, and generate reports.

### Usage
```bash
# Run each configuration 5 times across all block sizes and operations:
./average_benchmarks.py -n 5

# Or use the convenient shell wrapper:
./run_averages.sh 5

# Test a specific block size across all 4 configurations (5 runs each):
./average_benchmarks.py -n 5 -s 256

# Test only vector addition:
./average_benchmarks.py -n 5 --op vector

# Test only a specific configuration:
./average_benchmarks.py -n 5 --mem u --model gs --op vector
```

### Generated Artifacts
- **Terminal Display:** High-resolution formatted tables showing `Init (ms)`, `Add (ms)`, `Memcpy (ms)`, `Total GPU (ms)` ($\text{mean} \pm \text{stddev}$), `Speedup`, and `Status` (PASS/FAIL).
- **Summary Matrix:** Displays the optimal block size and peak speedup for each of the 4 implementations.
- **CSV Data File:** Saves raw and averaged statistics to `benchmark_results_averaged.csv`.
- **Markdown Report:** Generates an executive findings report to `benchmark_findings.md`.

---

## 4. Building the Project

A universal `Makefile` is included:

```bash
# Auto-detect CUDA compiler and architecture:
make

# Target specifically for NVIDIA GTX 980 (Maxwell, sm_52, CUDA 11.4):
make gtx980

# Target specifically for NVIDIA RTX 4080 (Ada Lovelace, sm_89, CUDA 13.0):
make rtx4080

# Universal fatbin build (sm_52 + compute_52 PTX):
make universal

# Clean binaries:
make clean
```

---

## 5. Profiling with `nvprof` and `ncu`

### On the Dual GTX 980 Machine (CUDA 11.4):
Per the assignment requirements, use `nvprof` on single-run executions:

```bash
# Profile Vector Addition with block size 256:
nvprof ./lab1 s m vector 256

# Profile Matrix Addition with block size 256:
nvprof ./lab1 s m matrix 256

# Or using the legacy binary symlink:
nvprof ./standard_monolithic vector 256
```

`nvprof` outputs the exact activity calls:
- `vectorInit` / `vectorInitGSL` (Calls: 3)
- `vectorAdd` / `vectorAddGSL` (Calls: 1)
- `[CUDA memcpy DtoH]` (Calls: 1, for Standard Memory)

### On Modern GPUs (Compute Capability $\ge 8.0$, e.g., RTX 4080):
Use NVIDIA Nsight Compute (`ncu`) or the built-in high-precision CUDA event timers:
```bash
ncu --target-processes all ./lab1 s m vector 256
```
