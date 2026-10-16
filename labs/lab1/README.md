# Lab 1: First CUDA Applications

**Course:** Massively Parallel Programming on GPUs  
**Instructor:** Donato D'Ambrosio  
**Department:** Mathematics and Computer Science, University of Calabria  

---

## 1. Overview & Architecture

This directory contains the refactored CUDA implementation, compilation system, and automated multi-run benchmark suite for **Lab 1**.

The CUDA binary (`./lab1`) executes a single chosen configuration directly with minimal overhead, suitable for direct execution or profiling tools (such as `nvprof` and `ncu`). All parameter sweeps, multi-run repetitions, and statistical comparisons are orchestrated by Python (`average_benchmarks.py`).

```bash
./lab1 <u|s> <m|gs> <vector|matrix> [blockSize] [options]
```

### The 4 Implementations

| Parameter 1 (`u`/`s`) | Parameter 2 (`m`/`gs`) | Operation (`vector`/`matrix`) | Kernel Names | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`u` (Unified)** | **`m` (Monolithic)** | `vector` / `matrix` | `vectorInit`, `vectorAdd`, `matrixInit`, `matrixAdd` | Managed memory (`cudaMallocManaged`) with 1-to-1 thread grid mapping |
| **`u` (Unified)** | **`gs` (Grid-Stride)** | `vector` / `matrix` | `vectorInitGSL`, `vectorAddGSL`, `matrixInitGSL`, `matrixAddGSL` | Managed memory with decoupled SM-scaled grid loops |
| **`s` (Standard)** | **`m` (Monolithic)** | `vector` / `matrix` | `vectorInit`, `vectorAdd`, `matrixInit`, `matrixAdd` | Explicit host/device allocations and `cudaMemcpy` (DtoH) transfer |
| **`s` (Standard)** | **`gs` (Grid-Stride)** | `vector` / `matrix` | `vectorInitGSL`, `vectorAddGSL`, `matrixInitGSL`, `matrixAddGSL` | Explicit memory with decoupled SM-scaled grid loops |

---

## 2. Command Line Interface: `./lab1`

### Syntax
```bash
./lab1 <u|s> <m|gs> <vector|matrix> [blockSize] [options]
```

### Positional Arguments (Single Configuration)
1. **Memory Model (`u` or `s`):**
   - `u` / `unified`: Unified Memory (`cudaMallocManaged`)
   - `s` / `standard`: Standard Memory (`cudaMalloc` + host `malloc` + explicit `cudaMemcpy` DtoH)
2. **Kernel Execution Model (`m` or `gs`):**
   - `m` / `monolithic`: Monolithic 1-to-1 thread mapping
   - `gs` / `grid_stride`: Grid-Stride Loops (`stride = gridDim * blockDim`, grid scaled to $32 \times \text{numSMs}$)
3. **Operation (`vector` or `matrix`):**
   - `vector` / `vec` / `v`: Double-precision vector addition ($n = 2^{24} = 16,777,216$ elements, 128 MB/vector)
   - `matrix` / `mat` / `m`: Double-precision matrix addition ($2048 \times 4096 = 8,388,608$ elements, 64 MB/matrix)
4. **`[blockSize]` (Optional Block Size):**
   - `32`, `64`, `128`, `256`, `512`, `1024`: Thread block size (default: 256)

### Options
- `--device <id>`: Select CUDA device ID (default: 0)
- `--csv`: Output pure CSV line for automated scripting and benchmarking
- `-h`, `--help`: Display usage and options

### Single-Run Examples (Ideal for Profiling)
```bash
./lab1 u m vector 256        # Unified memory, monolithic, vector addition, block size 256
./lab1 s gs matrix 512       # Standard memory, grid-stride loop, matrix addition, block size 512
./lab1 u gs vector 1024      # Unified memory, grid-stride loop, vector addition, block size 1024
```

---

## 3. Automated Multi-Run Benchmark Suite (`average_benchmarks.py`)

All parameter sweeps, repetitions, and comparisons are handled by `average_benchmarks.py`:

```bash
# Run full benchmark suite across all configurations, operations, and block sizes (5 runs each):
python3 average_benchmarks.py -n 5

# Or use the shell helper:
./run_comparisons.sh 5

# Test a specific block size across all 4 configurations:
python3 average_benchmarks.py -n 5 -s 256

# Test only vector addition:
python3 average_benchmarks.py -n 5 --op vector

# Test only unified memory with grid-stride loops:
python3 average_benchmarks.py -n 5 --mem u --model gs
```

### Metrics Reported
- **`Init (ms)`**: GPU time for 3x initialization kernel launches.
- **`Add (ms)`**: GPU time for the 1x vector/matrix addition kernel launch.
- **`Memcpy (ms)`**: Explicit `cudaMemcpy` (DtoH) transfer time for Standard Memory (0.000 ms for Unified Memory).
- **`Device Time (ms)`**: Actual time the GPU device spent working (kernels + explicit transfers via CUDA events).
- **`Total Run Time (ms)`**: Total end-to-end execution time including host verification. For Unified Memory, this captures on-demand page fault migrations across the PCIe bus when the host accesses data.
- **`Status`**: `PASS` if verified ($C[i] = 3.0$), `FAIL` otherwise.

### Generated Artifacts
- **Console Tables:** Formatted summary tables showing means and standard deviations ($\text{mean} \pm \text{std}$).
- **CSV Data File:** Exported to `benchmark_results_averaged.csv`.
- **Markdown Report:** Comprehensive report and architectural analysis exported to `benchmark_findings.md`.

---

## 4. Building the Project

A clean `Makefile` is provided:

```bash
# Auto-detect CUDA compiler and architecture:
make

# Target specifically for NVIDIA GTX 980 (Maxwell, sm_52, CUDA 11.4):
make gtx980

# Target specifically for NVIDIA RTX 4080 (Ada Lovelace, sm_89, CUDA 13.0):
make rtx4080

# Universal fatbin build (sm_52 + compute_52 PTX):
make universal

# Run a quick test:
make run

# Clean binaries:
make clean
```

---

## 5. Profiling with `nvprof` and `ncu`

### NVIDIA Profiler (`nvprof`) on CUDA $\le 11.x$ (e.g., GTX 980):
```bash
# Profile Vector Addition with block size 256:
nvprof ./lab1 s m vector 256

# Profile Matrix Addition with block size 256:
nvprof ./lab1 s gs matrix 256
```

### NVIDIA Nsight Compute (`ncu`) on modern GPUs (e.g., RTX 4080):
```bash
ncu --target-processes all ./lab1 s m vector 256
ncu --target-processes all ./lab1 u gs vector 256
```
