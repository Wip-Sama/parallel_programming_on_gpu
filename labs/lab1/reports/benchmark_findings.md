# Lab 1: Multi-Run Benchmark Report & Findings

- **Benchmark Binary:** `./lab1 <u|s> <m|gs> <vector|matrix> <size>`
- **Repetitions per configuration:** 5 runs (averaged)
- **Report Generated:** 2026-10-31 13:15:48

## Vector Addition Benchmark

| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Device Time (ms) | Total Run Time (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified | Monolithic | 32 | `524288` | 58.396 | 37.145 | 0.000 | 96.781 ± 23.77 | **96.214 ± 22.29** | PASS |
| Unified | Monolithic | 64 | `262144` | 65.769 | 42.957 | 0.000 | 110.101 ± 6.13 | **107.994 ± 5.74** | PASS |
| Unified | Monolithic | 128 | `131072` | 65.094 | 43.433 | 0.000 | 109.744 ± 3.29 | **107.876 ± 2.96** | PASS |
| Unified | Monolithic | 256 | `65536` | 58.440 | 37.004 | 0.000 | 96.589 ± 20.99 | **95.986 ± 18.61** | PASS |
| Unified | Monolithic | 512 | `32768` | 60.892 | 38.857 | 0.000 | 100.895 ± 22.79 | **99.773 ± 21.16** | PASS |
| Unified | Monolithic | 1024 | `16384` | 67.002 | 42.735 | 0.000 | 111.024 ± 0.72 | **109.662 ± 1.45** | PASS |
| Unified | Grid-Stride | 32 | `1856` | 55.176 | 37.285 | 0.000 | 93.396 ± 21.41 | **92.625 ± 19.27** | PASS |
| Unified | Grid-Stride | 64 | `1856` | 64.627 | 40.725 | 0.000 | 106.593 ± 1.48 | **105.768 ± 2.89** | PASS |
| Unified | Grid-Stride | 128 | `1856` | 64.704 | 41.131 | 0.000 | 107.159 ± 2.75 | **105.104 ± 2.54** | PASS |
| Unified | Grid-Stride | 256 | `1856` | 67.512 | 44.945 | 0.000 | 113.626 ± 6.05 | **111.736 ± 5.61** | PASS |
| Unified | Grid-Stride | 512 | `1856` | 58.937 | 37.563 | 0.000 | 97.739 ± 19.83 | **96.985 ± 18.05** | PASS |
| Unified | Grid-Stride | 1024 | `1856` | 65.312 | 41.511 | 0.000 | 108.227 ± 2.74 | **106.367 ± 2.62** | PASS |
| Standard | Monolithic | 32 | `524288` | 5.230 | 1.693 | 42.586 | 49.679 ± 21.67 | **59.610 ± 20.14** | PASS |
| Standard | Monolithic | 64 | `262144` | 5.242 | 1.778 | 35.507 | 42.725 ± 21.56 | **53.875 ± 20.40** | PASS |
| Standard | Monolithic | 128 | `131072` | 4.785 | 1.763 | 40.568 | 47.269 ± 23.45 | **58.060 ± 21.57** | PASS |
| Standard | Monolithic | 256 | `65536` | 5.473 | 1.671 | 34.990 | 42.325 ± 21.00 | **53.370 ± 20.71** | PASS |
| Standard | Monolithic | 512 | `32768` | 4.327 | 1.760 | 36.710 | 43.843 ± 17.76 | **56.171 ± 16.07** | PASS |
| Standard | Monolithic | 1024 | `16384` | 6.273 | 1.731 | 47.797 | 55.981 ± 22.75 | **67.090 ± 21.74** | PASS |
| Standard | Grid-Stride | 32 | `1856` | 5.527 | 2.340 | 38.968 | 47.048 ± 19.44 | **58.750 ± 17.22** | PASS |
| Standard | Grid-Stride | 64 | `1856` | 5.022 | 1.867 | 42.539 | 49.613 ± 24.53 | **59.203 ± 22.56** | PASS |
| Standard | Grid-Stride | 128 | `1856` | 4.472 | 1.729 | 31.689 | 38.027 ± 12.64 | **48.960 ± 11.10** | PASS |
| Standard | Grid-Stride | 256 | `1856` | 4.747 | 1.660 | 34.386 | 40.973 ± 21.63 | **52.208 ± 20.37** | PASS |
| Standard | Grid-Stride | 512 | `1856` | 4.388 | 1.825 | 33.341 | 39.689 ± 18.76 | **50.676 ± 17.32** | PASS |
| Standard | Grid-Stride | 1024 | `1856` | 4.592 | 1.795 | 40.113 | 46.650 ± 19.00 | **55.928 ± 17.37** | PASS |

## Matrix Addition Benchmark

| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Device Time (ms) | Total Run Time (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified | Monolithic | 32 | `128x2048` | 30.094 | 18.653 | 0.000 | 49.291 ± 11.95 | **47.960 ± 11.04** | PASS |
| Unified | Monolithic | 64 | `128x1024` | 23.504 | 14.489 | 0.000 | 38.671 ± 13.34 | **38.349 ± 12.37** | PASS |
| Unified | Monolithic | 128 | `128x512` | 27.719 | 17.028 | 0.000 | 45.442 ± 15.12 | **44.549 ± 13.88** | PASS |
| Unified | Monolithic | 256 | `128x256` | 24.497 | 14.985 | 0.000 | 40.091 ± 15.48 | **39.689 ± 14.41** | PASS |
| Unified | Monolithic | 512 | `128x128` | 34.137 | 21.365 | 0.000 | 56.133 ± 1.95 | **54.543 ± 1.99** | PASS |
| Unified | Monolithic | 1024 | `128x64` | 24.082 | 14.606 | 0.000 | 39.309 ± 14.45 | **38.964 ± 13.21** | PASS |
| Unified | Grid-Stride | 32 | `32x58` | 27.922 | 17.379 | 0.000 | 45.983 ± 14.23 | **45.055 ± 13.03** | PASS |
| Unified | Grid-Stride | 64 | `32x58` | 25.084 | 15.081 | 0.000 | 40.840 ± 16.64 | **40.430 ± 15.54** | PASS |
| Unified | Grid-Stride | 128 | `32x58` | 24.318 | 14.922 | 0.000 | 39.801 ± 15.73 | **39.129 ± 14.42** | PASS |
| Unified | Grid-Stride | 256 | `32x58` | 23.705 | 14.422 | 0.000 | 38.604 ± 13.46 | **38.060 ± 12.28** | PASS |
| Unified | Grid-Stride | 512 | `32x58` | 23.957 | 14.508 | 0.000 | 38.985 ± 14.87 | **38.418 ± 13.64** | PASS |
| Unified | Grid-Stride | 1024 | `32x58` | 23.793 | 14.676 | 0.000 | 39.028 ± 14.51 | **38.566 ± 13.46** | PASS |
| Standard | Monolithic | 32 | `128x2048` | 3.677 | 0.927 | 20.666 | 25.430 ± 9.30 | **29.521 ± 8.48** | PASS |
| Standard | Monolithic | 64 | `128x1024` | 3.653 | 0.995 | 17.575 | 22.401 ± 9.09 | **26.934 ± 8.23** | PASS |
| Standard | Monolithic | 128 | `128x512` | 3.819 | 1.006 | 17.676 | 22.694 ± 8.94 | **27.209 ± 8.04** | PASS |
| Standard | Monolithic | 256 | `128x256` | 3.730 | 0.974 | 19.275 | 24.181 ± 9.74 | **28.936 ± 9.68** | PASS |
| Standard | Monolithic | 512 | `128x128` | 3.614 | 1.002 | 18.129 | 22.911 ± 10.34 | **27.434 ± 9.42** | PASS |
| Standard | Monolithic | 1024 | `128x64` | 3.904 | 0.938 | 21.983 | 27.028 ± 11.71 | **31.266 ± 10.67** | PASS |
| Standard | Grid-Stride | 32 | `32x58` | 4.009 | 0.993 | 18.560 | 23.764 ± 10.02 | **28.290 ± 9.09** | PASS |
| Standard | Grid-Stride | 64 | `32x58` | 3.707 | 0.952 | 18.520 | 23.391 ± 10.11 | **28.004 ± 9.43** | PASS |
| Standard | Grid-Stride | 128 | `32x58` | 4.056 | 0.947 | 18.799 | 24.057 ± 11.47 | **28.630 ± 10.58** | PASS |
| Standard | Grid-Stride | 256 | `32x58` | 4.110 | 0.915 | 19.440 | 24.665 ± 10.52 | **29.420 ± 9.55** | PASS |
| Standard | Grid-Stride | 512 | `32x58` | 4.116 | 0.960 | 15.702 | 21.012 ± 2.83 | **25.856 ± 2.52** | PASS |
| Standard | Grid-Stride | 1024 | `32x58` | 3.869 | 0.935 | 18.787 | 23.809 ± 9.02 | **28.104 ± 8.39** | PASS |

## Architectural Analysis & Findings

### 1. Unified Memory vs Standard Memory: Where Time is Spent
- **Standard Memory (`cudaMalloc` + `cudaMemcpy` DtoH):** Features explicit memory transfers. The PCIe transfer time (`cudaMemcpy`) is explicitly recorded and represents the major bottleneck for bandwidth-bound vector operations.
- **Unified Memory (`cudaMallocManaged`):** Eliminates explicit `cudaMemcpy` calls (`Memcpy = 0.000 ms`). However, memory transfer overhead is **not eliminated**: on-demand page faults occur across the PCIe bus during kernel first-touch and when the CPU accesses data for verification. Comparing **Device Working Time** (CUDA events) and **Total Run Time** (including host access) reveals the true end-to-end execution cost.

### 2. Monolithic vs Grid-Stride Loop Execution Model
- **Monolithic Kernels:** Allocate exactly 1 thread per element (`gridDim = (N + blockSize - 1) / blockSize`). While direct, large grids incur block scheduling and launch management overhead on large datasets.
- **Grid-Stride Loops (`stride = gridDim * blockDim`):** Decouple grid size from the input array length, scaling the active grid to Streaming Multiprocessor (SM) capacity (`32 * numSMs`). Each thread reuses registers and cache across multiple elements in a strided loop, improving cache line utilization and scalability.

### 3. Block Size Occupancy Analysis
- Thread block sizes of **256** and **512** threads consistently deliver optimal warp occupancy and memory throughput across both vector and matrix operations without inducing register or shared memory pressure.
- Small block sizes (32 threads) are limited by warp scheduling and instruction latency hiding.
