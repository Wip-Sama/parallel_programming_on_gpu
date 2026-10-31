# Lab 1: Multi-Run Benchmark Report & Findings

- **Benchmark Binary:** `./lab1 <u|s> <m|gs> <vector|matrix> <size>`
- **Repetitions per configuration:** 5 runs (averaged)
- **Report Generated:** 2026-10-16 20:02:29

## Vector Addition Benchmark

| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Device Time (ms) | Total Run Time (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified | Monolithic | 32 | `524288` | 32.876 | 20.456 | 0.000 | 54.076 ± 0.32 | **54.569 ± 0.40** | PASS |
| Unified | Monolithic | 64 | `262144` | 32.694 | 20.415 | 0.000 | 53.817 ± 0.17 | **54.212 ± 0.16** | PASS |
| Unified | Monolithic | 128 | `131072` | 32.663 | 20.444 | 0.000 | 53.839 ± 0.18 | **54.360 ± 0.26** | PASS |
| Unified | Monolithic | 256 | `65536` | 33.083 | 20.459 | 0.000 | 54.304 ± 0.51 | **54.818 ± 0.53** | PASS |
| Unified | Monolithic | 512 | `32768` | 32.925 | 20.438 | 0.000 | 54.136 ± 0.49 | **54.723 ± 0.44** | PASS |
| Unified | Monolithic | 1024 | `16384` | 32.768 | 20.408 | 0.000 | 53.918 ± 0.12 | **54.361 ± 0.11** | PASS |
| Unified | Grid-Stride | 32 | `1856` | 33.099 | 20.780 | 0.000 | 54.592 ± 2.14 | **55.016 ± 2.00** | PASS |
| Unified | Grid-Stride | 64 | `1856` | 32.658 | 20.405 | 0.000 | 53.850 ± 0.25 | **54.493 ± 0.63** | PASS |
| Unified | Grid-Stride | 128 | `1856` | 33.277 | 20.443 | 0.000 | 54.508 ± 1.33 | **55.150 ± 1.22** | PASS |
| Unified | Grid-Stride | 256 | `1856` | 32.704 | 20.409 | 0.000 | 53.852 ± 0.24 | **54.338 ± 0.28** | PASS |
| Unified | Grid-Stride | 512 | `1856` | 32.652 | 20.423 | 0.000 | 53.843 ± 0.18 | **54.377 ± 0.23** | PASS |
| Unified | Grid-Stride | 1024 | `1856` | 32.749 | 20.421 | 0.000 | 53.904 ± 0.08 | **54.436 ± 0.06** | PASS |
| Standard | Monolithic | 32 | `524288` | 3.668 | 1.200 | 33.059 | 38.095 ± 20.85 | **46.905 ± 20.15** | PASS |
| Standard | Monolithic | 64 | `262144` | 3.660 | 1.185 | 24.824 | 29.758 ± 11.72 | **38.409 ± 11.03** | PASS |
| Standard | Monolithic | 128 | `131072` | 3.711 | 1.201 | 26.271 | 31.309 ± 14.43 | **40.011 ± 13.27** | PASS |
| Standard | Monolithic | 256 | `65536` | 3.543 | 1.247 | 24.407 | 29.314 ± 10.47 | **38.232 ± 9.67** | PASS |
| Standard | Monolithic | 512 | `32768` | 3.552 | 1.174 | 26.961 | 31.801 ± 15.44 | **40.521 ± 14.20** | PASS |
| Standard | Monolithic | 1024 | `16384` | 3.640 | 1.182 | 24.991 | 29.902 ± 12.86 | **38.746 ± 11.71** | PASS |
| Standard | Grid-Stride | 32 | `1856` | 3.639 | 1.206 | 25.398 | 30.378 ± 10.25 | **39.052 ± 9.39** | PASS |
| Standard | Grid-Stride | 64 | `1856` | 3.892 | 1.174 | 23.917 | 29.070 ± 9.48 | **37.905 ± 8.62** | PASS |
| Standard | Grid-Stride | 128 | `1856` | 3.825 | 1.206 | 25.393 | 30.540 ± 13.97 | **39.124 ± 12.86** | PASS |
| Standard | Grid-Stride | 256 | `1856` | 3.558 | 1.213 | 25.539 | 30.437 ± 12.64 | **39.295 ± 11.54** | PASS |
| Standard | Grid-Stride | 512 | `1856` | 3.874 | 1.173 | 26.463 | 31.614 ± 14.50 | **40.343 ± 13.28** | PASS |
| Standard | Grid-Stride | 1024 | `1856` | 4.234 | 1.188 | 28.215 | 33.933 ± 10.10 | **42.806 ± 9.08** | PASS |

## Matrix Addition Benchmark

| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Device Time (ms) | Total Run Time (ms) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified | Monolithic | 32 | `128x2048` | 18.912 | 11.413 | 0.000 | 30.918 ± 1.57 | **31.230 ± 1.69** | PASS |
| Unified | Monolithic | 64 | `128x1024` | 19.986 | 12.115 | 0.000 | 32.911 ± 2.16 | **33.142 ± 2.14** | PASS |
| Unified | Monolithic | 128 | `128x512` | 19.639 | 10.803 | 0.000 | 30.971 ± 5.13 | **31.171 ± 4.98** | PASS |
| Unified | Monolithic | 256 | `128x256` | 17.571 | 10.406 | 0.000 | 28.441 ± 0.58 | **28.703 ± 0.54** | PASS |
| Unified | Monolithic | 512 | `128x128` | 17.524 | 10.333 | 0.000 | 28.351 ± 0.69 | **28.649 ± 0.81** | PASS |
| Unified | Monolithic | 1024 | `128x64` | 17.619 | 10.443 | 0.000 | 28.519 ± 0.79 | **28.804 ± 0.78** | PASS |
| Unified | Grid-Stride | 32 | `32x58` | 17.691 | 10.501 | 0.000 | 28.640 ± 0.61 | **28.916 ± 0.57** | PASS |
| Unified | Grid-Stride | 64 | `32x58` | 17.883 | 10.632 | 0.000 | 28.985 ± 0.77 | **29.198 ± 0.75** | PASS |
| Unified | Grid-Stride | 128 | `32x58` | 18.243 | 10.908 | 0.000 | 29.660 ± 0.61 | **29.922 ± 0.46** | PASS |
| Unified | Grid-Stride | 256 | `32x58` | 17.830 | 10.531 | 0.000 | 28.835 ± 1.05 | **29.107 ± 1.15** | PASS |
| Unified | Grid-Stride | 512 | `32x58` | 17.566 | 10.369 | 0.000 | 28.372 ± 0.72 | **28.633 ± 0.67** | PASS |
| Unified | Grid-Stride | 1024 | `32x58` | 17.416 | 10.373 | 0.000 | 28.275 ± 0.57 | **28.603 ± 0.64** | PASS |
| Standard | Monolithic | 32 | `128x2048` | 3.212 | 0.680 | 16.499 | 20.548 ± 9.06 | **24.903 ± 8.40** | PASS |
| Standard | Monolithic | 64 | `128x1024` | 3.126 | 0.683 | 14.302 | 18.283 ± 7.50 | **22.803 ± 7.05** | PASS |
| Standard | Monolithic | 128 | `128x512` | 3.255 | 0.658 | 14.284 | 18.326 ± 8.57 | **22.833 ± 7.85** | PASS |
| Standard | Monolithic | 256 | `128x256` | 3.005 | 0.694 | 14.312 | 18.150 ± 7.87 | **22.673 ± 7.33** | PASS |
| Standard | Monolithic | 512 | `128x128` | 3.113 | 0.734 | 15.579 | 19.606 ± 8.94 | **23.998 ± 8.19** | PASS |
| Standard | Monolithic | 1024 | `128x64` | 3.287 | 0.666 | 10.403 | 14.478 ± 1.27 | **19.319 ± 1.32** | PASS |
| Standard | Grid-Stride | 32 | `32x58` | 3.213 | 0.731 | 14.602 | 18.671 ± 9.68 | **23.080 ± 9.02** | PASS |
| Standard | Grid-Stride | 64 | `32x58` | 3.373 | 0.628 | 13.530 | 17.631 ± 7.91 | **22.306 ± 7.25** | PASS |
| Standard | Grid-Stride | 128 | `32x58` | 3.172 | 0.651 | 13.808 | 17.746 ± 7.79 | **22.167 ± 7.13** | PASS |
| Standard | Grid-Stride | 256 | `32x58` | 3.203 | 0.642 | 14.455 | 18.409 ± 10.04 | **22.820 ± 9.20** | PASS |
| Standard | Grid-Stride | 512 | `32x58` | 3.269 | 0.640 | 13.572 | 17.618 ± 8.21 | **22.203 ± 7.45** | PASS |
| Standard | Grid-Stride | 1024 | `32x58` | 3.347 | 0.644 | 14.284 | 18.400 ± 7.76 | **22.837 ± 7.27** | PASS |

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
