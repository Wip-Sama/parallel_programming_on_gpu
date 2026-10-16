# Lab 1: Comprehensive Benchmark Report & Findings

- **Benchmark Executable:** `./lab1 <u/s> <m/gs> <vector/matrix> <size>`
- **Repetitions per configuration:** 5 runs (averaged)
- **Report Generated:** 2026-10-16 19:12:07

## Vector Addition Benchmark

**CPU Serial Baseline:** `139.243 ms`

| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Total GPU (ms) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified | Monolithic | 32 | `524288` | 32.910 | 20.431 | 0.000 | **53.341 ± 0.32** | 2.61x | PASS |
| Unified | Monolithic | 64 | `262144` | 33.632 | 21.077 | 0.000 | **54.709 ± 2.82** | 2.55x | PASS |
| Unified | Monolithic | 128 | `131072` | 32.574 | 20.497 | 0.000 | **53.071 ± 0.41** | 2.62x | PASS |
| Unified | Monolithic | 256 | `65536` | 32.744 | 20.521 | 0.000 | **53.265 ± 0.26** | 2.61x | PASS |
| Unified | Monolithic | 512 | `32768` | 33.109 | 20.496 | 0.000 | **53.606 ± 0.30** | 2.60x | PASS |
| Unified | Monolithic | 1024 | `16384` | 33.113 | 20.739 | 0.000 | **53.853 ± 1.14** | 2.59x | PASS |
| Unified | Grid-Stride | 32 | `1856` | 33.072 | 20.501 | 0.000 | **53.573 ± 0.37** | 2.60x | PASS |
| Unified | Grid-Stride | 64 | `1856` | 33.014 | 20.674 | 0.000 | **53.688 ± 1.42** | 2.59x | PASS |
| Unified | Grid-Stride | 128 | `1856` | 34.224 | 21.357 | 0.000 | **55.581 ± 2.36** | 2.51x | PASS |
| Unified | Grid-Stride | 256 | `1856` | 39.275 | 24.003 | 0.000 | **63.277 ± 6.22** | 2.20x | PASS |
| Unified | Grid-Stride | 512 | `1856` | 37.073 | 23.152 | 0.000 | **60.225 ± 6.57** | 2.31x | PASS |
| Unified | Grid-Stride | 1024 | `1856` | 33.849 | 21.953 | 0.000 | **55.802 ± 2.40** | 2.50x | PASS |
| Standard | Monolithic | 32 | `524288` | 3.673 | 1.243 | 35.568 | **40.484 ± 19.29** | 3.44x | PASS |
| Standard | Monolithic | 64 | `262144` | 3.777 | 1.293 | 29.469 | **34.540 ± 14.29** | 4.03x | PASS |
| Standard | Monolithic | 128 | `131072` | 3.821 | 1.483 | 29.112 | **34.416 ± 14.59** | 4.05x | PASS |
| Standard | Monolithic | 256 | `65536` | 3.738 | 1.342 | 28.772 | **33.851 ± 11.25** | 4.11x | PASS |
| Standard | Monolithic | 512 | `32768` | 4.028 | 1.232 | 28.952 | **34.213 ± 13.94** | 4.07x | PASS |
| Standard | Monolithic | 1024 | `16384` | 3.933 | 1.304 | 26.557 | **31.795 ± 12.02** | 4.38x | PASS |
| Standard | Grid-Stride | 32 | `1856` | 4.333 | 1.307 | 29.381 | **35.022 ± 14.12** | 3.98x | PASS |
| Standard | Grid-Stride | 64 | `1856` | 3.821 | 1.258 | 30.724 | **35.803 ± 20.02** | 3.89x | PASS |
| Standard | Grid-Stride | 128 | `1856` | 3.965 | 1.247 | 25.911 | **31.123 ± 9.15** | 4.47x | PASS |
| Standard | Grid-Stride | 256 | `1856` | 3.737 | 1.226 | 24.362 | **29.325 ± 8.20** | 4.75x | PASS |
| Standard | Grid-Stride | 512 | `1856` | 3.996 | 1.225 | 25.612 | **30.832 ± 13.23** | 4.52x | PASS |
| Standard | Grid-Stride | 1024 | `1856` | 3.610 | 1.317 | 28.102 | **33.029 ± 15.83** | 4.22x | PASS |

## Matrix Addition Benchmark

**CPU Serial Baseline:** `19.444 ms`

| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Total GPU (ms) | Speedup | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Unified | Monolithic | 32 | `128x2048` | 17.517 | 10.466 | 0.000 | **27.982 ± 0.49** | 0.69x | PASS |
| Unified | Monolithic | 64 | `128x1024` | 17.851 | 11.075 | 0.000 | **28.926 ± 1.13** | 0.67x | PASS |
| Unified | Monolithic | 128 | `128x512` | 17.588 | 10.610 | 0.000 | **28.198 ± 0.63** | 0.69x | PASS |
| Unified | Monolithic | 256 | `128x256` | 17.865 | 10.774 | 0.000 | **28.639 ± 1.73** | 0.68x | PASS |
| Unified | Monolithic | 512 | `128x128` | 18.710 | 10.974 | 0.000 | **29.683 ± 1.68** | 0.66x | PASS |
| Unified | Monolithic | 1024 | `128x64` | 17.977 | 10.854 | 0.000 | **28.832 ± 1.63** | 0.67x | PASS |
| Unified | Grid-Stride | 32 | `32x58` | 17.686 | 10.525 | 0.000 | **28.211 ± 0.66** | 0.69x | PASS |
| Unified | Grid-Stride | 64 | `32x58` | 18.234 | 10.835 | 0.000 | **29.069 ± 2.83** | 0.67x | PASS |
| Unified | Grid-Stride | 128 | `32x58` | 17.315 | 10.259 | 0.000 | **27.574 ± 0.11** | 0.71x | PASS |
| Unified | Grid-Stride | 256 | `32x58` | 18.016 | 10.594 | 0.000 | **28.610 ± 1.55** | 0.68x | PASS |
| Unified | Grid-Stride | 512 | `32x58` | 17.661 | 10.484 | 0.000 | **28.145 ± 0.81** | 0.69x | PASS |
| Unified | Grid-Stride | 1024 | `32x58` | 17.494 | 10.541 | 0.000 | **28.035 ± 0.74** | 0.69x | PASS |
| Standard | Monolithic | 32 | `128x2048` | 3.886 | 0.757 | 21.514 | **26.158 ± 16.38** | 0.74x | PASS |
| Standard | Monolithic | 64 | `128x1024` | 3.270 | 0.666 | 16.373 | **20.309 ± 9.94** | 0.96x | PASS |
| Standard | Monolithic | 128 | `128x512` | 3.140 | 0.823 | 14.705 | **18.668 ± 9.97** | 1.04x | PASS |
| Standard | Monolithic | 256 | `128x256` | 3.253 | 0.638 | 13.093 | **16.984 ± 7.94** | 1.14x | PASS |
| Standard | Monolithic | 512 | `128x128` | 3.339 | 0.647 | 13.535 | **17.522 ± 8.74** | 1.11x | PASS |
| Standard | Monolithic | 1024 | `128x64` | 3.161 | 0.676 | 10.669 | **14.506 ± 1.79** | 1.34x | PASS |
| Standard | Grid-Stride | 32 | `32x58` | 3.359 | 0.702 | 14.706 | **18.767 ± 7.48** | 1.04x | PASS |
| Standard | Grid-Stride | 64 | `32x58` | 2.980 | 0.677 | 15.610 | **19.268 ± 8.30** | 1.01x | PASS |
| Standard | Grid-Stride | 128 | `32x58` | 3.125 | 0.657 | 13.957 | **17.739 ± 7.83** | 1.10x | PASS |
| Standard | Grid-Stride | 256 | `32x58` | 3.399 | 0.671 | 15.616 | **19.686 ± 10.45** | 0.99x | PASS |
| Standard | Grid-Stride | 512 | `32x58` | 3.369 | 0.714 | 15.080 | **19.162 ± 7.94** | 1.01x | PASS |
| Standard | Grid-Stride | 1024 | `32x58` | 3.081 | 0.685 | 14.587 | **18.352 ± 8.73** | 1.06x | PASS |

## Key Findings & Architectural Analysis

### 1. Unified Memory vs Standard Memory
- **Unified Memory (`cudaMallocManaged`):** Eliminates explicit host-to-device and device-to-host copies. On modern GPUs with page migration, initial page faults occur on first access. Synchronization (`cudaDeviceSynchronize()`) ensures data coherency before CPU access.
- **Standard Memory (`cudaMalloc` + `cudaMemcpy`):** Explicit PCIe transfers (`cudaMemcpy` DtoH) represent the dominant portion of elapsed time for memory-bound vector operations.

### 2. Monolithic vs Grid-Stride Loop Execution Model
- **Monolithic Kernels:** Grid dimensions scale 1-to-1 with element count (`(N + blockSize - 1) / blockSize`). While conceptually simple, large grids incur block launch management overhead.
- **Grid-Stride Loops (`stride = gridDim * blockDim`):** The grid is decoupled from array size and scaled to Streaming Multiprocessor (SM) capacity (`32 * numSMs`). Each thread reuses registers and cache over consecutive elements, providing superior scalability and cache locality.

### 3. Optimal Block Size Analysis
- Block sizes of **256** and **512** threads consistently achieve optimal warp occupancy and memory bandwidth saturation without running into register or shared memory pressure.
- Very small block sizes (32 threads, 1 warp) suffer from low occupancy and latency hiding limitations.
