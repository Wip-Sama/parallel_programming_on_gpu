# Lab 3: Naive Matrix Multiplication Performance Benchmark Report

- **Course**: Massively Parallel Programming on GPUs
- **Teacher**: Donato D'Ambrosio
- **Dimensions**: $M=2048, K=1024, N=512$ ($2.147\text{ GFLOP}$)
- **Repetitions**: 4 runs (following 1 warm-up run)
- **CPU Threads**: 8 OpenMP threads

## Consolidated Benchmark Results

| Implementation | Block / Threads | Kernel Time (ms) | Total Time (ms) | Throughput (GFLOP/s) | Speedup vs CPU Std | Speedup vs CPU Opt | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CPU OpenMP (Standard) | - (8) | 312.6489 ± 4.3013 | 312.6489 ± 4.3013 | 6.87 | 1.00x | 14.88x | PASS |
| CPU OpenMP (Optimized I-K-J) | - (8) | 21.0125 ± 3.9534 | 21.0125 ± 3.9534 | 102.20 | 14.88x | 1.00x | PASS |
| CUDA Naive 8x8 | 8x8 (64) | 161.5675 ± 22.0410 | 168.4563 ± 22.8717 | 13.29 | 1.94x | 0.13x | PASS |
| CUDA Naive 16x16 | 16x16 (256) | 57.8962 ± 5.1631 | 64.4457 ± 5.3046 | 37.09 | 5.40x | 0.36x | PASS |
| CUDA Naive 32x32 | 32x32 (1024) | 31.4960 ± 2.0127 | 38.9293 ± 2.6534 | 68.18 | 9.93x | 0.67x | PASS |


## Key Observations

1. **CPU Memory Optimization**: Loop interchange (i-k-j) improves CPU performance by **14.88x** over the standard i-j-k baseline due to stride-1 continuous cache-line streaming across matrix $B$.
2. **CUDA Block Dimension Impact**: Block size **32x32** (1024 threads/block) achieved the fastest GPU execution time of **31.4960 ms** (68.18 GFLOP/s), providing a **9.93x** speedup over the standard CPU baseline.
3. **Memory Coalescing in Naive Kernel**: In the global memory naive implementation, column threads access consecutive elements of $B$, but each thread redundantly loads $A$ from global memory across the $K$ iterations. Tiled matrix multiplication with shared memory (Lab 3 Point 4.2) will address this reuse pattern.
