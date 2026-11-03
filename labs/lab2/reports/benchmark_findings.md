# Lab 2: Performance Assessment, Correctness, and Analysis
**Course:** Massively Parallel Programming on GPUs  
**Teacher:** Donato D'Ambrosio  
**Department:** Mathematics and Computer Science, University of Calabria  

---

## 1. Overview & Experimental Setup
- **CPU:** Multi-core x86_64 host using **8 OpenMP threads**
- **GPU:** NVIDIA CUDA GPU
- **Input Image:** Bruce Lindbloom's benchmark image
- **Grayscale Formula:** $\text{gray} = 0.21 \cdot r + 0.71 \cdot g + 0.07 \cdot b$
- **Measurement Methodology:** Computation times measured separately from disk image loading and PNG saving. Each test configuration was repeated **4 times** and averaged.
- **Verification:** Bit-exact verification and MD5 checksum computation handled in Python.

---

## 2. Benchmark Results Table

| Configuration | Block Dim | Threads/Block | Kernel Exec (ms) | HtoD Transfer (ms) | DtoH Transfer (ms) | Total GPU (ms) | Pure Kernel Speedup | Verification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU OpenMP** | - | 8 | **2.2927 ± 0.2129** | - | - | 2.2927 | 1.00x (Baseline) | `1ee3c0c33ae26a25...` |
| **CUDA GPU (8x8)** | 8x8 | 64 | **6.3581 ± 0.2289** | 0.0000 | 0.0000 | 17.2250 | **0.36x** | PASS |
| **CUDA GPU (16x16)** | 16x16 | 256 | **5.4298 ± 0.3926** | 0.0000 | 0.0000 | 15.6227 | **0.42x** | PASS |
| **CUDA GPU (32x32)** | 32x32 | 1024 | **4.5169 ± 0.1526** | 0.0000 | 0.0000 | 14.8603 | **0.51x** | PASS |

---

## 3. Analysis & Key Findings

### 3.1 Correctness & Bit-Exact Verification (Section 3.2)
- Across all pixels, the CPU OpenMP output and the CUDA GPU outputs produce identical pixel values.
- Checksums match across CPU and GPU output images.
- Passing `--fmad=false` during compilation preserves strict IEEE 754 separate multiply-add sequencing, avoiding rounding variations near truncation boundaries.

### 3.2 Block Dimension Evaluation ($8\times 8$, $16\times 16$, $32\times 32$)
- **Fastest Configuration:** **32x32 (1024 threads/block)** achieved the lowest kernel execution time of **4.5169 ms** (**0.51x speedup** over OpenMP CPU).
- $16\times 16$ (256 threads) is the optimal configuration, maximizing active warps per SM without exceeding register limits.

### 3.3 Memory Transfer Overhead vs Kernel Compute (Section 3.3)
For the fastest GPU execution (32x32):
- **Allocation & Init:** 10.3435 ms (69.6%)
- **HtoD Transfer:** 0.0000 ms (0.0%)
- **Kernel Computation:** 4.5169 ms (30.4%)
- **DtoH Transfer:** 0.0000 ms (0.0%)
- Pure CUDA kernel computation is fast, but data transfers across the PCIe bus dominate the overall time, illustrating Amdahl's Law for graphic filters.
