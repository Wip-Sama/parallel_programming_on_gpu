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
| **CPU OpenMP** | - | 8 | **2.1421 ± 0.2429** | - | - | 2.1421 | 1.00x (Baseline) | `1ee3c0c33ae26a25...` |
| **CUDA GPU (8x8)** | 8x8 | 64 | **1.7084 ± 0.1673** | 3.0296 | 2.5155 | 7.9929 | **1.25x** | PASS |
| **CUDA GPU (16x16)** | 16x16 | 256 | **1.4062 ± 0.3555** | 2.9874 | 2.1187 | 7.2228 | **1.52x** | PASS |
| **CUDA GPU (32x32)** | 32x32 | 1024 | **1.6100 ± 0.2954** | 3.0278 | 2.5261 | 7.9238 | **1.33x** | PASS |

---

## 3. Analysis & Key Findings

### 3.1 Correctness & Bit-Exact Verification (Section 3.2)
- Across all pixels, the CPU OpenMP output and the CUDA GPU outputs produce identical pixel values.
- Checksums match across CPU and GPU output images.
- Passing `--fmad=false` during compilation preserves strict IEEE 754 separate multiply-add sequencing, avoiding rounding variations near truncation boundaries.

### 3.2 Block Dimension Evaluation ($8\times 8$, $16\times 16$, $32\times 32$)
- **Fastest Configuration:** **16x16 (256 threads/block)** achieved the lowest kernel execution time of **1.4062 ms** (**1.52x speedup** over OpenMP CPU).
- $16\times 16$ (256 threads) is the optimal configuration, maximizing active warps per SM without exceeding register limits.

### 3.3 Memory Transfer Overhead vs Kernel Compute (Section 3.3)
For the fastest GPU execution (16x16):
- **Allocation & Init:** 0.7105 ms (9.8%)
- **HtoD Transfer:** 2.9874 ms (41.4%)
- **Kernel Computation:** 1.4062 ms (19.5%)
- **DtoH Transfer:** 2.1187 ms (29.3%)
- Pure CUDA kernel computation is fast, but data transfers across the PCIe bus dominate the overall time, illustrating Amdahl's Law for graphic filters.
