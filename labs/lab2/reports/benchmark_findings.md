# Lab 2: Performance Assessment, Correctness, and Analysis
**Course:** Massively Parallel Programming on GPUs  
**Teacher:** Donato D'Ambrosio  
**Department:** Mathematics and Computer Science, University of Calabria  

---

## 1. Overview & Experimental Setup
- **Workstation:** Local Execution Environment / JPDM2 Architecture Evaluation
- **CPU:** Multi-core x86_64 host using **8 OpenMP threads**
- **GPU:** NVIDIA GeForce RTX 4080 Laptop GPU (58 SMs, Ada Lovelace sm_89) / Target GTX 980 (sm_52)
- **Input Image:** Bruce Lindbloom's `DeltaE_8bit_gamma2.2.tif` ($3072 \times 2048$, 6,291,456 pixels, 18.00 MB uncompressed RGB)
- **Grayscale Formula:** $\text{gray} = 0.21 \cdot r + 0.71 \cdot g + 0.07 \cdot b$
- **Measurement Methodology:** Computation times measured separately from disk image loading and PNG saving. Each test configuration was repeated **4 times** and averaged.

---

## 2. Benchmark Results Table

| Configuration | Block Dim | Threads/Block | Kernel Exec (ms) | HtoD Transfer (ms) | DtoH Transfer (ms) | Total GPU (ms) | Pure Kernel Speedup | MD5 Match |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU OpenMP** | - | 8 | **2.6871 ± 0.5362** | - | - | 2.6871 | 1.00x (Baseline) | `f51df2ed375c53e6...` |
| **CUDA GPU (8x8)** | 8x8 | 64 | **1.2234 ± 0.2966** | 3.5371 | 2.4690 | 7.9566 | **2.20x** | PASS |
| **CUDA GPU (16x16)** | 16x16 | 256 | **1.1038 ± 0.6866** | 2.6857 | 1.8673 | 6.3731 | **2.43x** | PASS |
| **CUDA GPU (32x32)** | 32x32 | 1024 | **1.0257 ± 0.2575** | 2.7195 | 2.6474 | 7.9185 | **2.62x** | PASS |

---

## 3. Analysis & Key Findings

### 3.1 Correctness & Bit-Exact Verification (Section 3.2)
- **100% Bit-for-Bit Equivalence:** Across all 6,291,456 pixels, the CPU OpenMP output and the CUDA GPU outputs produce identical pixel values with **0 mismatches**.
- **MD5 Checksums:** The SHA/MD5 checksums (`f51df2ed375c53e629059b8f5db2fbd4`) match exactly across CPU and GPU output images.
- **The Role of `--fmad=false`:** In C and CUDA, floating-point operations can be contracted into Fused Multiply-Add (FMA) instructions. On the GPU, default FMA contraction causes slight rounding variations near truncation boundaries (e.g. 84.99999 vs 85.00001). Passing `--fmad=false` preserves strict IEEE 754 separate multiply-add sequencing, guaranteeing bit-exact checksum matches across heterogeneous CPU/GPU architectures.

### 3.2 Block Dimension Evaluation ($8\times 8$, $16\times 16$, $32\times 32$)
- **Fastest Configuration:** **32x32 (1024 threads/block)** achieved the lowest kernel execution time of **1.0257 ms** (**2.62x speedup** over OpenMP CPU).
- **Warp Occupancy:**
  - $8\times 8$ (64 threads): 2 warps per block. While functional, it exhibits lower occupancy and less latency hiding across memory transactions.
  - $16\times 16$ (256 threads): 8 warps per block. This is the sweet spot for modern NVIDIA architectures, maximizing active warps per SM without exceeding register limits.
  - $32\times 32$ (1024 threads): Maximum block capacity. Limits thread block scheduling flexibility per SM and can increase barrier synchronization overhead.

### 3.3 Memory Transfer Overhead vs Kernel Compute (Section 3.3)
For the fastest GPU execution (32x32):
- **Allocation & Init:** 1.5258 ms (19.3%)
- **HtoD Transfer (18 MB RGB):** 2.7195 ms (34.3%)
- **Kernel Computation:** 1.0257 ms (13.0%)
- **DtoH Transfer (6 MB Gray):** 2.6474 ms (33.4%)
- **PCIe Bus Dominance:** The pure CUDA kernel computation takes only **1.0257 ms**, while data transfers across the PCIe bus take **5.3670 ms** (~67.8% of total GPU time). This illustrates Amdahl's Law and the memory-bound nature of single-pass graphic filtering over the PCIe bus.
