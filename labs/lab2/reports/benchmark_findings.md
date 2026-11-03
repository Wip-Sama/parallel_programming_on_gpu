# Lab 2: Performance Assessment, Correctness, and Analysis
**Course:** Massively Parallel Programming on GPUs  
**Teacher:** Donato D'Ambrosio  
**Department:** Mathematics and Computer Science, University of Calabria  

---

## 1. Overview & Experimental Setup
- **CPU:** Multi-core x86_64 host using **8 OpenMP threads**
- **GPU:** NVIDIA CUDA GPU
- **Input Image:** Bruce Lindbloom's benchmark image (3072x2048, 18.88 MB)
- **Grayscale Formula:** $\text{gray} = 0.21 \cdot r + 0.71 \cdot g + 0.07 \cdot b$
- **Memory Architectures Evaluated:**
  1. **Standard Memory (`cudaMalloc` + `cudaMemcpy`):** Explicit host-to-device and device-to-host transfers across the PCIe bus.
  2. **Unified Memory (`cudaMallocManaged`):** Shared virtual address space with hardware-assisted on-demand page migration.
- **Measurement Methodology:** Computation times measured separately from disk image loading and PNG saving. Each test configuration was repeated **4 times** and averaged.
- **Verification:** Bit-exact verification and MD5 checksum computation handled in Python.

---

## 2. Benchmark Results Table

| Configuration | Memory Model | Block Dim | Threads/Block | Kernel Exec (ms) | HtoD Transfer (ms) | DtoH Transfer (ms) | Total GPU (ms) | Kernel Speedup | Verification |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CPU OpenMP** | - | - | 8 | **2.0807 ± 0.2610** | - | - | 2.0807 | 1.00x (Baseline) | `1ee3c0c33ae26a25...` |
| **Standard CUDA (8x8)** | Standard | 8x8 | 64 | **0.9344 ± 0.1665** | 1.8224 | 2.4076 | 5.8792 | **2.23x** | PASS |
| **Standard CUDA (16x16)** | Standard | 16x16 | 256 | **0.7997 ± 0.1600** | 1.6880 | 2.2066 | 5.4468 | **2.60x** | PASS |
| **Standard CUDA (32x32)** | Standard | 32x32 | 1024 | **0.7332 ± 0.0428** | 1.7714 | 2.2528 | 5.3951 | **2.84x** | PASS |
| **Unified CUDA (8x8)** | Unified | 8x8 | 64 | **6.0799 ± 0.5093** | 0.0000 | 0.0000 | 16.3599 | **0.34x** | PASS |
| **Unified CUDA (16x16)** | Unified | 16x16 | 256 | **4.9953 ± 0.5271** | 0.0000 | 0.0000 | 14.3647 | **0.42x** | PASS |
| **Unified CUDA (32x32)** | Unified | 32x32 | 1024 | **4.4464 ± 0.2989** | 0.0000 | 0.0000 | 13.3740 | **0.47x** | PASS |

---

## 3. Analysis & Key Findings

### 3.1 Correctness & Bit-Exact Verification (Section 3.2)
- Across all pixels, both the Standard Memory (`lab2_standard`) and Unified Memory (`lab2`) outputs match the CPU OpenMP baseline bit-for-bit with 0 byte mismatches.
- Checksums match across CPU and GPU output images: `1ee3c0c33ae26a25c63458fd2231d7a5`.
- Passing `--fmad=false` during compilation preserves strict IEEE 754 separate multiply-add sequencing, avoiding rounding variations near truncation boundaries.

### 3.2 Standard Memory vs. Unified Memory Comparison
1. **Explicit Transfers vs On-Demand Paging:**
   - **Standard Memory (`cudaMalloc` + `cudaMemcpy`):** Pre-copies the full 18.88 MB RGB buffer in a single batched DMA transfer across PCIe (**~1.77 ms**). The kernel then executes entirely out of high-bandwidth GPU VRAM, achieving pure kernel execution of **~0.73 ms** (**2.84x speedup**).
   - **Unified Memory (`cudaMallocManaged`):** Eliminates explicit `cudaMemcpy` calls (`HtoD = 0.00 ms`). However, because memory pages reside in host RAM when the kernel begins, the GPU hardware page fault handler must stall warps and migrate 4KB pages over PCIe on first-touch. This increases kernel duration to **~4.45 ms**.
2. **Allocation Latency:**
   - Standard `cudaMalloc` requires only **~0.64 ms**.
   - `cudaMallocManaged` incurs driver-level virtual memory mapping overhead on the OS and GPU MMU, taking **~8.93 ms**.

### 3.3 Block Dimension Evaluation ($8\times 8$, $16\times 16$, $32\times 32$)
- $16\times 16$ (256 threads) and $32\times 32$ (1024 threads) provide optimal occupancy, with $16\times 16$ maximizing active warps without exceeding per-SM register limits.

### 3.4 Summary: When to Use Which Architecture
- **Standard Memory:** Ideal for high-performance throughput pipelines where DMA transfer latency can be overlapped with streams or hidden behind large compute workloads.
- **Unified Memory:** Ideal for rapid prototyping, complex linked data structures (graphs/trees), and codebases prioritizing simplicity without manual dual-buffer management.
