/**
 * Lab 1: First CUDA Applications - Unified Driver
 * Course: Massively Parallel Programming on GPUs
 * 
 * Provides all 4 CUDA implementations switchable via CLI:
 *   Memory Model:  Unified Memory (u) vs Standard Memory (s)
 *   Kernel Model:  Monolithic (m) vs Grid-Stride Loops (gs)
 *   Operation:     Vector Addition (vector) vs Matrix Addition (matrix)
 * 
 * Usage:
 *   ./lab1 <u|s> <m|gs> <vector|matrix> [size] [options]
 */

#include "benchmark.h"

// ---------------------------------------------------------------------------
// 1. CUDA Monolithic Kernels (1 thread per element, 1-to-1 grid mapping)
// ---------------------------------------------------------------------------
__global__ void vectorInit(double *data, double val, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = val;
}

__global__ void vectorAdd(const double *A, const double *B, double *C, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

__global__ void matrixInit(double *M, double val, size_t nrows, size_t ncols) {
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (row < nrows && col < ncols) M[row * ncols + col] = val;
}

__global__ void matrixAdd(const double *A, const double *B, double *C, size_t nrows, size_t ncols) {
    size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    if (row < nrows && col < ncols) {
        size_t idx = row * ncols + col;
        C[idx] = A[idx] + B[idx];
    }
}

// ---------------------------------------------------------------------------
// 2. CUDA Grid-Stride Loop Kernels (stride = gridDim * blockDim)
// Ref: Mark Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"
// ---------------------------------------------------------------------------
__global__ void vectorInitGSL(double *data, double val, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        data[i] = val;
    }
}

__global__ void vectorAddGSL(const double *A, const double *B, double *C, size_t n) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)blockDim.x * gridDim.x;
    for (size_t i = tid; i < n; i += stride) {
        C[i] = A[i] + B[i];
    }
}

__global__ void matrixInitGSL(double *M, double val, size_t nrows, size_t ncols) {
    size_t startCol = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t strideCol = (size_t)blockDim.x * gridDim.x;
    size_t startRow = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t strideRow = (size_t)blockDim.y * gridDim.y;

    for (size_t r = startRow; r < nrows; r += strideRow) {
        for (size_t c = startCol; c < ncols; c += strideCol) {
            M[r * ncols + c] = val;
        }
    }
}

__global__ void matrixAddGSL(const double *A, const double *B, double *C, size_t nrows, size_t ncols) {
    size_t startCol = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t strideCol = (size_t)blockDim.x * gridDim.x;
    size_t startRow = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
    size_t strideRow = (size_t)blockDim.y * gridDim.y;

    for (size_t r = startRow; r < nrows; r += strideRow) {
        for (size_t c = startCol; c < ncols; c += strideCol) {
            size_t idx = r * ncols + c;
            C[idx] = A[idx] + B[idx];
        }
    }
}

// ---------------------------------------------------------------------------
// Main Driver
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    return run_benchmark_dispatcher(
        argc, argv,
        vectorInit, vectorAdd, matrixInit, matrixAdd,
        vectorInitGSL, vectorAddGSL, matrixInitGSL, matrixAddGSL);
}
