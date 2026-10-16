/**
 * Implementation 1: Unified-Memory / Monolithic-Kernels
 * 
 * - Memory: Unified Memory (cudaMallocManaged)
 * - Kernel Model: Monolithic (1 thread per element, grid covers all elements)
 * - Initialization: 3x vectorInit / matrixInit on GPU
 * - Computation: 1x vectorAdd / matrixAdd on GPU
 */

#include "benchmark.h"

// ---------------------------------------------------------------------------
// CUDA Monolithic Kernels (1 thread per element)
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
// Main Driver
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    return run_benchmark(argc, argv,
                         "Lab 1: Implementation 1 - Unified-Memory / Monolithic-Kernels",
                         "Unified-Memory / Monolithic-Kernels",
                         "Unified / Monolithic",
                         MEM_UNIFIED, MODEL_MONOLITHIC,
                         vectorInit, vectorAdd, matrixInit, matrixAdd);
}
