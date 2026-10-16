/**
 * Implementation 2: Unified-Memory / Grid-Stride-Loop-Kernels
 * 
 * - Memory: Unified Memory (cudaMallocManaged)
 * - Kernel Model: Grid-Stride Loops (stride = gridDim * blockDim)
 * - Decoupled Grid: Grid size scaled to SM count (32 * numSMs)
 * - Initialization: 3x vectorInitGSL / matrixInitGSL on GPU
 * - Computation: 1x vectorAddGSL / matrixAddGSL on GPU
 */

#include "benchmark.h"

// ---------------------------------------------------------------------------
// CUDA Grid-Stride Loop Kernels
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
    return run_benchmark(argc, argv,
                         "Lab 1: Implementation 2 - Unified-Memory / Grid-Stride-Loop-Kernels",
                         "Unified-Memory / Grid-Stride-Loop-Kernels",
                         "Unified / Grid-Stride",
                         MEM_UNIFIED, MODEL_GRID_STRIDE,
                         vectorInitGSL, vectorAddGSL, matrixInitGSL, matrixAddGSL);
}
