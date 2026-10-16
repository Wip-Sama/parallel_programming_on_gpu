/**
 * Implementation 4: Standard-Memory / Grid-Stride-Loop-Kernels
 * 
 * - Memory: Standard Memory (cudaMalloc on Device, malloc on Host)
 * - Kernel Model: Grid-Stride Loops (stride = gridDim * blockDim)
 * - Decoupled Grid: Grid size scaled to SM count (32 * numSMs)
 * - Transfers: Explicit cudaMemcpy (DtoH) of result buffer C
 * - Initialization: 3x vectorInitGSL / matrixInitGSL on GPU
 * - Computation: 1x vectorAddGSL / matrixAddGSL on GPU
 */

#include "common.h"

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
// Benchmark Execution
// ---------------------------------------------------------------------------
GpuBenchResult1D run_vector(size_t n, int blockSize, int numSMs, double cpuMs) {
    GpuBenchResult1D res;
    res.blockSize = blockSize;
    res.cpuTimeMs = cpuMs;

    int maxBlocks = numSMs * 32;
    int neededBlocks = (int)((n + blockSize - 1) / blockSize);
    res.numBlocks = (neededBlocks < maxBlocks) ? neededBlocks : maxBlocks;

    size_t bytes = n * sizeof(double);
    double *d_A, *d_B, *d_C;
    double *h_C = (double*)malloc(bytes);
    if (!h_C) { fprintf(stderr, "Host allocation failed\n"); exit(EXIT_FAILURE); }

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t sInit, eInit, sAdd, eAdd, sCpy, eCpy;
    CUDA_CHECK(cudaEventCreate(&sInit)); CUDA_CHECK(cudaEventCreate(&eInit));
    CUDA_CHECK(cudaEventCreate(&sAdd));  CUDA_CHECK(cudaEventCreate(&eAdd));
    CUDA_CHECK(cudaEventCreate(&sCpy));  CUDA_CHECK(cudaEventCreate(&eCpy));

    dim3 block(blockSize);
    dim3 grid(res.numBlocks);

    // 1. Initialise GPU-side (3 calls)
    CUDA_CHECK(cudaEventRecord(sInit, 0));
    vectorInitGSL<<<grid, block>>>(d_A, 1.0, n);
    vectorInitGSL<<<grid, block>>>(d_B, 2.0, n);
    vectorInitGSL<<<grid, block>>>(d_C, 0.0, n);
    CUDA_CHECK(cudaEventRecord(eInit, 0));
    CUDA_CHECK(cudaEventSynchronize(eInit));

    // 2. Vector addition (1 call)
    CUDA_CHECK(cudaEventRecord(sAdd, 0));
    vectorAddGSL<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(eAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(eAdd));

    // 3. Explicit memory copy to host
    CUDA_CHECK(cudaEventRecord(sCpy, 0));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(eCpy, 0));
    CUDA_CHECK(cudaEventSynchronize(eCpy));

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, sInit, eInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, sAdd, eAdd));
    CUDA_CHECK(cudaEventElapsedTime(&res.memcpyTimeMs, sCpy, eCpy));

    res.totalGpuTimeMs = res.initTimeMs + res.addTimeMs + res.memcpyTimeMs;
    res.speedup = (cpuMs > 0.0) ? (cpuMs / res.totalGpuTimeMs) : 0.0;
    res.verified = verify_vector(h_C, n, 3.0);

    CUDA_CHECK(cudaEventDestroy(sInit)); CUDA_CHECK(cudaEventDestroy(eInit));
    CUDA_CHECK(cudaEventDestroy(sAdd));  CUDA_CHECK(cudaEventDestroy(eAdd));
    CUDA_CHECK(cudaEventDestroy(sCpy));  CUDA_CHECK(cudaEventDestroy(eCpy));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_C);
    return res;
}

GpuBenchResult2D run_matrix(size_t nrows, size_t ncols, int blockSize, int numSMs, double cpuMs) {
    GpuBenchResult2D res;
    res.blockSize = blockSize;
    res.cpuTimeMs = cpuMs;

    int bx = 32, by = (blockSize / 32 > 0) ? (blockSize / 32) : 1;
    res.blockDim = dim3(bx, by);

    int maxGridX = 32;
    int maxGridY = (numSMs * 32 + maxGridX - 1) / maxGridX;
    int gx = (int)((ncols + bx - 1) / bx);
    int gy = (int)((nrows + by - 1) / by);
    if (gx > maxGridX) gx = maxGridX;
    if (gy > maxGridY) gy = maxGridY;
    res.gridDim = dim3(gx, gy);

    size_t bytes = nrows * ncols * sizeof(double);
    double *d_A, *d_B, *d_C;
    double *h_C = (double*)malloc(bytes);
    if (!h_C) { fprintf(stderr, "Host allocation failed\n"); exit(EXIT_FAILURE); }

    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    cudaEvent_t sInit, eInit, sAdd, eAdd, sCpy, eCpy;
    CUDA_CHECK(cudaEventCreate(&sInit)); CUDA_CHECK(cudaEventCreate(&eInit));
    CUDA_CHECK(cudaEventCreate(&sAdd));  CUDA_CHECK(cudaEventCreate(&eAdd));
    CUDA_CHECK(cudaEventCreate(&sCpy));  CUDA_CHECK(cudaEventCreate(&eCpy));

    CUDA_CHECK(cudaEventRecord(sInit, 0));
    matrixInitGSL<<<res.gridDim, res.blockDim>>>(d_A, 1.0, nrows, ncols);
    matrixInitGSL<<<res.gridDim, res.blockDim>>>(d_B, 2.0, nrows, ncols);
    matrixInitGSL<<<res.gridDim, res.blockDim>>>(d_C, 0.0, nrows, ncols);
    CUDA_CHECK(cudaEventRecord(eInit, 0));
    CUDA_CHECK(cudaEventSynchronize(eInit));

    CUDA_CHECK(cudaEventRecord(sAdd, 0));
    matrixAddGSL<<<res.gridDim, res.blockDim>>>(d_A, d_B, d_C, nrows, ncols);
    CUDA_CHECK(cudaEventRecord(eAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(eAdd));

    CUDA_CHECK(cudaEventRecord(sCpy, 0));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(eCpy, 0));
    CUDA_CHECK(cudaEventSynchronize(eCpy));

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, sInit, eInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, sAdd, eAdd));
    CUDA_CHECK(cudaEventElapsedTime(&res.memcpyTimeMs, sCpy, eCpy));

    res.totalGpuTimeMs = res.initTimeMs + res.addTimeMs + res.memcpyTimeMs;
    res.speedup = (cpuMs > 0.0) ? (cpuMs / res.totalGpuTimeMs) : 0.0;
    res.verified = verify_matrix(h_C, nrows, ncols, 3.0);

    CUDA_CHECK(cudaEventDestroy(sInit)); CUDA_CHECK(cudaEventDestroy(eInit));
    CUDA_CHECK(cudaEventDestroy(sAdd));  CUDA_CHECK(cudaEventDestroy(eAdd));
    CUDA_CHECK(cudaEventDestroy(sCpy));  CUDA_CHECK(cudaEventDestroy(eCpy));
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    free(h_C);
    return res;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    AppConfig cfg = parse_app_args(argc, argv);
    cudaDeviceProp prop = print_app_header("Lab 1: Implementation 4 - Standard-Memory / Grid-Stride-Loop-Kernels", cfg.devId);

    // Single run mode for profiling (nvprof / ncu)
    if (cfg.singleBlockSize > 0) {
        if (strcmp(cfg.mode, "matrix") == 0) {
            GpuBenchResult2D r = run_matrix(cfg.matRows, cfg.matCols, cfg.singleBlockSize, prop.multiProcessorCount, 0.0);
            print_single_run_2d("Matrix Addition - Standard / Grid-Stride", cfg.matRows, cfg.matCols, &r);
        } else {
            GpuBenchResult1D r = run_vector(cfg.vecN, cfg.singleBlockSize, prop.multiProcessorCount, 0.0);
            print_single_run_1d("Vector Addition - Standard / Grid-Stride", cfg.vecN, &r);
        }
        return 0;
    }

    // Full comparison sweep mode
    printf("\n>>> Measuring Serial CPU Baseline for Vector Addition (n = %zu)...\n", cfg.vecN);
    double cpuVecMs = measure_cpu_vector_baseline(cfg.vecN);
    printf("CPU Serial Elapsed Time (vectorInit + vectorAdd): %.3f ms\n", cpuVecMs);

    GpuBenchResult1D vecResults[NUM_BLOCK_SIZES];
    for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
        vecResults[i] = run_vector(cfg.vecN, BLOCK_SIZES[i], prop.multiProcessorCount, cpuVecMs);
    }
    print_sweep_table_1d("Standard-Memory / Grid-Stride-Loop-Kernels", 1, vecResults, NUM_BLOCK_SIZES);

    printf("\n>>> Measuring Serial CPU Baseline for Matrix Addition (%zu x %zu)...\n", cfg.matRows, cfg.matCols);
    double cpuMatMs = measure_cpu_matrix_baseline(cfg.matRows, cfg.matCols);
    printf("CPU Serial Elapsed Time (matrixInit + matrixAdd): %.3f ms\n", cpuMatMs);

    GpuBenchResult2D matResults[NUM_BLOCK_SIZES];
    for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
        matResults[i] = run_matrix(cfg.matRows, cfg.matCols, BLOCK_SIZES[i], prop.multiProcessorCount, cpuMatMs);
    }
    print_sweep_table_2d("Standard-Memory / Grid-Stride-Loop-Kernels", 1, matResults, NUM_BLOCK_SIZES);

    return 0;
}
