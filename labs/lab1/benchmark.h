#ifndef LAB1_BENCHMARK_H
#define LAB1_BENCHMARK_H

#include "common.h"

// ---------------------------------------------------------------------------
// Execution & Memory Model Enumerations
// ---------------------------------------------------------------------------
enum MemoryModel {
    MEM_UNIFIED,
    MEM_STANDARD
};

enum KernelModel {
    MODEL_MONOLITHIC,
    MODEL_GRID_STRIDE
};

// ---------------------------------------------------------------------------
// Grid & Block Dimension Computation Utilities
// ---------------------------------------------------------------------------
static inline int calculate_vector_grid_size(size_t n, int blockSize, int numSMs, KernelModel model) {
    int neededBlocks = (int)((n + (size_t)blockSize - 1) / (size_t)blockSize);
    if (model == MODEL_GRID_STRIDE) {
        int smCount = (numSMs > 0) ? numSMs : 1;
        int maxBlocks = smCount * 32;
        return (neededBlocks < maxBlocks) ? neededBlocks : maxBlocks;
    }
    return neededBlocks;
}

static inline void calculate_matrix_dims(size_t nrows, size_t ncols, int blockSize, int numSMs,
                                         KernelModel model, dim3 &blockDim, dim3 &gridDim) {
    int bx = 32;
    int by = (blockSize / 32 > 0) ? (blockSize / 32) : 1;
    blockDim = dim3(bx, by);

    int gx = (int)((ncols + bx - 1) / bx);
    int gy = (int)((nrows + by - 1) / by);

    if (model == MODEL_GRID_STRIDE) {
        int smCount = (numSMs > 0) ? numSMs : 1;
        int maxGridX = 32;
        int maxGridY = (smCount * 32 + maxGridX - 1) / maxGridX;
        if (gx > maxGridX) gx = maxGridX;
        if (gy > maxGridY) gy = maxGridY;
    }
    gridDim = dim3(gx, gy);
}

// ---------------------------------------------------------------------------
// Generic Benchmark Runner: Vector Addition (1D)
// ---------------------------------------------------------------------------
template <typename InitKernel, typename AddKernel>
GpuBenchResult1D run_vector_benchmark(size_t n, int blockSize, int numSMs, double cpuMs,
                                      MemoryModel memModel, KernelModel kernelModel,
                                      InitKernel initKernel, AddKernel addKernel) {
    GpuBenchResult1D res;
    memset(&res, 0, sizeof(res));
    res.blockSize = blockSize;
    res.numBlocks = calculate_vector_grid_size(n, blockSize, numSMs, kernelModel);
    res.cpuTimeMs = cpuMs;

    size_t bytes = n * sizeof(double);
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    double *h_C = NULL;

    if (memModel == MEM_STANDARD) {
        h_C = (double*)malloc(bytes);
        if (!h_C) {
            fprintf(stderr, "Host allocation failed for vector buffer\n");
            exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    } else {
        CUDA_CHECK(cudaMallocManaged((void**)&d_A, bytes));
        CUDA_CHECK(cudaMallocManaged((void**)&d_B, bytes));
        CUDA_CHECK(cudaMallocManaged((void**)&d_C, bytes));
    }

    cudaEvent_t sInit, eInit, sAdd, eAdd, sCpy, eCpy;
    CUDA_CHECK(cudaEventCreate(&sInit)); CUDA_CHECK(cudaEventCreate(&eInit));
    CUDA_CHECK(cudaEventCreate(&sAdd));  CUDA_CHECK(cudaEventCreate(&eAdd));
    if (memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventCreate(&sCpy)); CUDA_CHECK(cudaEventCreate(&eCpy));
    }

    dim3 block(blockSize);
    dim3 grid(res.numBlocks);

    // 1. Initialise GPU-side (3 calls)
    CUDA_CHECK(cudaEventRecord(sInit, 0));
    initKernel<<<grid, block>>>(d_A, 1.0, n);
    initKernel<<<grid, block>>>(d_B, 2.0, n);
    initKernel<<<grid, block>>>(d_C, 0.0, n);
    CUDA_CHECK(cudaEventRecord(eInit, 0));
    CUDA_CHECK(cudaEventSynchronize(eInit));

    // 2. Vector addition (1 call)
    CUDA_CHECK(cudaEventRecord(sAdd, 0));
    addKernel<<<grid, block>>>(d_A, d_B, d_C, n);
    CUDA_CHECK(cudaEventRecord(eAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(eAdd));

    // 3. Post-processing: Explicit copy or synchronization
    if (memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventRecord(sCpy, 0));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(eCpy, 0));
        CUDA_CHECK(cudaEventSynchronize(eCpy));
        CUDA_CHECK(cudaEventElapsedTime(&res.memcpyTimeMs, sCpy, eCpy));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        res.memcpyTimeMs = 0.0f;
    }

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, sInit, eInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, sAdd, eAdd));
    res.totalGpuTimeMs = res.initTimeMs + res.addTimeMs + res.memcpyTimeMs;
    res.speedup = (cpuMs > 0.0) ? (cpuMs / res.totalGpuTimeMs) : 0.0;

    double *verifyBuf = (memModel == MEM_STANDARD) ? h_C : d_C;
    res.verified = verify_vector(verifyBuf, n, 3.0);

    CUDA_CHECK(cudaEventDestroy(sInit)); CUDA_CHECK(cudaEventDestroy(eInit));
    CUDA_CHECK(cudaEventDestroy(sAdd));  CUDA_CHECK(cudaEventDestroy(eAdd));
    if (memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventDestroy(sCpy)); CUDA_CHECK(cudaEventDestroy(eCpy));
        free(h_C);
    }
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    return res;
}

// ---------------------------------------------------------------------------
// Generic Benchmark Runner: Matrix Addition (2D)
// ---------------------------------------------------------------------------
template <typename InitKernel, typename AddKernel>
GpuBenchResult2D run_matrix_benchmark(size_t nrows, size_t ncols, int blockSize, int numSMs, double cpuMs,
                                      MemoryModel memModel, KernelModel kernelModel,
                                      InitKernel initKernel, AddKernel addKernel) {
    GpuBenchResult2D res;
    memset(&res, 0, sizeof(res));
    res.blockSize = blockSize;
    res.cpuTimeMs = cpuMs;
    calculate_matrix_dims(nrows, ncols, blockSize, numSMs, kernelModel, res.blockDim, res.gridDim);

    size_t bytes = nrows * ncols * sizeof(double);
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    double *h_C = NULL;

    if (memModel == MEM_STANDARD) {
        h_C = (double*)malloc(bytes);
        if (!h_C) {
            fprintf(stderr, "Host allocation failed for matrix buffer\n");
            exit(EXIT_FAILURE);
        }
        CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
        CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    } else {
        CUDA_CHECK(cudaMallocManaged((void**)&d_A, bytes));
        CUDA_CHECK(cudaMallocManaged((void**)&d_B, bytes));
        CUDA_CHECK(cudaMallocManaged((void**)&d_C, bytes));
    }

    cudaEvent_t sInit, eInit, sAdd, eAdd, sCpy, eCpy;
    CUDA_CHECK(cudaEventCreate(&sInit)); CUDA_CHECK(cudaEventCreate(&eInit));
    CUDA_CHECK(cudaEventCreate(&sAdd));  CUDA_CHECK(cudaEventCreate(&eAdd));
    if (memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventCreate(&sCpy)); CUDA_CHECK(cudaEventCreate(&eCpy));
    }

    // 1. Initialise GPU-side (3 calls)
    CUDA_CHECK(cudaEventRecord(sInit, 0));
    initKernel<<<res.gridDim, res.blockDim>>>(d_A, 1.0, nrows, ncols);
    initKernel<<<res.gridDim, res.blockDim>>>(d_B, 2.0, nrows, ncols);
    initKernel<<<res.gridDim, res.blockDim>>>(d_C, 0.0, nrows, ncols);
    CUDA_CHECK(cudaEventRecord(eInit, 0));
    CUDA_CHECK(cudaEventSynchronize(eInit));

    // 2. Matrix addition (1 call)
    CUDA_CHECK(cudaEventRecord(sAdd, 0));
    addKernel<<<res.gridDim, res.blockDim>>>(d_A, d_B, d_C, nrows, ncols);
    CUDA_CHECK(cudaEventRecord(eAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(eAdd));

    // 3. Post-processing: Explicit copy or synchronization
    if (memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventRecord(sCpy, 0));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(eCpy, 0));
        CUDA_CHECK(cudaEventSynchronize(eCpy));
        CUDA_CHECK(cudaEventElapsedTime(&res.memcpyTimeMs, sCpy, eCpy));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        res.memcpyTimeMs = 0.0f;
    }

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, sInit, eInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, sAdd, eAdd));
    res.totalGpuTimeMs = res.initTimeMs + res.addTimeMs + res.memcpyTimeMs;
    res.speedup = (cpuMs > 0.0) ? (cpuMs / res.totalGpuTimeMs) : 0.0;

    double *verifyBuf = (memModel == MEM_STANDARD) ? h_C : d_C;
    res.verified = verify_matrix(verifyBuf, nrows, ncols, 3.0);

    CUDA_CHECK(cudaEventDestroy(sInit)); CUDA_CHECK(cudaEventDestroy(eInit));
    CUDA_CHECK(cudaEventDestroy(sAdd));  CUDA_CHECK(cudaEventDestroy(eAdd));
    if (memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventDestroy(sCpy)); CUDA_CHECK(cudaEventDestroy(eCpy));
        free(h_C);
    }
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    return res;
}

// ---------------------------------------------------------------------------
// High-Level Benchmark Suite Driver
// ---------------------------------------------------------------------------
template <typename VecInitKernel, typename VecAddKernel, typename MatInitKernel, typename MatAddKernel>
int run_benchmark(int argc, char **argv,
                  const char *appHeaderTitle,
                  const char *sweepTableTitle,
                  const char *shortName,
                  MemoryModel memModel,
                  KernelModel kernelModel,
                  VecInitKernel vecInit,
                  VecAddKernel vecAdd,
                  MatInitKernel matInit,
                  MatAddKernel matAdd) {
    AppConfig cfg = parse_app_args(argc, argv);
    cudaDeviceProp prop = print_app_header(appHeaderTitle, cfg.devId);

    const char *label = (shortName != NULL) ? shortName : sweepTableTitle;

    // Single run mode for profiling (nvprof / ncu)
    if (cfg.singleBlockSize > 0) {
        if (strcmp(cfg.mode, "matrix") == 0) {
            GpuBenchResult2D r = run_matrix_benchmark(
                cfg.matRows, cfg.matCols, cfg.singleBlockSize, prop.multiProcessorCount, 0.0,
                memModel, kernelModel, matInit, matAdd);
            char singleTitle[128];
            snprintf(singleTitle, sizeof(singleTitle), "Matrix Addition - %s", label);
            print_single_run_2d(singleTitle, cfg.matRows, cfg.matCols, &r);
        } else {
            GpuBenchResult1D r = run_vector_benchmark(
                cfg.vecN, cfg.singleBlockSize, prop.multiProcessorCount, 0.0,
                memModel, kernelModel, vecInit, vecAdd);
            char singleTitle[128];
            snprintf(singleTitle, sizeof(singleTitle), "Vector Addition - %s", label);
            print_single_run_1d(singleTitle, cfg.vecN, &r);
        }
        return 0;
    }

    // Full comparison sweep mode
    printf("\n>>> Measuring Serial CPU Baseline for Vector Addition (n = %zu)...\n", cfg.vecN);
    double cpuVecMs = measure_cpu_vector_baseline(cfg.vecN);
    printf("CPU Serial Elapsed Time (vectorInit + vectorAdd): %.3f ms\n", cpuVecMs);

    GpuBenchResult1D vecResults[NUM_BLOCK_SIZES];
    for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
        vecResults[i] = run_vector_benchmark(
            cfg.vecN, BLOCK_SIZES[i], prop.multiProcessorCount, cpuVecMs,
            memModel, kernelModel, vecInit, vecAdd);
    }
    print_sweep_table_1d(sweepTableTitle, (memModel == MEM_STANDARD), vecResults, NUM_BLOCK_SIZES);

    printf("\n>>> Measuring Serial CPU Baseline for Matrix Addition (%zu x %zu)...\n", cfg.matRows, cfg.matCols);
    double cpuMatMs = measure_cpu_matrix_baseline(cfg.matRows, cfg.matCols);
    printf("CPU Serial Elapsed Time (matrixInit + matrixAdd): %.3f ms\n", cpuMatMs);

    GpuBenchResult2D matResults[NUM_BLOCK_SIZES];
    for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
        matResults[i] = run_matrix_benchmark(
            cfg.matRows, cfg.matCols, BLOCK_SIZES[i], prop.multiProcessorCount, cpuMatMs,
            memModel, kernelModel, matInit, matAdd);
    }
    print_sweep_table_2d(sweepTableTitle, (memModel == MEM_STANDARD), matResults, NUM_BLOCK_SIZES);

    return 0;
}

// Convenient overload with 2 titles
template <typename VecInitKernel, typename VecAddKernel, typename MatInitKernel, typename MatAddKernel>
static inline int run_benchmark(int argc, char **argv,
                                const char *appHeaderTitle,
                                const char *sweepTableTitle,
                                MemoryModel memModel,
                                KernelModel kernelModel,
                                VecInitKernel vecInit,
                                VecAddKernel vecAdd,
                                MatInitKernel matInit,
                                MatAddKernel matAdd) {
    return run_benchmark(argc, argv, appHeaderTitle, sweepTableTitle, sweepTableTitle,
                         memModel, kernelModel, vecInit, vecAdd, matInit, matAdd);
}

#endif // LAB1_BENCHMARK_H

