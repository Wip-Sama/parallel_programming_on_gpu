#ifndef LAB1_BENCHMARK_H
#define LAB1_BENCHMARK_H

#include "common.h"

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
// Averaged Benchmark Runners (N repetitions)
// ---------------------------------------------------------------------------
template <typename InitKernel, typename AddKernel>
GpuBenchResult1D run_vector_benchmark_avg(size_t n, int blockSize, int numSMs, double cpuMs,
                                          MemoryModel memModel, KernelModel kernelModel,
                                          InitKernel initKernel, AddKernel addKernel, int numRuns) {
    if (numRuns <= 1) {
        return run_vector_benchmark(n, blockSize, numSMs, cpuMs, memModel, kernelModel, initKernel, addKernel);
    }
    GpuBenchResult1D avgRes;
    memset(&avgRes, 0, sizeof(avgRes));
    int allPass = 1;

    for (int r = 0; r < numRuns; r++) {
        GpuBenchResult1D cur = run_vector_benchmark(n, blockSize, numSMs, cpuMs, memModel, kernelModel, initKernel, addKernel);
        avgRes.initTimeMs += cur.initTimeMs;
        avgRes.addTimeMs += cur.addTimeMs;
        avgRes.memcpyTimeMs += cur.memcpyTimeMs;
        avgRes.totalGpuTimeMs += cur.totalGpuTimeMs;
        avgRes.blockSize = cur.blockSize;
        avgRes.numBlocks = cur.numBlocks;
        allPass &= cur.verified;
    }
    avgRes.initTimeMs /= (float)numRuns;
    avgRes.addTimeMs /= (float)numRuns;
    avgRes.memcpyTimeMs /= (float)numRuns;
    avgRes.totalGpuTimeMs /= (float)numRuns;
    avgRes.cpuTimeMs = cpuMs;
    avgRes.speedup = (cpuMs > 0.0) ? (cpuMs / avgRes.totalGpuTimeMs) : 0.0;
    avgRes.verified = allPass;
    return avgRes;
}

template <typename InitKernel, typename AddKernel>
GpuBenchResult2D run_matrix_benchmark_avg(size_t nrows, size_t ncols, int blockSize, int numSMs, double cpuMs,
                                          MemoryModel memModel, KernelModel kernelModel,
                                          InitKernel initKernel, AddKernel addKernel, int numRuns) {
    if (numRuns <= 1) {
        return run_matrix_benchmark(nrows, ncols, blockSize, numSMs, cpuMs, memModel, kernelModel, initKernel, addKernel);
    }
    GpuBenchResult2D avgRes;
    memset(&avgRes, 0, sizeof(avgRes));
    int allPass = 1;

    for (int r = 0; r < numRuns; r++) {
        GpuBenchResult2D cur = run_matrix_benchmark(nrows, ncols, blockSize, numSMs, cpuMs, memModel, kernelModel, initKernel, addKernel);
        avgRes.initTimeMs += cur.initTimeMs;
        avgRes.addTimeMs += cur.addTimeMs;
        avgRes.memcpyTimeMs += cur.memcpyTimeMs;
        avgRes.totalGpuTimeMs += cur.totalGpuTimeMs;
        avgRes.blockSize = cur.blockSize;
        avgRes.blockDim = cur.blockDim;
        avgRes.gridDim = cur.gridDim;
        allPass &= cur.verified;
    }
    avgRes.initTimeMs /= (float)numRuns;
    avgRes.addTimeMs /= (float)numRuns;
    avgRes.memcpyTimeMs /= (float)numRuns;
    avgRes.totalGpuTimeMs /= (float)numRuns;
    avgRes.cpuTimeMs = cpuMs;
    avgRes.speedup = (cpuMs > 0.0) ? (cpuMs / avgRes.totalGpuTimeMs) : 0.0;
    avgRes.verified = allPass;
    return avgRes;
}

// ---------------------------------------------------------------------------
// Suite Executor for a specific (memModel, kernelModel) configuration
// ---------------------------------------------------------------------------
template <typename VecInitKernel, typename VecAddKernel, typename MatInitKernel, typename MatAddKernel>
void run_single_suite(const AppConfig &cfg, const cudaDeviceProp &prop,
                      MemoryModel memModel, KernelModel kernelModel,
                      VecInitKernel vecInit, VecAddKernel vecAdd,
                      MatInitKernel matInit, MatAddKernel matAdd,
                      double cpuVecMs, double cpuMatMs) {
    const char *memTitle = (memModel == MEM_UNIFIED) ? "Unified-Memory" : "Standard-Memory";
    const char *modelTitle = (kernelModel == MODEL_MONOLITHIC) ? "Monolithic-Kernels" : "Grid-Stride-Loop-Kernels";
    const char *memCode = (memModel == MEM_UNIFIED) ? "u" : "s";
    const char *modelCode = (kernelModel == MODEL_MONOLITHIC) ? "m" : "gs";

    char appHeader[64];
    snprintf(appHeader, sizeof(appHeader), "%s / %s", memTitle, modelTitle);

    if (!cfg.csvOutput) {
        printf("\n================================================================================\n");
        printf(" Configuration: %s\n", appHeader);
        printf("================================================================================\n");
    }

    if (cfg.singleBlockSize > 0) {
        if (strcmp(cfg.mode, "matrix") == 0) {
            GpuBenchResult2D r = run_matrix_benchmark_avg(
                cfg.matRows, cfg.matCols, cfg.singleBlockSize, prop.multiProcessorCount, cpuMatMs,
                memModel, kernelModel, matInit, matAdd, cfg.numRuns);
            if (cfg.csvOutput) {
                printf("CSV,%s,%s,matrix,%d,%dx%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                       memCode, modelCode, r.blockSize, r.gridDim.x, r.gridDim.y,
                       r.initTimeMs, r.addTimeMs, r.memcpyTimeMs, r.totalGpuTimeMs,
                       r.cpuTimeMs, r.speedup, r.verified ? "PASS" : "FAIL");
            } else {
                char title[256];
                snprintf(title, sizeof(title), "Matrix Addition - %s", appHeader);
                print_single_run_2d(title, cfg.matRows, cfg.matCols, &r, cfg.numRuns);
            }
        } else {
            GpuBenchResult1D r = run_vector_benchmark_avg(
                cfg.vecN, cfg.singleBlockSize, prop.multiProcessorCount, cpuVecMs,
                memModel, kernelModel, vecInit, vecAdd, cfg.numRuns);
            if (cfg.csvOutput) {
                printf("CSV,%s,%s,vector,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                       memCode, modelCode, r.blockSize, r.numBlocks,
                       r.initTimeMs, r.addTimeMs, r.memcpyTimeMs, r.totalGpuTimeMs,
                       r.cpuTimeMs, r.speedup, r.verified ? "PASS" : "FAIL");
            } else {
                char title[256];
                snprintf(title, sizeof(title), "Vector Addition - %s", appHeader);
                print_single_run_1d(title, cfg.vecN, &r, cfg.numRuns);
            }
        }
        return;
    }

    // Sweep mode
    if (strcmp(cfg.mode, "all") == 0 || strcmp(cfg.mode, "vector") == 0) {
        GpuBenchResult1D vecResults[NUM_BLOCK_SIZES];
        for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
            vecResults[i] = run_vector_benchmark_avg(
                cfg.vecN, BLOCK_SIZES[i], prop.multiProcessorCount, cpuVecMs,
                memModel, kernelModel, vecInit, vecAdd, cfg.numRuns);
            if (cfg.csvOutput) {
                printf("CSV,%s,%s,vector,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                       memCode, modelCode, vecResults[i].blockSize, vecResults[i].numBlocks,
                       vecResults[i].initTimeMs, vecResults[i].addTimeMs, vecResults[i].memcpyTimeMs,
                       vecResults[i].totalGpuTimeMs, vecResults[i].cpuTimeMs, vecResults[i].speedup,
                       vecResults[i].verified ? "PASS" : "FAIL");
            }
        }
        if (!cfg.csvOutput) {
            char tableTitle[256];
            if (cfg.numRuns > 1) {
                snprintf(tableTitle, sizeof(tableTitle), "%s (Average of %d runs)", appHeader, cfg.numRuns);
            } else {
                snprintf(tableTitle, sizeof(tableTitle), "%s", appHeader);
            }
            print_sweep_table_1d(tableTitle, (memModel == MEM_STANDARD), vecResults, NUM_BLOCK_SIZES);
        }
    }

    if (strcmp(cfg.mode, "all") == 0 || strcmp(cfg.mode, "matrix") == 0) {
        GpuBenchResult2D matResults[NUM_BLOCK_SIZES];
        for (int i = 0; i < NUM_BLOCK_SIZES; i++) {
            matResults[i] = run_matrix_benchmark_avg(
                cfg.matRows, cfg.matCols, BLOCK_SIZES[i], prop.multiProcessorCount, cpuMatMs,
                memModel, kernelModel, matInit, matAdd, cfg.numRuns);
            if (cfg.csvOutput) {
                printf("CSV,%s,%s,matrix,%d,%dx%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%s\n",
                       memCode, modelCode, matResults[i].blockSize, matResults[i].gridDim.x, matResults[i].gridDim.y,
                       matResults[i].initTimeMs, matResults[i].addTimeMs, matResults[i].memcpyTimeMs,
                       matResults[i].totalGpuTimeMs, matResults[i].cpuTimeMs, matResults[i].speedup,
                       matResults[i].verified ? "PASS" : "FAIL");
            }
        }
        if (!cfg.csvOutput) {
            char tableTitle[256];
            if (cfg.numRuns > 1) {
                snprintf(tableTitle, sizeof(tableTitle), "%s (Average of %d runs)", appHeader, cfg.numRuns);
            } else {
                snprintf(tableTitle, sizeof(tableTitle), "%s", appHeader);
            }
            print_sweep_table_2d(tableTitle, (memModel == MEM_STANDARD), matResults, NUM_BLOCK_SIZES);
        }
    }
}

// ---------------------------------------------------------------------------
// High-Level Unified Dispatcher
// ---------------------------------------------------------------------------
template <typename VecInitM, typename VecAddM, typename MatInitM, typename MatAddM,
          typename VecInitG, typename VecAddG, typename MatInitG, typename MatAddG>
int run_benchmark_dispatcher(int argc, char **argv,
                             VecInitM vecInitM, VecAddM vecAddM, MatInitM matInitM, MatAddM matAddM,
                             VecInitG vecInitG, VecAddG vecAddG, MatInitG matInitG, MatAddG matAddG) {
    AppConfig cfg = parse_app_args(argc, argv);
    if (cfg.showHelp) {
        print_usage(argv[0]);
        return 0;
    }

    CUDA_CHECK(cudaSetDevice(cfg.devId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.devId));
    if (!cfg.csvOutput) {
        prop = print_app_header("Lab 1: First CUDA Applications", cfg.devId);
    }

    // Measure CPU baselines once if in sweep mode
    double cpuVecMs = 0.0;
    double cpuMatMs = 0.0;
    if (cfg.singleBlockSize < 0) {
        if (strcmp(cfg.mode, "all") == 0 || strcmp(cfg.mode, "vector") == 0) {
            if (!cfg.csvOutput) {
                printf("\n>>> Measuring Serial CPU Baseline for Vector Addition (n = %zu)...\n", cfg.vecN);
            }
            cpuVecMs = measure_cpu_vector_baseline(cfg.vecN);
            if (!cfg.csvOutput) {
                printf("CPU Serial Elapsed Time (vectorInit + vectorAdd): %.3f ms\n", cpuVecMs);
            }
        }
        if (strcmp(cfg.mode, "all") == 0 || strcmp(cfg.mode, "matrix") == 0) {
            if (!cfg.csvOutput) {
                printf("\n>>> Measuring Serial CPU Baseline for Matrix Addition (%zu x %zu)...\n", cfg.matRows, cfg.matCols);
            }
            cpuMatMs = measure_cpu_matrix_baseline(cfg.matRows, cfg.matCols);
            if (!cfg.csvOutput) {
                printf("CPU Serial Elapsed Time (matrixInit + matrixAdd): %.3f ms\n", cpuMatMs);
            }
        }
    }

    if (cfg.csvOutput) {
        printf("CSV_HEADER,mem,kernel,op,blockSize,gridSize,initMs,addMs,memcpyMs,totalGpuMs,cpuMs,speedup,status\n");
    }

    MemoryModel mems[2];
    int numMems = 0;
    if (cfg.memModel == MEM_ALL) {
        mems[0] = MEM_UNIFIED;
        mems[1] = MEM_STANDARD;
        numMems = 2;
    } else {
        mems[0] = cfg.memModel;
        numMems = 1;
    }

    KernelModel models[2];
    int numModels = 0;
    if (cfg.kernelModel == MODEL_ALL) {
        models[0] = MODEL_MONOLITHIC;
        models[1] = MODEL_GRID_STRIDE;
        numModels = 2;
    } else {
        models[0] = cfg.kernelModel;
        numModels = 1;
    }

    for (int mi = 0; mi < numMems; mi++) {
        for (int ki = 0; ki < numModels; ki++) {
            MemoryModel m = mems[mi];
            KernelModel k = models[ki];
            if (k == MODEL_MONOLITHIC) {
                run_single_suite(cfg, prop, m, k, vecInitM, vecAddM, matInitM, matAddM, cpuVecMs, cpuMatMs);
            } else {
                run_single_suite(cfg, prop, m, k, vecInitG, vecAddG, matInitG, matAddG, cpuVecMs, cpuMatMs);
            }
        }
    }

    return 0;
}

#endif // LAB1_BENCHMARK_H
