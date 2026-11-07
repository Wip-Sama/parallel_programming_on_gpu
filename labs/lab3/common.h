#ifndef LAB3_COMMON_H
#define LAB3_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

// Common CUDA utilities (error checking, timers, device info)
#include "cuda_common.h"

// ---------------------------------------------------------------------------
// Constants & Defaults (from Lab 3 Assignment Specification)
// ---------------------------------------------------------------------------
// Matrix Dimensions from Section 5.1: (M, K, N) = (2^11, 2^10, 2^9)
// A: M x K (2048 x 1024)
// B: K x N (1024 x 512)
// C: M x N (2048 x 512)
static const int DEFAULT_M = 2048;
static const int DEFAULT_K = 1024;
static const int DEFAULT_N = 512;

static const int DEFAULT_BLOCK_SIZE = 16;   // 8, 16, or 32
static const int DEFAULT_TILE_NUMBER = 2;   // Coarsening factor nt (typically 2 or 4)
static const int DEFAULT_NUM_THREADS = 8;   // 8 CPU cores as recommended in Sec 5.4
static const float DEFAULT_TOLERANCE = 1e-4f;

// ---------------------------------------------------------------------------
// Enumerations & Structs
// ---------------------------------------------------------------------------
enum RunMode {
    MODE_CPU_STD,       // CPU baseline standard (i-j-k loop)
    MODE_CPU_OPT,       // CPU optimized loop-interchanged (i-k-j loop)
    MODE_CPU_BOTH,      // Both CPU implementations
    MODE_GPU_NAIVE,     // Point 4.1: CUDA Naive (Global memory only)
    MODE_GPU_TILED,     // Point 4.2: CUDA Tiled (Shared memory)
    MODE_GPU_COARSE,    // Point 4.3: CUDA Coarse-Grained Tiled (Shared memory + block coarsening)
    MODE_GPU_ALL,       // All GPU implementations (Naive + Tiled + Coarse)
    MODE_ALL            // CPU Standard + CPU Optimized + GPU Naive + GPU Tiled + GPU Coarse
};

typedef struct {
    RunMode mode;
    int M;
    int K;
    int N;
    int blockSize;          // 8, 16, or 32 (BxB 2D block dimension)
    int tileWidth;          // Tile width for shared memory (default: matches blockSize)
    int tileNumber;         // Coarsening factor nt (number of tiles per block, default: 2)
    int numThreads;         // OpenMP CPU threads (default: 8)
    int devId;              // CUDA device ID (default: 0)
    int csvOutput;          // Output machine-readable CSV format
    int warmup;             // Run warm-up before timing
    float tolerance;        // Correctness tolerance
} Config;

typedef struct {
    float allocMs;
    float htodMs;
    float kernelMs;
    float dtohMs;
    float computeMs;        // Kernel compute time
    float totalMs;          // Alloc + Kernel + DtoH
    dim3 blockDim;
    dim3 gridDim;
} GpuTiming;

typedef struct {
    double stdMs;
    double optMs;
} CpuTiming;

// ---------------------------------------------------------------------------
// Performance Metrics Helpers
// ---------------------------------------------------------------------------
/**
 * @brief Computes throughput in GFLOP/s for matrix multiplication C = A * B.
 * Total operations: 2 * M * N * K FLOPs (1 multiply + 1 add per element).
 */
static inline double compute_gflops(int M, int K, int N, double timeMs) {
    if (timeMs <= 0.0) return 0.0;
    double totalFlops = 2.0 * (double)M * (double)N * (double)K;
    return (totalFlops * 1e-6) / timeMs;
}

// ---------------------------------------------------------------------------
// Verification Helpers
// ---------------------------------------------------------------------------
static inline bool check_matrix_equality(
    const float *ref,
    const float *test,
    int M,
    int N,
    float tol = DEFAULT_TOLERANCE
) {
    size_t total = (size_t)M * N;
    for (size_t i = 0; i < total; i++) {
        if (fabsf(ref[i] - test[i]) > tol) {
            size_t row = i / N;
            size_t col = i % N;
            fprintf(stderr, "Verification FAILED at [%zu, %zu]: ref=%f, got=%f, diff=%e\n",
                    row, col, ref[i], test[i], fabsf(ref[i] - test[i]));
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// CLI Argument Parsing
// ---------------------------------------------------------------------------
static inline void print_usage(const char *prog) {
    printf("================================================================================\n");
    printf(" Lab 3: Naive, Tiled, and Coarse-Grained Matrix Multiplication\n");
    printf(" Course: Massively Parallel Programming on GPUs - Teacher: Donato D'Ambrosio\n");
    printf("================================================================================\n");
    printf("Usage:\n");
    printf("  %s [mode] [blockSize] [tileNumber] [M] [K] [N] [options]\n\n", prog);
    printf("Positional Arguments (optional):\n");
    printf("  mode        : Execution mode ('all', 'cpu', 'cpu_opt', 'cpu_both', 'gpu_naive', 'gpu_tiled', 'gpu_coarse', 'gpu_all') [default: all]\n");
    printf("  blockSize   : CUDA 2D block dimension ('8', '16', or '32' for BxB) [default: 16]\n");
    printf("  tileNumber  : Coarsening factor nt ('1', '2', or '4' tiles/block) [default: 2]\n");
    printf("  M           : Number of rows in A and C [default: 2048]\n");
    printf("  K           : Columns in A and rows in B [default: 1024]\n");
    printf("  N           : Columns in B and C [default: 512]\n\n");
    printf("Options:\n");
    printf("  --mode <m>       : Execution mode ('all', 'cpu', 'cpu_opt', 'gpu_naive', 'gpu_tiled', 'gpu_coarse', 'gpu_all')\n");
    printf("  --block <b>      : CUDA 2D block dimension (8, 16, 32)\n");
    printf("  --tile <t>       : Shared memory tile width (default: matches block dimension)\n");
    printf("  --nt <num>       : Coarsening factor nt (1, 2, 4) [default: 2]\n");
    printf("  --threads <t>    : Number of OpenMP CPU threads [default: 8]\n");
    printf("  -M <int>         : Matrix M dimension (rows of A & C)\n");
    printf("  -K <int>         : Matrix K dimension (cols of A, rows of B)\n");
    printf("  -N <int>         : Matrix N dimension (cols of B & C)\n");
    printf("  --device <id>    : CUDA device ID [default: 0]\n");
    printf("  --csv            : Output machine-readable CSV format\n");
    printf("  --no-warmup      : Skip initial warm-up run\n");
    printf("  -h, --help       : Show this help message and exit\n");
    printf("================================================================================\n");
}

static inline Config parse_args(int argc, char **argv) {
    Config cfg;
    cfg.mode = MODE_ALL;
    cfg.M = DEFAULT_M;
    cfg.K = DEFAULT_K;
    cfg.N = DEFAULT_N;
    cfg.blockSize = DEFAULT_BLOCK_SIZE;
    cfg.tileWidth = DEFAULT_BLOCK_SIZE;
    cfg.tileNumber = DEFAULT_TILE_NUMBER;
    cfg.numThreads = DEFAULT_NUM_THREADS;
    cfg.devId = 0;
    cfg.csvOutput = 0;
    cfg.warmup = 1;
    cfg.tolerance = DEFAULT_TOLERANCE;

    int pos_idx = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char *m = argv[++i];
            if (strcasecmp(m, "cpu") == 0) cfg.mode = MODE_CPU_STD;
            else if (strcasecmp(m, "cpu_opt") == 0) cfg.mode = MODE_CPU_OPT;
            else if (strcasecmp(m, "cpu_both") == 0) cfg.mode = MODE_CPU_BOTH;
            else if (strcasecmp(m, "gpu_naive") == 0 || strcasecmp(m, "naive") == 0) cfg.mode = MODE_GPU_NAIVE;
            else if (strcasecmp(m, "gpu_tiled") == 0 || strcasecmp(m, "tiled") == 0) cfg.mode = MODE_GPU_TILED;
            else if (strcasecmp(m, "gpu_coarse") == 0 || strcasecmp(m, "coarse") == 0) cfg.mode = MODE_GPU_COARSE;
            else if (strcasecmp(m, "gpu") == 0 || strcasecmp(m, "gpu_all") == 0) cfg.mode = MODE_GPU_ALL;
            else cfg.mode = MODE_ALL;
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            cfg.blockSize = atoi(argv[++i]);
            cfg.tileWidth = cfg.blockSize;
        } else if (strcmp(argv[i], "--tile") == 0 && i + 1 < argc) {
            cfg.tileWidth = atoi(argv[++i]);
            cfg.blockSize = cfg.tileWidth;
        } else if ((strcmp(argv[i], "--nt") == 0 || strcmp(argv[i], "--tile-num") == 0) && i + 1 < argc) {
            cfg.tileNumber = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.numThreads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-M") == 0 && i + 1 < argc) {
            cfg.M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-K") == 0 && i + 1 < argc) {
            cfg.K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) {
            cfg.N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            cfg.devId = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--csv") == 0) {
            cfg.csvOutput = 1;
        } else if (strcmp(argv[i], "--no-warmup") == 0) {
            cfg.warmup = 0;
        } else if (argv[i][0] != '-') {
            if (pos_idx == 0) {
                if (strcasecmp(argv[i], "cpu") == 0) cfg.mode = MODE_CPU_STD;
                else if (strcasecmp(argv[i], "cpu_opt") == 0) cfg.mode = MODE_CPU_OPT;
                else if (strcasecmp(argv[i], "cpu_both") == 0) cfg.mode = MODE_CPU_BOTH;
                else if (strcasecmp(argv[i], "gpu_naive") == 0 || strcasecmp(argv[i], "naive") == 0) cfg.mode = MODE_GPU_NAIVE;
                else if (strcasecmp(argv[i], "gpu_tiled") == 0 || strcasecmp(argv[i], "tiled") == 0) cfg.mode = MODE_GPU_TILED;
                else if (strcasecmp(argv[i], "gpu_coarse") == 0 || strcasecmp(argv[i], "coarse") == 0) cfg.mode = MODE_GPU_COARSE;
                else if (strcasecmp(argv[i], "gpu") == 0 || strcasecmp(argv[i], "gpu_all") == 0) cfg.mode = MODE_GPU_ALL;
                else if (strcasecmp(argv[i], "all") == 0) cfg.mode = MODE_ALL;
                else if (atoi(argv[i]) == 8 || atoi(argv[i]) == 16 || atoi(argv[i]) == 32) {
                    cfg.blockSize = atoi(argv[i]);
                    cfg.tileWidth = cfg.blockSize;
                } else cfg.M = atoi(argv[i]);
            } else if (pos_idx == 1) {
                if (atoi(argv[i]) == 8 || atoi(argv[i]) == 16 || atoi(argv[i]) == 32) {
                    cfg.blockSize = atoi(argv[i]);
                    cfg.tileWidth = cfg.blockSize;
                } else if (atoi(argv[i]) == 1 || atoi(argv[i]) == 2 || atoi(argv[i]) == 4) {
                    cfg.tileNumber = atoi(argv[i]);
                } else cfg.K = atoi(argv[i]);
            } else if (pos_idx == 2) {
                if (atoi(argv[i]) == 1 || atoi(argv[i]) == 2 || atoi(argv[i]) == 4) {
                    cfg.tileNumber = atoi(argv[i]);
                } else cfg.M = atoi(argv[i]);
            } else if (pos_idx == 3) {
                cfg.K = atoi(argv[i]);
            } else if (pos_idx == 4) {
                cfg.N = atoi(argv[i]);
            }
            pos_idx++;
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Results Printing (Terminal & CSV)
// ---------------------------------------------------------------------------
static inline void print_results(
    const Config &cfg,
    double cpuStdMs,
    double cpuOptMs,
    const GpuTiming &gpuNaiveTiming,
    const GpuTiming &gpuTiledTiming,
    const GpuTiming &gpuCoarseTiming,
    bool verifiedStd,
    bool verifiedOpt,
    bool verifiedNaive,
    bool verifiedTiled,
    bool verifiedCoarse
) {
    double totalFlops  = 2.0 * (double)cfg.M * (double)cfg.N * (double)cfg.K;
    double gflopsStd   = compute_gflops(cfg.M, cfg.K, cfg.N, cpuStdMs);
    double gflopsOpt   = compute_gflops(cfg.M, cfg.K, cfg.N, cpuOptMs);
    double gflopsNaive = compute_gflops(cfg.M, cfg.K, cfg.N, gpuNaiveTiming.kernelMs);
    double gflopsTiled = compute_gflops(cfg.M, cfg.K, cfg.N, gpuTiledTiming.kernelMs);
    double gflopsCoarse= compute_gflops(cfg.M, cfg.K, cfg.N, gpuCoarseTiming.kernelMs);

    if (cfg.csvOutput) {
        if (cfg.mode == MODE_CPU_STD) {
            printf("CSV,cpu_std,%d,%d,%d,%d,%.4f,%.2f,%s\n",
                   cfg.numThreads, cfg.M, cfg.K, cfg.N, cpuStdMs, gflopsStd,
                   verifiedStd ? "PASS" : "FAIL");
        } else if (cfg.mode == MODE_CPU_OPT) {
            printf("CSV,cpu_opt,%d,%d,%d,%d,%.4f,%.2f,%s\n",
                   cfg.numThreads, cfg.M, cfg.K, cfg.N, cpuOptMs, gflopsOpt,
                   verifiedOpt ? "PASS" : "FAIL");
        } else if (cfg.mode == MODE_GPU_NAIVE) {
            printf("CSV,gpu_naive,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%s\n",
                   cfg.blockSize, gpuNaiveTiming.gridDim.x, gpuNaiveTiming.gridDim.y,
                   cfg.numThreads, cfg.M, cfg.K, cfg.N,
                   gpuNaiveTiming.allocMs, gpuNaiveTiming.htodMs, gpuNaiveTiming.kernelMs, gpuNaiveTiming.dtohMs,
                   gpuNaiveTiming.computeMs, gpuNaiveTiming.totalMs, gflopsNaive,
                   verifiedNaive ? "PASS" : "FAIL");
        } else if (cfg.mode == MODE_GPU_TILED) {
            printf("CSV,gpu_tiled,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%s\n",
                   cfg.blockSize, gpuTiledTiming.gridDim.x, gpuTiledTiming.gridDim.y,
                   cfg.numThreads, cfg.M, cfg.K, cfg.N,
                   gpuTiledTiming.allocMs, gpuTiledTiming.htodMs, gpuTiledTiming.kernelMs, gpuTiledTiming.dtohMs,
                   gpuTiledTiming.computeMs, gpuTiledTiming.totalMs, gflopsTiled,
                   verifiedTiled ? "PASS" : "FAIL");
        } else if (cfg.mode == MODE_GPU_COARSE) {
            printf("CSV,gpu_coarse,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%s\n",
                   cfg.blockSize, cfg.tileNumber, gpuCoarseTiming.gridDim.x, gpuCoarseTiming.gridDim.y,
                   cfg.numThreads, cfg.M, cfg.K, cfg.N,
                   gpuCoarseTiming.allocMs, gpuCoarseTiming.htodMs, gpuCoarseTiming.kernelMs, gpuCoarseTiming.dtohMs,
                   gpuCoarseTiming.computeMs, gpuCoarseTiming.totalMs, gflopsCoarse,
                   verifiedCoarse ? "PASS" : "FAIL");
        } else {
            // Full consolidated row
            printf("CSV,all,%d,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%.2f,%.2f,%s\n",
                   cfg.blockSize, cfg.tileNumber, cfg.numThreads, cfg.M, cfg.K, cfg.N,
                   cpuStdMs, cpuOptMs,
                   gpuNaiveTiming.kernelMs, gpuNaiveTiming.totalMs,
                   gpuTiledTiming.kernelMs, gpuTiledTiming.totalMs,
                   gpuCoarseTiming.kernelMs, gpuCoarseTiming.totalMs,
                   gflopsStd, gflopsOpt, gflopsNaive, gflopsTiled, gflopsCoarse,
                   (verifiedStd && verifiedOpt && verifiedNaive && verifiedTiled && verifiedCoarse) ? "PASS" : "FAIL");
        }
    } else {
        printf("================================================================================\n");
        printf(" Lab 3: Matrix Multiplication Performance Results\n");
        printf(" Matrix Dimensions      : M=%d, K=%d, N=%d\n", cfg.M, cfg.K, cfg.N);
        printf(" Total Operations       : %.3f GFLOP (%.0f FLOPs)\n", totalFlops * 1e-9, totalFlops);
        printf(" Memory per Matrix      : A=%.2f MB, B=%.2f MB, C=%.2f MB (Total: %.2f MB)\n",
               (double)(cfg.M * cfg.K * sizeof(float)) / (1024.0 * 1024.0),
               (double)(cfg.K * cfg.N * sizeof(float)) / (1024.0 * 1024.0),
               (double)(cfg.M * cfg.N * sizeof(float)) / (1024.0 * 1024.0),
               (double)((cfg.M*cfg.K + cfg.K*cfg.N + cfg.M*cfg.N) * sizeof(float)) / (1024.0 * 1024.0));
        printf("--------------------------------------------------------------------------------\n");

        if (cfg.mode == MODE_CPU_STD || cfg.mode == MODE_CPU_BOTH || cfg.mode == MODE_ALL) {
            printf(" CPU Baseline (OpenMP)  : %8.4f ms | %8.2f GFLOP/s (%d threads)\n",
                   cpuStdMs, gflopsStd, cfg.numThreads);
        }
        if (cfg.mode == MODE_CPU_OPT || cfg.mode == MODE_CPU_BOTH || cfg.mode == MODE_ALL) {
            double speedupOverStd = (cpuOptMs > 0.0) ? (cpuStdMs / cpuOptMs) : 0.0;
            printf(" CPU Optimized (Loop I-K-J): %8.4f ms | %8.2f GFLOP/s (Speedup vs Std: %.2fx, %s)\n",
                   cpuOptMs, gflopsOpt, speedupOverStd, verifiedOpt ? "PASSED" : "FAILED");
        }

        // Point 4.1: GPU Naive
        if (cfg.mode == MODE_GPU_NAIVE || cfg.mode == MODE_GPU_ALL || cfg.mode == MODE_ALL) {
            printf("--------------------------------------------------------------------------------\n");
            printf(" Point 4.1: CUDA Naive Kernel (Global Memory Only)\n");
            printf("   Block Dimensions     : %d x %d (%d threads/block)\n",
                   gpuNaiveTiming.blockDim.x, gpuNaiveTiming.blockDim.y, cfg.blockSize * cfg.blockSize);
            printf("   Grid Dimensions      : %d x %d (%d blocks)\n",
                   gpuNaiveTiming.gridDim.x, gpuNaiveTiming.gridDim.y, gpuNaiveTiming.gridDim.x * gpuNaiveTiming.gridDim.y);
            printf("   Pure Kernel Time     : %8.4f ms | %8.2f GFLOP/s\n", gpuNaiveTiming.kernelMs, gflopsNaive);
            printf("   Total GPU Time       : %8.4f ms (Alloc + Kernel)\n", gpuNaiveTiming.totalMs);
            if (cpuStdMs > 0.0 && gpuNaiveTiming.kernelMs > 0.0) {
                printf("   Speedup vs CPU Std   : %8.2fx (Kernel) | %8.2fx (Total)\n",
                       cpuStdMs / gpuNaiveTiming.kernelMs, cpuStdMs / gpuNaiveTiming.totalMs);
            }
            if (cpuOptMs > 0.0 && gpuNaiveTiming.kernelMs > 0.0) {
                printf("   Speedup vs CPU Opt   : %8.2fx (Kernel) | %8.2fx (Total)\n",
                       cpuOptMs / gpuNaiveTiming.kernelMs, cpuOptMs / gpuNaiveTiming.totalMs);
            }
            printf("   Verification vs CPU  : %s\n", verifiedNaive ? "PASSED" : "FAILED");
        }

        // Point 4.2: GPU Tiled
        if (cfg.mode == MODE_GPU_TILED || cfg.mode == MODE_GPU_ALL || cfg.mode == MODE_ALL) {
            printf("--------------------------------------------------------------------------------\n");
            printf(" Point 4.2: CUDA Tiled Kernel (Global & Shared Memory)\n");
            printf("   Tile / Block Dim     : %d x %d (%d threads, 2x%dx%d shared floats)\n",
                   cfg.tileWidth, cfg.tileWidth, cfg.blockSize * cfg.blockSize, cfg.tileWidth, cfg.tileWidth);
            printf("   Grid Dimensions      : %d x %d (%d blocks)\n",
                   gpuTiledTiming.gridDim.x, gpuTiledTiming.gridDim.y, gpuTiledTiming.gridDim.x * gpuTiledTiming.gridDim.y);
            printf("   Pure Kernel Time     : %8.4f ms | %8.2f GFLOP/s\n", gpuTiledTiming.kernelMs, gflopsTiled);
            printf("   Total GPU Time       : %8.4f ms (Alloc + Kernel)\n", gpuTiledTiming.totalMs);
            if (cpuStdMs > 0.0 && gpuTiledTiming.kernelMs > 0.0) {
                printf("   Speedup vs CPU Std   : %8.2fx (Kernel) | %8.2fx (Total)\n",
                       cpuStdMs / gpuTiledTiming.kernelMs, cpuStdMs / gpuTiledTiming.totalMs);
            }
            if (gpuNaiveTiming.kernelMs > 0.0 && gpuTiledTiming.kernelMs > 0.0) {
                printf("   Speedup vs Naive GPU : %8.2fx (Kernel Speedup)\n",
                       gpuNaiveTiming.kernelMs / gpuTiledTiming.kernelMs);
            }
            printf("   Verification vs CPU  : %s\n", verifiedTiled ? "PASSED" : "FAILED");
        }

        // Point 4.3: GPU Coarse-Grained Tiled
        if (cfg.mode == MODE_GPU_COARSE || cfg.mode == MODE_GPU_ALL || cfg.mode == MODE_ALL) {
            printf("--------------------------------------------------------------------------------\n");
            printf(" Point 4.3: CUDA Coarse-Grained Tiled Kernel (Shared Mem + nt=%d Coarsening)\n", cfg.tileNumber);
            printf("   Tile / Block Dim     : %d x %d (%d threads, (1+%d)x%dx%d shared floats)\n",
                   cfg.tileWidth, cfg.tileWidth, cfg.blockSize * cfg.blockSize, cfg.tileNumber, cfg.tileWidth, cfg.tileWidth);
            printf("   Grid Dimensions      : %d x %d (%d blocks)\n",
                   gpuCoarseTiming.gridDim.x, gpuCoarseTiming.gridDim.y, gpuCoarseTiming.gridDim.x * gpuCoarseTiming.gridDim.y);
            printf("   Pure Kernel Time     : %8.4f ms | %8.2f GFLOP/s\n", gpuCoarseTiming.kernelMs, gflopsCoarse);
            printf("   Total GPU Time       : %8.4f ms (Alloc + Kernel)\n", gpuCoarseTiming.totalMs);
            if (cpuStdMs > 0.0 && gpuCoarseTiming.kernelMs > 0.0) {
                printf("   Speedup vs CPU Std   : %8.2fx (Kernel) | %8.2fx (Total)\n",
                       cpuStdMs / gpuCoarseTiming.kernelMs, cpuStdMs / gpuCoarseTiming.totalMs);
            }
            if (gpuNaiveTiming.kernelMs > 0.0 && gpuCoarseTiming.kernelMs > 0.0) {
                printf("   Speedup vs Naive GPU : %8.2fx (Kernel Speedup)\n",
                       gpuNaiveTiming.kernelMs / gpuCoarseTiming.kernelMs);
            }
            if (gpuTiledTiming.kernelMs > 0.0 && gpuCoarseTiming.kernelMs > 0.0) {
                printf("   Speedup vs Tiled GPU : %8.2fx (Kernel Speedup)\n",
                       gpuTiledTiming.kernelMs / gpuCoarseTiming.kernelMs);
            }
            printf("   Verification vs CPU  : %s\n", verifiedCoarse ? "PASSED" : "FAILED");
        }
        printf("================================================================================\n");
    }
}

#endif // LAB3_COMMON_H
