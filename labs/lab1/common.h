#ifndef LAB1_COMMON_H
#define LAB1_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error Checking Macros (Ref: Proper CUDA Error Checking)
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Constants & Defaults
// ---------------------------------------------------------------------------
// Vector size: n = 2^24 = 16,777,216 elements (as specified in PDF)
static const size_t DEFAULT_VEC_N = 1ULL << 24;

// Matrix size: (nrows, ncols) = (2048, 4096), maintaining (n/2, n) ratio
// while fitting safely in GPU memory (192 MB for 3 matrices)
static const size_t DEFAULT_MAT_ROWS = 2048;
static const size_t DEFAULT_MAT_COLS = 4096;

static const int BLOCK_SIZES[] = {32, 64, 128, 256, 512, 1024};
static const int NUM_BLOCK_SIZES = 6;

// ---------------------------------------------------------------------------
// Execution & Memory Model Enumerations
// ---------------------------------------------------------------------------
enum MemoryModel {
    MEM_UNIFIED,
    MEM_STANDARD,
    MEM_ALL
};

enum KernelModel {
    MODEL_MONOLITHIC,
    MODEL_GRID_STRIDE,
    MODEL_ALL
};

// ---------------------------------------------------------------------------
// Timing & CPU Reference Implementations
// ---------------------------------------------------------------------------
static inline double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static inline void cpu_vector_init(double *data, double val, size_t n) {
    for (size_t i = 0; i < n; i++) data[i] = val;
}

static inline void cpu_vector_add(const double *A, const double *B, double *C, size_t n) {
    for (size_t i = 0; i < n; i++) C[i] = A[i] + B[i];
}

static inline void cpu_matrix_init(double *M, double val, size_t nrows, size_t ncols) {
    size_t total = nrows * ncols;
    for (size_t i = 0; i < total; i++) M[i] = val;
}

static inline void cpu_matrix_add(const double *A, const double *B, double *C, size_t nrows, size_t ncols) {
    size_t total = nrows * ncols;
    for (size_t i = 0; i < total; i++) C[i] = A[i] + B[i];
}

// ---------------------------------------------------------------------------
// Verification Helpers
// ---------------------------------------------------------------------------
static inline int verify_vector(const double *C, size_t n, double expected) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(C[i] - expected) > 1e-9) {
            fprintf(stderr, "Verification FAILED at index %zu: got %f, expected %f\n",
                    i, C[i], expected);
            return 0;
        }
    }
    return 1;
}

static inline int verify_matrix(const double *C, size_t nrows, size_t ncols, double expected) {
    size_t total = nrows * ncols;
    for (size_t i = 0; i < total; i++) {
        if (fabs(C[i] - expected) > 1e-9) {
            fprintf(stderr, "Verification FAILED at index %zu (row %zu, col %zu): got %f, expected %f\n",
                    i, i / ncols, i % ncols, C[i], expected);
            return 0;
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Result Structures
// ---------------------------------------------------------------------------
typedef struct {
    int blockSize;
    int numBlocks;
    float initTimeMs;
    float addTimeMs;
    float memcpyTimeMs;
    float totalGpuTimeMs;
    double cpuTimeMs;
    double speedup;
    int verified;
} GpuBenchResult1D;

typedef struct {
    int blockSize;
    dim3 blockDim;
    dim3 gridDim;
    float initTimeMs;
    float addTimeMs;
    float memcpyTimeMs;
    float totalGpuTimeMs;
    double cpuTimeMs;
    double speedup;
    int verified;
} GpuBenchResult2D;

// ---------------------------------------------------------------------------
// CLI Argument Parsing & Help
// ---------------------------------------------------------------------------
typedef struct {
    int devId;
    int singleBlockSize;      // -1 for sweep (all), or specific block size (e.g. 256)
    const char *mode;         // "vector", "matrix", "all"
    MemoryModel memModel;     // MEM_UNIFIED, MEM_STANDARD, MEM_ALL
    KernelModel kernelModel;  // MODEL_MONOLITHIC, MODEL_GRID_STRIDE, MODEL_ALL
    size_t vecN;
    size_t matRows;
    size_t matCols;
    int numRuns;              // default 1 (or N for averaging)
    int csvOutput;            // 1 if CSV output requested
    int showHelp;             // 1 if help banner requested
} AppConfig;

static inline void print_usage(const char *progName) {
    printf("================================================================================\n");
    printf(" Lab 1: First CUDA Applications Benchmark Driver\n");
    printf("================================================================================\n");
    printf("Usage:\n");
    printf("  %s <u|s> <m|gs> <vector|matrix> [size] [options]\n\n", progName);
    printf("Arguments:\n");
    printf("  1. Memory Model (u|s):\n");
    printf("       u,  unified       : Unified Memory (cudaMallocManaged)\n");
    printf("       s,  standard      : Standard Memory (cudaMalloc + explicit cudaMemcpy DtoH)\n");
    printf("  2. Kernel Execution Model (m|gs):\n");
    printf("       m,  monolithic    : Monolithic 1-to-1 thread-to-element grid mapping\n");
    printf("       gs, grid_stride   : Decoupled Grid-Stride Loop execution (SM-scaled grid)\n");
    printf("  3. Target Operation (vector|matrix):\n");
    printf("       vector, vec, v    : Double-precision Vector Addition (n = 2^24 elements)\n");
    printf("       matrix, mat, m    : Double-precision Matrix Addition (2048 x 4096 elements)\n");
    printf("       all               : Run both vector and matrix operations\n");
    printf("  4. [size] (optional block size):\n");
    printf("       32..1024          : Run single block size (e.g. 32, 64, 128, 256, 512, 1024)\n");
    printf("       all (or omitted)  : Sweep across all standard block sizes\n\n");
    printf("Options:\n");
    printf("  -n, --runs <N>         : Repeat benchmark N times and average findings (default: 1)\n");
    printf("  --device <id>          : Select CUDA device ID (default: 0)\n");
    printf("  --csv                  : Output results in CSV format for automated parsing\n");
    printf("  --vec-n <N>            : Custom vector size (default: 16777216)\n");
    printf("  --mat-rows <R>         : Custom matrix rows (default: 2048)\n");
    printf("  --mat-cols <C>         : Custom matrix cols (default: 4096)\n");
    printf("  -h, --help             : Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s u m vector 256       # Unified memory, monolithic kernel, vector, blockSize 256\n", progName);
    printf("  %s s gs matrix 512      # Standard memory, grid-stride kernel, matrix, blockSize 512\n", progName);
    printf("  %s u gs vector          # Sweep all block sizes for unified grid-stride vector\n", progName);
    printf("  %s s m vector 256 -n 5  # Run 5 times and average results\n", progName);
    printf("  %s all                  # Run full comparison sweep for all 4 implementations\n", progName);
    printf("================================================================================\n");
}

static inline AppConfig parse_app_args(int argc, char **argv) {
    AppConfig cfg;
    cfg.devId = 0;
    cfg.singleBlockSize = -1;
    cfg.mode = "all";
    cfg.memModel = MEM_UNIFIED;
    cfg.kernelModel = MODEL_MONOLITHIC;
    cfg.vecN = DEFAULT_VEC_N;
    cfg.matRows = DEFAULT_MAT_ROWS;
    cfg.matCols = DEFAULT_MAT_COLS;
    cfg.numRuns = 1;
    cfg.csvOutput = 0;
    cfg.showHelp = 0;

    if (argc <= 1) {
        cfg.showHelp = 1;
        return cfg;
    }

    // Check if called via legacy binary name / symlink
    const char *prog = argv[0];
    const char *slash = strrchr(prog, '/');
    if (slash) prog = slash + 1;
    int isLegacyName = 0;
    if (strcmp(prog, "unified_monolithic") == 0) {
        cfg.memModel = MEM_UNIFIED;
        cfg.kernelModel = MODEL_MONOLITHIC;
        isLegacyName = 1;
    } else if (strcmp(prog, "unified_grid_stride") == 0) {
        cfg.memModel = MEM_UNIFIED;
        cfg.kernelModel = MODEL_GRID_STRIDE;
        isLegacyName = 1;
    } else if (strcmp(prog, "standard_monolithic") == 0) {
        cfg.memModel = MEM_STANDARD;
        cfg.kernelModel = MODEL_MONOLITHIC;
        isLegacyName = 1;
    } else if (strcmp(prog, "standard_grid_stride") == 0) {
        cfg.memModel = MEM_STANDARD;
        cfg.kernelModel = MODEL_GRID_STRIDE;
        isLegacyName = 1;
    }

    int posIdx = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            cfg.showHelp = 1;
            return cfg;
        } else if (strcmp(arg, "--csv") == 0) {
            cfg.csvOutput = 1;
        } else if ((strcmp(arg, "-n") == 0 || strcmp(arg, "--runs") == 0) && i + 1 < argc) {
            cfg.numRuns = atoi(argv[++i]);
            if (cfg.numRuns < 1) cfg.numRuns = 1;
        } else if (strcmp(arg, "--device") == 0 && i + 1 < argc) {
            cfg.devId = atoi(argv[++i]);
        } else if (strcmp(arg, "--vec-n") == 0 && i + 1 < argc) {
            cfg.vecN = (size_t)atoll(argv[++i]);
        } else if (strcmp(arg, "--mat-rows") == 0 && i + 1 < argc) {
            cfg.matRows = (size_t)atoll(argv[++i]);
        } else if (strcmp(arg, "--mat-cols") == 0 && i + 1 < argc) {
            cfg.matCols = (size_t)atoll(argv[++i]);
        } else if (strcmp(arg, "all") == 0 && posIdx == 0 && !isLegacyName) {
            cfg.memModel = MEM_ALL;
            cfg.kernelModel = MODEL_ALL;
            cfg.mode = "all";
            posIdx = 4;
        } else {
            if (isLegacyName) {
                // Legacy: [vector|matrix] [blockSize] [matRows] [matCols]
                if (posIdx == 0) {
                    if (strcmp(arg, "vector") == 0 || strcmp(arg, "v") == 0) {
                        cfg.mode = "vector";
                    } else if (strcmp(arg, "matrix") == 0 || strcmp(arg, "m") == 0) {
                        cfg.mode = "matrix";
                    } else if (strcmp(arg, "all") == 0) {
                        cfg.mode = "all";
                    } else if (arg[0] >= '0' && arg[0] <= '9') {
                        cfg.mode = "vector";
                        cfg.singleBlockSize = atoi(arg);
                    }
                    posIdx++;
                } else if (posIdx == 1) {
                    if (arg[0] >= '0' && arg[0] <= '9') {
                        cfg.singleBlockSize = atoi(arg);
                    }
                    posIdx++;
                } else if (posIdx == 2 && strcmp(cfg.mode, "matrix") == 0 && arg[0] >= '0' && arg[0] <= '9') {
                    cfg.matRows = (size_t)atoll(arg);
                    posIdx++;
                } else if (posIdx == 3 && strcmp(cfg.mode, "matrix") == 0 && arg[0] >= '0' && arg[0] <= '9') {
                    cfg.matCols = (size_t)atoll(arg);
                    posIdx++;
                }
            } else {
                // New unified layout: ./lab1 u/s m/gs vector/matrix <size> [runs]
                if (posIdx == 0) {
                    if (strcmp(arg, "u") == 0 || strcasecmp(arg, "unified") == 0) {
                        cfg.memModel = MEM_UNIFIED;
                    } else if (strcmp(arg, "s") == 0 || strcasecmp(arg, "standard") == 0) {
                        cfg.memModel = MEM_STANDARD;
                    } else if (strcmp(arg, "all") == 0) {
                        cfg.memModel = MEM_ALL;
                    } else {
                        fprintf(stderr, "Error: Unknown memory model '%s'. Use 'u' or 's'.\n", arg);
                        cfg.showHelp = 1;
                        return cfg;
                    }
                    posIdx++;
                } else if (posIdx == 1) {
                    if (strcmp(arg, "m") == 0 || strcasecmp(arg, "monolithic") == 0) {
                        cfg.kernelModel = MODEL_MONOLITHIC;
                    } else if (strcmp(arg, "gs") == 0 || strcasecmp(arg, "grid_stride") == 0 || strcasecmp(arg, "grid-stride") == 0) {
                        cfg.kernelModel = MODEL_GRID_STRIDE;
                    } else if (strcmp(arg, "all") == 0) {
                        cfg.kernelModel = MODEL_ALL;
                    } else {
                        fprintf(stderr, "Error: Unknown kernel model '%s'. Use 'm' or 'gs'.\n", arg);
                        cfg.showHelp = 1;
                        return cfg;
                    }
                    posIdx++;
                } else if (posIdx == 2) {
                    if (strcmp(arg, "vector") == 0 || strcmp(arg, "vec") == 0 || strcmp(arg, "v") == 0) {
                        cfg.mode = "vector";
                    } else if (strcmp(arg, "matrix") == 0 || strcmp(arg, "mat") == 0 || strcmp(arg, "m") == 0) {
                        cfg.mode = "matrix";
                    } else if (strcmp(arg, "all") == 0) {
                        cfg.mode = "all";
                    } else {
                        fprintf(stderr, "Error: Unknown operation '%s'. Use 'vector' or 'matrix'.\n", arg);
                        cfg.showHelp = 1;
                        return cfg;
                    }
                    posIdx++;
                } else if (posIdx == 3) {
                    if (strcmp(arg, "all") == 0 || strcmp(arg, "sweep") == 0) {
                        cfg.singleBlockSize = -1;
                    } else if (arg[0] >= '0' && arg[0] <= '9') {
                        cfg.singleBlockSize = atoi(arg);
                    }
                    posIdx++;
                } else if (posIdx == 4) {
                    if (arg[0] >= '0' && arg[0] <= '9') {
                        cfg.numRuns = atoi(arg);
                        if (cfg.numRuns < 1) cfg.numRuns = 1;
                    }
                    posIdx++;
                }
            }
        }
    }

    return cfg;
}

// ---------------------------------------------------------------------------
// Header and Banner Display
// ---------------------------------------------------------------------------
static inline cudaDeviceProp print_app_header(const char *title, int devId) {
    CUDA_CHECK(cudaSetDevice(devId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));

    printf("================================================================================\n");
    printf(" %s\n", title);
    printf(" Target Device [%d]: %s (Compute %d.%d, %d SMs, %.2f GB Global Mem)\n",
           devId, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("================================================================================\n");
    return prop;
}

// ---------------------------------------------------------------------------
// Single-Run Report (for nvprof / ncu profiling or single-block testing)
// ---------------------------------------------------------------------------
static inline void print_single_run_1d(const char *title, size_t n, const GpuBenchResult1D *res, int numRuns) {
    size_t bytes = n * sizeof(double);
    printf("\n=======================================================\n");
    if (numRuns > 1) {
        printf("  Single Run (Average of %d runs) (%s)\n", numRuns, title);
    } else {
        printf("  Single Run (%s)\n", title);
    }
    printf("=======================================================\n");
    printf(" Vector size (n)         : %zu elements (%.2f MB per vector)\n", n, (double)bytes / (1024.0 * 1024.0));
    printf(" Block size              : %d threads\n", res->blockSize);
    printf(" Grid size               : %d blocks\n", res->numBlocks);
    printf(" Time 3x vectorInit      : %8.3f ms%s\n", res->initTimeMs, (numRuns > 1) ? " (avg)" : "");
    printf(" Time 1x vectorAdd       : %8.3f ms%s\n", res->addTimeMs, (numRuns > 1) ? " (avg)" : "");
    if (res->memcpyTimeMs > 0.0f) {
        printf(" Time cudaMemcpy (DtoH)  : %8.3f ms%s\n", res->memcpyTimeMs, (numRuns > 1) ? " (avg)" : "");
    }
    printf(" Total GPU Elapsed Time  : %8.3f ms%s\n", res->totalGpuTimeMs, (numRuns > 1) ? " (avg)" : "");
    if (res->cpuTimeMs > 0.0) {
        printf(" CPU Serial Time (calc)  : %8.3f ms\n", res->cpuTimeMs);
        printf(" Speedup vs CPU Serial   : %8.2fx\n", res->speedup);
    }
    printf(" Correctness Check       : %s\n", res->verified ? "PASSED (all C[i] == 3.0)" : "FAILED");
    printf("=======================================================\n");
}

static inline void print_single_run_2d(const char *title, size_t nrows, size_t ncols, const GpuBenchResult2D *res, int numRuns) {
    size_t total = nrows * ncols;
    size_t bytes = total * sizeof(double);
    printf("\n=======================================================\n");
    if (numRuns > 1) {
        printf("  Single Run (Average of %d runs) (%s)\n", numRuns, title);
    } else {
        printf("  Single Run (%s)\n", title);
    }
    printf("=======================================================\n");
    printf(" Matrix dimensions       : %zu x %zu (%zu elements, %.2f MB)\n",
           nrows, ncols, total, (double)bytes / (1024.0 * 1024.0));
    printf(" Block dimensions        : (%d, %d) = %d threads\n",
           res->blockDim.x, res->blockDim.y, res->blockSize);
    printf(" Grid dimensions         : (%d, %d) blocks\n", res->gridDim.x, res->gridDim.y);
    printf(" Time 3x matrixInit      : %8.3f ms%s\n", res->initTimeMs, (numRuns > 1) ? " (avg)" : "");
    printf(" Time 1x matrixAdd       : %8.3f ms%s\n", res->addTimeMs, (numRuns > 1) ? " (avg)" : "");
    if (res->memcpyTimeMs > 0.0f) {
        printf(" Time cudaMemcpy (DtoH)  : %8.3f ms%s\n", res->memcpyTimeMs, (numRuns > 1) ? " (avg)" : "");
    }
    printf(" Total GPU Elapsed Time  : %8.3f ms%s\n", res->totalGpuTimeMs, (numRuns > 1) ? " (avg)" : "");
    if (res->cpuTimeMs > 0.0) {
        printf(" CPU Serial Time (calc)  : %8.3f ms\n", res->cpuTimeMs);
        printf(" Speedup vs CPU Serial   : %8.2fx\n", res->speedup);
    }
    printf(" Correctness Check       : %s\n", res->verified ? "PASSED (all C[i][j] == 3.0)" : "FAILED");
    printf("=======================================================\n");
}

// ---------------------------------------------------------------------------
// CPU Baseline Measurement Helpers
// ---------------------------------------------------------------------------
static inline double measure_cpu_vector_baseline(size_t n) {
    size_t bytes = n * sizeof(double);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Host memory allocation failed for CPU vector baseline\n");
        exit(EXIT_FAILURE);
    }
    double t0 = get_time_sec();
    cpu_vector_init(A, 1.0, n);
    cpu_vector_init(B, 2.0, n);
    cpu_vector_init(C, 0.0, n);
    cpu_vector_add(A, B, C, n);
    double t1 = get_time_sec();
    free(A); free(B); free(C);
    return (t1 - t0) * 1000.0;
}

static inline double measure_cpu_matrix_baseline(size_t nrows, size_t ncols) {
    size_t bytes = nrows * ncols * sizeof(double);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    if (!A || !B || !C) {
        fprintf(stderr, "Host memory allocation failed for CPU matrix baseline\n");
        exit(EXIT_FAILURE);
    }
    double t0 = get_time_sec();
    cpu_matrix_init(A, 1.0, nrows, ncols);
    cpu_matrix_init(B, 2.0, nrows, ncols);
    cpu_matrix_init(C, 0.0, nrows, ncols);
    cpu_matrix_add(A, B, C, nrows, ncols);
    double t1 = get_time_sec();
    free(A); free(B); free(C);
    return (t1 - t0) * 1000.0;
}

// ---------------------------------------------------------------------------
// Sweep Table Printing Helpers
// ---------------------------------------------------------------------------
static inline void print_sweep_table_1d(const char *title, int hasMemcpy,
                                       const GpuBenchResult1D *results, int numResults) {
    printf("\n----------------------------------------------------------------------------------------------------\n");
    printf(" Vector Addition: %s\n", title);
    printf(" (Double Precision, n = 2^24, 384 MB Device)\n");
    printf("----------------------------------------------------------------------------------------------------\n");
    if (hasMemcpy) {
        printf("%-10s %-10s %-13s %-13s %-13s %-14s %-12s %-10s %-6s\n",
               "blockSize", "gridSize", "Init 3x (ms)", "Add 1x (ms)", "Memcpy (ms)",
               "Total GPU(ms)", "CPU Ser(ms)", "Speedup", "Status");
    } else {
        printf("%-10s %-12s %-14s %-14s %-14s %-14s %-10s %-8s\n",
               "blockSize", "gridSize", "Init 3x (ms)", "Add 1x (ms)", "Total GPU(ms)",
               "CPU Ser(ms)", "Speedup", "Status");
    }
    printf("----------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < numResults; i++) {
        const GpuBenchResult1D *r = &results[i];
        char speedupStr[16];
        snprintf(speedupStr, sizeof(speedupStr), "%.2fx", r->speedup);
        if (hasMemcpy) {
            printf("%-10d %-10d %-13.3f %-13.3f %-13.3f %-14.3f %-12.3f %-10s %-6s\n",
                   r->blockSize, r->numBlocks, r->initTimeMs, r->addTimeMs, r->memcpyTimeMs,
                   r->totalGpuTimeMs, r->cpuTimeMs, speedupStr, r->verified ? "PASS" : "FAIL");
        } else {
            printf("%-10d %-12d %-14.3f %-14.3f %-14.3f %-14.3f %-10s %-8s\n",
                   r->blockSize, r->numBlocks, r->initTimeMs, r->addTimeMs,
                   r->totalGpuTimeMs, r->cpuTimeMs, speedupStr, r->verified ? "PASS" : "FAIL");
        }
    }
    printf("----------------------------------------------------------------------------------------------------\n");
}

static inline void print_sweep_table_2d(const char *title, int hasMemcpy,
                                       const GpuBenchResult2D *results, int numResults) {
    printf("\n----------------------------------------------------------------------------------------------------\n");
    printf(" Matrix Addition: %s\n", title);
    printf(" (Double Precision, 192 MB Device)\n");
    printf("----------------------------------------------------------------------------------------------------\n");
    if (hasMemcpy) {
        printf("%-10s %-12s %-13s %-13s %-13s %-14s %-12s %-10s %-6s\n",
               "blockSize", "blockDim", "Init 3x (ms)", "Add 1x (ms)", "Memcpy (ms)",
               "Total GPU(ms)", "CPU Ser(ms)", "Speedup", "Status");
    } else {
        printf("%-10s %-14s %-14s %-14s %-14s %-14s %-10s %-8s\n",
               "blockSize", "blockDim", "Init 3x (ms)", "Add 1x (ms)", "Total GPU(ms)",
               "CPU Ser(ms)", "Speedup", "Status");
    }
    printf("----------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < numResults; i++) {
        const GpuBenchResult2D *r = &results[i];
        char blockDimStr[32];
        snprintf(blockDimStr, sizeof(blockDimStr), "(%d,%d)", r->blockDim.x, r->blockDim.y);
        char speedupStr[16];
        snprintf(speedupStr, sizeof(speedupStr), "%.2fx", r->speedup);
        if (hasMemcpy) {
            printf("%-10d %-12s %-13.3f %-13.3f %-13.3f %-14.3f %-12.3f %-10s %-6s\n",
                   r->blockSize, blockDimStr, r->initTimeMs, r->addTimeMs, r->memcpyTimeMs,
                   r->totalGpuTimeMs, r->cpuTimeMs, speedupStr, r->verified ? "PASS" : "FAIL");
        } else {
            printf("%-10d %-14s %-14.3f %-14.3f %-14.3f %-14.3f %-10s %-8s\n",
                   r->blockSize, blockDimStr, r->initTimeMs, r->addTimeMs,
                   r->totalGpuTimeMs, r->cpuTimeMs, speedupStr, r->verified ? "PASS" : "FAIL");
        }
    }
    printf("----------------------------------------------------------------------------------------------------\n");
}

#endif // LAB1_COMMON_H
