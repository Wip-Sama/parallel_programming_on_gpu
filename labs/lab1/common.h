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
// Error Checking Macro for checking even cuda dignity
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) cuda_check_error(call, #call, __FILE__, __LINE__)

inline void cuda_check_error(cudaError_t err, const char *call, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",
                call, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------
// Constants & Defaults
// ---------------------------------------------------------------------------
// Vector size: n = 2^24 = 16,777,216 elements (128 MB per vector)
static const size_t DEFAULT_VEC_N = 1ULL << 24;

// Matrix size: (nrows, ncols) = (2048, 4096), maintaining (n/2, n) ratio (64 MB per matrix)
static const size_t DEFAULT_MAT_ROWS = 2048;
static const size_t DEFAULT_MAT_COLS = 4096;

static const int DEFAULT_BLOCK_SIZE = 256;

// ---------------------------------------------------------------------------
// Enumerations
// ---------------------------------------------------------------------------
enum MemoryModel {
    MEM_UNIFIED,
    MEM_STANDARD
};

enum KernelModel {
    MODEL_MONOLITHIC,
    MODEL_GRID_STRIDE
};

enum OperationType {
    OP_VECTOR,
    OP_MATRIX
};

// ---------------------------------------------------------------------------
// Configuration & Results Structs
// ---------------------------------------------------------------------------
typedef struct {
    MemoryModel memModel;
    KernelModel kernelModel;
    OperationType opType;
    int blockSize;
    int devId;
    int csvOutput;
    size_t vecN;
    size_t matRows;
    size_t matCols;
} RunConfig;

typedef struct {
    int blockSize;
    dim3 blockDim;
    dim3 gridDim;
    float initTimeMs;
    float addTimeMs;
    float memcpyTimeMs;
    float deviceTimeMs;
    double totalRunTimeMs;
    int verified;
} RunResult;

// ---------------------------------------------------------------------------
// High-Resolution Wall-Clock Timer
// ---------------------------------------------------------------------------
static inline double get_wall_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
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
// Headers and Formatted Output Displays
// ---------------------------------------------------------------------------
static inline void print_usage(const char *progName) {
    printf("================================================================================\n");
    printf(" Lab 1: First CUDA Applications - Single Configuration Driver\n");
    printf("================================================================================\n");
    printf("Usage:\n");
    printf("  %s <u|s> <m|gs> <vector|matrix> [blockSize] [options]\n\n", progName);
    printf("Positional Arguments (Single Configuration):\n");
    printf("  1. Memory Model (u|s):\n");
    printf("       u,  unified       : Unified Memory (cudaMallocManaged)\n");
    printf("       s,  standard      : Standard Memory (cudaMalloc + host malloc + cudaMemcpy DtoH)\n");
    printf("  2. Kernel Execution Model (m|gs):\n");
    printf("       m,  monolithic    : Monolithic 1-to-1 thread mapping\n");
    printf("       gs, grid_stride   : Grid-Stride Loops (stride = gridDim * blockDim, SM-scaled grid)\n");
    printf("  3. Target Operation (vector|matrix):\n");
    printf("       vector, vec, v    : Double-precision Vector Addition (n = 2^24 elements, 128 MB/vector)\n");
    printf("       matrix, mat, m    : Double-precision Matrix Addition (2048 x 4096 elements, 64 MB/matrix)\n");
    printf("  4. [blockSize] (optional):\n");
    printf("       32..1024          : Thread block size (default: 256)\n\n");
    printf("Options:\n");
    printf("  --device <id>          : Select CUDA device ID (default: 0)\n");
    printf("  --csv                  : Output result in CSV format for automated benchmarking\n");
    printf("  -h, --help             : Show this help message\n\n");
    printf("Examples:\n");
    printf("  %s u m vector 256       # Unified memory, monolithic kernel, vector addition, block 256\n", progName);
    printf("  %s s gs matrix 512      # Standard memory, grid-stride kernel, matrix addition, block 512\n\n", progName);
    printf("Note:\n");
    printf("  This executable executes only a single configuration. Multi-run benchmark\n");
    printf("  sweeps and statistical analyses are orchestrated by 'average_benchmarks.py'.\n");
    printf("================================================================================\n");
}

static inline void print_device_header(int devId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));

    printf("================================================================================\n");
    printf(" Lab 1: First CUDA Applications\n");
    printf(" Target Device [%d]: %s (Compute %d.%d, %d SMs, %.2f GB Global Mem)\n",
            devId, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("================================================================================\n");
}

static inline void print_run_report(const RunConfig &cfg, const RunResult &res) {
    const char *memTitle = (cfg.memModel == MEM_UNIFIED) ? "Unified Memory" : "Standard Memory";
    const char *modelTitle = (cfg.kernelModel == MODEL_MONOLITHIC) ? "Monolithic" : "Grid-Stride";
    const char *opTitle = (cfg.opType == OP_VECTOR) ? "Vector Addition" : "Matrix Addition";

    printf("\n================================================================================\n");
    printf(" Execution Report: %s / %s (%s)\n", memTitle, modelTitle, opTitle);
    printf("================================================================================\n");

    if (cfg.opType == OP_VECTOR) {
        size_t bytes = cfg.vecN * sizeof(double);
        printf(" Vector Size (n)        : %zu elements (%.2f MB per vector)\n",
               cfg.vecN, (double)bytes / (1024.0 * 1024.0));
        printf(" Block Dimensions       : %d threads\n", res.blockSize);
        printf(" Grid Dimensions        : %d blocks\n", res.gridDim.x);
    } else {
        size_t total = cfg.matRows * cfg.matCols;
        size_t bytes = total * sizeof(double);
        printf(" Matrix Dimensions      : %zu x %zu (%zu elements, %.2f MB per matrix)\n",
               cfg.matRows, cfg.matCols, total, (double)bytes / (1024.0 * 1024.0));
        printf(" Block Dimensions       : (%d, %d) = %d threads\n",
                res.blockDim.x, res.blockDim.y, res.blockSize);
        printf(" Grid Dimensions        : (%d, %d) blocks\n", res.gridDim.x, res.gridDim.y);
    }

    printf(" Time 3x Init Kernels   : %8.3f ms\n", res.initTimeMs);
    printf(" Time 1x Add Kernel     : %8.3f ms\n", res.addTimeMs);
    if (cfg.memModel == MEM_STANDARD) {
        printf(" Time cudaMemcpy (DtoH) : %8.3f ms\n", res.memcpyTimeMs);
    } else {
        printf(" Time cudaMemcpy (DtoH) : %8.3f ms (N/A for Unified Memory)\n", 0.0f);
    }
    printf(" Device Working Time    : %8.3f ms (CUDA events: kernels + transfers)\n", res.deviceTimeMs);
    printf(" Total Run Time         : %8.3f ms (wall-clock: execution + host verification)\n", res.totalRunTimeMs);
    printf(" Correctness Check      : %s\n", res.verified ? "PASSED (all results match expected)" : "FAILED");
    printf("================================================================================\n");
}

static inline void print_csv_result(const RunConfig &cfg, const RunResult &res) {
    const char *memCode = (cfg.memModel == MEM_UNIFIED) ? "u" : "s";
    const char *modelCode = (cfg.kernelModel == MODEL_MONOLITHIC) ? "m" : "gs";
    const char *opCode = (cfg.opType == OP_VECTOR) ? "vector" : "matrix";

    char gridStr[32];
    if (cfg.opType == OP_VECTOR) {
        snprintf(gridStr, sizeof(gridStr), "%d", res.gridDim.x);
    } else {
        snprintf(gridStr, sizeof(gridStr), "%dx%d", res.gridDim.x, res.gridDim.y);
    }

    // CSV format: CSV,mem,kernel,op,blockSize,gridSize,initMs,addMs,memcpyMs,deviceTimeMs,totalRunTimeMs,status
    printf("CSV,%s,%s,%s,%d,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%s\n",
            memCode, modelCode, opCode, res.blockSize, gridStr,
            res.initTimeMs, res.addTimeMs, res.memcpyTimeMs,
            res.deviceTimeMs, res.totalRunTimeMs,
            res.verified ? "PASS" : "FAIL");
}

#endif // LAB1_COMMON_H
