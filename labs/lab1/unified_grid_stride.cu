/**
 * ============================================================================
 * Lab 1 - Activity 2 & 4: First CUDA Applications
 * Course: Massively Parallel Programming on GPUs
 * 
 * Implementation 2: Unified-Memory / Grid-Stride-Loop-Kernels
 * 
 * Features:
 *  - Memory: Unified Memory (cudaMallocManaged)
 *  - Kernel Style: Grid-Stride Loops (threads loop with stride = gridDim * blockDim)
 *  - Decoupled Grid: Grid size is decoupled from problem size, scaled to SM count
 *  - Kernels: vectorInitGSL (3 calls), vectorAddGSL (1 call)
 *             matrixInitGSL (3 calls), matrixAddGSL (1 call)
 *  - Comparisons:
 *      * Serial CPU baseline vs CUDA GPU execution
 *      * Block sizes: 32, 64, 128, 256, 512, 1024
 *      * Verification of correctness on CPU (C[i] == 3.0)
 *      * Compatibility with nvprof (GTX 980 / CUDA 11.4) and ncu (RTX 4080)
 * 
 * Usage:
 *   ./unified_grid_stride                        # Runs full comparison sweep (all block sizes)
 *   ./unified_grid_stride vector <blockSize>     # Single run for vector addition (ideal for nvprof)
 *   ./unified_grid_stride matrix <blockSize>     # Single run for matrix addition (ideal for nvprof)
 *   ./unified_grid_stride <blockSize>            # Single run for vector addition with blockSize
 *   Options:
 *     --device <dev_id>                          # Select GPU device (e.g., 0 or 1 on dual GTX 980)
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// Error Checking Macros (Reference: Proper CUDA Error Checking)
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

#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                      \
        cudaError_t err = cudaGetLastError();                                 \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel launch error at %s:%d: %s\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
        err = cudaDeviceSynchronize();                                        \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA kernel execution error at %s:%d: %s\n",     \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Constants and Defaults
// ---------------------------------------------------------------------------
static const size_t DEFAULT_VEC_N = 1ULL << 24;      // n = 2^24 = 16,777,216
static const size_t DEFAULT_MAT_ROWS = 2048;        // (n_matrix/2, n_matrix)
static const size_t DEFAULT_MAT_COLS = 4096;

static const int BLOCK_SIZES[] = {32, 64, 128, 256, 512, 1024};
static const int NUM_BLOCK_SIZES = 6;

// ---------------------------------------------------------------------------
// CPU High-Resolution Timer Helper
// ---------------------------------------------------------------------------
static double get_time_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

// ---------------------------------------------------------------------------
// Serial CPU Implementations (for baseline comparison)
// ---------------------------------------------------------------------------
void cpu_vector_init(double *data, double val, size_t n) {
    for (size_t i = 0; i < n; i++) {
        data[i] = val;
    }
}

void cpu_vector_add(const double *A, const double *B, double *C, size_t n) {
    for (size_t i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

void cpu_matrix_init(double *M, double val, size_t nrows, size_t ncols) {
    size_t total = nrows * ncols;
    for (size_t i = 0; i < total; i++) {
        M[i] = val;
    }
}

void cpu_matrix_add(const double *A, const double *B, double *C, size_t nrows, size_t ncols) {
    size_t total = nrows * ncols;
    for (size_t i = 0; i < total; i++) {
        C[i] = A[i] + B[i];
    }
}

// ---------------------------------------------------------------------------
// CUDA Grid-Stride Loop Kernels
// Reference: Mark Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops"
// Benefits:
//  1. Flexibility & scalability: decoupled grid size from problem size.
//  2. High SM occupancy and cache reuse.
//  3. Debuggable: runs correctly even with a grid of 1 block.
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
// Correctness Verification Helpers
// ---------------------------------------------------------------------------
int verify_vector_result(const double *C, size_t n, double expected) {
    for (size_t i = 0; i < n; i++) {
        if (fabs(C[i] - expected) > 1e-9) {
            fprintf(stderr, "Verification FAILED at index %zu: got %f, expected %f\n",
                    i, C[i], expected);
            return 0;
        }
    }
    return 1;
}

int verify_matrix_result(const double *C, size_t nrows, size_t ncols, double expected) {
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
// Benchmark Result Structures
// ---------------------------------------------------------------------------
typedef struct {
    int blockSize;
    int numBlocks;
    float initTimeMs;
    float addTimeMs;
    float totalGpuTimeMs;
    double cpuTimeMs;
    double speedup;
    int verified;
} BenchResult1D;

typedef struct {
    int blockSize;
    dim3 blockDim;
    dim3 gridDim;
    float initTimeMs;
    float addTimeMs;
    float totalGpuTimeMs;
    double cpuTimeMs;
    double speedup;
    int verified;
} BenchResult2D;

// ---------------------------------------------------------------------------
// Vector Benchmark Runner (Unified Memory + Grid-Stride Loop)
// ---------------------------------------------------------------------------
BenchResult1D run_vector_benchmark(size_t n, int blockSize, int numSMs, double cpuBaselineMs, int isSingleRun) {
    BenchResult1D res;
    res.blockSize = blockSize;
    res.cpuTimeMs = cpuBaselineMs;

    // Grid-stride loop grid sizing: common best-practice is 32 * numSMs blocks,
    // bounded by problem ceiling
    int maxBlocks = numSMs * 32;
    int neededBlocks = (int)((n + blockSize - 1) / blockSize);
    res.numBlocks = (neededBlocks < maxBlocks) ? neededBlocks : maxBlocks;

    size_t bytes = n * sizeof(double);
    double *A = NULL, *B = NULL, *C = NULL;

    CUDA_CHECK(cudaMallocManaged((void**)&A, bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&B, bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&C, bytes));

    cudaEvent_t startInit, stopInit, startAdd, stopAdd;
    CUDA_CHECK(cudaEventCreate(&startInit));
    CUDA_CHECK(cudaEventCreate(&stopInit));
    CUDA_CHECK(cudaEventCreate(&startAdd));
    CUDA_CHECK(cudaEventCreate(&stopAdd));

    dim3 block(blockSize);
    dim3 grid(res.numBlocks);

    // 1. Initialise GPU-side by 3 calls to vectorInitGSL()
    CUDA_CHECK(cudaEventRecord(startInit, 0));
    vectorInitGSL<<<grid, block>>>(A, 1.0, n);
    vectorInitGSL<<<grid, block>>>(B, 2.0, n);
    vectorInitGSL<<<grid, block>>>(C, 0.0, n);
    CUDA_CHECK(cudaEventRecord(stopInit, 0));
    CUDA_CHECK(cudaEventSynchronize(stopInit));

    // 2. Compute vector addition on GPU by 1 call to vectorAddGSL()
    CUDA_CHECK(cudaEventRecord(startAdd, 0));
    vectorAddGSL<<<grid, block>>>(A, B, C, n);
    CUDA_CHECK(cudaEventRecord(stopAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(stopAdd));

    // Sync before CPU touches managed memory
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, startInit, stopInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, startAdd, stopAdd));
    res.totalGpuTimeMs = res.initTimeMs + res.addTimeMs;
    res.speedup = (cpuBaselineMs > 0.0) ? (cpuBaselineMs / res.totalGpuTimeMs) : 0.0;

    // 3. Verify correctness
    res.verified = verify_vector_result(C, n, 3.0);

    if (isSingleRun) {
        printf("\n=======================================================\n");
        printf("  Single Run (Vector Addition - Unified / Grid-Stride)\n");
        printf("=======================================================\n");
        printf(" Vector size (n)         : %zu elements (%.2f MB per vector)\n", n, (double)bytes / (1024.0 * 1024.0));
        printf(" Block size              : %d threads\n", blockSize);
        printf(" Grid size               : %d blocks (decoupled, scaled to %d SMs)\n", res.numBlocks, numSMs);
        printf(" Stride                  : %zu threads\n", (size_t)res.numBlocks * blockSize);
        printf(" Time 3x vectorInitGSL   : %8.3f ms\n", res.initTimeMs);
        printf(" Time 1x vectorAddGSL    : %8.3f ms\n", res.addTimeMs);
        printf(" Total GPU Elapsed Time  : %8.3f ms\n", res.totalGpuTimeMs);
        if (cpuBaselineMs > 0.0) {
            printf(" CPU Serial Time (calc)  : %8.3f ms\n", cpuBaselineMs);
            printf(" Speedup vs CPU Serial   : %8.2fx\n", res.speedup);
        }
        printf(" Correctness Check       : %s\n", res.verified ? "PASSED (all C[i] == 3.0)" : "FAILED");
        printf("=======================================================\n");
    }

    CUDA_CHECK(cudaEventDestroy(startInit));
    CUDA_CHECK(cudaEventDestroy(stopInit));
    CUDA_CHECK(cudaEventDestroy(startAdd));
    CUDA_CHECK(cudaEventDestroy(stopAdd));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return res;
}

// ---------------------------------------------------------------------------
// Matrix Benchmark Runner (Unified Memory + Grid-Stride Loop)
// ---------------------------------------------------------------------------
BenchResult2D run_matrix_benchmark(size_t nrows, size_t ncols, int blockSize, int numSMs, double cpuBaselineMs, int isSingleRun) {
    BenchResult2D res;
    res.blockSize = blockSize;
    res.cpuTimeMs = cpuBaselineMs;

    int bx = 32;
    int by = blockSize / 32;
    if (by == 0) by = 1;
    res.blockDim = dim3(bx, by);

    // Grid-stride loop 2D grid sizing
    int maxGridX = 32;
    int maxGridY = (numSMs * 32 + maxGridX - 1) / maxGridX;
    int gx = (int)((ncols + bx - 1) / bx);
    int gy = (int)((nrows + by - 1) / by);
    if (gx > maxGridX) gx = maxGridX;
    if (gy > maxGridY) gy = maxGridY;
    res.gridDim = dim3(gx, gy);

    size_t totalElements = nrows * ncols;
    size_t bytes = totalElements * sizeof(double);
    double *A = NULL, *B = NULL, *C = NULL;

    CUDA_CHECK(cudaMallocManaged((void**)&A, bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&B, bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&C, bytes));

    cudaEvent_t startInit, stopInit, startAdd, stopAdd;
    CUDA_CHECK(cudaEventCreate(&startInit));
    CUDA_CHECK(cudaEventCreate(&stopInit));
    CUDA_CHECK(cudaEventCreate(&startAdd));
    CUDA_CHECK(cudaEventCreate(&stopAdd));

    // 1. Initialise GPU-side by 3 calls to matrixInitGSL()
    CUDA_CHECK(cudaEventRecord(startInit, 0));
    matrixInitGSL<<<res.gridDim, res.blockDim>>>(A, 1.0, nrows, ncols);
    matrixInitGSL<<<res.gridDim, res.blockDim>>>(B, 2.0, nrows, ncols);
    matrixInitGSL<<<res.gridDim, res.blockDim>>>(C, 0.0, nrows, ncols);
    CUDA_CHECK(cudaEventRecord(stopInit, 0));
    CUDA_CHECK(cudaEventSynchronize(stopInit));

    // 2. Compute matrix addition on GPU by 1 call to matrixAddGSL()
    CUDA_CHECK(cudaEventRecord(startAdd, 0));
    matrixAddGSL<<<res.gridDim, res.blockDim>>>(A, B, C, nrows, ncols);
    CUDA_CHECK(cudaEventRecord(stopAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(stopAdd));

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, startInit, stopInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, startAdd, stopAdd));
    res.totalGpuTimeMs = res.initTimeMs + res.addTimeMs;
    res.speedup = (cpuBaselineMs > 0.0) ? (cpuBaselineMs / res.totalGpuTimeMs) : 0.0;

    // 3. Verify correctness
    res.verified = verify_matrix_result(C, nrows, ncols, 3.0);

    if (isSingleRun) {
        printf("\n=======================================================\n");
        printf("  Single Run (Matrix Addition - Unified / Grid-Stride)\n");
        printf("=======================================================\n");
        printf(" Matrix dimensions       : %zu x %zu (%zu elements, %.2f MB)\n",
               nrows, ncols, totalElements, (double)bytes / (1024.0 * 1024.0));
        printf(" Block dimensions        : (%d, %d) = %d threads\n", bx, by, blockSize);
        printf(" Grid dimensions         : (%d, %d) blocks (decoupled)\n", gx, gy);
        printf(" Time 3x matrixInitGSL   : %8.3f ms\n", res.initTimeMs);
        printf(" Time 1x matrixAddGSL    : %8.3f ms\n", res.addTimeMs);
        printf(" Total GPU Elapsed Time  : %8.3f ms\n", res.totalGpuTimeMs);
        if (cpuBaselineMs > 0.0) {
            printf(" CPU Serial Time (calc)  : %8.3f ms\n", cpuBaselineMs);
            printf(" Speedup vs CPU Serial   : %8.2fx\n", res.speedup);
        }
        printf(" Correctness Check       : %s\n", res.verified ? "PASSED (all C[i][j] == 3.0)" : "FAILED");
        printf("=======================================================\n");
    }

    CUDA_CHECK(cudaEventDestroy(startInit));
    CUDA_CHECK(cudaEventDestroy(stopInit));
    CUDA_CHECK(cudaEventDestroy(startAdd));
    CUDA_CHECK(cudaEventDestroy(stopAdd));

    CUDA_CHECK(cudaFree(A));
    CUDA_CHECK(cudaFree(B));
    CUDA_CHECK(cudaFree(C));

    return res;
}

// ---------------------------------------------------------------------------
// Main & CLI Handling
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int devId = 0;
    int singleBlockSize = -1;
    const char *mode = "all";
    size_t vecN = DEFAULT_VEC_N;
    size_t matRows = DEFAULT_MAT_ROWS;
    size_t matCols = DEFAULT_MAT_COLS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            devId = atoi(argv[++i]);
        } else if (strcmp(argv[i], "vector") == 0) {
            mode = "vector";
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                singleBlockSize = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "matrix") == 0) {
            mode = "matrix";
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                singleBlockSize = atoi(argv[++i]);
            }
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                matRows = (size_t)atoll(argv[++i]);
            }
            if (i + 1 < argc && argv[i + 1][0] >= '0' && argv[i + 1][0] <= '9') {
                matCols = (size_t)atoll(argv[++i]);
            }
        } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
            singleBlockSize = atoi(argv[i]);
            mode = "vector";
        }
    }

    CUDA_CHECK(cudaSetDevice(devId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));

    printf("================================================================================\n");
    printf(" Lab 1: Implementation 2 - Unified-Memory / Grid-Stride-Loop-Kernels\n");
    printf(" Target Device [%d]: %s (Compute %d.%d, %d SMs, %.2f GB Global Mem)\n",
           devId, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("================================================================================\n");

    // Single run mode for profiling
    if (singleBlockSize > 0) {
        if (strcmp(mode, "matrix") == 0) {
            run_matrix_benchmark(matRows, matCols, singleBlockSize, prop.multiProcessorCount, 0.0, 1);
        } else {
            run_vector_benchmark(vecN, singleBlockSize, prop.multiProcessorCount, 0.0, 1);
        }
        return 0;
    }

    // Full comparison sweep mode
    printf("\n>>> Measuring Serial CPU Baseline for Vector Addition (n = 2^24 = %zu)...\n", vecN);
    size_t vecBytes = vecN * sizeof(double);
    double *h_A = (double*)malloc(vecBytes);
    double *h_B = (double*)malloc(vecBytes);
    double *h_C = (double*)malloc(vecBytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed for CPU baseline\n");
        exit(EXIT_FAILURE);
    }

    double t0 = get_time_sec();
    cpu_vector_init(h_A, 1.0, vecN);
    cpu_vector_init(h_B, 2.0, vecN);
    cpu_vector_init(h_C, 0.0, vecN);
    cpu_vector_add(h_A, h_B, h_C, vecN);
    double t1 = get_time_sec();
    double cpuVecMs = (t1 - t0) * 1000.0;
    printf("CPU Serial Elapsed Time (vectorInit + vectorAdd): %.3f ms\n", cpuVecMs);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n--------------------------------------------------------------------------------\n");
    printf(" Vector Addition Comparison: Unified-Memory / Grid-Stride-Loop-Kernels\n");
    printf(" (Double Precision, n = 2^24, 384 MB total for A, B, C)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("%-10s %-12s %-14s %-14s %-14s %-14s %-10s %-8s\n",
           "blockSize", "gridSize", "Init 3x (ms)", "Add 1x (ms)", "Total GPU(ms)",
           "CPU Ser(ms)", "Speedup", "Status");
    printf("--------------------------------------------------------------------------------\n");

    for (int b = 0; b < NUM_BLOCK_SIZES; b++) {
        int bs = BLOCK_SIZES[b];
        BenchResult1D res = run_vector_benchmark(vecN, bs, prop.multiProcessorCount, cpuVecMs, 0);
        char speedupStr[16];
        snprintf(speedupStr, sizeof(speedupStr), "%.2fx", res.speedup);
        printf("%-10d %-12d %-14.3f %-14.3f %-14.3f %-14.3f %-10s %-8s\n",
               res.blockSize, res.numBlocks, res.initTimeMs, res.addTimeMs,
               res.totalGpuTimeMs, res.cpuTimeMs, speedupStr,
               res.verified ? "PASS" : "FAIL");
    }
    printf("--------------------------------------------------------------------------------\n");

    // Matrix Addition Benchmark Sweep
    printf("\n>>> Measuring Serial CPU Baseline for Matrix Addition (%zu x %zu = %zu elements)...\n",
           matRows, matCols, matRows * matCols);
    size_t matBytes = matRows * matCols * sizeof(double);
    h_A = (double*)malloc(matBytes);
    h_B = (double*)malloc(matBytes);
    h_C = (double*)malloc(matBytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed for Matrix CPU baseline\n");
        exit(EXIT_FAILURE);
    }

    t0 = get_time_sec();
    cpu_matrix_init(h_A, 1.0, matRows, matCols);
    cpu_matrix_init(h_B, 2.0, matRows, matCols);
    cpu_matrix_init(h_C, 0.0, matRows, matCols);
    cpu_matrix_add(h_A, h_B, h_C, matRows, matCols);
    t1 = get_time_sec();
    double cpuMatMs = (t1 - t0) * 1000.0;
    printf("CPU Serial Elapsed Time (matrixInit + matrixAdd): %.3f ms\n", cpuMatMs);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("\n--------------------------------------------------------------------------------\n");
    printf(" Matrix Addition Comparison: Unified-Memory / Grid-Stride-Loop-Kernels\n");
    printf(" (Double Precision, nrows=%zu, ncols=%zu, 192 MB total for A, B, C)\n", matRows, matCols);
    printf("--------------------------------------------------------------------------------\n");
    printf("%-10s %-14s %-14s %-14s %-14s %-14s %-10s %-8s\n",
           "blockSize", "blockDim", "Init 3x (ms)", "Add 1x (ms)", "Total GPU(ms)",
           "CPU Ser(ms)", "Speedup", "Status");
    printf("--------------------------------------------------------------------------------\n");

    for (int b = 0; b < NUM_BLOCK_SIZES; b++) {
        int bs = BLOCK_SIZES[b];
        BenchResult2D res = run_matrix_benchmark(matRows, matCols, bs, prop.multiProcessorCount, cpuMatMs, 0);
        char blockDimStr[32];
        snprintf(blockDimStr, sizeof(blockDimStr), "(%d,%d)", res.blockDim.x, res.blockDim.y);
        char speedupStr[16];
        snprintf(speedupStr, sizeof(speedupStr), "%.2fx", res.speedup);
        printf("%-10d %-14s %-14.3f %-14.3f %-14.3f %-14.3f %-10s %-8s\n",
               res.blockSize, blockDimStr, res.initTimeMs, res.addTimeMs,
               res.totalGpuTimeMs, res.cpuTimeMs, speedupStr,
               res.verified ? "PASS" : "FAIL");
    }
    printf("--------------------------------------------------------------------------------\n");
    printf("Note: For profiling with nvprof, run with a single block size, e.g.:\n");
    printf("      nvprof ./unified_grid_stride vector 256\n");
    printf("      nvprof ./unified_grid_stride matrix 256\n");
    printf("================================================================================\n");

    return 0;
}

