/**
 * Lab 1: First CUDA Applications - Single Configuration Driver
 * Course: Massively Parallel Programming on GPUs
 * 
 * Executes a single CUDA configuration directly:
 *   Memory Model:  Unified Memory (u) vs Standard Memory (s)
 *   Kernel Model:  Monolithic (m) vs Grid-Stride Loops (gs)
 *   Operation:     Vector Addition (vector) vs Matrix Addition (matrix)
 * 
 * Usage:
 *   ./lab1 <u|s> <m|gs> <vector|matrix> [blockSize] [options]
 */

#include "common.h"

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
// Grid & Block Dimension Computation
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
// Execution Functions: Vector Addition
// ---------------------------------------------------------------------------
static RunResult execute_vector(const RunConfig &cfg, int numSMs) {
    RunResult res;
    memset(&res, 0, sizeof(res));
    res.blockSize = cfg.blockSize;
    res.blockDim = dim3(cfg.blockSize);
    int numBlocks = calculate_vector_grid_size(cfg.vecN, cfg.blockSize, numSMs, cfg.kernelModel);
    res.gridDim = dim3(numBlocks);

    size_t bytes = cfg.vecN * sizeof(double);
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    double *h_C = NULL;

    if (cfg.memModel == MEM_STANDARD) {
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

    cudaEvent_t sInit, eInit, sAdd, eAdd, sCpy, eCpy, sDev, eDev;
    CUDA_CHECK(cudaEventCreate(&sInit)); CUDA_CHECK(cudaEventCreate(&eInit));
    CUDA_CHECK(cudaEventCreate(&sAdd));  CUDA_CHECK(cudaEventCreate(&eAdd));
    CUDA_CHECK(cudaEventCreate(&sDev));  CUDA_CHECK(cudaEventCreate(&eDev));
    if (cfg.memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventCreate(&sCpy)); CUDA_CHECK(cudaEventCreate(&eCpy));
    }

    double t0 = get_wall_time_ms();

    CUDA_CHECK(cudaEventRecord(sDev, 0));

    // 1. Initialise GPU-side (3 kernel launches)
    CUDA_CHECK(cudaEventRecord(sInit, 0));
    if (cfg.kernelModel == MODEL_MONOLITHIC) {
        vectorInit<<<res.gridDim, res.blockDim>>>(d_A, 1.0, cfg.vecN);
        vectorInit<<<res.gridDim, res.blockDim>>>(d_B, 2.0, cfg.vecN);
        vectorInit<<<res.gridDim, res.blockDim>>>(d_C, 0.0, cfg.vecN);
    } else {
        vectorInitGSL<<<res.gridDim, res.blockDim>>>(d_A, 1.0, cfg.vecN);
        vectorInitGSL<<<res.gridDim, res.blockDim>>>(d_B, 2.0, cfg.vecN);
        vectorInitGSL<<<res.gridDim, res.blockDim>>>(d_C, 0.0, cfg.vecN);
    }
    CUDA_CHECK(cudaEventRecord(eInit, 0));
    CUDA_CHECK(cudaEventSynchronize(eInit));

    // 2. Vector addition (1 kernel launch)
    CUDA_CHECK(cudaEventRecord(sAdd, 0));
    if (cfg.kernelModel == MODEL_MONOLITHIC) {
        vectorAdd<<<res.gridDim, res.blockDim>>>(d_A, d_B, d_C, cfg.vecN);
    } else {
        vectorAddGSL<<<res.gridDim, res.blockDim>>>(d_A, d_B, d_C, cfg.vecN);
    }
    CUDA_CHECK(cudaEventRecord(eAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(eAdd));

    // 3. Explicit copy or device synchronization
    if (cfg.memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventRecord(sCpy, 0));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(eCpy, 0));
        CUDA_CHECK(cudaEventSynchronize(eCpy));
        CUDA_CHECK(cudaEventElapsedTime(&res.memcpyTimeMs, sCpy, eCpy));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        res.memcpyTimeMs = 0.0f;
    }

    CUDA_CHECK(cudaEventRecord(eDev, 0));
    CUDA_CHECK(cudaEventSynchronize(eDev));

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, sInit, eInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, sAdd, eAdd));
    CUDA_CHECK(cudaEventElapsedTime(&res.deviceTimeMs, sDev, eDev));

    // 4. Host verification (triggers page migration for Unified Memory)
    double *verifyBuf = (cfg.memModel == MEM_STANDARD) ? h_C : d_C;
    res.verified = verify_vector(verifyBuf, cfg.vecN, 3.0);

    double t1 = get_wall_time_ms();
    res.totalRunTimeMs = t1 - t0;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(sInit)); CUDA_CHECK(cudaEventDestroy(eInit));
    CUDA_CHECK(cudaEventDestroy(sAdd));  CUDA_CHECK(cudaEventDestroy(eAdd));
    CUDA_CHECK(cudaEventDestroy(sDev));  CUDA_CHECK(cudaEventDestroy(eDev));
    if (cfg.memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventDestroy(sCpy)); CUDA_CHECK(cudaEventDestroy(eCpy));
        free(h_C);
    }
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    return res;
}

// ---------------------------------------------------------------------------
// Execution Functions: Matrix Addition
// ---------------------------------------------------------------------------
static RunResult execute_matrix(const RunConfig &cfg, int numSMs) {
    RunResult res;
    memset(&res, 0, sizeof(res));
    res.blockSize = cfg.blockSize;
    calculate_matrix_dims(cfg.matRows, cfg.matCols, cfg.blockSize, numSMs,
                          cfg.kernelModel, res.blockDim, res.gridDim);

    size_t total = cfg.matRows * cfg.matCols;
    size_t bytes = total * sizeof(double);
    double *d_A = NULL, *d_B = NULL, *d_C = NULL;
    double *h_C = NULL;

    if (cfg.memModel == MEM_STANDARD) {
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

    cudaEvent_t sInit, eInit, sAdd, eAdd, sCpy, eCpy, sDev, eDev;
    CUDA_CHECK(cudaEventCreate(&sInit)); CUDA_CHECK(cudaEventCreate(&eInit));
    CUDA_CHECK(cudaEventCreate(&sAdd));  CUDA_CHECK(cudaEventCreate(&eAdd));
    CUDA_CHECK(cudaEventCreate(&sDev));  CUDA_CHECK(cudaEventCreate(&eDev));
    if (cfg.memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventCreate(&sCpy)); CUDA_CHECK(cudaEventCreate(&eCpy));
    }

    double t0 = get_wall_time_ms();

    CUDA_CHECK(cudaEventRecord(sDev, 0));

    // 1. Initialise GPU-side (3 kernel launches)
    CUDA_CHECK(cudaEventRecord(sInit, 0));
    if (cfg.kernelModel == MODEL_MONOLITHIC) {
        matrixInit<<<res.gridDim, res.blockDim>>>(d_A, 1.0, cfg.matRows, cfg.matCols);
        matrixInit<<<res.gridDim, res.blockDim>>>(d_B, 2.0, cfg.matRows, cfg.matCols);
        matrixInit<<<res.gridDim, res.blockDim>>>(d_C, 0.0, cfg.matRows, cfg.matCols);
    } else {
        matrixInitGSL<<<res.gridDim, res.blockDim>>>(d_A, 1.0, cfg.matRows, cfg.matCols);
        matrixInitGSL<<<res.gridDim, res.blockDim>>>(d_B, 2.0, cfg.matRows, cfg.matCols);
        matrixInitGSL<<<res.gridDim, res.blockDim>>>(d_C, 0.0, cfg.matRows, cfg.matCols);
    }
    CUDA_CHECK(cudaEventRecord(eInit, 0));
    CUDA_CHECK(cudaEventSynchronize(eInit));

    // 2. Matrix addition (1 kernel launch)
    CUDA_CHECK(cudaEventRecord(sAdd, 0));
    if (cfg.kernelModel == MODEL_MONOLITHIC) {
        matrixAdd<<<res.gridDim, res.blockDim>>>(d_A, d_B, d_C, cfg.matRows, cfg.matCols);
    } else {
        matrixAddGSL<<<res.gridDim, res.blockDim>>>(d_A, d_B, d_C, cfg.matRows, cfg.matCols);
    }
    CUDA_CHECK(cudaEventRecord(eAdd, 0));
    CUDA_CHECK(cudaEventSynchronize(eAdd));

    // 3. Explicit copy or device synchronization
    if (cfg.memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventRecord(sCpy, 0));
        CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(eCpy, 0));
        CUDA_CHECK(cudaEventSynchronize(eCpy));
        CUDA_CHECK(cudaEventElapsedTime(&res.memcpyTimeMs, sCpy, eCpy));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
        res.memcpyTimeMs = 0.0f;
    }

    CUDA_CHECK(cudaEventRecord(eDev, 0));
    CUDA_CHECK(cudaEventSynchronize(eDev));

    CUDA_CHECK(cudaEventElapsedTime(&res.initTimeMs, sInit, eInit));
    CUDA_CHECK(cudaEventElapsedTime(&res.addTimeMs, sAdd, eAdd));
    CUDA_CHECK(cudaEventElapsedTime(&res.deviceTimeMs, sDev, eDev));

    // 4. Host verification (triggers page migration for Unified Memory)
    double *verifyBuf = (cfg.memModel == MEM_STANDARD) ? h_C : d_C;
    res.verified = verify_matrix(verifyBuf, cfg.matRows, cfg.matCols, 3.0);

    double t1 = get_wall_time_ms();
    res.totalRunTimeMs = t1 - t0;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(sInit)); CUDA_CHECK(cudaEventDestroy(eInit));
    CUDA_CHECK(cudaEventDestroy(sAdd));  CUDA_CHECK(cudaEventDestroy(eAdd));
    CUDA_CHECK(cudaEventDestroy(sDev));  CUDA_CHECK(cudaEventDestroy(eDev));
    if (cfg.memModel == MEM_STANDARD) {
        CUDA_CHECK(cudaEventDestroy(sCpy)); CUDA_CHECK(cudaEventDestroy(eCpy));
        free(h_C);
    }
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));

    return res;
}

// ---------------------------------------------------------------------------
// CLI Argument Parsing
// ---------------------------------------------------------------------------
static int parse_arguments(int argc, char **argv, RunConfig &cfg) {
    if (argc < 4) {
        return -1;
    }

    int pos = 0;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            return -1;
        } else if (strcmp(arg, "--csv") == 0) {
            cfg.csvOutput = 1;
        } else if (strcmp(arg, "--device") == 0 && i + 1 < argc) {
            cfg.devId = atoi(argv[++i]);
        } else {
            if (pos == 0) {
                if (strcmp(arg, "u") == 0 || strcasecmp(arg, "unified") == 0) {
                    cfg.memModel = MEM_UNIFIED;
                } else if (strcmp(arg, "s") == 0 || strcasecmp(arg, "standard") == 0) {
                    cfg.memModel = MEM_STANDARD;
                } else {
                    fprintf(stderr, "Error: Unknown memory model '%s'. Must be 'u' (unified) or 's' (standard).\n", arg);
                    return -1;
                }
                pos++;
            } else if (pos == 1) {
                if (strcmp(arg, "m") == 0 || strcasecmp(arg, "monolithic") == 0) {
                    cfg.kernelModel = MODEL_MONOLITHIC;
                } else if (strcmp(arg, "gs") == 0 || strcasecmp(arg, "grid_stride") == 0 || strcasecmp(arg, "grid-stride") == 0) {
                    cfg.kernelModel = MODEL_GRID_STRIDE;
                } else {
                    fprintf(stderr, "Error: Unknown kernel model '%s'. Must be 'm' (monolithic) or 'gs' (grid_stride).\n", arg);
                    return -1;
                }
                pos++;
            } else if (pos == 2) {
                if (strcmp(arg, "vector") == 0 || strcmp(arg, "vec") == 0 || strcmp(arg, "v") == 0) {
                    cfg.opType = OP_VECTOR;
                } else if (strcmp(arg, "matrix") == 0 || strcmp(arg, "mat") == 0 || strcmp(arg, "m") == 0) {
                    cfg.opType = OP_MATRIX;
                } else {
                    fprintf(stderr, "Error: Unknown operation '%s'. Must be 'vector' or 'matrix'.\n", arg);
                    return -1;
                }
                pos++;
            } else if (pos == 3) {
                int sz = atoi(arg);
                if (sz > 0) {
                    cfg.blockSize = sz;
                } else {
                    fprintf(stderr, "Error: Invalid block size '%s'. Must be a positive integer.\n", arg);
                    return -1;
                }
                pos++;
            }
        }
    }

    if (pos < 3) {
        fprintf(stderr, "Error: Missing required positional arguments.\n");
        return -1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Main Driver (Single Configuration Execution)
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    RunConfig cfg;
    cfg.memModel = MEM_UNIFIED;
    cfg.kernelModel = MODEL_MONOLITHIC;
    cfg.opType = OP_VECTOR;
    cfg.blockSize = DEFAULT_BLOCK_SIZE;
    cfg.devId = 0;
    cfg.csvOutput = 0;
    cfg.vecN = DEFAULT_VEC_N;
    cfg.matRows = DEFAULT_MAT_ROWS;
    cfg.matCols = DEFAULT_MAT_COLS;

    if (parse_arguments(argc, argv, cfg) != 0) {
        print_usage(argv[0]);
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(cfg.devId));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.devId));

    if (!cfg.csvOutput) {
        print_device_header(cfg.devId);
    }

    RunResult res;
    if (cfg.opType == OP_VECTOR) {
        res = execute_vector(cfg, prop.multiProcessorCount);
    } else {
        res = execute_matrix(cfg, prop.multiProcessorCount);
    }

    if (cfg.csvOutput) {
        print_csv_result(cfg, res);
    } else {
        print_run_report(cfg, res);
    }

    return res.verified ? 0 : 1;
}
