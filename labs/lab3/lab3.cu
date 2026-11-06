#include "common.h"

// ---------------------------------------------------------------------------
// 1. Matrix Initialization & CPU Reference Implementations (OpenMP)
// ---------------------------------------------------------------------------

/**
 * @brief Initializes matrices according to Lab 3 specification:
 * - A: all elements set to 0.5 * 2^-10 = 2^-11 = 0x1p-11f (0.00048828125f)
 * - B: all elements set to 2.0f
 * - C: all elements set to 0.0f
 * Expected dot product per element: K * (2^-11 * 2) = K * 2^-10.
 * For K = 1024, expected C[i, j] = 1.0f.
 */
void cpu_init(
    float *A,
    float *B,
    float *C_std,
    float *C_opt,
    float *C_gpu,
    int M,
    int K,
    int N,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            A[i * K + k] = 0x1p-11f;
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            B[k * N + j] = 2.0f;
        }
    }

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            if (C_std) C_std[i * N + j] = 0.0f;
            if (C_opt) C_opt[i * N + j] = 0.0f;
            if (C_gpu) C_gpu[i * N + j] = 0.0f;
        }
    }
}

/**
 * @brief Baseline CPU matrix multiplication using OpenMP (standard i-j-k loop order).
 */
void matrix_multiplication_cpu(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Cache-optimized CPU matrix multiplication using OpenMP (i-k-j loop interchange).
 * Exploits continuous memory layout in row-major matrix B and accumulates in C.
 */
void matrix_multiplication_cpu_optimized(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a * B[k * N + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. CUDA Kernel Implementation: Naive Matrix Multiplication (Point 4.1)
// ---------------------------------------------------------------------------

/**
 * @brief Point 4.1: Naive Matrix Multiplication (Global Memory only).
 * Assigns one CUDA thread to each output element C_ij.
 * Implements a 2D grid-stride loop covering arbitrary M, N, and K dimensions,
 * including dimensions not evenly divisible by block size.
 */
__global__ void naive_matrix_multiplication_cuda(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N
) {
    int strideRow = blockDim.y * gridDim.y;
    int strideCol = blockDim.x * gridDim.x;

    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < M; row += strideRow) {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < N; col += strideCol) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

/**
 * @brief Driver function to run the Naive CUDA kernel and record fine-grained timings.
 */
void run_gpu_global_memory(
    const float *A,
    const float *B,
    float *C,
    int M,
    int K,
    int N,
    int block_dim_size,
    GpuTiming *timing
) {
    size_t size_A = (size_t)M * K * sizeof(float);
    size_t size_B = (size_t)K * N * sizeof(float);
    size_t size_C = (size_t)M * N * sizeof(float);

    dim3 blockDim(block_dim_size, block_dim_size);
    dim3 gridDim(
        (N + block_dim_size - 1) / block_dim_size,
        (M + block_dim_size - 1) / block_dim_size
    );

    if (timing != nullptr) {
        timing->blockDim = blockDim;
        timing->gridDim = gridDim;
    }

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Measure memory allocation
    double t_alloc_start = get_wall_time_ms();
    float *dev_A = nullptr, *dev_B = nullptr, *dev_C = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&dev_A, size_A));
    CUDA_CHECK(cudaMallocManaged((void**)&dev_B, size_B));
    CUDA_CHECK(cudaMallocManaged((void**)&dev_C, size_C));
    float allocMs = static_cast<float>(get_wall_time_ms() - t_alloc_start);

    // Copy input matrices
    memcpy(dev_A, A, size_A);
    memcpy(dev_B, B, size_B);
    memset(dev_C, 0, size_C);

    // Warm-up cache / page tables if necessary, then time pure kernel execution
    CUDA_CHECK(cudaEventRecord(ev_start));
    naive_matrix_multiplication_cuda<<<gridDim, blockDim>>>(
        dev_A, dev_B, dev_C, M, K, N
    );
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    float kernelMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernelMs, ev_start, ev_stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(C, dev_C, size_C);

    if (timing != nullptr) {
        timing->allocMs   = allocMs;
        timing->htodMs    = 0.0f;
        timing->kernelMs  = kernelMs;
        timing->dtohMs    = 0.0f;
        timing->computeMs = kernelMs;
        timing->totalMs   = allocMs + kernelMs;
    }

    CUDA_CHECK(cudaFree(dev_A));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_C));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// 3. Main Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    // CUDA device initialization
    if (cfg.mode == MODE_GPU || cfg.mode == MODE_ALL) {
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount == 0) {
            fprintf(stderr, "Error: No CUDA capable devices detected.\n");
            return EXIT_FAILURE;
        }
        if (cfg.devId >= devCount) cfg.devId = 0;
        CUDA_CHECK(cudaSetDevice(cfg.devId));
        CUDA_CHECK(cudaFree(0)); // Initialize CUDA context
    }

    if (!cfg.csvOutput) {
        print_cuda_device_info(cfg.devId);
    }

    // Allocate host memory
    size_t size_A = (size_t)cfg.M * cfg.K * sizeof(float);
    size_t size_B = (size_t)cfg.K * cfg.N * sizeof(float);
    size_t size_C = (size_t)cfg.M * cfg.N * sizeof(float);

    float *A = (float*)malloc(size_A);
    float *B = (float*)malloc(size_B);
    float *C_std = (float*)malloc(size_C);
    float *C_opt = (float*)malloc(size_C);
    float *C_gpu = (float*)malloc(size_C);

    if (!A || !B || !C_std || !C_opt || !C_gpu) {
        fprintf(stderr, "Error: Failed to allocate host memory for matrices.\n");
        return EXIT_FAILURE;
    }

    // Initialize inputs
    cpu_init(A, B, C_std, C_opt, C_gpu, cfg.M, cfg.K, cfg.N, cfg.numThreads);

    // Optional Warm-Up run
    if (cfg.warmup) {
        if (cfg.mode == MODE_GPU || cfg.mode == MODE_ALL) {
            GpuTiming warmTiming;
            run_gpu_global_memory(A, B, C_gpu, cfg.M, cfg.K, cfg.N, cfg.blockSize, &warmTiming);
            memset(C_gpu, 0, size_C);
        }
    }

    // 1. CPU Standard Benchmark
    double cpuStdMs = 0.0;
    bool verifiedStd = true;
    if (cfg.mode == MODE_CPU_STD || cfg.mode == MODE_CPU_BOTH || cfg.mode == MODE_ALL) {
        double t0 = omp_get_wtime();
        matrix_multiplication_cpu(A, B, C_std, cfg.M, cfg.K, cfg.N, cfg.numThreads);
        cpuStdMs = (omp_get_wtime() - t0) * 1000.0;
        // Verify against analytical value: K * 2^-10
        float expectedVal = (float)cfg.K * 0x1p-10f;
        if (fabsf(C_std[0] - expectedVal) > cfg.tolerance) {
            verifiedStd = false;
        }
    }

    // 2. CPU Optimized Benchmark
    double cpuOptMs = 0.0;
    bool verifiedOpt = true;
    if (cfg.mode == MODE_CPU_OPT || cfg.mode == MODE_CPU_BOTH || cfg.mode == MODE_ALL) {
        double t0 = omp_get_wtime();
        matrix_multiplication_cpu_optimized(A, B, C_opt, cfg.M, cfg.K, cfg.N, cfg.numThreads);
        cpuOptMs = (omp_get_wtime() - t0) * 1000.0;

        if (cfg.mode == MODE_ALL || cfg.mode == MODE_CPU_BOTH) {
            verifiedOpt = check_matrix_equality(C_std, C_opt, cfg.M, cfg.N, cfg.tolerance);
        } else {
            float expectedVal = (float)cfg.K * 0x1p-10f;
            verifiedOpt = (fabsf(C_opt[0] - expectedVal) <= cfg.tolerance);
        }
    }

    // 3. GPU Naive Benchmark
    GpuTiming gpuTiming;
    memset(&gpuTiming, 0, sizeof(gpuTiming));
    bool verifiedGpu = true;
    if (cfg.mode == MODE_GPU || cfg.mode == MODE_ALL) {
        run_gpu_global_memory(A, B, C_gpu, cfg.M, cfg.K, cfg.N, cfg.blockSize, &gpuTiming);

        if (cfg.mode == MODE_ALL) {
            verifiedGpu = check_matrix_equality(C_std, C_gpu, cfg.M, cfg.N, cfg.tolerance);
        } else {
            float expectedVal = (float)cfg.K * 0x1p-10f;
            verifiedGpu = (fabsf(C_gpu[0] - expectedVal) <= cfg.tolerance);
        }
    }

    // Output results
    print_results(cfg, cpuStdMs, cpuOptMs, gpuTiming, verifiedStd, verifiedOpt, verifiedGpu);

    // Clean up
    free(A);
    free(B);
    free(C_std);
    free(C_opt);
    free(C_gpu);

    return EXIT_SUCCESS;
}
