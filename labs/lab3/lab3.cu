#include "common.h"



// ---------------------------------------------------------------------------
// 1. CPU Reference Implementation (OpenMP)
// ---------------------------------------------------------------------------
void cpu_init(
    float* A,
    float* B,
    float* C_host,
    float* C_device,
    int N,
    int M,
    int K,
    int num_threads
) {
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A[i * K + j] = 0x1p-11f;
        }
    }
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B[i * N + j] = 2;
        }
    }
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C_host[i * N + j] = 0;
            C_device[i * N + j] = 0;
        }
    }
}

void matrix_multiplication_cpu(
    const float* A,
    const float* B,
    float* C,
    int N,
    int M,
    int K,
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

void matrix_multiplication_cpu_optimized(
    const float* A,
    const float* B,
    float* C,
    int N,
    int M,
    int K,
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

bool check_result(
    const float* H, 
    const float* D, 
    int N, 
    int M,
    int num_threads
) {
    int size = M * N;

    for (int i = 0; i < size; i++) {
        if (fabs(H[i] - D[i]) > 1e-5) {
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 2. CUDA Kernel Implementation (2D Grid-Stride Loop)
// ---------------------------------------------------------------------------
__global__ void global_memory_matrix_multiplication_cuda(
    const float* A,
    const float* B,
    float* C,
    int N,
    int M,
    int K
) {
    int strideY = blockDim.y * gridDim.y;
    int strideX = blockDim.x * gridDim.x;

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < N; y += strideY) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < M; x += strideX) {
            float sum = 0;

            for (int k = 0; k < K; k++) {
                sum += A[y * K + k] * B[k * N + x];
            }

            C[y * N + x] = sum;
        }
    }
}

void run_gpu_global_memory(
    const float* A,
    const float* B,
    float* C,
    int N,
    int M,
    int K,
    int block_dim_size, 
    GpuTiming *timing
) {
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    timing->blockDim = dim3(block_dim_size, block_dim_size);
    timing->gridDim = dim3(
        (N + block_dim_size - 1) / block_dim_size, 
        (M + block_dim_size - 1) / block_dim_size
    );

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    double t_alloc_0 = get_wall_time_ms();
    unsigned char *dev_A = nullptr, *dev_B = nullptr, *dev_C = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&dev_A, size_A));
    CUDA_CHECK(cudaMallocManaged((void**)&dev_B, size_B));
    CUDA_CHECK(cudaMallocManaged((void**)&dev_C, size_C));
    timing->allocMs = static_cast<float>(get_wall_time_ms() - t_alloc_0);
    
    memcpy(dev_A, A, size_A);
    memcpy(dev_B, B, size_B);
    timing->htodMs = 0.0f;

    CUDA_CHECK(cudaEventRecord(ev_start));
    global_memory_matrix_multiplication_cuda<<<gridDim, blockDim>>>(
        (float*)dev_A, 
        (float*)dev_B, 
        (float*)dev_C, 
        N, 
        M, 
        K
    );
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&timing->kernelMs, ev_start, ev_stop));

    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(C, dev_C, size_C);
    timing->dtohMs = 0.0f;

    timing->computeMs = timing->kernelMs;
    timing->totalMs   = timing->allocMs + timing->computeMs;

    CUDA_CHECK(cudaFree(dev_A));
    CUDA_CHECK(cudaFree(dev_B));
    CUDA_CHECK(cudaFree(dev_C));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    int N = 1024; // Number of columns in B and C
    int M = 1024; // Number of rows in A and C
    int K = 1024; // Number of columns in A and rows in B
    int num_threads = 4; // Number of threads to use
    int block_dim_size = 16; // Block dimension size for CUDA kernel
    float *A, *B;
    float *C_host;
    float *C_device;
    A = (float*)malloc(M * K * sizeof(float));
    B = (float*)malloc(K * N * sizeof(float));
    C_host = (float*)malloc(M * N * sizeof(float));
    C_device = (float*)malloc(M * N * sizeof(float));
    cpu_init(A, B, C_host, C_device, N, M, K, num_threads);
    matrix_multiplication_cpu(A, B, C_host, N, M, K, num_threads);
    GpuTiming timing;
    run_gpu_global_memory(A, B, C_device, N, M, K, block_dim_size, &timing);
    check_result(C_host, C_device, N, M, num_threads) ? printf("Results match!\n") : printf("Results do not match!\n");
    free(A);
    free(B);
    free(C_host);
    free(C_device);
}
