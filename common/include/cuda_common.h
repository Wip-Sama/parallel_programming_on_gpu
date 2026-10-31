#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

/**
 * @brief Macro for robust CUDA API error checking.
 */
#define CUDA_CHECK(call) cuda_check_error(call, #call, __FILE__, __LINE__)

static inline void cuda_check_error(cudaError_t err, const char *call, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",
                call, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/**
 * @brief High-resolution monotonic wall-clock timer returning elapsed milliseconds.
 */
static inline double get_wall_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

/**
 * @brief Lightweight C++ wrapper around CUDA events for GPU timing.
 */
struct CudaTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    bool initialized;

    CudaTimer() : initialized(false) {}

    void init() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
        initialized = true;
    }

    void start(cudaStream_t stream = 0) {
        if (!initialized) init();
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms;
    }

    void destroy() {
        if (initialized) {
            cudaEventDestroy(start_event);
            cudaEventDestroy(stop_event);
            initialized = false;
        }
    }

    ~CudaTimer() {
        destroy();
    }
};

/**
 * @brief Print CUDA device properties header.
 */
static inline void print_cuda_device_info(int devId) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));
    printf("================================================================================\n");
    printf(" CUDA Target Device [%d]: %s (Compute %d.%d, %d SMs, %.2f GB Global Mem)\n",
           devId, prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("================================================================================\n");
}

#endif // CUDA_COMMON_H

