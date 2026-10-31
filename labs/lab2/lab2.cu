/**
 * @file lab2.cu
 * @brief Massively Parallel Programming on GPUs - Lab 2: Color-to-Gray Filter
 * 
 * Course: Massively Parallel Programming on GPUs
 * Teacher: Donato D'Ambrosio
 * Department of Mathematics and Computer Science, University of Calabria
 */

#include "common.h"

// ---------------------------------------------------------------------------
// 1. CPU Reference Implementation (OpenMP)
// ---------------------------------------------------------------------------
/**
 * @brief CPU OpenMP Color-to-Gray filter.
 * Formula: gray = 0.21 * r + 0.71 * g + 0.07 * b
 * 
 * @param rgb         Input RGB image buffer (3 bytes per pixel: R, G, B)
 * @param gray        Output Grayscale buffer (1 byte per pixel)
 * @param width       Image width in pixels
 * @param height      Image height in pixels
 * @param num_threads Number of OpenMP threads to use
 * 
 * TODO: Implement the parallel CPU color-to-gray transformation using OpenMP.
 */
void color_to_gray_cpu(
    const unsigned char *rgb, 
    unsigned char *gray,
    int width, 
    int height, 
    int num_threads
) {
    int size = width * height;

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < size; i++) {
        gray[i] = 0.21f * rgb[3 * i] + 0.71f * rgb[ 3 * i + 1] + 0.07f * rgb[3 * i + 2];
    }
}

// ---------------------------------------------------------------------------
// 2. CUDA Kernel Implementation (2D Grid-Stride Loop)
// ---------------------------------------------------------------------------
/**
 * @brief CUDA 2D Grid-Stride Color-to-Gray Kernel.
 * Formula: gray = 0.21 * r + 0.71 * g + 0.07 * b
 * 
 * @param rgb    Device pointer to RGB image buffer
 * @param gray   Device pointer to Grayscale output buffer
 * @param width  Image width in pixels
 * @param height Image height in pixels
 * 
 * TODO: Implement the CUDA kernel with 2D grid-stride loops.
 */
__global__ void color_to_gray_cuda(
    const unsigned char *rgb, 
    unsigned char *gray,
    int width, 
    int height
) {
    
    int strideX = blockDim.x * gridDim.x;
    int strideY = blockDim.y * gridDim.y;

    // We take the block at position blockIdx
    // We take teh dimnesion of the block
    // We take out thread position in the block
    // pos += stridePos make jumps of size (blockDim * gridDim) 
    //      to cover the specific element of out thread in each tile
    // Keep in mind that a block is a CUDA execution concept, just a group of threads
    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += strideY) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += strideX) {
            int idx = y * width + x;
            int rgb_idx = 3 * idx;

            float val = 0.21f * rgb[rgb_idx] + 
                        0.71f * rgb[rgb_idx + 1] + 
                        0.07f * rgb[rgb_idx + 2];

            gray[idx] = static_cast<unsigned char>(val);

            // Alternative implementation (commented out for clarity)
            // int idx = y * width + x;
            // gray[idx] = 0.21f * rgb[3 * idx] + 0.71f * rgb[ 3 * idx + 1] + 0.07f * rgb[3 * idx + 2];
        }
    }
}

// ---------------------------------------------------------------------------
// 3. GPU Runner (Allocation, Transfers, Launch, Timing)
// ---------------------------------------------------------------------------
/**
 * @brief Executes the complete CUDA pipeline:
 * 1. Allocate device memory for RGB and Gray buffers (cudaMalloc)
 * 2. Transfer input RGB data from host to device (cudaMemcpy HtoD)
 * 3. Configure 2D block and grid dimensions
 * 4. Launch color_to_gray_cuda kernel
 * 5. Transfer result gray image from device to host (cudaMemcpy DtoH)
 * 6. Record exact timings for each phase with CUDA events
 * 7. Clean up device memory and CUDA events
 * 
 * TODO: Implement device allocations, transfers, timing events, and kernel launch.
 */
void run_gpu(
    const unsigned char *h_rgb, 
    unsigned char *h_gray,
    int width, 
    int height, 
    int block_dim_size, 
    GpuTiming *timing
) {
    // Calculate bytes using size_t to prevent integer overflow
    size_t rgb_bytes  = static_cast<size_t>(width) * height * 3 * sizeof(unsigned char);
    size_t gray_bytes = static_cast<size_t>(width) * height * sizeof(unsigned char);

    // Set block & grid dimensions
    // Note: CUDA blocks cannot exceed 1024 threads total (block_dim_size must be <= 32)
    timing->blockDim = dim3(block_dim_size, block_dim_size);
    timing->gridDim = dim3(
        (width  + block_dim_size - 1) / block_dim_size,
        (height + block_dim_size - 1) / block_dim_size
    );

    // Set up CUDA Events for precise GPU timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 1. Device memory allocation (timed with high-resolution wall-clock)
    double t_alloc_0 = get_wall_time_ms();
    unsigned char *d_rgb = nullptr;
    unsigned char *d_gray = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_rgb, rgb_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray, gray_bytes));
    timing->allocMs = (float)(get_wall_time_ms() - t_alloc_0);

    // 2. Host-to-Device transfer
    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, rgb_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timing->htodMs, ev_start, ev_stop));

    // 3. Kernel Execution
    CUDA_CHECK(cudaEventRecord(ev_start));
    color_to_gray_cuda<<<timing->gridDim, timing->blockDim>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&timing->kernelMs, ev_start, ev_stop));

    // 4. Device-to-Host transfer
    CUDA_CHECK(cudaEventRecord(ev_start));
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timing->dtohMs, ev_start, ev_stop));

    // Compute composite timings
    timing->computeMs = timing->htodMs + timing->kernelMs + timing->dtohMs;
    timing->totalMs   = timing->allocMs + timing->computeMs;

    // Free resources and cleanup events
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    return run_lab2_driver(argc, argv, color_to_gray_cpu, run_gpu);
}
