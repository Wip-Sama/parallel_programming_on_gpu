/**
 * @file lab2.cu
 * @brief Massively Parallel Programming on GPUs - Lab 2: Color-to-Gray Filter
 *        Unified Memory Implementation (cudaMallocManaged)
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
        gray[i] = 0.21f * rgb[3 * i] + 0.71f * rgb[3 * i + 1] + 0.07f * rgb[3 * i + 2];
    }
}

// ---------------------------------------------------------------------------
// 2. CUDA Kernel Implementation (2D Grid-Stride Loop)
// ---------------------------------------------------------------------------
/**
 * @brief CUDA 2D Grid-Stride Color-to-Gray Kernel.
 * Formula: gray = 0.21 * r + 0.71 * g + 0.07 * b
 * 
 * @param rgb    Pointer to managed RGB image buffer
 * @param gray   Pointer to managed Grayscale output buffer
 * @param width  Image width in pixels
 * @param height Image height in pixels
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
// 3. GPU Runner (Unified Memory: cudaMallocManaged)
// ---------------------------------------------------------------------------
/**
 * @brief Executes the complete CUDA pipeline using Unified Memory:
 * 1. Allocate managed memory for RGB and Gray buffers (cudaMallocManaged)
 * 2. CPU populates managed RGB buffer directly (no explicit cudaMemcpy HtoD)
 * 3. Configure 2D block and grid dimensions
 * 4. Launch color_to_gray_cuda kernel (pages migrate on-demand via hardware page faults)
 * 5. Synchronize device with cudaDeviceSynchronize() before host access
 * 6. Copy output to destination buffer
 * 7. Clean up managed memory and CUDA events
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
    timing->blockDim = dim3(block_dim_size, block_dim_size);
    timing->gridDim = dim3(
        (width  + block_dim_size - 1) / block_dim_size,
        (height + block_dim_size - 1) / block_dim_size
    );

    // Set up CUDA Events for precise kernel execution timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // Unified Memory allocation (cudaMallocManaged)
    double t_alloc_0 = get_wall_time_ms();
    unsigned char *u_rgb = nullptr;
    unsigned char *u_gray = nullptr;
    CUDA_CHECK(cudaMallocManaged((void**)&u_rgb, rgb_bytes));
    CUDA_CHECK(cudaMallocManaged((void**)&u_gray, gray_bytes));
    timing->allocMs = static_cast<float>(get_wall_time_ms() - t_alloc_0);

    // Populate managed RGB input buffer from host data
    // In Unified Memory, CPU writes directly to managed memory; no explicit cudaMemcpy HtoD needed
    memcpy(u_rgb, h_rgb, rgb_bytes);
    timing->htodMs = 0.0f;

    // Kernel Execution
    // As GPU threads access u_rgb and u_gray, hardware page faults transparently migrate
    // pages across the PCIe bus on demand
    CUDA_CHECK(cudaEventRecord(ev_start));
    color_to_gray_cuda<<<timing->gridDim, timing->blockDim>>>(u_rgb, u_gray, width, height);
    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&timing->kernelMs, ev_start, ev_stop));

    // Device Synchronization and Host Access
    // Explicit sync ensures GPU kernel has fully finished before CPU reads u_gray
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(h_gray, u_gray, gray_bytes);
    timing->dtohMs = 0.0f;

    // Compute composite timings
    timing->computeMs = timing->kernelMs;
    timing->totalMs   = timing->allocMs + timing->computeMs;

    // Free managed resources and cleanup events
    CUDA_CHECK(cudaFree(u_rgb));
    CUDA_CHECK(cudaFree(u_gray));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    return run_lab2_driver(argc, argv, color_to_gray_cpu, run_gpu);
}
