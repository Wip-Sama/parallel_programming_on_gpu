/**
 * @file lab2.cu
 * @brief Massively Parallel Programming on GPUs - Lab 2: Color-to-Gray Graphic Filter
 * 
 * Course: Massively Parallel Programming on GPUs
 * Teacher: Donato D'Ambrosio
 * Department of Mathematics and Computer Science, University of Calabria
 * 
 * Features:
 * 1. CPU Reference: OpenMP baseline color-to-gray filter using 8 cores.
 * 2. GPU CUDA: Standard-memory (cudaMalloc/cudaMemcpy) color-to-gray kernel
 *    using 2D grid-stride loops for block sizes B in { 8x8, 16x16, 32x32 }.
 * 3. Exact timing of:
 *    - CPU computation (OpenMP)
 *    - GPU device memory allocation
 *    - Host-to-Device memory transfer (cudaMemcpy HtoD)
 *    - CUDA kernel execution
 *    - Device-to-Host memory transfer (cudaMemcpy DtoH)
 *    Timing measured separately from image loading and saving.
 * 4. Image I/O:
 *    - stb_image.h for primary image loading
 *    - Built-in zero-dependency uncompressed TIFF decoder fallback for
 *      Bruce Lindbloom's 3072x2048 DeltaE_8bit_gamma2.2.tif reference image
 *    - stb_image_write.h for saving output grayscale PNGs
 * 5. Bit-exact verification:
 *    - Pixel-by-pixel mismatch counter
 *    - Built-in MD5 checksum computation matching CPU and GPU output images
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// STB Image Libraries
// ---------------------------------------------------------------------------
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// ---------------------------------------------------------------------------
// CUDA Error Checking
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call) cuda_check_error(call, #call, __FILE__, __LINE__)

static inline void cuda_check_error(cudaError_t err, const char *call, const char *file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",
                call, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ---------------------------------------------------------------------------
// High-Resolution Wall-Clock Timer
// ---------------------------------------------------------------------------
static inline double get_wall_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

// ---------------------------------------------------------------------------
// Standalone MD5 Implementation (RFC 1321) for Bit-Exact Verification
// ---------------------------------------------------------------------------
typedef struct {
    uint32_t state[4];
    uint32_t count[2];
    uint8_t buffer[64];
} MD5_CTX;

#define MD5_F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define MD5_G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | (~z)))
#define MD5_ROTL(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

#define MD5_FF(a, b, c, d, x, s, ac) { \
    (a) += MD5_F((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = MD5_ROTL((a), (s)); (a) += (b); \
}
#define MD5_GG(a, b, c, d, x, s, ac) { \
    (a) += MD5_G((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = MD5_ROTL((a), (s)); (a) += (b); \
}
#define MD5_HH(a, b, c, d, x, s, ac) { \
    (a) += MD5_H((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = MD5_ROTL((a), (s)); (a) += (b); \
}
#define MD5_II(a, b, c, d, x, s, ac) { \
    (a) += MD5_I((b), (c), (d)) + (x) + (uint32_t)(ac); \
    (a) = MD5_ROTL((a), (s)); (a) += (b); \
}

static void md5_transform(uint32_t state[4], const uint8_t block[64]) {
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3], x[16];
    for (int i = 0, j = 0; j < 64; i++, j += 4)
        x[i] = ((uint32_t)block[j]) | (((uint32_t)block[j+1]) << 8) |
               (((uint32_t)block[j+2]) << 16) | (((uint32_t)block[j+3]) << 24);

    MD5_FF(a, b, c, d, x[ 0],  7, 0xd76aa478); MD5_FF(d, a, b, c, x[ 1], 12, 0xe8c7b756);
    MD5_FF(c, d, a, b, x[ 2], 17, 0x242070db); MD5_FF(b, c, d, a, x[ 3], 22, 0xc1bdceee);
    MD5_FF(a, b, c, d, x[ 4],  7, 0xf57c0faf); MD5_FF(d, a, b, c, x[ 5], 12, 0x4787c62a);
    MD5_FF(c, d, a, b, x[ 6], 17, 0xa8304613); MD5_FF(b, c, d, a, x[ 7], 22, 0xfd469501);
    MD5_FF(a, b, c, d, x[ 8],  7, 0x698098d8); MD5_FF(d, a, b, c, x[ 9], 12, 0x8b44f7af);
    MD5_FF(c, d, a, b, x[10], 17, 0xffff5bb1); MD5_FF(b, c, d, a, x[11], 22, 0x895cd7be);
    MD5_FF(a, b, c, d, x[12],  7, 0x6b901122); MD5_FF(d, a, b, c, x[13], 12, 0xfd987193);
    MD5_FF(c, d, a, b, x[14], 17, 0xa679438e); MD5_FF(b, c, d, a, x[15], 22, 0x49b40821);

    MD5_GG(a, b, c, d, x[ 1],  5, 0xf61e2562); MD5_GG(d, a, b, c, x[ 6],  9, 0xc040b340);
    MD5_GG(c, d, a, b, x[11], 14, 0x265e5a51); MD5_GG(b, c, d, a, x[ 0], 20, 0xe9b6c7aa);
    MD5_GG(a, b, c, d, x[ 5],  5, 0xd62f105d); MD5_GG(d, a, b, c, x[10],  9, 0x02441453);
    MD5_GG(c, d, a, b, x[15], 14, 0xd8a1e681); MD5_GG(b, c, d, a, x[ 4], 20, 0xe7d3fbc8);
    MD5_GG(a, b, c, d, x[ 9],  5, 0x21e1cde6); MD5_GG(d, a, b, c, x[14],  9, 0xc33707d6);
    MD5_GG(c, d, a, b, x[ 3], 14, 0xf4d50d87); MD5_GG(b, c, d, a, x[ 8], 20, 0x455a14ed);
    MD5_GG(a, b, c, d, x[13],  5, 0xa9e3e905); MD5_GG(d, a, b, c, x[ 2],  9, 0xfcefa3f8);
    MD5_GG(c, d, a, b, x[ 7], 14, 0x676f02d9); MD5_GG(b, c, d, a, x[12], 20, 0x8d2a4c8a);

    MD5_HH(a, b, c, d, x[ 5],  4, 0xfffa3942); MD5_HH(d, a, b, c, x[ 8], 11, 0x8771f681);
    MD5_HH(c, d, a, b, x[11], 16, 0x6d9d6122); MD5_HH(b, c, d, a, x[14], 23, 0xfde5380c);
    MD5_HH(a, b, c, d, x[ 1],  4, 0xa4beea44); MD5_HH(d, a, b, c, x[ 4], 11, 0x4bdecfa9);
    MD5_HH(c, d, a, b, x[ 7], 16, 0xf6bb4b60); MD5_HH(b, c, d, a, x[10], 23, 0xbebfbc70);
    MD5_HH(a, b, c, d, x[13],  4, 0x289b7ec6); MD5_HH(d, a, b, c, x[ 0], 11, 0xeaa127fa);
    MD5_HH(c, d, a, b, x[ 3], 16, 0xd4ef3085); MD5_HH(b, c, d, a, x[ 6], 23, 0x04881d05);
    MD5_HH(a, b, c, d, x[ 9],  4, 0xd9d4d039); MD5_HH(d, a, b, c, x[12], 11, 0xe6db99e5);
    MD5_HH(c, d, a, b, x[15], 16, 0x1fa27cf8); MD5_HH(b, c, d, a, x[ 2], 23, 0xc4ac5665);

    MD5_II(a, b, c, d, x[ 0],  6, 0xf4292244); MD5_II(d, a, b, c, x[ 7], 10, 0x432aff97);
    MD5_II(c, d, a, b, x[14], 15, 0xab9423a7); MD5_II(b, c, d, a, x[ 5], 21, 0xfc93a039);
    MD5_II(a, b, c, d, x[12],  6, 0x655b59c3); MD5_II(d, a, b, c, x[ 3], 10, 0x8f0ccc92);
    MD5_II(c, d, a, b, x[10], 15, 0xffeff47d); MD5_II(b, c, d, a, x[ 1], 21, 0x85845dd1);
    MD5_II(a, b, c, d, x[ 8],  6, 0x6fa87e4f); MD5_II(d, a, b, c, x[15], 10, 0xfe2ce6e0);
    MD5_II(c, d, a, b, x[ 6], 15, 0xa3014314); MD5_II(b, c, d, a, x[13], 21, 0x4e0811a1);
    MD5_II(a, b, c, d, x[ 4],  6, 0xf7537e82); MD5_II(d, a, b, c, x[11], 10, 0xbd3af235);
    MD5_II(c, d, a, b, x[ 2], 15, 0x2ad7d2bb); MD5_II(b, c, d, a, x[ 9], 21, 0xeb86d391);

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
}

static void md5_init(MD5_CTX *context) {
    context->count[0] = context->count[1] = 0;
    context->state[0] = 0x67452301;
    context->state[1] = 0xefcdab89;
    context->state[2] = 0x98badcfe;
    context->state[3] = 0x10325476;
}

static void md5_update(MD5_CTX *context, const uint8_t *input, size_t inputLen) {
    size_t i, index, partLen;
    index = (size_t)((context->count[0] >> 3) & 0x3F);
    if ((context->count[0] += ((uint32_t)inputLen << 3)) < ((uint32_t)inputLen << 3))
        context->count[1]++;
    context->count[1] += (uint32_t)(inputLen >> 29);
    partLen = 64 - index;
    if (inputLen >= partLen) {
        memcpy(&context->buffer[index], input, partLen);
        md5_transform(context->state, context->buffer);
        for (i = partLen; i + 63 < inputLen; i += 64)
            md5_transform(context->state, &input[i]);
        index = 0;
    } else {
        i = 0;
    }
    memcpy(&context->buffer[index], &input[i], inputLen - i);
}

static void md5_final(uint8_t digest[16], MD5_CTX *context) {
    static const uint8_t PADDING[64] = { 0x80 };
    uint8_t bits[8];
    for (int i = 0; i < 4; i++) {
        bits[i]   = (uint8_t)((context->count[0] >> (i * 8)) & 0xFF);
        bits[i+4] = (uint8_t)((context->count[1] >> (i * 8)) & 0xFF);
    }
    size_t index = (size_t)((context->count[0] >> 3) & 0x3F);
    size_t padLen = (index < 56) ? (56 - index) : (120 - index);
    md5_update(context, PADDING, padLen);
    md5_update(context, bits, 8);
    for (int i = 0; i < 4; i++) {
        digest[i]    = (uint8_t)((context->state[0] >> (i * 8)) & 0xFF);
        digest[i+4]  = (uint8_t)((context->state[1] >> (i * 8)) & 0xFF);
        digest[i+8]  = (uint8_t)((context->state[2] >> (i * 8)) & 0xFF);
        digest[i+12] = (uint8_t)((context->state[3] >> (i * 8)) & 0xFF);
    }
}

static void compute_md5(const uint8_t *data, size_t len, char out_hex[33]) {
    MD5_CTX ctx;
    uint8_t digest[16];
    md5_init(&ctx);
    md5_update(&ctx, data, len);
    md5_final(digest, &ctx);
    for (int i = 0; i < 16; i++) {
        sprintf(out_hex + i * 2, "%02x", digest[i]);
    }
    out_hex[32] = '\0';
}

// ---------------------------------------------------------------------------
// Image Loader: stb_image.h with Uncompressed TIFF Decoder Fallback
// ---------------------------------------------------------------------------
// Bruce Lindbloom's DeltaE_8bit_gamma2.2.tif is an uncompressed TIFF with
// strips of raw RGB pixels. stb_image.h natively lacks TIFF parsing.
// This decoder seamlessly loads uncompressed TIFF without external dependencies.
static unsigned char* load_uncompressed_tiff_rgb(const char* filename, int* out_w, int* out_h, int* out_channels) {
    FILE* f = fopen(filename, "rb");
    if (!f) return NULL;

    unsigned char header[8];
    if (fread(header, 1, 8, f) != 8) { fclose(f); return NULL; }

    int is_be = 0;
    if (header[0] == 'M' && header[1] == 'M') is_be = 1;
    else if (header[0] == 'I' && header[1] == 'I') is_be = 0;
    else { fclose(f); return NULL; }

    #define TIFF_R16(b) (is_be ? (((uint16_t)(b)[0] << 8) | (b)[1]) : (((uint16_t)(b)[1] << 8) | (b)[0]))
    #define TIFF_R32(b) (is_be ? (((uint32_t)(b)[0] << 24) | ((uint32_t)(b)[1] << 16) | ((uint32_t)(b)[2] << 8) | (b)[3]) : \
                                 (((uint32_t)(b)[3] << 24) | ((uint32_t)(b)[2] << 16) | ((uint32_t)(b)[1] << 8) | (b)[0]))

    if (TIFF_R16(header + 2) != 42) { fclose(f); return NULL; }
    uint32_t ifd_offset = TIFF_R32(header + 4);
    if (fseek(f, ifd_offset, SEEK_SET) != 0) { fclose(f); return NULL; }

    unsigned char num_entries_buf[2];
    if (fread(num_entries_buf, 1, 2, f) != 2) { fclose(f); return NULL; }
    uint16_t num_entries = TIFF_R16(num_entries_buf);

    uint32_t width = 0, height = 0, compression = 1, samples_per_pixel = 3;
    uint32_t strip_offsets_offset = 0, strip_offsets_count = 0;
    uint32_t strip_bytes_offset = 0, strip_bytes_count = 0;

    for (int i = 0; i < num_entries; i++) {
        unsigned char entry[12];
        if (fread(entry, 1, 12, f) != 12) { fclose(f); return NULL; }
        uint16_t tag = TIFF_R16(entry);
        uint16_t type = TIFF_R16(entry + 2);
        uint32_t count = TIFF_R32(entry + 4);

        if (tag == 0x0100) { // ImageWidth
            width = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0101) { // ImageHeight
            height = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0103) { // Compression
            compression = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0115) { // SamplesPerPixel
            samples_per_pixel = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0111) { // StripOffsets
            strip_offsets_count = count;
            strip_offsets_offset = (count == 1) ? ((type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8)) : TIFF_R32(entry + 8);
        } else if (tag == 0x0117) { // StripByteCounts
            strip_bytes_count = count;
            strip_bytes_offset = (count == 1) ? ((type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8)) : TIFF_R32(entry + 8);
        }
    }

    if (compression != 1 || width == 0 || height == 0 || samples_per_pixel != 3) {
        fclose(f);
        return NULL;
    }

    uint32_t* strip_offsets = (uint32_t*)malloc(strip_offsets_count * sizeof(uint32_t));
    uint32_t* strip_bytes = (uint32_t*)malloc(strip_bytes_count * sizeof(uint32_t));

    if (strip_offsets_count == 1) {
        strip_offsets[0] = strip_offsets_offset;
    } else {
        fseek(f, strip_offsets_offset, SEEK_SET);
        for (uint32_t i = 0; i < strip_offsets_count; i++) {
            unsigned char b[4];
            if (fread(b, 1, 4, f) != 4) break;
            strip_offsets[i] = TIFF_R32(b);
        }
    }

    if (strip_bytes_count == 1) {
        strip_bytes[0] = strip_bytes_offset;
    } else {
        fseek(f, strip_bytes_offset, SEEK_SET);
        for (uint32_t i = 0; i < strip_bytes_count; i++) {
            unsigned char b[4];
            if (fread(b, 1, 4, f) != 4) break;
            strip_bytes[i] = TIFF_R32(b);
        }
    }

    size_t total_bytes = (size_t)width * height * 3;
    unsigned char* data = (unsigned char*)malloc(total_bytes);
    if (!data) {
        free(strip_offsets);
        free(strip_bytes);
        fclose(f);
        return NULL;
    }

    size_t current_pos = 0;
    for (uint32_t i = 0; i < strip_offsets_count && current_pos < total_bytes; i++) {
        fseek(f, strip_offsets[i], SEEK_SET);
        size_t to_read = strip_bytes[i];
        if (current_pos + to_read > total_bytes) to_read = total_bytes - current_pos;
        current_pos += fread(data + current_pos, 1, to_read, f);
    }

    free(strip_offsets);
    free(strip_bytes);
    fclose(f);

    *out_w = (int)width;
    *out_h = (int)height;
    *out_channels = 3;
    return data;
}

typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
    int is_stb_allocated;
} LoadedImage;

static LoadedImage load_image(const char* filepath) {
    LoadedImage img;
    memset(&img, 0, sizeof(img));

    // Try stb_image first (supports PNG, JPG, BMP, PPM, etc.)
    img.data = stbi_load(filepath, &img.width, &img.height, &img.channels, 3);
    if (img.data) {
        img.channels = 3;
        img.is_stb_allocated = 1;
        return img;
    }

    // Fallback: uncompressed TIFF loader (DeltaE_8bit_gamma2.2.tif)
    img.data = load_uncompressed_tiff_rgb(filepath, &img.width, &img.height, &img.channels);
    if (img.data) {
        img.channels = 3;
        img.is_stb_allocated = 0;
        return img;
    }

    return img;
}

static void free_loaded_image(LoadedImage* img) {
    if (img && img->data) {
        if (img->is_stb_allocated) {
            stbi_image_free(img->data);
        } else {
            free(img->data);
        }
        img->data = NULL;
    }
}

// ---------------------------------------------------------------------------
// 1. CPU Reference Implementation (OpenMP)
// ---------------------------------------------------------------------------
/**
 * Formula: gray = 0.21 * r + 0.71 * g + 0.07 * b
 * Evaluated across 2D coordinates in row-major order.
 * Parallelized with OpenMP static scheduling.
 */
void color_to_gray_cpu_omp(const unsigned char *__restrict__ rgb,
                           unsigned char *__restrict__ gray,
                           int width, int height, int num_threads) {
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int gray_idx = y * width + x;
            int rgb_idx = gray_idx * 3;
            float r = (float)rgb[rgb_idx];
            float g = (float)rgb[rgb_idx + 1];
            float b = (float)rgb[rgb_idx + 2];
            gray[gray_idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

// ---------------------------------------------------------------------------
// 2. CUDA Kernel Implementation (2D Grid-Stride Loop)
// ---------------------------------------------------------------------------
/**
 * Standard memory model kernel with 2D grid-stride loops:
 * - x stride: blockDim.x * gridDim.x
 * - y stride: blockDim.y * gridDim.y
 * Decouples thread count and block sizing from arbitrary image geometries.
 */
__global__ void color_to_gray_cuda_grid_stride(const unsigned char *__restrict__ rgb,
                                               unsigned char *__restrict__ gray,
                                               int width, int height) {
    int x_stride = blockDim.x * gridDim.x;
    int y_stride = blockDim.y * gridDim.y;

    for (int y = blockIdx.y * blockDim.y + threadIdx.y; y < height; y += y_stride) {
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < width; x += x_stride) {
            int gray_idx = y * width + x;
            int rgb_idx = gray_idx * 3;
            float r = (float)rgb[rgb_idx];
            float g = (float)rgb[rgb_idx + 1];
            float b = (float)rgb[rgb_idx + 2];
            gray[gray_idx] = (unsigned char)(0.21f * r + 0.71f * g + 0.07f * b);
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration & Results Structs
// ---------------------------------------------------------------------------
enum RunMode {
    MODE_CPU,
    MODE_GPU,
    MODE_BOTH
};

typedef struct {
    RunMode mode;
    int blockSize;          // 8, 16, or 32 (defines BxB block dimensions)
    int numThreads;         // OpenMP threads (default 8)
    int devId;              // CUDA device ID (default 0)
    int saveOutput;         // Whether to write PNG output to disk
    int csvOutput;          // Output CSV line for automated harness
    char inputPath[512];    // Input image path
    char outputPath[512];   // Output grayscale PNG path
} Config;

typedef struct {
    double cpuTimeMs;
    char cpuMd5[33];

    float gpuAllocMs;
    float gpuHtodMs;
    float gpuKernelMs;
    float gpuDtohMs;
    float gpuComputeMs;     // HtoD + Kernel + DtoH
    float gpuTotalMs;       // Alloc + HtoD + Kernel + DtoH
    char gpuMd5[33];

    int verified;           // 1 if CPU and GPU match exactly, 0 otherwise
    int mismatchCount;
    dim3 blockDim;
    dim3 gridDim;
} BenchmarkResult;

// ---------------------------------------------------------------------------
// GPU Benchmark Execution
// ---------------------------------------------------------------------------
void run_gpu_benchmark(const unsigned char *h_rgb, unsigned char *h_gray,
                       int width, int height, int block_dim_val, BenchmarkResult *res) {
    size_t img_bytes = (size_t)width * height * 3;
    size_t gray_bytes = (size_t)width * height;

    dim3 block(block_dim_val, block_dim_val);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    res->blockDim = block;
    res->gridDim = grid;

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));

    // 1. Device Memory Allocation & Initialization
    double t_alloc_start = get_wall_time_ms();
    unsigned char *d_rgb = NULL;
    unsigned char *d_gray = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_rgb, img_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_gray, gray_bytes));
    double t_alloc_end = get_wall_time_ms();
    res->gpuAllocMs = (float)(t_alloc_end - t_alloc_start);

    // 2. Host-to-Device Memory Transfer
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    CUDA_CHECK(cudaMemcpy(d_rgb, h_rgb, img_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&res->gpuHtodMs, ev_start, ev_stop));

    // 3. Kernel Execution
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    color_to_gray_cuda_grid_stride<<<grid, block>>>(d_rgb, d_gray, width, height);
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventElapsedTime(&res->gpuKernelMs, ev_start, ev_stop));

    // 4. Device-to-Host Memory Transfer
    CUDA_CHECK(cudaEventRecord(ev_start, 0));
    CUDA_CHECK(cudaMemcpy(h_gray, d_gray, gray_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev_stop, 0));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaEventElapsedTime(&res->gpuDtohMs, ev_start, ev_stop));

    // Total timings
    res->gpuComputeMs = res->gpuHtodMs + res->gpuKernelMs + res->gpuDtohMs;
    res->gpuTotalMs = res->gpuAllocMs + res->gpuComputeMs;

    // MD5 of GPU output buffer
    compute_md5(h_gray, gray_bytes, res->gpuMd5);

    // Cleanup device resources
    CUDA_CHECK(cudaFree(d_rgb));
    CUDA_CHECK(cudaFree(d_gray));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
}

// ---------------------------------------------------------------------------
// CLI Parser & Help
// ---------------------------------------------------------------------------
static void print_usage(const char* prog) {
    printf("Usage:\n");
    printf("  %s [mode] [blockSize] [imagePath] [options]\n\n", prog);
    printf("Positional Arguments (optional):\n");
    printf("  mode        : Execution mode ('cpu', 'gpu', or 'both') [default: both]\n");
    printf("  blockSize   : CUDA 2D block dimension ('8', '16', or '32' for BxB) [default: 16]\n");
    printf("  imagePath   : Path to input RGB image [default: DeltaE_8bit_gamma2.2.tif]\n\n");
    printf("Options:\n");
    printf("  --mode <m>       : Execution mode ('cpu', 'gpu', 'both')\n");
    printf("  --block <b>      : CUDA 2D block dimension (8, 16, 32)\n");
    printf("  --threads <t>    : Number of OpenMP CPU threads [default: 8]\n");
    printf("  --device <id>    : CUDA device ID [default: 0]\n");
    printf("  --input <path>   : Input image path\n");
    printf("  --out <path>     : Output grayscale PNG file path [default: output.png]\n");
    printf("  --save           : Save grayscale PNG output(s) to disk [default]\n");
    printf("  --no-save        : Do not write PNG to disk (for pure benchmark sweeps)\n");
    printf("  --csv            : Output machine-readable CSV for automated scripting\n");
    printf("  -h, --help       : Show this help message and exit\n\n");
    printf("Examples:\n");
    printf("  %s both 16                             # Run CPU & GPU (16x16) and verify\n", prog);
    printf("  %s gpu 8 DeltaE_8bit_gamma2.2.tif      # Run GPU with 8x8 block size\n", prog);
    printf("  %s cpu 8                               # Run OpenMP CPU reference (8 threads)\n", prog);
    printf("  %s both 16 --csv                       # Output CSV row for benchmarks\n", prog);
}

static Config parse_args(int argc, char** argv) {
    Config cfg;
    cfg.mode = MODE_BOTH;
    cfg.blockSize = 16;
    cfg.numThreads = 8;
    cfg.devId = 0;
    cfg.saveOutput = 1;
    cfg.csvOutput = 0;
    strcpy(cfg.inputPath, "DeltaE_8bit_gamma2.2.tif");
    strcpy(cfg.outputPath, "output.png");

    int pos_idx = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(EXIT_SUCCESS);
        } else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) {
            const char* m = argv[++i];
            if (strcasecmp(m, "cpu") == 0) cfg.mode = MODE_CPU;
            else if (strcasecmp(m, "gpu") == 0) cfg.mode = MODE_GPU;
            else cfg.mode = MODE_BOTH;
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            cfg.blockSize = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg.numThreads = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
            cfg.devId = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            strncpy(cfg.inputPath, argv[++i], sizeof(cfg.inputPath) - 1);
        } else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
            strncpy(cfg.outputPath, argv[++i], sizeof(cfg.outputPath) - 1);
        } else if (strcmp(argv[i], "--save") == 0) {
            cfg.saveOutput = 1;
        } else if (strcmp(argv[i], "--no-save") == 0) {
            cfg.saveOutput = 0;
        } else if (strcmp(argv[i], "--csv") == 0) {
            cfg.csvOutput = 1;
        } else if (argv[i][0] != '-') {
            // Positional argument
            if (pos_idx == 0) {
                if (strcasecmp(argv[i], "cpu") == 0) cfg.mode = MODE_CPU;
                else if (strcasecmp(argv[i], "gpu") == 0) cfg.mode = MODE_GPU;
                else if (strcasecmp(argv[i], "both") == 0) cfg.mode = MODE_BOTH;
                else if (atoi(argv[i]) == 8 || atoi(argv[i]) == 16 || atoi(argv[i]) == 32) {
                    cfg.blockSize = atoi(argv[i]);
                } else {
                    strncpy(cfg.inputPath, argv[i], sizeof(cfg.inputPath) - 1);
                }
            } else if (pos_idx == 1) {
                if (atoi(argv[i]) == 8 || atoi(argv[i]) == 16 || atoi(argv[i]) == 32) {
                    cfg.blockSize = atoi(argv[i]);
                } else {
                    strncpy(cfg.inputPath, argv[i], sizeof(cfg.inputPath) - 1);
                }
            } else if (pos_idx == 2) {
                strncpy(cfg.inputPath, argv[i], sizeof(cfg.inputPath) - 1);
            }
            pos_idx++;
        }
    }

    if (cfg.blockSize != 8 && cfg.blockSize != 16 && cfg.blockSize != 32) {
        if (!cfg.csvOutput) {
            fprintf(stderr, "[!] Warning: block size %d is not in {8, 16, 32}. Using %d.\n",
                    cfg.blockSize, cfg.blockSize);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Main Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    // Check device if GPU will be used
    if (cfg.mode == MODE_GPU || cfg.mode == MODE_BOTH) {
        int devCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&devCount));
        if (devCount == 0) {
            fprintf(stderr, "Error: No CUDA capable devices detected.\n");
            return EXIT_FAILURE;
        }
        if (cfg.devId >= devCount) cfg.devId = 0;
        CUDA_CHECK(cudaSetDevice(cfg.devId));
        // Initialize CUDA context to eliminate lazy driver initialization noise
        CUDA_CHECK(cudaFree(0));
    }

    // 1. Load Input Image
    double t_load_start = get_wall_time_ms();
    LoadedImage img = load_image(cfg.inputPath);
    if (!img.data) {
        // Try fallback to search in local directory or labs/lab2
        char alt_path[600];
        snprintf(alt_path, sizeof(alt_path), "labs/lab2/%s", cfg.inputPath);
        img = load_image(alt_path);
    }
    if (!img.data) {
        fprintf(stderr, "Error: Failed to load image '%s'\n", cfg.inputPath);
        return EXIT_FAILURE;
    }
    double t_load_end = get_wall_time_ms();
    double loadTimeMs = t_load_end - t_load_start;

    int width = img.width;
    int height = img.height;
    size_t gray_bytes = (size_t)width * height;

    unsigned char* h_gray_cpu = NULL;
    unsigned char* h_gray_gpu = NULL;

    BenchmarkResult res;
    memset(&res, 0, sizeof(res));
    res.verified = 1;

    // 2. CPU OpenMP Reference Execution
    if (cfg.mode == MODE_CPU || cfg.mode == MODE_BOTH) {
        h_gray_cpu = (unsigned char*)malloc(gray_bytes);
        if (!h_gray_cpu) {
            fprintf(stderr, "Error: Host allocation for CPU gray buffer failed.\n");
            free_loaded_image(&img);
            return EXIT_FAILURE;
        }

        double t0 = omp_get_wtime();
        color_to_gray_cpu_omp(img.data, h_gray_cpu, width, height, cfg.numThreads);
        double t1 = omp_get_wtime();
        res.cpuTimeMs = (t1 - t0) * 1000.0;

        compute_md5(h_gray_cpu, gray_bytes, res.cpuMd5);
    }

    // 3. GPU CUDA Execution
    if (cfg.mode == MODE_GPU || cfg.mode == MODE_BOTH) {
        h_gray_gpu = (unsigned char*)malloc(gray_bytes);
        if (!h_gray_gpu) {
            fprintf(stderr, "Error: Host allocation for GPU gray buffer failed.\n");
            if (h_gray_cpu) free(h_gray_cpu);
            free_loaded_image(&img);
            return EXIT_FAILURE;
        }

        run_gpu_benchmark(img.data, h_gray_gpu, width, height, cfg.blockSize, &res);
    }

    // 4. Verification & Correctness Checking
    if (cfg.mode == MODE_BOTH && h_gray_cpu && h_gray_gpu) {
        int mismatches = 0;
        for (size_t i = 0; i < gray_bytes; i++) {
            if (h_gray_cpu[i] != h_gray_gpu[i]) mismatches++;
        }
        res.mismatchCount = mismatches;
        res.verified = (mismatches == 0 && strcmp(res.cpuMd5, res.gpuMd5) == 0);
    }

    // 5. Saving Grayscale Image Outputs
    double saveTimeMs = 0.0;
    if (cfg.saveOutput) {
        double t_save_start = get_wall_time_ms();
        if (cfg.mode == MODE_CPU && h_gray_cpu) {
            stbi_write_png(cfg.outputPath, width, height, 1, h_gray_cpu, width);
        } else if (cfg.mode == MODE_GPU && h_gray_gpu) {
            stbi_write_png(cfg.outputPath, width, height, 1, h_gray_gpu, width);
        } else if (cfg.mode == MODE_BOTH) {
            stbi_write_png("output_cpu.png", width, height, 1, h_gray_cpu, width);
            stbi_write_png("output_gpu.png", width, height, 1, h_gray_gpu, width);
        }
        double t_save_end = get_wall_time_ms();
        saveTimeMs = t_save_end - t_save_start;
    }

    // 6. Formatted Output (CSV or Human Console)
    if (cfg.csvOutput) {
        if (cfg.mode == MODE_CPU) {
            printf("CSV,cpu,%d,%d,%d,%.4f,%s\n",
                   cfg.numThreads, width, height, res.cpuTimeMs, res.cpuMd5);
        } else if (cfg.mode == MODE_GPU) {
            printf("CSV,gpu,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%s\n",
                   cfg.blockSize, res.gridDim.x, res.gridDim.y, width, height,
                   res.gpuAllocMs, res.gpuHtodMs, res.gpuKernelMs, res.gpuDtohMs,
                   res.gpuComputeMs, res.gpuTotalMs, res.gpuMd5);
        } else { // BOTH
            float speedupKernel = (float)(res.cpuTimeMs / res.gpuKernelMs);
            float speedupCompute = (float)(res.cpuTimeMs / res.gpuComputeMs);
            float speedupTotal = (float)(res.cpuTimeMs / res.gpuTotalMs);
            printf("CSV,both,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.2f,%.2f,%.2f,%s,%s,%s\n",
                   cfg.blockSize, cfg.numThreads, width, height,
                   res.cpuTimeMs, res.gpuAllocMs, res.gpuHtodMs, res.gpuKernelMs, res.gpuDtohMs,
                   res.gpuComputeMs, res.gpuTotalMs, speedupKernel, speedupCompute, speedupTotal,
                   res.cpuMd5, res.gpuMd5, res.verified ? "PASS" : "FAIL");
        }
    } else {
        printf("\n======================================================================\n");
        printf(" Massively Parallel Programming on GPUs - Lab 2: Color-to-Gray Filter\n");
        printf("======================================================================\n");
        printf(" [Input Image]  : %s (%dx%d, %zu pixels, %.2f MB RGB)\n",
               cfg.inputPath, width, height, (size_t)width * height,
               (width * height * 3.0) / (1024.0 * 1024.0));
        printf(" [I/O Timing]   : Load = %.2f ms | Save = %.2f ms\n", loadTimeMs, saveTimeMs);
        printf("----------------------------------------------------------------------\n");

        if (cfg.mode == MODE_CPU || cfg.mode == MODE_BOTH) {
            printf(" [CPU OpenMP]   : %d Cores / Threads\n", cfg.numThreads);
            printf("   Elapsed Time : %8.4f ms (Color-to-Gray Computation Only)\n", res.cpuTimeMs);
            printf("   MD5 Checksum : %s\n", res.cpuMd5);
            printf("----------------------------------------------------------------------\n");
        }

        if (cfg.mode == MODE_GPU || cfg.mode == MODE_BOTH) {
            cudaDeviceProp prop;
            CUDA_CHECK(cudaGetDeviceProperties(&prop, cfg.devId));
            printf(" [CUDA GPU]     : %s (SM %d.%d, %d SMs)\n",
                   prop.name, prop.major, prop.minor, prop.multiProcessorCount);
            printf("   Block Config : %dx%d (%d threads/block)\n",
                   res.blockDim.x, res.blockDim.y, res.blockDim.x * res.blockDim.y);
            printf("   Grid Config  : %dx%d (%d total blocks)\n",
                   res.gridDim.x, res.gridDim.y, res.gridDim.x * res.gridDim.y);
            printf("   -- Breakdown --\n");
            printf("   cudaMalloc   : %8.4f ms (Device Buffers Allocation)\n", res.gpuAllocMs);
            printf("   HtoD Transfer: %8.4f ms (Host -> Device, %.2f MB)\n",
                   res.gpuHtodMs, (width * height * 3.0) / (1024.0 * 1024.0));
            printf("   Kernel Exec  : %8.4f ms (color_to_gray_cuda_grid_stride)\n", res.gpuKernelMs);
            printf("   DtoH Transfer: %8.4f ms (Device -> Host, %.2f MB)\n",
                   res.gpuDtohMs, (width * height * 1.0) / (1024.0 * 1024.0));
            printf("   Compute Time : %8.4f ms (Transfers + Kernel)\n", res.gpuComputeMs);
            printf("   Total GPU    : %8.4f ms (Alloc + Transfers + Kernel)\n", res.gpuTotalMs);
            printf("   MD5 Checksum : %s\n", res.gpuMd5);
            printf("----------------------------------------------------------------------\n");
        }

        if (cfg.mode == MODE_BOTH) {
            printf(" [Verification & Correctness]\n");
            printf("   Pixel Check  : %s (%d mismatches out of %zu pixels)\n",
                   res.mismatchCount == 0 ? "EXACT MATCH (100% BIT-FOR-BIT)" : "FAILED MISMATCH",
                   res.mismatchCount, gray_bytes);
            printf("   MD5 Match    : %s\n",
                   strcmp(res.cpuMd5, res.gpuMd5) == 0 ? "IDENTICAL CHECKSUM (PASS)" : "CHECKSUM MISMATCH (FAIL)");
            printf("   Status       : [%s]\n", res.verified ? "SUCCESS / PASS" : "FAILURE / FAIL");
            printf("----------------------------------------------------------------------\n");
            printf(" [Performance Speedup]\n");
            printf("   Pure Kernel Speedup  : %8.2fx (CPU Computation vs GPU Kernel)\n",
                   res.cpuTimeMs / res.gpuKernelMs);
            printf("   Transfer+Kernel Speed: %8.2fx (CPU Computation vs HtoD+Kernel+DtoH)\n",
                   res.cpuTimeMs / res.gpuComputeMs);
            printf("   End-to-End Speedup   : %8.2fx (CPU Computation vs Total GPU Time)\n",
                   res.cpuTimeMs / res.gpuTotalMs);
            printf("----------------------------------------------------------------------\n");
            if (cfg.saveOutput) {
                printf(" [Saved Outputs]: output_cpu.png and output_gpu.png\n");
            }
        } else if (cfg.saveOutput) {
            printf(" [Saved Output] : %s\n", cfg.outputPath);
        }
        printf("======================================================================\n\n");
    }

    // Cleanup host allocations
    if (h_gray_cpu) free(h_gray_cpu);
    if (h_gray_gpu) free(h_gray_gpu);
    free_loaded_image(&img);

    return res.verified ? EXIT_SUCCESS : EXIT_FAILURE;
}
