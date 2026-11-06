#ifndef LAB3_COMMON_H
#define LAB3_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <omp.h>
#include <cuda_runtime.h>

// Common CUDA utilities (error checking, timers, device info)
#include "cuda_common.h"
#endif // LAB3_COMMON_H

// ---------------------------------------------------------------------------
// Enumerations & Structs
// ---------------------------------------------------------------------------
enum RunMode {
    MODE_CPU,
    MODE_GPU,
    MODE_BOTH
};

typedef struct {
    RunMode mode;
    int blockSize;          // 8, 16, or 32 (BxB thread block dimension)
    int numThreads;         // OpenMP threads (default: 8)
    int devId;              // CUDA device ID (default: 0)
    int saveOutput;         // Whether to write PNG outputs to disk
    int csvOutput;          // Output machine-readable CSV format
    char inputPath[512];    // Input image path
    char outputPath[512];   // Output grayscale PNG path
} Config;

typedef struct {
    float allocMs;
    float htodMs;
    float kernelMs;
    float dtohMs;
    float computeMs;        // HtoD + Kernel + DtoH
    float totalMs;          // Alloc + HtoD + Kernel + DtoH
    dim3 blockDim;
    dim3 gridDim;
} GpuTiming;

typedef struct {
    unsigned char *data;
    int width;
    int height;
    int channels;
    int is_stb_allocated;
} LoadedImage;