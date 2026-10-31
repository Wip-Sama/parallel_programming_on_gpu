#ifndef LAB2_COMMON_H
#define LAB2_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <omp.h>
#include <cuda_runtime.h>

// STB Image Libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Common CUDA utilities (error checking, timers, device info)
#include "cuda_common.h"

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

// ---------------------------------------------------------------------------
// Uncompressed TIFF Loader (Bruce Lindbloom's DeltaE_8bit_gamma2.2.tif)
// ---------------------------------------------------------------------------
static inline unsigned char* load_uncompressed_tiff_rgb(const char* filename, int* out_w, int* out_h, int* out_channels) {
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

        if (tag == 0x0100) {
            width = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0101) {
            height = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0103) {
            compression = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0115) {
            samples_per_pixel = (type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8);
        } else if (tag == 0x0111) {
            strip_offsets_count = count;
            strip_offsets_offset = (count == 1) ? ((type == 3) ? TIFF_R16(entry + 8) : TIFF_R32(entry + 8)) : TIFF_R32(entry + 8);
        } else if (tag == 0x0117) {
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

// ---------------------------------------------------------------------------
// Image Management Helpers
// ---------------------------------------------------------------------------
static inline LoadedImage load_image_from_file(const char *path) {
    LoadedImage img;
    memset(&img, 0, sizeof(img));

    // Try stb_image first (supports PNG, JPG, BMP, PPM, etc.)
    img.data = stbi_load(path, &img.width, &img.height, &img.channels, 3);
    if (img.data) {
        img.channels = 3;
        img.is_stb_allocated = 1;
        return img;
    }

    // Try uncompressed TIFF fallback
    img.data = load_uncompressed_tiff_rgb(path, &img.width, &img.height, &img.channels);
    if (img.data) {
        img.channels = 3;
        img.is_stb_allocated = 0;
        return img;
    }

    return img;
}

static inline LoadedImage load_image(const char *path) {
    LoadedImage img = load_image_from_file(path);
    if (!img.data) {
        char alt_path[600];
        snprintf(alt_path, sizeof(alt_path), "labs/lab2/%s", path);
        img = load_image_from_file(alt_path);
    }
    if (!img.data) {
        img = load_image_from_file("labs/lab2/DeltaE_8bit_gamma2.2.tif");
    }
    if (!img.data) {
        img = load_image_from_file("DeltaE_8bit_gamma2.2.tif");
    }
    return img;
}

static inline void free_image(LoadedImage *img) {
    if (img && img->data) {
        if (img->is_stb_allocated) {
            stbi_image_free(img->data);
        } else {
            free(img->data);
        }
        img->data = NULL;
    }
}

static inline void save_outputs(const Config &cfg, int width, int height,
                                const unsigned char *h_gray_cpu,
                                const unsigned char *h_gray_gpu) {
    if (!cfg.saveOutput) return;

    if (cfg.mode == MODE_CPU && h_gray_cpu) {
        stbi_write_png(cfg.outputPath, width, height, 1, h_gray_cpu, width);
    } else if (cfg.mode == MODE_GPU && h_gray_gpu) {
        stbi_write_png(cfg.outputPath, width, height, 1, h_gray_gpu, width);
    } else if (cfg.mode == MODE_BOTH) {
        if (h_gray_cpu) stbi_write_png("output_cpu.png", width, height, 1, h_gray_cpu, width);
        if (h_gray_gpu) {
            stbi_write_png("output_gpu.png", width, height, 1, h_gray_gpu, width);
            stbi_write_png(cfg.outputPath, width, height, 1, h_gray_gpu, width);
        }
    }
}

// ---------------------------------------------------------------------------
// CLI Argument Parsing
// ---------------------------------------------------------------------------
static inline void print_usage(const char *prog) {
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
    printf("  --no-save        : Do not write PNG to disk (for pure benchmarking)\n");
    printf("  --csv            : Output machine-readable CSV format\n");
    printf("  -h, --help       : Show this help message and exit\n");
}

static inline Config parse_args(int argc, char **argv) {
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
            const char *m = argv[++i];
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
            if (pos_idx == 0) {
                if (strcasecmp(argv[i], "cpu") == 0) cfg.mode = MODE_CPU;
                else if (strcasecmp(argv[i], "gpu") == 0) cfg.mode = MODE_GPU;
                else if (strcasecmp(argv[i], "both") == 0) cfg.mode = MODE_BOTH;
                else if (atoi(argv[i]) == 8 || atoi(argv[i]) == 16 || atoi(argv[i]) == 32) cfg.blockSize = atoi(argv[i]);
                else strncpy(cfg.inputPath, argv[i], sizeof(cfg.inputPath) - 1);
            } else if (pos_idx == 1) {
                if (atoi(argv[i]) == 8 || atoi(argv[i]) == 16 || atoi(argv[i]) == 32) cfg.blockSize = atoi(argv[i]);
                else strncpy(cfg.inputPath, argv[i], sizeof(cfg.inputPath) - 1);
            } else if (pos_idx == 2) {
                strncpy(cfg.inputPath, argv[i], sizeof(cfg.inputPath) - 1);
            }
            pos_idx++;
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Results Printing (Terminal & CSV)
// ---------------------------------------------------------------------------
static inline void print_results(const Config &cfg, int width, int height,
                                 double cpuTimeMs, const GpuTiming &gpuTiming) {
    if (cfg.csvOutput) {
        if (cfg.mode == MODE_CPU) {
            printf("CSV,cpu,%d,%d,%d,%.4f\n", cfg.numThreads, width, height, cpuTimeMs);
        } else if (cfg.mode == MODE_GPU) {
            printf("CSV,gpu,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   cfg.blockSize, gpuTiming.gridDim.x, gpuTiming.gridDim.y, width, height,
                   gpuTiming.allocMs, gpuTiming.htodMs, gpuTiming.kernelMs, gpuTiming.dtohMs,
                   gpuTiming.computeMs, gpuTiming.totalMs);
        } else {
            printf("CSV,both,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                   cfg.blockSize, cfg.numThreads, width, height,
                   cpuTimeMs, gpuTiming.allocMs, gpuTiming.htodMs, gpuTiming.kernelMs, gpuTiming.dtohMs,
                   gpuTiming.computeMs, gpuTiming.totalMs);
        }
    } else {
        printf("======================================================================\n");
        printf(" Lab 2: Color-to-Gray Filter\n");
        printf(" Image Dimensions : %d x %d (%zu pixels)\n", width, height, (size_t)width * height);
        if (cfg.mode == MODE_CPU || cfg.mode == MODE_BOTH) {
            printf(" CPU Time (OpenMP): %8.4f ms (%d threads)\n", cpuTimeMs, cfg.numThreads);
        }
        if (cfg.mode == MODE_GPU || cfg.mode == MODE_BOTH) {
            printf(" GPU Block Dim    : %d x %d\n", gpuTiming.blockDim.x, gpuTiming.blockDim.y);
            printf(" GPU Grid Dim     : %d x %d\n", gpuTiming.gridDim.x, gpuTiming.gridDim.y);
            printf(" cudaMalloc       : %8.4f ms\n", gpuTiming.allocMs);
            printf(" HtoD Transfer    : %8.4f ms\n", gpuTiming.htodMs);
            printf(" Kernel Exec      : %8.4f ms\n", gpuTiming.kernelMs);
            printf(" DtoH Transfer    : %8.4f ms\n", gpuTiming.dtohMs);
            printf(" Compute Time     : %8.4f ms (HtoD + Kernel + DtoH)\n", gpuTiming.computeMs);
            printf(" Total GPU Time   : %8.4f ms (Alloc + Compute)\n", gpuTiming.totalMs);
        }
        printf("======================================================================\n");
    }
}

// ---------------------------------------------------------------------------
// Common Driver Pipeline
// ---------------------------------------------------------------------------
typedef void (*CpuFilterFunc)(const unsigned char *rgb, unsigned char *gray,
                              int width, int height, int num_threads);
typedef void (*GpuFilterFunc)(const unsigned char *h_rgb, unsigned char *h_gray,
                              int width, int height, int block_dim_size, GpuTiming *timing);

static inline int run_lab2_driver(int argc, char **argv,
                                  CpuFilterFunc cpu_fn,
                                  GpuFilterFunc gpu_fn) {
    Config cfg = parse_args(argc, argv);

    if (cfg.mode == MODE_GPU || cfg.mode == MODE_BOTH) {
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

    LoadedImage img = load_image(cfg.inputPath);
    if (!img.data) {
        fprintf(stderr, "Error: Failed to load image '%s'\n", cfg.inputPath);
        return EXIT_FAILURE;
    }

    size_t gray_bytes = (size_t)img.width * img.height;
    unsigned char *h_gray_cpu = (cfg.mode != MODE_GPU) ? (unsigned char*)malloc(gray_bytes) : NULL;
    unsigned char *h_gray_gpu = (cfg.mode != MODE_CPU) ? (unsigned char*)malloc(gray_bytes) : NULL;

    double cpuTimeMs = 0.0;
    if (h_gray_cpu && cpu_fn) {
        double t0 = omp_get_wtime();
        cpu_fn(img.data, h_gray_cpu, img.width, img.height, cfg.numThreads);
        cpuTimeMs = (omp_get_wtime() - t0) * 1000.0;
    }

    GpuTiming gpuTiming;
    memset(&gpuTiming, 0, sizeof(gpuTiming));
    if (h_gray_gpu && gpu_fn) {
        gpu_fn(img.data, h_gray_gpu, img.width, img.height, cfg.blockSize, &gpuTiming);
    }

    save_outputs(cfg, img.width, img.height, h_gray_cpu, h_gray_gpu);
    print_results(cfg, img.width, img.height, cpuTimeMs, gpuTiming);

    if (h_gray_cpu) free(h_gray_cpu);
    if (h_gray_gpu) free(h_gray_gpu);
    free_image(&img);

    return EXIT_SUCCESS;
}

#endif // LAB2_COMMON_H
