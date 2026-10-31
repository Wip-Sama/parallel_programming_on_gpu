# Common Utilities for GPU Programming Labs

This directory provides generalized, reusable utilities shared across all GPU programming laboratories (`lab1`, `lab2`, and future labs).

---

## 1. Directory Structure

```
common/
├── include/
│   ├── cuda_common.h       # Robust CUDA error checking, timers, and device introspection
│   ├── stb_image.h         # Sean Barrett's header-only image loading library
│   └── stb_image_write.h   # Header-only image writing library (PNG output)
├── python/
│   ├── __init__.py         # Python package entry point
│   ├── plot_utils.py       # Reusable Pycairo chart generation (grouped bars, breakdown, axes, legends)
│   ├── verify_utils.py     # Python-level verification (MD5 checksums, bit-exact mismatch counting)
│   └── bench_utils.py      # Statistical aggregation (mean, std, min, max) and CSV I/O helpers
├── common.h                # Master C++ convenience include header
└── README.md               # This documentation
```

---

## 2. C++ / CUDA Utilities (`common/include/cuda_common.h`)

All CUDA/C++ code includes `cuda_common.h` (`-I../common/include -I../../common/include`):

- **`CUDA_CHECK(call)`**: Macro that wraps any CUDA API call, checking for errors with line number, file name, and exact error string on failure.
- **`get_wall_time_ms()`**: High-resolution monotonic clock timer returning elapsed time in milliseconds.
- **`CudaTimer`**: Lightweight RAII / struct wrapper for recording and synchronizing CUDA events across streams.
- **`print_cuda_device_info(int devId)`**: Formatted GPU architecture and hardware specifications summary.

---

## 3. Python Utilities (`common/python/`)

Complex analysis, multi-run statistical sweeps, checksums, image verification, and graph plotting are handled in Python rather than cluttering CUDA C++ source code.

### Verification (`verify_utils.py`)
```python
from common.python import verify_image_files, compute_file_md5

# Computes MD5 checksums and counts byte/pixel mismatches:
res = verify_image_files("output_cpu.png", "output_gpu.png")
print("Matched:", res["matched"], "Mismatches:", res["mismatches"])
```

### Graph Plotting (`plot_utils.py`)
```python
from common.python import create_surface, draw_title, draw_axes, draw_grouped_bars, save_surface, PALETTE

surface, ctx = create_surface(900, 550)
draw_title(ctx, "Experiment Title", "Descriptive subtitle")
# Render publication-quality charts using Pycairo
save_surface(surface, "reports/plot.png")
```

### Benchmarking & Statistics (`bench_utils.py`)
```python
from common.python import aggregate_metrics, save_csv, load_csv

stats = aggregate_metrics([1.2, 1.4, 1.1, 1.3])
print(f"Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
```

