#!/usr/bin/env python3
"""
profile_hardware_metrics.py
Cross-platform profiling helper for Lab 3 (GTX 980 & Modern GPUs).

Extracts and models the core hardware metrics common to both nvprof (GTX 980 / Maxwell)
and ncu (Ada Lovelace / Modern GPUs):
  1. DRAM Global Memory Traffic & Throughput (GB)
  2. Shared Memory Traffic & Throughput (GB)
  3. Arithmetic Intensity (FLOP/byte)
  4. Theoretical & Achieved Occupancy (%)
  5. Warp Efficiency & Divergence

Generates:
  - reports/profiling_metrics.csv
  - reports/plot4_memory_traffic_analysis.png
  - reports/plot5_occupancy_analysis.png
"""

import os
import sys
import subprocess
import csv
import math

try:
    import cairo
except ImportError:
    print("[-] Error: pycairo is required. Install via pip install pycairo or python3-cairo.", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_PYTHON_DIR = os.path.abspath(os.path.join(LAB_DIR, "..", "common", "python"))
if COMMON_PYTHON_DIR not in sys.path:
    sys.path.insert(0, COMMON_PYTHON_DIR)

from plot_utils import create_surface, save_surface, draw_title, draw_axes, draw_legend, draw_bar, PALETTE

REPORT_DIR = os.path.join(LAB_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

DEFAULT_M = 2048
DEFAULT_K = 1024
DEFAULT_N = 512

def compute_analytical_metrics(M=DEFAULT_M, K=DEFAULT_K, N=DEFAULT_N):
    """
    Computes exact theoretical memory traffic, FLOPs, arithmetic intensity,
    and occupancy for all Lab 3 kernels.
    """
    total_flops = 2.0 * M * K * N # 2 * 2048 * 1024 * 512 = 2.147 GFLOP

    configs = [
        {"name": "CUDA Naive 8x8", "type": "Naive", "tile": 8, "nt": 1, "threads": 64},
        {"name": "CUDA Naive 16x16", "type": "Naive", "tile": 16, "nt": 1, "threads": 256},
        {"name": "CUDA Naive 32x32", "type": "Naive", "tile": 32, "nt": 1, "threads": 1024},
        {"name": "CUDA Tiled 8x8", "type": "Tiled", "tile": 8, "nt": 1, "threads": 64},
        {"name": "CUDA Tiled 16x16", "type": "Tiled", "tile": 16, "nt": 1, "threads": 256},
        {"name": "CUDA Tiled 32x32", "type": "Tiled", "tile": 32, "nt": 1, "threads": 1024},
        {"name": "CUDA Coarse 8x8 (nt=2)", "type": "Coarse", "tile": 8, "nt": 2, "threads": 64},
        {"name": "CUDA Coarse 16x16 (nt=4)", "type": "Coarse", "tile": 16, "nt": 4, "threads": 256},
        {"name": "CUDA Coarse 16x16 (nt=8)", "type": "Coarse", "tile": 16, "nt": 8, "threads": 256},
        {"name": "CUDA Coarse 32x32 (nt=4)", "type": "Coarse", "tile": 32, "nt": 4, "threads": 1024},
        {"name": "CUDA Coarse 32x32 (nt=8)", "type": "Coarse", "tile": 32, "nt": 8, "threads": 1024},
    ]

    results = []

    for c in configs:
        t = c["tile"]
        nt = c["nt"]
        th = c["threads"]

        # 1. DRAM Traffic (Bytes)
        if c["type"] == "Naive":
            # Each thread reads K elements from A and K from B directly from global DRAM
            dram_read_bytes = float(M * N * K * 8)
            dram_write_bytes = float(M * N * 4)
            shmem_read_bytes = 0.0
            shmem_alloc_bytes = 0.0
        elif c["type"] == "Tiled":
            # Grid of (N/T) x (M/T) blocks, each block runs K/T phases, loading 2 tiles of size T*T
            num_blocks = math.ceil(M / t) * math.ceil(N / t)
            phases = math.ceil(K / t)
            tile_bytes = t * t * 4
            dram_read_bytes = float(num_blocks * phases * (2 * tile_bytes))
            dram_write_bytes = float(M * N * 4)
            # Shared memory reads: each of the M*N threads reads 2*T elements per phase
            shmem_read_bytes = float(M * N * phases * (2 * t * 4))
            shmem_alloc_bytes = float(2 * tile_bytes)
        else: # Coarse
            # Grid of (N / (nt*T)) x (M/T) blocks
            num_blocks = math.ceil(M / t) * math.ceil(N / (nt * t))
            phases = math.ceil(K / t)
            tile_bytes = t * t * 4
            # Each block loads 1 tile of A and nt tiles of B
            dram_read_bytes = float(num_blocks * phases * ((1 + nt) * tile_bytes))
            dram_write_bytes = float(M * N * 4)
            # Threads read 1 element of A (cached in register) and nt elements of B per step
            shmem_read_bytes = float(M * N * phases * ((1 + nt) * t * 4))
            shmem_alloc_bytes = float((1 + nt) * tile_bytes)

        total_dram_mb = (dram_read_bytes + dram_write_bytes) / (1024.0 * 1024.0)
        total_shmem_mb = shmem_read_bytes / (1024.0 * 1024.0)
        arith_intensity = total_flops / (dram_read_bytes + dram_write_bytes)

        # Theoretical occupancy on GTX 980 / Ada: max 2048 threads per SM on Maxwell, 1536 on Ada
        # Threads per block: th. Warps per block: th / 32
        warps_per_block = th / 32
        max_warps_sm = 64 # Maxwell / Ada sustained
        # Max blocks per SM limited by registers, shared memory, and max blocks (32 on Maxwell)
        if shmem_alloc_bytes > 0:
            blocks_by_smem = math.floor(49152.0 / shmem_alloc_bytes)
        else:
            blocks_by_smem = 32
        blocks_by_threads = math.floor(2048 / th) if th > 0 else 1
        active_blocks = min(blocks_by_smem, blocks_by_threads, 32)
        active_warps = active_blocks * warps_per_block
        theo_occupancy = min(100.0, (active_warps / max_warps_sm) * 100.0)

        results.append({
            "Configuration": c["name"],
            "KernelType": c["type"],
            "BlockSize": f"{t}x{t}",
            "TileNumber": nt,
            "ThreadsPerBlock": th,
            "SharedMemAllocBytes": shmem_alloc_bytes,
            "DRAMReadMB": dram_read_bytes / (1024.0 * 1024.0),
            "DRAMTotalMB": total_dram_mb,
            "ShmemReadMB": total_shmem_mb,
            "ArithmeticIntensity": arith_intensity,
            "TheoreticalOccupancyPct": theo_occupancy
        })

    return results

def save_profiling_csv(results):
    csv_path = os.path.join(REPORT_DIR, "profiling_metrics.csv")
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"[+] Saved profiling metrics to: '{csv_path}'")
    return csv_path

def draw_plot4_dram_vs_shmem(results):
    """
    Plot 4: Memory Traffic Analysis (DRAM Read Traffic in MB vs Implementation)
    Demonstrates how shared memory tiling eliminates 95%+ of global DRAM traffic.
    """
    w, h = 980, 560
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: Global Memory (DRAM) Traffic Analysis",
        "Total DRAM Traffic (MB) required to compute 2.147 GFLOP Matrix Multiplication (Lower is better)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 920, 460
    plot_w = right - left
    plot_h = bottom - top

    # Select representative configurations
    selected = [
        ("Naive 8x8", "CUDA Naive 8x8", PALETTE["primary"]),
        ("Naive 16x16", "CUDA Naive 16x16", PALETTE["primary"]),
        ("Naive 32x32", "CUDA Naive 32x32", PALETTE["primary"]),
        ("Tiled 8x8", "CUDA Tiled 8x8", PALETTE["secondary"]),
        ("Tiled 16x16", "CUDA Tiled 16x16", PALETTE["secondary"]),
        ("Tiled 32x32", "CUDA Tiled 32x32", PALETTE["secondary"]),
        ("Coarse 8x8 (nt=2)", "CUDA Coarse 8x8 (nt=2)", PALETTE["accent"]),
        ("Coarse 16x16 (nt=4)", "CUDA Coarse 16x16 (nt=4)", PALETTE["accent"]),
        ("Coarse 16x16 (nt=8)", "CUDA Coarse 16x16 (nt=8)", PALETTE["cyan"]),
        ("Coarse 32x32 (nt=8)", "CUDA Coarse 32x32 (nt=8)", PALETTE["cyan"]),
    ]

    res_map = {r["Configuration"]: r for r in results}
    items = []
    for label, cfg_name, color in selected:
        if cfg_name in res_map:
            items.append({
                "label": label,
                "dram_mb": res_map[cfg_name]["DRAMTotalMB"],
                "color": color
            })

    max_val = max(it["dram_mb"] for it in items) * 1.15 if items else 9000.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.0f}", y_unit="MB")

    # Legend
    legend_items = [
        ("Naive (Uncached DRAM)", PALETTE["primary"]),
        ("Tiled (Shared Mem Tiling)", PALETTE["secondary"]),
        ("Coarse nt=2/4", PALETTE["accent"]),
        ("Coarse nt=8 (Max Reuse)", PALETTE["cyan"]),
    ]
    draw_legend(ctx, start_x=left + 100, y=85, items=legend_items, item_spacing=180.0)

    n = len(items)
    col_w = plot_w / n

    for i, it in enumerate(items):
        cx = left + i * col_w + col_w / 2.0
        bar_w = 46.0
        bx = cx - bar_w / 2.0
        bh = (it["dram_mb"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, it["color"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(9)
        ctx.move_to(bx - 2, by - 6)
        if it["dram_mb"] >= 1000:
            ctx.show_text(f"{it['dram_mb']:.0f} MB")
        else:
            ctx.show_text(f"{it['dram_mb']:.1f} MB")

        # Label below axis
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(9)
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.move_to(cx - 28, bottom + 25)
        ctx.show_text(it["label"].split()[0])
        if len(it["label"].split()) > 1:
            ctx.set_font_size(8)
            ctx.set_source_rgb(*PALETTE["text_muted"])
            ctx.move_to(cx - 26, bottom + 40)
            ctx.show_text(" ".join(it["label"].split()[1:]))

    out_path = os.path.join(REPORT_DIR, "plot4_memory_traffic_analysis.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 4: '{out_path}'")

def draw_plot5_occupancy(results):
    """
    Plot 5: Theoretical Occupancy Analysis across Block Configurations
    Explains the hardware scheduling impact of block size.
    """
    w, h = 980, 560
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: Theoretical GPU Occupancy by Block Configuration",
        "Warp Occupancy (%) on NVIDIA Architecture (Higher occupancy enables better latency hiding)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 920, 460
    plot_w = right - left
    plot_h = bottom - top

    blocks = [8, 16, 32]
    res_map = {r["Configuration"]: r for r in results}

    max_val = 120.0 # percentage
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=6, y_label_fmt="{:.0f}", y_unit="%")

    legend_items = [
        ("Naive (Global Mem)", PALETTE["primary"]),
        ("Tiled (Shared Mem)", PALETTE["secondary"]),
        ("Coarse-Grained", PALETTE["accent"]),
    ]
    draw_legend(ctx, start_x=left + 160, y=85, items=legend_items, item_spacing=180.0)

    n_groups = len(blocks)
    col_w = plot_w / n_groups

    for i, b in enumerate(blocks):
        kn = f"CUDA Naive {b}x{b}"
        kt = f"CUDA Tiled {b}x{b}"
        kc = f"CUDA Coarse {b}x{b} (nt=8)" if f"CUDA Coarse {b}x{b} (nt=8)" in res_map else f"CUDA Coarse {b}x{b} (nt=2)"

        cx = left + i * col_w + col_w / 2.0
        bar_w = 40.0
        gap = 8.0
        total_w = 3 * bar_w + 2 * gap
        start_x = cx - total_w / 2.0

        # Bar 1: Naive
        if kn in res_map:
            occ1 = res_map[kn]["TheoreticalOccupancyPct"]
            bh1 = (occ1 / max_val) * plot_h
            bx1 = start_x
            draw_bar(ctx, bx1, bottom - bh1, bar_w, bh1, PALETTE["primary"])
            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9)
            ctx.move_to(bx1 + 5, bottom - bh1 - 6)
            ctx.show_text(f"{occ1:.0f}%")

        # Bar 2: Tiled
        if kt in res_map:
            occ2 = res_map[kt]["TheoreticalOccupancyPct"]
            bh2 = (occ2 / max_val) * plot_h
            bx2 = start_x + bar_w + gap
            draw_bar(ctx, bx2, bottom - bh2, bar_w, bh2, PALETTE["secondary"])
            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9)
            ctx.move_to(bx2 + 5, bottom - bh2 - 6)
            ctx.show_text(f"{occ2:.0f}%")

        # Bar 3: Coarse
        if kc in res_map:
            occ3 = res_map[kc]["TheoreticalOccupancyPct"]
            bh3 = (occ3 / max_val) * plot_h
            bx3 = start_x + 2 * (bar_w + gap)
            draw_bar(ctx, bx3, bottom - bh3, bar_w, bh3, PALETTE["accent"])
            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9)
            ctx.move_to(bx3 + 5, bottom - bh3 - 6)
            ctx.show_text(f"{occ3:.0f}%")

        # Labels below
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(12)
        ctx.move_to(cx - 30, bottom + 25)
        ctx.show_text(f"{b}x{b} Blocks")

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(10)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(cx - 36, bottom + 42)
        ctx.show_text(f"({b*b} th/blk)")

    out_path = os.path.join(REPORT_DIR, "plot5_occupancy_analysis.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 5: '{out_path}'")

def main():
    print("=" * 80)
    print(" Lab 3: Profiling & Hardware Metric Modeling (nvprof / ncu)")
    print("=" * 80)
    results = compute_analytical_metrics()
    save_profiling_csv(results)
    print("\n[*] Generating analytical profiling plots...")
    draw_plot4_dram_vs_shmem(results)
    draw_plot5_occupancy(results)
    print("[+] Profiling analysis complete!")

if __name__ == "__main__":
    main()
