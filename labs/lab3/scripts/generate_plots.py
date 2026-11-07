#!/usr/bin/env python3
"""
generate_plots.py - Generates Publication-Quality Analytical Charts for Lab 3
Course: Massively Parallel Programming on GPUs - Lab 3: Naive, Tiled & Coarse-Grained Matrix Multiplication
Teacher: Donato D'Ambrosio

Charts Generated (Saved in reports/ as required by PDF Section 5.5):
1. plot1_gpu_block_sizes.png: Comparison of Naive (Point 4.1), Tiled (Point 4.2), and Coarse-Grained (Point 4.3) across block sizes (8x8, 16x16, 32x32)
2. plot2_cpu_vs_gpu_comparison.png: Comparison of all implementations (CPU Standard, CPU Optimized, Best Naive, Best Tiled, Best Coarse)
3. plot3_gflops_comparison.png: Computational throughput (GFLOP/s) across all implementations
"""

import os
import sys
import csv
import cairo

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_PYTHON_DIR = os.path.abspath(os.path.join(LAB_DIR, "..", "common", "python"))
if COMMON_PYTHON_DIR not in sys.path:
    sys.path.insert(0, COMMON_PYTHON_DIR)

from plot_utils import (
    create_surface,
    draw_title,
    draw_axes,
    draw_legend,
    save_surface,
    draw_bar,
    PALETTE,
)

REPORT_DIR = os.path.join(LAB_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

CSV_PATH = os.path.join(REPORT_DIR, "benchmark_results_averaged.csv")

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"[-] CSV file '{CSV_PATH}' not found. Please run average_benchmarks.py first.")
        sys.exit(1)

    data = {}
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cfg = row["Configuration"]
            data[cfg] = {
                "type": row.get("Type", "GPU"),
                "kernelType": row.get("KernelType", "GPU"),
                "blockSize": row.get("BlockSize", "-"),
                "tileNumber": int(row.get("TileNumber", 1)),
                "threads": int(row.get("Threads", 0)),
                "M": int(row.get("M", 2048)),
                "K": int(row.get("K", 1024)),
                "N": int(row.get("N", 512)),
                "kernelMean": float(row.get("KernelMeanMs", 0.0)),
                "kernelStd": float(row.get("KernelStdMs", 0.0)),
                "allocMean": float(row.get("AllocMeanMs", 0.0)),
                "totalMean": float(row.get("TotalMeanMs", 0.0)),
                "totalStd": float(row.get("TotalStdMs", 0.0)),
                "gflops": float(row.get("GFlops", 0.0)),
                "speedupVsStd": float(row.get("SpeedupVsStd", 1.0)),
                "speedupVsOpt": float(row.get("SpeedupVsOpt", 1.0)),
                "speedupVsNaive": float(row.get("SpeedupVsNaive", 1.0)),
                "status": row.get("Status", "PASS"),
            }
    return data

def draw_plot1_block_sizes(data):
    """
    Plot 1: Section 5.5 Requirement:
    'A bar plot for each CUDA implementation reporting the execution times for each block configuration'
    Compares Naive (Point 4.1), Tiled (Point 4.2), and Coarse-Grained (Point 4.3, using Best nt) across 8x8, 16x16, 32x32.
    """
    w, h = 980, 560
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: CUDA Execution Times across Block Configurations",
        "Comparison of Naive (Point 4.1), Tiled (Point 4.2), and Coarse-Grained (Point 4.3, Best nt) across Block Configurations",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 920, 460
    plot_w = right - left
    plot_h = bottom - top

    blocks = [8, 16, 32]
    
    # Helper to find best coarse configuration for a given block size
    def get_best_coarse(b):
        candidates = [k for k in data if k.startswith(f"CUDA Coarse {b}x{b}")]
        if not candidates:
            return None
        return min(candidates, key=lambda k: data[k]["kernelMean"])

    vals = []
    for b in blocks:
        kn = f"CUDA Naive {b}x{b}"
        kt = f"CUDA Tiled {b}x{b}"
        kc = get_best_coarse(b)
        if kn in data: vals.append(data[kn]["kernelMean"])
        if kt in data: vals.append(data[kt]["kernelMean"])
        if kc and kc in data: vals.append(data[kc]["kernelMean"])

    max_val = (max(vals) * 1.25) if vals else 100.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    # Legend
    legend_items = [
        ("Naive (Global Mem)", PALETTE["primary"]),
        ("Tiled (Shared Mem)", PALETTE["secondary"]),
        ("Coarse-Grained (Best nt)", PALETTE["accent"]),
    ]
    draw_legend(ctx, start_x=left + 160, y=85, items=legend_items, item_spacing=180.0)

    n_groups = len(blocks)
    col_w = plot_w / n_groups

    for i, b in enumerate(blocks):
        kn = f"CUDA Naive {b}x{b}"
        kt = f"CUDA Tiled {b}x{b}"
        kc = get_best_coarse(b)
        best_nt = data[kc]["tileNumber"] if kc and kc in data else 2
        cx = left + i * col_w + col_w / 2.0

        bar_w = 40.0
        gap = 8.0
        total_w = 3 * bar_w + 2 * gap
        start_x = cx - total_w / 2.0

        # Bar 1: Naive
        if kn in data:
            bh1 = (data[kn]["kernelMean"] / max_val) * plot_h
            by1 = bottom - bh1
            bx1 = start_x
            draw_bar(ctx, bx1, by1, bar_w, bh1, PALETTE["primary"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9)
            ctx.move_to(bx1 + 1, by1 - 6)
            ctx.show_text(f"{data[kn]['kernelMean']:.1f}")

        # Bar 2: Tiled
        if kt in data:
            bh2 = (data[kt]["kernelMean"] / max_val) * plot_h
            by2 = bottom - bh2
            bx2 = start_x + bar_w + gap
            draw_bar(ctx, bx2, by2, bar_w, bh2, PALETTE["secondary"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9)
            ctx.move_to(bx2 + 1, by2 - 6)
            ctx.show_text(f"{data[kt]['kernelMean']:.1f}")

        # Bar 3: Coarse (Best nt)
        if kc and kc in data:
            bh3 = (data[kc]["kernelMean"] / max_val) * plot_h
            by3 = bottom - bh3
            bx3 = start_x + 2 * (bar_w + gap)
            draw_bar(ctx, bx3, by3, bar_w, bh3, PALETTE["accent"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
            ctx.set_font_size(9)
            ctx.move_to(bx3 + 1, by3 - 6)
            ctx.show_text(f"{data[kc]['kernelMean']:.1f}")

        # Group label below axis
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

        if kn in data and kc and kc in data and data[kc]["kernelMean"] > 0:
            sp = data[kn]["kernelMean"] / data[kc]["kernelMean"]
            ctx.move_to(cx - 52, bottom + 57)
            ctx.show_text(f"Best nt={best_nt} ({sp:.2f}x speedup)")

    out_path = os.path.join(REPORT_DIR, "plot1_gpu_block_sizes.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 1: '{out_path}'")

def draw_plot2_comparison(data):
    """
    Plot 2: Section 5.5 Requirement:
    'A bar plot comparing the execution times of all implementations, including those of the CPU implementation.
     For the CUDA implementations, choose the minimum elapsed time among those obtained by varying the block configuration.'
    """
    w, h = 980, 560
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: Execution Time Comparison across All Implementations",
        "OpenMP CPU (Standard i-j-k vs Optimized i-k-j) vs Best Naive vs Best Tiled vs Best Coarse (M=2048, K=1024, N=512)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 920, 460
    plot_w = right - left
    plot_h = bottom - top

    # Find best GPU block configs
    naive_keys = [k for k in data.keys() if "CUDA Naive" in k]
    best_naive_key = min(naive_keys, key=lambda k: data[k]["kernelMean"]) if naive_keys else None

    tiled_keys = [k for k in data.keys() if "CUDA Tiled" in k]
    best_tiled_key = min(tiled_keys, key=lambda k: data[k]["kernelMean"]) if tiled_keys else None

    coarse_keys = [k for k in data.keys() if "CUDA Coarse" in k]
    best_coarse_key = min(coarse_keys, key=lambda k: data[k]["kernelMean"]) if coarse_keys else None

    groups = []
    if "CPU OpenMP (Standard)" in data:
        groups.append({
            "name": "CPU Baseline",
            "sub": "OpenMP Standard (i-j-k)",
            "val": data["CPU OpenMP (Standard)"]["kernelMean"],
            "color": PALETTE["red"],
            "gflops": data["CPU OpenMP (Standard)"]["gflops"]
        })
    if "CPU OpenMP (Optimized I-K-J)" in data:
        groups.append({
            "name": "CPU Optimized",
            "sub": "Loop Interchange (i-k-j)",
            "val": data["CPU OpenMP (Optimized I-K-J)"]["kernelMean"],
            "color": PALETTE["purple"],
            "gflops": data["CPU OpenMP (Optimized I-K-J)"]["gflops"]
        })
    if best_naive_key and best_naive_key in data:
        best_naive = data[best_naive_key]
        groups.append({
            "name": "CUDA Naive",
            "sub": f"Global Mem ({best_naive['blockSize']})",
            "val": best_naive["kernelMean"],
            "color": PALETTE["primary"],
            "gflops": best_naive["gflops"]
        })
    if best_tiled_key and best_tiled_key in data:
        best_tiled = data[best_tiled_key]
        groups.append({
            "name": "CUDA Tiled",
            "sub": f"Shared Mem ({best_tiled['blockSize']})",
            "val": best_tiled["kernelMean"],
            "color": PALETTE["secondary"],
            "gflops": best_tiled["gflops"]
        })
    if best_coarse_key and best_coarse_key in data:
        best_coarse = data[best_coarse_key]
        groups.append({
            "name": "CUDA Coarse",
            "sub": f"Shared nt={best_coarse['tileNumber']} ({best_coarse['blockSize']})",
            "val": best_coarse["kernelMean"],
            "color": PALETTE["accent"],
            "gflops": best_coarse["gflops"]
        })

    max_val = max(g["val"] for g in groups) * 1.25 if groups else 100.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    n_groups = len(groups)
    col_w = plot_w / n_groups

    for i, grp in enumerate(groups):
        cx = left + i * col_w + col_w / 2.0
        bar_w = 64.0
        bx = cx - bar_w / 2.0
        bh = (grp["val"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, grp["color"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(10)
        ctx.move_to(bx + 2, by - 8)
        ctx.show_text(f"{grp['val']:.2f} ms")

        ctx.move_to(cx - 36, bottom + 25)
        ctx.show_text(grp["name"])

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(9)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(cx - 46, bottom + 42)
        ctx.show_text(grp["sub"])

        ctx.move_to(cx - 26, bottom + 57)
        ctx.show_text(f"{grp['gflops']:.1f} GFLOP/s")

    out_path = os.path.join(REPORT_DIR, "plot2_cpu_vs_gpu_comparison.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 2: '{out_path}'")

def draw_plot3_gflops(data):
    """
    Plot 3: Computational Throughput (GFLOP/s) across all implementations.
    Higher is better.
    """
    w, h = 980, 560
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: Computational Throughput (GFLOP/s) Comparison",
        "Higher is better | Work = 2.147 GFLOP (Single-Precision Floating-Point, M=2048, K=1024, N=512)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 920, 460
    plot_w = right - left
    plot_h = bottom - top

    items = []
    if "CPU OpenMP (Standard)" in data:
        items.append({
            "name": "CPU Std",
            "sub": "8 cores",
            "gflops": data["CPU OpenMP (Standard)"]["gflops"],
            "color": PALETTE["red"]
        })
    if "CPU OpenMP (Optimized I-K-J)" in data:
        items.append({
            "name": "CPU Opt",
            "sub": "8 cores",
            "gflops": data["CPU OpenMP (Optimized I-K-J)"]["gflops"],
            "color": PALETTE["purple"]
        })
    for b in [8, 16, 32]:
        kn = f"CUDA Naive {b}x{b}"
        if kn in data:
            items.append({
                "name": f"Naive {b}",
                "sub": f"{b*b} th",
                "gflops": data[kn]["gflops"],
                "color": PALETTE["primary"]
            })
    for b in [8, 16, 32]:
        kt = f"CUDA Tiled {b}x{b}"
        if kt in data:
            items.append({
                "name": f"Tiled {b}",
                "sub": f"{b*b} th",
                "gflops": data[kt]["gflops"],
                "color": PALETTE["secondary"]
            })
    for (b, nt) in [(8, 2), (16, 2), (32, 2), (16, 4), (32, 4), (16, 8), (32, 8)]:
        kc = f"CUDA Coarse {b}x{b} (nt={nt})"
        if kc in data:
            items.append({
                "name": f"Crs {b}",
                "sub": f"nt={nt}",
                "gflops": data[kc]["gflops"],
                "color": PALETTE["accent"]
            })

    max_val = max(it["gflops"] for it in items) * 1.25 if items else 200.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.0f}", y_unit="GFLOP/s")

    n_items = len(items)
    col_w = plot_w / n_items

    for i, it in enumerate(items):
        cx = left + i * col_w + col_w / 2.0
        bar_w = 34.0
        bx = cx - bar_w / 2.0
        bh = (it["gflops"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, it["color"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(9)
        ctx.move_to(bx, by - 6)
        ctx.show_text(f"{it['gflops']:.0f}")

        ctx.move_to(cx - 20, bottom + 25)
        ctx.show_text(it["name"])

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(8)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(cx - 16, bottom + 40)
        ctx.show_text(it["sub"])

    out_path = os.path.join(REPORT_DIR, "plot3_gflops_comparison.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 3: '{out_path}'")

def main():
    print("[*] Loading benchmark dataset from CSV...")
    data = load_data()
    print(f"[+] Loaded {len(data)} configurations.")

    print("[*] Generating analytical plots in reports/ ...")
    draw_plot1_block_sizes(data)
    draw_plot2_comparison(data)
    draw_plot3_gflops(data)
    print("[+] All plots generated successfully!")

if __name__ == "__main__":
    main()
