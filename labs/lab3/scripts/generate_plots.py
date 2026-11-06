#!/usr/bin/env python3
"""
generate_plots.py - Generates Publication-Quality Analytical Charts for Lab 3
Course: Massively Parallel Programming on GPUs - Lab 3: Naive & Tiled Matrix Multiplication
Teacher: Donato D'Ambrosio

Charts Generated (Saved in reports/ as required by PDF Section 5.5):
1. plot1_gpu_block_sizes.png: Execution times across CUDA block configurations (8x8, 16x16, 32x32)
2. plot2_cpu_vs_gpu_comparison.png: Comparison of all implementations (CPU Standard, CPU Optimized, and Best CUDA Naive)
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
                "blockSize": row.get("BlockSize", "-"),
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
                "status": row.get("Status", "PASS"),
            }
    return data

def draw_plot1_block_sizes(data):
    """
    Plot 1: Section 5.5 Requirement:
    'A bar plot for each CUDA implementation reporting the execution times for each block configuration'
    Compares Pure Kernel Time vs Total GPU Time for 8x8, 16x16, 32x32.
    """
    w, h = 900, 520
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: CUDA Naive Matrix Multiplication across Block Configurations",
        "Execution Times for Global Memory Naive Kernel across 8x8, 16x16, and 32x32 Thread Blocks (M=2048, K=1024, N=512)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 840, 440
    plot_w = right - left
    plot_h = bottom - top

    blocks = [8, 16, 32]
    vals = []
    for b in blocks:
        k = f"CUDA Naive {b}x{b}"
        if k in data:
            vals.extend([data[k]["kernelMean"], data[k]["totalMean"]])

    max_val = (max(vals) * 1.22) if vals else 100.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    # Legend
    legend_items = [
        ("Pure Kernel Time", PALETTE["primary"]),
        ("Total GPU Time (Alloc + Kernel)", PALETTE["purple"]),
    ]
    draw_legend(ctx, start_x=left + 180, y=85, items=legend_items, item_spacing=240.0)

    n_groups = len(blocks)
    col_w = plot_w / n_groups

    for i, b in enumerate(blocks):
        k = f"CUDA Naive {b}x{b}"
        if k not in data:
            continue

        item = data[k]
        cx = left + i * col_w + col_w / 2.0

        bar_w = 48.0
        gap = 12.0
        total_w = 2 * bar_w + gap
        start_x = cx - total_w / 2.0

        # Bar 1: Pure Kernel Time
        bh1 = (item["kernelMean"] / max_val) * plot_h
        by1 = bottom - bh1
        bx1 = start_x
        draw_bar(ctx, bx1, by1, bar_w, bh1, PALETTE["primary"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(10)
        ctx.move_to(bx1 + 2, by1 - 6)
        ctx.show_text(f"{item['kernelMean']:.1f} ms")

        # Bar 2: Total Time
        bh2 = (item["totalMean"] / max_val) * plot_h
        by2 = bottom - bh2
        bx2 = start_x + bar_w + gap
        draw_bar(ctx, bx2, by2, bar_w, bh2, PALETTE["purple"])

        ctx.move_to(bx2 + 2, by2 - 6)
        ctx.show_text(f"{item['totalMean']:.1f} ms")

        # Group label below axis
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(12)
        ctx.move_to(cx - 30, bottom + 25)
        ctx.show_text(f"{b}x{b} Blocks")

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(10)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(cx - 38, bottom + 42)
        ctx.show_text(f"({b*b} threads/block)")

        ctx.move_to(cx - 35, bottom + 57)
        ctx.show_text(f"{item['gflops']:.1f} GFLOP/s")

    out_path = os.path.join(REPORT_DIR, "plot1_gpu_block_sizes.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 1: '{out_path}'")

def draw_plot2_comparison(data):
    """
    Plot 2: Section 5.5 Requirement:
    'A bar plot comparing the execution times of all implementations, including those of the CPU implementation.
     For the CUDA implementations, choose the minimum elapsed time among those obtained by varying the block configuration.'
    """
    w, h = 900, 520
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: Execution Time Comparison across All Implementations",
        "OpenMP CPU Baseline (Standard i-j-k vs Optimized i-k-j) vs Best CUDA Naive Kernel (M=2048, K=1024, N=512)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 840, 440
    plot_w = right - left
    plot_h = bottom - top

    # Find best GPU block config
    gpu_configs = [k for k in data.keys() if "CUDA Naive" in k]
    best_gpu_key = min(gpu_configs, key=lambda k: data[k]["kernelMean"]) if gpu_configs else None

    groups = []
    if "CPU OpenMP (Standard)" in data:
        groups.append({
            "name": "CPU Baseline",
            "sub": "OpenMP Standard (i-j-k)",
            "threads": "8 cores",
            "val": data["CPU OpenMP (Standard)"]["kernelMean"],
            "color": PALETTE["red"],
            "gflops": data["CPU OpenMP (Standard)"]["gflops"]
        })
    if "CPU OpenMP (Optimized I-K-J)" in data:
        groups.append({
            "name": "CPU Optimized",
            "sub": "Loop Interchange (i-k-j)",
            "threads": "8 cores",
            "val": data["CPU OpenMP (Optimized I-K-J)"]["kernelMean"],
            "color": PALETTE["secondary"],
            "gflops": data["CPU OpenMP (Optimized I-K-J)"]["gflops"]
        })
    if best_gpu_key and best_gpu_key in data:
        best_item = data[best_gpu_key]
        groups.append({
            "name": "CUDA Naive (Best)",
            "sub": f"Global Memory ({best_item['blockSize']})",
            "threads": f"{best_item['threads']} threads/blk",
            "val": best_item["kernelMean"],
            "color": PALETTE["primary"],
            "gflops": best_item["gflops"]
        })

    max_val = max(g["val"] for g in groups) * 1.25 if groups else 100.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    n_groups = len(groups)
    col_w = plot_w / n_groups

    for i, grp in enumerate(groups):
        cx = left + i * col_w + col_w / 2.0
        bar_w = 75.0
        bx = cx - bar_w / 2.0
        bh = (grp["val"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, grp["color"])

        # Value label on top of bar
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(11)
        ctx.move_to(bx + 4, by - 8)
        ctx.show_text(f"{grp['val']:.2f} ms")

        # Labels below axis
        ctx.move_to(cx - 40, bottom + 25)
        ctx.show_text(grp["name"])

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(10)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(cx - 55, bottom + 42)
        ctx.show_text(grp["sub"])

        ctx.move_to(cx - 30, bottom + 57)
        ctx.show_text(f"{grp['gflops']:.1f} GFLOP/s")

    out_path = os.path.join(REPORT_DIR, "plot2_cpu_vs_gpu_comparison.png")
    save_surface(surface, out_path)
    print(f"  [+] Generated Plot 2: '{out_path}'")

def draw_plot3_gflops(data):
    """
    Plot 3: Computational Throughput (GFLOP/s) across all implementations.
    Higher is better.
    """
    w, h = 900, 520
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 3: Computational Throughput (GFLOP/s) Comparison",
        "Higher is better | Total Work = 2.147 GFLOP (Single-Precision Floating-Point, M=2048, K=1024, N=512)",
        x=50, y=45
    )

    left, top, right, bottom = 80, 115, 840, 440
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
            "color": PALETTE["secondary"]
        })
    for b in [8, 16, 32]:
        k = f"CUDA Naive {b}x{b}"
        if k in data:
            items.append({
                "name": f"CUDA {b}x{b}",
                "sub": f"{b*b} th/blk",
                "gflops": data[k]["gflops"],
                "color": PALETTE["primary"]
            })

    max_val = max(it["gflops"] for it in items) * 1.25 if items else 200.0
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.0f}", y_unit="GFLOP/s")

    n_items = len(items)
    col_w = plot_w / n_items

    for i, it in enumerate(items):
        cx = left + i * col_w + col_w / 2.0
        bar_w = 60.0
        bx = cx - bar_w / 2.0
        bh = (it["gflops"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, it["color"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(11)
        ctx.move_to(bx + 4, by - 8)
        ctx.show_text(f"{it['gflops']:.1f}")

        ctx.move_to(cx - 30, bottom + 25)
        ctx.show_text(it["name"])

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(10)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(cx - 24, bottom + 42)
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
