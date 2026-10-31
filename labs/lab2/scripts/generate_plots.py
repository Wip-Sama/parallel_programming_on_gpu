#!/usr/bin/env python3
"""
generate_plots.py - Generates analytical charts for Lab 2
Course: Massively Parallel Programming on GPUs - Lab 2: Color-to-Gray Filter
Teacher: Donato D'Ambrosio

Plots Generated (saved in reports/):
1. plot1_cpu_vs_gpu_elapsed.png: CPU vs GPU elapsed times across block configurations.
2. plot2_fastest_gpu_breakdown.png: Memory allocation, transfers, and kernel breakdown for fastest GPU.
3. plot3_gpu_speedup.png: Pure kernel and end-to-end speedups relative to OpenMP baseline.
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
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(LAB_DIR, "benchmark_results_averaged.csv")

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
                "blockSize": row["BlockSize"],
                "threads": int(row["Threads"]),
                "kernelMean": float(row["KernelMeanMs"]),
                "kernelStd": float(row["KernelStdMs"]),
                "allocMean": float(row["AllocMeanMs"]),
                "htodMean": float(row["HtoDMeanMs"]),
                "dtohMean": float(row["DtoHMeanMs"]),
                "totalGpuMean": float(row["TotalGpuMeanMs"]),
                "totalGpuStd": float(row["TotalGpuStdMs"]),
                "speedupKernel": float(row["SpeedupKernel"]),
                "speedupTotal": float(row["SpeedupTotal"]),
            }
    return data

def draw_plot1(data):
    """Plot 1: CPU vs GPU elapsed times across tested block sizes (8x8, 16x16, 32x32)"""
    w, h = 900, 550
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 2: CPU vs GPU Elapsed Execution Times",
        "Comparing OpenMP (8 threads) and CUDA GPU across Block Configurations (8x8, 16x16, 32x32)",
        x=50, y=45
    )

    left, top, right, bottom = 100, 110, 840, 470
    plot_w = right - left
    plot_h = bottom - top

    categories = [
        {"name": "CPU OpenMP\n(8 Cores)", "kernel": data["CPU_OpenMP"]["kernelMean"], "total": None, "color_k": PALETTE["primary"]},
        {"name": "CUDA GPU\n(8x8)", "kernel": data["CUDA_8x8"]["kernelMean"], "total": data["CUDA_8x8"]["totalGpuMean"], "color_k": PALETTE["secondary"], "color_t": PALETTE["accent"]},
        {"name": "CUDA GPU\n(16x16)", "kernel": data["CUDA_16x16"]["kernelMean"], "total": data["CUDA_16x16"]["totalGpuMean"], "color_k": PALETTE["secondary"], "color_t": PALETTE["accent"]},
        {"name": "CUDA GPU\n(32x32)", "kernel": data["CUDA_32x32"]["kernelMean"], "total": data["CUDA_32x32"]["totalGpuMean"], "color_k": PALETTE["secondary"], "color_t": PALETTE["accent"]},
    ]

    max_val = max(c["total"] if c["total"] else c["kernel"] for c in categories) * 1.25
    if max_val <= 0: max_val = 10.0

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    # Legend
    legend_items = [
        ("CPU OpenMP", PALETTE["primary"]),
        ("GPU Kernel", PALETTE["secondary"]),
        ("Total GPU (Transfers + Kernel)", PALETTE["accent"]),
    ]
    draw_legend(ctx, start_x=450, y=85, items=legend_items, item_spacing=130.0)

    n_cat = len(categories)
    cat_width = plot_w / n_cat
    bar_width = 45.0

    for i, cat in enumerate(categories):
        center_x = left + i * cat_width + cat_width / 2.0

        if cat["total"] is None:
            bx = center_x - bar_width / 2.0
            bh = (cat["kernel"] / max_val) * plot_h
            by = bottom - bh
            draw_bar(ctx, bx, by, bar_width, bh, cat["color_k"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(12)
            ctx.move_to(bx - 5, by - 8)
            ctx.show_text(f"{cat['kernel']:.2f} ms")
        else:
            bx1 = center_x - bar_width - 4.0
            bh1 = (cat["kernel"] / max_val) * plot_h
            by1 = bottom - bh1
            draw_bar(ctx, bx1, by1, bar_width, bh1, cat["color_k"])

            bx2 = center_x + 4.0
            bh2 = (cat["total"] / max_val) * plot_h
            by2 = bottom - bh2
            draw_bar(ctx, bx2, by2, bar_width, bh2, cat["color_t"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(11)
            ctx.move_to(bx1 - 2, by1 - 6)
            ctx.show_text(f"{cat['kernel']:.2f}")

            ctx.move_to(bx2 - 2, by2 - 6)
            ctx.show_text(f"{cat['total']:.2f}")

        lines = cat["name"].split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(12)
        ctx.move_to(center_x - 35, bottom + 20)
        ctx.show_text(lines[0])
        if len(lines) > 1:
            ctx.move_to(center_x - 30, bottom + 36)
            ctx.show_text(lines[1])

    png_path = os.path.join(REPORT_DIR, "plot1_cpu_vs_gpu_elapsed.png")
    save_surface(surface, png_path)

def draw_plot2(data):
    """Plot 2: Breakdown of costs for fastest GPU execution (Alloc, HtoD, Kernel, DtoH)"""
    w, h = 850, 520
    surface, ctx = create_surface(w, h)

    fastest_k = min(["CUDA_8x8", "CUDA_16x16", "CUDA_32x32"], key=lambda k: data[k]["kernelMean"])
    fg = data[fastest_k]
    cfg_name = fastest_k.replace("CUDA_", "")

    draw_title(
        ctx,
        f"Lab 2: Fastest GPU Execution Timing Breakdown ({cfg_name})",
        "Breakdown of Memory Allocation, Host-to-Device Transfer, Kernel Execution, and Device-to-Host Transfer",
        x=50, y=45
    )

    left, top, right, bottom = 120, 110, 780, 440
    plot_w = right - left
    plot_h = bottom - top

    stages = [
        {"name": "Allocation / Init\n(cudaMalloc)", "val": fg["allocMean"], "color": PALETTE["gray"]},
        {"name": "HtoD Transfer\n(18 MB RGB)", "val": fg["htodMean"], "color": PALETTE["accent"]},
        {"name": "Kernel Execution\n(Color-to-Gray)", "val": fg["kernelMean"], "color": PALETTE["secondary"]},
        {"name": "DtoH Transfer\n(6 MB Gray)", "val": fg["dtohMean"], "color": PALETTE["cyan"]},
    ]

    max_val = max(s["val"] for s in stages) * 1.35
    if max_val <= 0: max_val = 5.0

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.2f}", y_unit="ms")

    n_stages = len(stages)
    col_w = plot_w / n_stages
    bar_w = 75.0
    tot_time = sum(s["val"] for s in stages)

    for i, s in enumerate(stages):
        cx = left + i * col_w + col_w / 2.0
        bx = cx - bar_w / 2.0
        bh = (s["val"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, s["color"])

        pct = (s["val"] / tot_time) * 100 if tot_time > 0 else 0
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(12)
        ctx.move_to(bx + 10, by - 22)
        ctx.show_text(f"{s['val']:.3f} ms")

        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.set_font_size(11)
        ctx.move_to(bx + 16, by - 7)
        ctx.show_text(f"({pct:.1f}%)")

        lines = s["name"].split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.set_font_size(11)
        ctx.move_to(cx - 40, bottom + 20)
        ctx.show_text(lines[0])
        if len(lines) > 1:
            ctx.move_to(cx - 35, bottom + 35)
            ctx.show_text(lines[1])

    png_path = os.path.join(REPORT_DIR, "plot2_fastest_gpu_breakdown.png")
    save_surface(surface, png_path)

def draw_plot3(data):
    """Plot 3: Speedup achieved by each CUDA execution with respect to OpenMP"""
    w, h = 850, 520
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 2: Speedup vs OpenMP CPU Baseline (8 Cores)",
        "Comparing Pure Kernel Speedup and End-to-End Speedup (including PCIe Data Transfers)",
        x=50, y=45
    )

    left, top, right, bottom = 100, 110, 780, 440
    plot_w = right - left
    plot_h = bottom - top

    items = [
        {"name": "CUDA GPU (8x8)\n64 Threads/Block", "sp_k": data["CUDA_8x8"]["speedupKernel"], "sp_t": data["CUDA_8x8"]["speedupTotal"]},
        {"name": "CUDA GPU (16x16)\n256 Threads/Block", "sp_k": data["CUDA_16x16"]["speedupKernel"], "sp_t": data["CUDA_16x16"]["speedupTotal"]},
        {"name": "CUDA GPU (32x32)\n1024 Threads/Block", "sp_k": data["CUDA_32x32"]["speedupKernel"], "sp_t": data["CUDA_32x32"]["speedupTotal"]},
    ]

    max_val = max(it["sp_k"] for it in items) * 1.3
    if max_val < 2.0: max_val = 2.5

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="x")

    # Baseline 1.0x line
    y_1x = bottom - (1.0 / max_val) * plot_h
    ctx.set_source_rgb(*PALETTE["red"])
    ctx.set_line_width(1.5)
    ctx.set_dash([4.0, 4.0])
    ctx.move_to(left, y_1x)
    ctx.line_to(right, y_1x)
    ctx.stroke()
    ctx.set_dash([])
    ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
    ctx.set_font_size(10)
    ctx.move_to(right - 105, y_1x - 5)
    ctx.show_text("CPU Baseline (1.00x)")

    # Legend
    legend_items = [
        ("Pure Kernel Speedup", PALETTE["secondary"]),
        ("Total GPU Speedup (w/ PCIe)", PALETTE["accent"]),
    ]
    draw_legend(ctx, start_x=450, y=85, items=legend_items, item_spacing=180.0)

    n_items = len(items)
    col_w = plot_w / n_items
    bar_w = 50.0

    for i, it in enumerate(items):
        cx = left + i * col_w + col_w / 2.0

        bx1 = cx - bar_w - 6.0
        bh1 = (it["sp_k"] / max_val) * plot_h
        by1 = bottom - bh1
        draw_bar(ctx, bx1, by1, bar_w, bh1, PALETTE["secondary"])

        bx2 = cx + 6.0
        bh2 = (it["sp_t"] / max_val) * plot_h
        by2 = bottom - bh2
        draw_bar(ctx, bx2, by2, bar_w, bh2, PALETTE["accent"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(12)
        ctx.move_to(bx1 + 5, by1 - 8)
        ctx.show_text(f"{it['sp_k']:.2f}x")

        ctx.move_to(bx2 + 5, by2 - 8)
        ctx.show_text(f"{it['sp_t']:.2f}x")

        lines = it["name"].split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.set_font_size(11)
        ctx.move_to(cx - 50, bottom + 20)
        ctx.show_text(lines[0])
        if len(lines) > 1:
            ctx.move_to(cx - 45, bottom + 35)
            ctx.show_text(lines[1])

    png_path = os.path.join(REPORT_DIR, "plot3_gpu_speedup.png")
    save_surface(surface, png_path)

def main():
    data = load_data()
    draw_plot1(data)
    draw_plot2(data)
    draw_plot3(data)
    print(f"[+] All 3 analytical plots generated successfully in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
