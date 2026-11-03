#!/usr/bin/env python3
"""
generate_plots.py - Generates analytical charts for Lab 2
Course: Massively Parallel Programming on GPUs - Lab 2: Color-to-Gray Filter
Teacher: Donato D'Ambrosio

Plots Generated (saved in reports/):
1. plot1_cpu_vs_gpu_elapsed.png: CPU vs GPU elapsed times across block configurations (Standard vs Unified Memory).
2. plot2_fastest_gpu_breakdown.png: Memory allocation, transfers, and kernel breakdown for fastest GPU execution (Standard vs Unified).
3. plot3_gpu_speedup.png: Pure kernel and end-to-end speedups relative to OpenMP baseline for both memory models.
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
                "memModel": row.get("MemoryModel", "Standard"),
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
    """Plot 1: CPU vs GPU elapsed times across tested block sizes (Standard vs Unified)"""
    w, h = 980, 560
    surface, ctx = create_surface(w, h)

    has_unified = any("Unified" in k for k in data.keys())
    has_standard = any("Standard" in k or k in ["CUDA_8x8", "CUDA_16x16", "CUDA_32x32"] for k in data.keys())

    draw_title(
        ctx,
        "Lab 2: CPU vs GPU Elapsed Execution Times",
        "Comparing OpenMP (8 threads), Standard Memory CUDA, and Unified Memory CUDA (8x8, 16x16, 32x32)",
        x=50, y=45
    )

    left, top, right, bottom = 90, 110, 920, 480
    plot_w = right - left
    plot_h = bottom - top

    cpu_val = data["CPU_OpenMP"]["kernelMean"] if "CPU_OpenMP" in data else 3.0

    # Collect maximum value for scaling
    all_vals = [cpu_val]
    for b in [8, 16, 32]:
        std_k = f"CUDA_Standard_{b}x{b}" if f"CUDA_Standard_{b}x{b}" in data else f"CUDA_{b}x{b}"
        uni_k = f"CUDA_Unified_{b}x{b}"
        if std_k in data:
            all_vals.extend([data[std_k]["kernelMean"], data[std_k]["totalGpuMean"]])
        if uni_k in data:
            all_vals.extend([data[uni_k]["kernelMean"], data[uni_k]["totalGpuMean"]])

    max_val = max(all_vals) * 1.25
    if max_val <= 0: max_val = 10.0

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    # Legend
    legend_items = [
        ("CPU OpenMP", PALETTE["primary"]),
    ]
    if has_standard:
        legend_items.extend([
            ("Std Kernel", PALETTE["secondary"]),
            ("Std Total", PALETTE["accent"]),
        ])
    if has_unified:
        legend_items.extend([
            ("Uni Kernel (w/ faults)", PALETTE["cyan"]),
            ("Uni Total (Alloc+Faults)", PALETTE["purple"]),
        ])

    draw_legend(ctx, start_x=left + 180, y=85, items=legend_items, item_spacing=145.0)

    groups = [
        {"name": "CPU OpenMP\n(8 Cores)", "is_cpu": True, "val": cpu_val},
        {"name": "Block 8x8\n(64 Threads)", "is_cpu": False, "b": 8},
        {"name": "Block 16x16\n(256 Threads)", "is_cpu": False, "b": 16},
        {"name": "Block 32x32\n(1024 Threads)", "is_cpu": False, "b": 32},
    ]

    n_groups = len(groups)
    col_w = plot_w / n_groups

    for i, grp in enumerate(groups):
        cx = left + i * col_w + col_w / 2.0

        if grp["is_cpu"]:
            bar_w = 45.0
            bx = cx - bar_w / 2.0
            bh = (grp["val"] / max_val) * plot_h
            by = bottom - bh
            draw_bar(ctx, bx, by, bar_w, bh, PALETTE["primary"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(11)
            ctx.move_to(bx - 4, by - 8)
            ctx.show_text(f"{grp['val']:.2f} ms")
        else:
            b = grp["b"]
            std_k = f"CUDA_Standard_{b}x{b}" if f"CUDA_Standard_{b}x{b}" in data else f"CUDA_{b}x{b}"
            uni_k = f"CUDA_Unified_{b}x{b}"

            bars_to_draw = []
            if std_k in data:
                bars_to_draw.append({"val": data[std_k]["kernelMean"], "color": PALETTE["secondary"], "lbl": f"{data[std_k]['kernelMean']:.1f}"})
                bars_to_draw.append({"val": data[std_k]["totalGpuMean"], "color": PALETTE["accent"], "lbl": f"{data[std_k]['totalGpuMean']:.1f}"})
            if uni_k in data:
                bars_to_draw.append({"val": data[uni_k]["kernelMean"], "color": PALETTE["cyan"], "lbl": f"{data[uni_k]['kernelMean']:.1f}"})
                bars_to_draw.append({"val": data[uni_k]["totalGpuMean"], "color": PALETTE["purple"], "lbl": f"{data[uni_k]['totalGpuMean']:.1f}"})

            n_bars = len(bars_to_draw)
            bar_w = 26.0 if n_bars > 2 else 38.0
            gap = 5.0
            tot_cluster_w = n_bars * bar_w + (n_bars - 1) * gap
            start_bx = cx - tot_cluster_w / 2.0

            for j, b_info in enumerate(bars_to_draw):
                bx = start_bx + j * (bar_w + gap)
                bh = (b_info["val"] / max_val) * plot_h
                by = bottom - bh
                draw_bar(ctx, bx, by, bar_w, bh, b_info["color"])

                ctx.set_source_rgb(*PALETTE["text_dark"])
                ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
                ctx.set_font_size(10)
                ctx.move_to(bx - 2, by - 6)
                ctx.show_text(b_info["lbl"])

        # Category label
        lines = grp["name"].split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(12)
        ctx.move_to(cx - 35, bottom + 20)
        ctx.show_text(lines[0])
        if len(lines) > 1:
            ctx.move_to(cx - 30, bottom + 36)
            ctx.show_text(lines[1])

    png_path = os.path.join(REPORT_DIR, "plot1_cpu_vs_gpu_elapsed.png")
    save_surface(surface, png_path)

def draw_plot2(data):
    """Plot 2: Breakdown of costs for fastest GPU execution (Standard vs Unified side-by-side)"""
    w, h = 950, 530
    surface, ctx = create_surface(w, h)

    has_unified = any("Unified" in k for k in data.keys())

    # Find fastest standard config
    std_keys = [k for k in ["CUDA_Standard_8x8", "CUDA_Standard_16x16", "CUDA_Standard_32x32", "CUDA_8x8", "CUDA_16x16", "CUDA_32x32"] if k in data]
    fastest_std_k = min(std_keys, key=lambda k: data[k]["kernelMean"]) if std_keys else None
    f_std = data[fastest_std_k] if fastest_std_k else None

    # Find fastest unified config
    uni_keys = [k for k in ["CUDA_Unified_8x8", "CUDA_Unified_16x16", "CUDA_Unified_32x32"] if k in data]
    fastest_uni_k = min(uni_keys, key=lambda k: data[k]["kernelMean"]) if uni_keys else None
    f_uni = data[fastest_uni_k] if fastest_uni_k else None

    std_name = fastest_std_k.replace("CUDA_Standard_", "").replace("CUDA_", "") if fastest_std_k else "16x16"
    uni_name = fastest_uni_k.replace("CUDA_Unified_", "") if fastest_uni_k else "16x16"

    draw_title(
        ctx,
        "Lab 2: Fastest GPU Execution Timing Breakdown",
        f"Comparing Stage Latencies: Standard Memory ({std_name}) vs. Unified Memory ({uni_name})",
        x=50, y=45
    )

    left, top, right, bottom = 100, 110, 880, 450
    plot_w = right - left
    plot_h = bottom - top

    # Define stages for both models
    stages = []
    if f_std:
        stages.extend([
            {"group": f"Standard ({std_name})", "name": "Alloc / Init\n(cudaMalloc)", "val": f_std["allocMean"], "color": PALETTE["gray"]},
            {"group": f"Standard ({std_name})", "name": "HtoD Transfer\n(18.9 MB RGB)", "val": f_std["htodMean"], "color": PALETTE["accent"]},
            {"group": f"Standard ({std_name})", "name": "Kernel Exec\n(In-VRAM)", "val": f_std["kernelMean"], "color": PALETTE["secondary"]},
            {"group": f"Standard ({std_name})", "name": "DtoH Transfer\n(6.3 MB Gray)", "val": f_std["dtohMean"], "color": PALETTE["cyan"]},
        ])
    if f_uni:
        stages.extend([
            {"group": f"Unified ({uni_name})", "name": "Alloc / Init\n(cudaMallocManaged)", "val": f_uni["allocMean"], "color": PALETTE["gray"]},
            {"group": f"Unified ({uni_name})", "name": "HtoD Transfer\n(On-Demand)", "val": f_uni["htodMean"], "color": PALETTE["accent"]},
            {"group": f"Unified ({uni_name})", "name": "Kernel Exec\n(w/ Page Faults)", "val": f_uni["kernelMean"], "color": PALETTE["purple"]},
            {"group": f"Unified ({uni_name})", "name": "Sync & Read\n(On-Demand)", "val": f_uni["dtohMean"], "color": PALETTE["cyan"]},
        ])

    max_val = max(s["val"] for s in stages) * 1.3
    if max_val <= 0: max_val = 10.0

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    n_stages = len(stages)
    col_w = plot_w / n_stages
    bar_w = 52.0

    for i, s in enumerate(stages):
        cx = left + i * col_w + col_w / 2.0
        bx = cx - bar_w / 2.0
        bh = (s["val"] / max_val) * plot_h
        by = bottom - bh

        draw_bar(ctx, bx, by, bar_w, bh, s["color"])

        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(11)
        ctx.move_to(bx + 4, by - 8)
        ctx.show_text(f"{s['val']:.2f} ms")

        lines = s["name"].split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.set_font_size(10)
        ctx.move_to(cx - 36, bottom + 18)
        ctx.show_text(lines[0])
        if len(lines) > 1:
            ctx.move_to(cx - 32, bottom + 32)
            ctx.show_text(lines[1])

    # Section dividers / labels
    if f_std and f_uni:
        mid_x = left + 4 * col_w
        ctx.set_source_rgb(*PALETTE["axis"])
        ctx.set_line_width(1.0)
        ctx.set_dash([3.0, 3.0])
        ctx.move_to(mid_x, top)
        ctx.line_to(mid_x, bottom)
        ctx.stroke()
        ctx.set_dash([])

        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(12)
        ctx.set_source_rgb(*PALETTE["text_muted"])
        ctx.move_to(left + 2 * col_w - 60, top - 8)
        ctx.show_text(f"Standard Memory Pipeline ({std_name})")
        ctx.move_to(left + 6 * col_w - 60, top - 8)
        ctx.show_text(f"Unified Memory Pipeline ({uni_name})")

    png_path = os.path.join(REPORT_DIR, "plot2_fastest_gpu_breakdown.png")
    save_surface(surface, png_path)

def draw_plot3(data):
    """Plot 3: Speedup achieved by each CUDA execution with respect to OpenMP (Standard vs Unified)"""
    w, h = 920, 520
    surface, ctx = create_surface(w, h)

    draw_title(
        ctx,
        "Lab 2: Speedup vs OpenMP CPU Baseline (8 Cores)",
        "Comparing Pure Kernel Speedup and End-to-End Speedup across Standard and Unified Memory Models",
        x=50, y=45
    )

    left, top, right, bottom = 90, 110, 850, 440
    plot_w = right - left
    plot_h = bottom - top

    blocks = [8, 16, 32]
    items = []
    all_sp = [1.0]

    for b in blocks:
        std_k = f"CUDA_Standard_{b}x{b}" if f"CUDA_Standard_{b}x{b}" in data else f"CUDA_{b}x{b}"
        uni_k = f"CUDA_Unified_{b}x{b}"

        entry = {"name": f"Block {b}x{b}\n({b*b} Threads)", "b": b}
        if std_k in data:
            entry["std_k"] = data[std_k]["speedupKernel"]
            entry["std_t"] = data[std_k]["speedupTotal"]
            all_sp.extend([entry["std_k"], entry["std_t"]])
        if uni_k in data:
            entry["uni_k"] = data[uni_k]["speedupKernel"]
            entry["uni_t"] = data[uni_k]["speedupTotal"]
            all_sp.extend([entry["uni_k"], entry["uni_t"]])
        items.append(entry)

    max_val = max(all_sp) * 1.3
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
        ("Std Kernel Speedup", PALETTE["secondary"]),
        ("Std Total Speedup", PALETTE["accent"]),
        ("Uni Kernel Speedup", PALETTE["cyan"]),
        ("Uni Total Speedup", PALETTE["purple"]),
    ]
    draw_legend(ctx, start_x=left + 150, y=85, items=legend_items, item_spacing=150.0)

    n_items = len(items)
    col_w = plot_w / n_items

    for i, it in enumerate(items):
        cx = left + i * col_w + col_w / 2.0
        
        bars_to_draw = []
        if "std_k" in it:
            bars_to_draw.append({"val": it["std_k"], "color": PALETTE["secondary"], "lbl": f"{it['std_k']:.2f}x"})
            bars_to_draw.append({"val": it["std_t"], "color": PALETTE["accent"], "lbl": f"{it['std_t']:.2f}x"})
        if "uni_k" in it:
            bars_to_draw.append({"val": it["uni_k"], "color": PALETTE["cyan"], "lbl": f"{it['uni_k']:.2f}x"})
            bars_to_draw.append({"val": it["uni_t"], "color": PALETTE["purple"], "lbl": f"{it['uni_t']:.2f}x"})

        n_b = len(bars_to_draw)
        b_w = 28.0 if n_b > 2 else 45.0
        gap = 5.0
        cluster_w = n_b * b_w + (n_b - 1) * gap
        start_x = cx - cluster_w / 2.0

        for j, b_info in enumerate(bars_to_draw):
            bx = start_x + j * (b_w + gap)
            bh = (b_info["val"] / max_val) * plot_h
            by = bottom - bh
            draw_bar(ctx, bx, by, b_w, bh, b_info["color"])

            ctx.set_source_rgb(*PALETTE["text_dark"])
            ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
            ctx.set_font_size(10)
            ctx.move_to(bx - 1, by - 6)
            ctx.show_text(b_info["lbl"])

        lines = it["name"].split("\n")
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.set_font_size(11)
        ctx.move_to(cx - 38, bottom + 20)
        ctx.show_text(lines[0])
        if len(lines) > 1:
            ctx.move_to(cx - 34, bottom + 35)
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
