#!/usr/bin/env python3
"""
generate_plots.py - Graph Plotting for Lab 1 Benchmarks
Course: Massively Parallel Programming on GPUs (Lab 1)

Generates analytical charts from multi-run benchmark results:
1. plot1_vector_runtime.png: Vector addition runtime across block sizes for all 4 configurations.
2. plot2_matrix_runtime.png: Matrix addition runtime across block sizes for all 4 configurations.
3. plot3_time_breakdown.png: Init vs Add vs Memcpy / Page fault overhead breakdown.
4. plot4_unified_vs_standard.png: Device working time vs Total end-to-end runtime.
"""

import os
import sys
import csv
import cairo

# Import reusable common utilities
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
    draw_grouped_bars,
    draw_bar,
    save_surface,
    PALETTE,
)

REPORT_DIR = os.path.join(LAB_DIR, "reports")
CSV_PATH = os.path.join(REPORT_DIR, "benchmark_results_averaged.csv")
if not os.path.exists(CSV_PATH):
    CSV_PATH = os.path.join(LAB_DIR, "benchmark_results_averaged.csv")

def load_data():
    if not os.path.exists(CSV_PATH):
        print(f"[-] Benchmark CSV not found at {CSV_PATH}. Run average_benchmarks.py first.")
        sys.exit(1)

    rows = []
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "op": r["op"],
                "mem": r["mem"],
                "kernel": r["kernel"],
                "blockSize": int(r["blockSize"]),
                "gridSize": r["gridSize"],
                "initMeanMs": float(r["initMeanMs"]),
                "addMeanMs": float(r["addMeanMs"]),
                "memcpyMeanMs": float(r["memcpyMeanMs"]),
                "deviceMeanMs": float(r["deviceMeanMs"]),
                "totalRunMeanMs": float(r["totalRunMeanMs"]),
                "totalRunStdMs": float(r["totalRunStdMs"]),
            })
    return rows

def generate_op_plot(rows, op_name, output_filename, chart_title, chart_subtitle):
    """Draws runtime comparison across block sizes for a given operation (vector or matrix)."""
    block_sizes = [32, 64, 128, 256, 512, 1024]
    categories = []

    for b in block_sizes:
        cat = {"label": f"{b}\nthreads"}
        for r in rows:
            if r["op"] == op_name and r["blockSize"] == b:
                key = f"{r['mem']}_{r['kernel']}"
                cat[key] = r["totalRunMeanMs"]
        categories.append(cat)

    series_keys = ["u_m", "u_gs", "s_m", "s_gs"]
    colors = [
        PALETTE["primary"],    # Unified Monolithic (Blue)
        PALETTE["cyan"],       # Unified Grid-Stride (Cyan)
        PALETTE["accent"],     # Standard Monolithic (Amber)
        PALETTE["secondary"],  # Standard Grid-Stride (Green)
    ]

    max_val = max(max(c.get(k, 0.0) for k in series_keys) for c in categories) * 1.25

    w, h = 920, 560
    surface, ctx = create_surface(w, h)
    draw_title(ctx, chart_title, chart_subtitle, x=50, y=45)

    legend_items = [
        ("Unified / Monolithic", PALETTE["primary"]),
        ("Unified / Grid-Stride", PALETTE["cyan"]),
        ("Standard / Monolithic", PALETTE["accent"]),
        ("Standard / Grid-Stride", PALETTE["secondary"]),
    ]
    draw_legend(ctx, start_x=160, y=85, items=legend_items, item_spacing=180.0)

    left, top, right, bottom = 100, 120, 860, 480
    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")
    draw_grouped_bars(ctx, left, top, right, bottom, max_val, categories, series_keys, colors, bar_width=18.0, gap=4.0)

    out_path = os.path.join(REPORT_DIR, output_filename)
    save_surface(surface, out_path)

def generate_breakdown_plot(rows):
    """Draws timing breakdown (Init vs Add vs Memcpy/Migration) for block size 256."""
    target_block = 256
    items = []
    
    # 4 cases: Vector Unified, Vector Standard, Matrix Unified, Matrix Standard
    cases = [
        ("vector", "u", "gs", "Vector Unified\n(Grid-Stride 256)"),
        ("vector", "s", "gs", "Vector Standard\n(Grid-Stride 256)"),
        ("matrix", "u", "gs", "Matrix Unified\n(Grid-Stride 256)"),
        ("matrix", "s", "gs", "Matrix Standard\n(Grid-Stride 256)"),
    ]

    for op, mem, kern, label in cases:
        match = next((r for r in rows if r["op"] == op and r["mem"] == mem and r["kernel"] == kern and r["blockSize"] == target_block), None)
        if match:
            # For Unified memory, memory transfer is embedded in kernel and page faults
            # Calculate residual as total - (init + add)
            residual = max(0.0, match["totalRunMeanMs"] - (match["initMeanMs"] + match["addMeanMs"]))
            items.append({
                "label": label,
                "init": match["initMeanMs"],
                "add": match["addMeanMs"],
                "transfer": match["memcpyMeanMs"] if mem == "s" else residual,
                "total": match["totalRunMeanMs"],
            })

    if not items:
        return

    w, h = 880, 540
    surface, ctx = create_surface(w, h)
    draw_title(
        ctx,
        "Lab 1: Execution Time Breakdown (Init vs Add vs Data Transfer)",
        "Comparing Init (3x kernels), Add (1x kernel), and Memory Transfer / Page Fault Overhead at Block Size 256",
        x=50, y=45
    )

    legend_items = [
        ("Init 3x Kernels", PALETTE["primary"]),
        ("Add 1x Kernel", PALETTE["secondary"]),
        ("Memcpy / Page Migration", PALETTE["accent"]),
    ]
    draw_legend(ctx, start_x=220, y=85, items=legend_items, item_spacing=200.0)

    left, top, right, bottom = 100, 120, 820, 460
    plot_w = right - left
    plot_h = bottom - top
    max_val = max(it["total"] for it in items) * 1.25

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")

    col_w = plot_w / len(items)
    bar_w = 60.0

    for i, it in enumerate(items):
        cx = left + i * col_w + col_w / 2.0
        bx = cx - bar_w / 2.0

        # Stacked bar: init (bottom), add (middle), transfer (top)
        bh_init = (it["init"] / max_val) * plot_h
        bh_add = (it["add"] / max_val) * plot_h
        bh_xfer = (it["transfer"] / max_val) * plot_h

        by_init = bottom - bh_init
        by_add = by_init - bh_add
        by_xfer = by_add - bh_xfer

        draw_bar(ctx, bx, by_init, bar_w, bh_init, PALETTE["primary"])
        draw_bar(ctx, bx, by_add, bar_w, bh_add, PALETTE["secondary"])
        draw_bar(ctx, bx, by_xfer, bar_w, bh_xfer, PALETTE["accent"])

        # Label total at top
        ctx.set_source_rgb(*PALETTE["text_dark"])
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(11)
        ctx.move_to(bx + 8, by_xfer - 6)
        ctx.show_text(f"{it['total']:.1f} ms")

        # Category label below
        lines = it["label"].split("\n")
        ctx.select_font_face("Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_font_size(11)
        for l_idx, line in enumerate(lines):
            ctx.move_to(cx - 45, bottom + 20 + l_idx * 16)
            ctx.show_text(line)

    out_path = os.path.join(REPORT_DIR, "plot3_time_breakdown.png")
    save_surface(surface, out_path)

def generate_unified_vs_standard_plot(rows):
    """Compares Device Working Time vs Total Runtime (highlighting hidden page fault cost)."""
    target_block = 256
    categories = []

    cases = [
        ("vector", "u", "Vector Unified"),
        ("vector", "s", "Vector Standard"),
        ("matrix", "u", "Matrix Unified"),
        ("matrix", "s", "Matrix Standard"),
    ]

    for op, mem, label in cases:
        match = next((r for r in rows if r["op"] == op and r["mem"] == mem and r["kernel"] == "gs" and r["blockSize"] == target_block), None)
        if match:
            categories.append({
                "label": label,
                "device": match["deviceMeanMs"],
                "total": match["totalRunMeanMs"],
            })

    if not categories:
        return

    w, h = 880, 540
    surface, ctx = create_surface(w, h)
    draw_title(
        ctx,
        "Lab 1: Device Working Time vs Total End-to-End Runtime",
        "CUDA Event Device Time vs Wall-Clock Total Time (Revealing Unified Memory Page Fault Overheads)",
        x=50, y=45
    )

    legend_items = [
        ("Device Working Time (CUDA Events)", PALETTE["secondary"]),
        ("Total Run Time (Host Clock + Verification)", PALETTE["primary"]),
    ]
    draw_legend(ctx, start_x=180, y=85, items=legend_items, item_spacing=260.0)

    left, top, right, bottom = 100, 120, 820, 460
    max_val = max(c["total"] for c in categories) * 1.25

    draw_axes(ctx, left, top, right, bottom, max_val, ticks=5, y_label_fmt="{:.1f}", y_unit="ms")
    draw_grouped_bars(
        ctx, left, top, right, bottom, max_val,
        categories, ["device", "total"],
        [PALETTE["secondary"], PALETTE["primary"]],
        bar_width=32.0, gap=6.0
    )

    out_path = os.path.join(REPORT_DIR, "plot4_unified_vs_standard.png")
    save_surface(surface, out_path)

def main():
    rows = load_data()
    print(f"[+] Loaded {len(rows)} benchmark entries from {CSV_PATH}")
    os.makedirs(REPORT_DIR, exist_ok=True)

    generate_op_plot(
        rows,
        op_name="vector",
        output_filename="plot1_vector_runtime.png",
        chart_title="Lab 1: Vector Addition - Total Runtime Across Block Sizes",
        chart_subtitle="Comparing Unified vs Standard Memory & Monolithic vs Grid-Stride (N = 2^24 elements, 128 MB)"
    )

    generate_op_plot(
        rows,
        op_name="matrix",
        output_filename="plot2_matrix_runtime.png",
        chart_title="Lab 1: Matrix Addition - Total Runtime Across Block Sizes",
        chart_subtitle="Comparing Unified vs Standard Memory & Monolithic vs Grid-Stride (2048 x 4096, 64 MB)"
    )

    generate_breakdown_plot(rows)
    generate_unified_vs_standard_plot(rows)

    print(f"[+] All 4 Lab 1 analytical plots generated successfully in: {REPORT_DIR}")

if __name__ == "__main__":
    main()
