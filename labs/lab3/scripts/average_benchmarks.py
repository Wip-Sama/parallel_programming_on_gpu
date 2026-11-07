#!/usr/bin/env python3
"""
average_benchmarks.py - Automated Multi-Run Benchmark Harness & Statistical Analyzer
Course: Massively Parallel Programming on GPUs - Lab 3: Naive, Tiled & Coarse-Grained Matrix Multiplication
Teacher: Donato D'Ambrosio

Evaluation Scope (Points 4.1, 4.2 & 4.3):
1. CPU OpenMP Baseline Standard (i-j-k loop, 8 threads)
2. CPU OpenMP Baseline Optimized (i-k-j loop, 8 threads)
3. Point 4.1: CUDA Naive (Global Memory only) across block sizes {8x8, 16x16, 32x32}
4. Point 4.2: CUDA Tiled (Global & Shared Memory) across block/tile sizes {8x8, 16x16, 32x32}
5. Point 4.3: CUDA Coarse-Grained Tiled (Shared Memory + Coarsening nt in {2, 4})
6. Statistical aggregation across 1 warm-up + 4 repetitions (Section 5.4)
7. Exports consolidated CSV, comparative Markdown report, and triggers Cairo plot generation
"""

import os
import sys
import argparse
import subprocess
import statistics
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAB_DIR = os.path.dirname(SCRIPT_DIR)
COMMON_PYTHON_DIR = os.path.abspath(os.path.join(LAB_DIR, "..", "common", "python"))
if COMMON_PYTHON_DIR not in sys.path:
    sys.path.insert(0, COMMON_PYTHON_DIR)

from bench_utils import ensure_dir, aggregate_metrics, save_csv

BLOCK_CONFIGS = [8, 16, 32]
COARSE_CONFIGS = [(8, 2), (16, 2), (32, 2), (16, 4), (32, 4), (16, 8), (32, 8)]
REPETITIONS = 4
REPORT_DIR = os.path.join(LAB_DIR, "reports")
ensure_dir(REPORT_DIR)

BINARY = os.path.join(LAB_DIR, "lab3")

DEFAULT_M = 2048
DEFAULT_K = 1024
DEFAULT_N = 512

def ensure_binary():
    """Ensures lab3 binary is compiled and up-to-date."""
    if not os.path.exists(BINARY) or not os.access(BINARY, os.X_OK):
        print(f"[+] Compiling missing binary via Makefile...")
        res = subprocess.run(["make", "-C", LAB_DIR, "lab3"], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[-] Build failed:\n{res.stderr}", file=sys.stderr)
            sys.exit(1)

def run_single(mode, block_size=16, nt=2, threads=8, M=DEFAULT_M, K=DEFAULT_K, N=DEFAULT_N):
    """
    Runs: ./lab3 --mode <mode> --block <block_size> --nt <nt> --threads <threads> -M <M> -K <K> -N <N> --csv
    Returns parsed dictionary of results.
    """
    cmd = [
        BINARY,
        "--mode", mode,
        "--block", str(block_size),
        "--nt", str(nt),
        "--threads", str(threads),
        "-M", str(M),
        "-K", str(K),
        "-N", str(N),
        "--csv"
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("CSV,"):
                parts = line.split(",")
                tag = parts[1]
                if tag == "cpu_std":
                    # CSV,cpu_std,threads,M,K,N,timeMs,gflops,status
                    return {
                        "mode": "cpu_std",
                        "threads": int(parts[2]),
                        "M": int(parts[3]),
                        "K": int(parts[4]),
                        "N": int(parts[5]),
                        "computeMs": float(parts[6]),
                        "gflops": float(parts[7]),
                        "status": parts[8]
                    }
                elif tag == "cpu_opt":
                    # CSV,cpu_opt,threads,M,K,N,timeMs,gflops,status
                    return {
                        "mode": "cpu_opt",
                        "threads": int(parts[2]),
                        "M": int(parts[3]),
                        "K": int(parts[4]),
                        "N": int(parts[5]),
                        "computeMs": float(parts[6]),
                        "gflops": float(parts[7]),
                        "status": parts[8]
                    }
                elif tag in ("gpu", "gpu_naive", "gpu_tiled"):
                    # CSV,gpu_tag,blockSize,gridX,gridY,threads,M,K,N,allocMs,htodMs,kernelMs,dtohMs,computeMs,totalMs,gflops,status
                    return {
                        "mode": tag,
                        "blockSize": int(parts[2]),
                        "tileNumber": 1,
                        "gridX": int(parts[3]),
                        "gridY": int(parts[4]),
                        "threads": int(parts[5]),
                        "M": int(parts[6]),
                        "K": int(parts[7]),
                        "N": int(parts[8]),
                        "allocMs": float(parts[9]),
                        "htodMs": float(parts[10]),
                        "kernelMs": float(parts[11]),
                        "dtohMs": float(parts[12]),
                        "computeMs": float(parts[13]),
                        "totalMs": float(parts[14]),
                        "gflops": float(parts[15]),
                        "status": parts[16]
                    }
                elif tag == "gpu_coarse":
                    # CSV,gpu_coarse,blockSize,tileNumber,gridX,gridY,threads,M,K,N,allocMs,htodMs,kernelMs,dtohMs,computeMs,totalMs,gflops,status
                    return {
                        "mode": tag,
                        "blockSize": int(parts[2]),
                        "tileNumber": int(parts[3]),
                        "gridX": int(parts[4]),
                        "gridY": int(parts[5]),
                        "threads": int(parts[6]),
                        "M": int(parts[7]),
                        "K": int(parts[8]),
                        "N": int(parts[9]),
                        "allocMs": float(parts[10]),
                        "htodMs": float(parts[11]),
                        "kernelMs": float(parts[12]),
                        "dtohMs": float(parts[13]),
                        "computeMs": float(parts[14]),
                        "totalMs": float(parts[15]),
                        "gflops": float(parts[16]),
                        "status": parts[17]
                    }
    except subprocess.CalledProcessError as e:
        print(f"[-] Execution error for {' '.join(cmd)}:\n{e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[-] Unexpected error: {e}", file=sys.stderr)
        return None
    return None

def benchmark_cuda_kernel(kernel_mode, kernel_name, repetitions, threads, M, K, N, total_ops, std_mean, opt_mean):
    """Benchmarks a standard CUDA kernel (naive or tiled) across all block sizes."""
    results = {}
    for b in BLOCK_CONFIGS:
        print(f"\n[*] Evaluating {kernel_name} with Block Size {b}x{b} ({b*b} threads/block)...")
        # 1 Warm-up run
        run_single(kernel_mode, block_size=b, threads=threads, M=M, K=K, N=N)

        k_times, tot_times, alloc_times = [], [], []
        last_status = "PASS"
        grid_dim_str = ""

        for i in range(repetitions):
            r = run_single(kernel_mode, block_size=b, threads=threads, M=M, K=K, N=N)
            if r:
                k_times.append(r["kernelMs"])
                tot_times.append(r["totalMs"])
                alloc_times.append(r["allocMs"])
                last_status = r["status"]
                grid_dim_str = f"{r['gridX']}x{r['gridY']}"
                print(f"    Run {i+1}: Kernel = {r['kernelMs']:.4f} ms ({r['gflops']:.2f} GFLOP/s) | Total = {r['totalMs']:.4f} ms [{r['status']}]")

        k_metrics = aggregate_metrics(k_times)
        tot_metrics = aggregate_metrics(tot_times)
        alloc_metrics = aggregate_metrics(alloc_times)

        k_mean = k_metrics["mean"]
        k_gflops = (total_ops * 1e-6) / k_mean if k_mean > 0 else 0.0
        tot_mean = tot_metrics["mean"]

        speedup_std = (std_mean / k_mean) if k_mean > 0 else 0.0
        speedup_opt = (opt_mean / k_mean) if k_mean > 0 else 0.0

        results[b] = {
            "kernelType": kernel_name,
            "mode": kernel_mode,
            "blockSize": b,
            "blockDim": f"{b}x{b}",
            "gridDim": grid_dim_str,
            "threadsPerBlock": b * b,
            "kernelMeanMs": k_mean,
            "kernelStdMs": k_metrics["std"],
            "allocMeanMs": alloc_metrics["mean"],
            "totalMeanMs": tot_mean,
            "totalStdMs": tot_metrics["std"],
            "gflops": k_gflops,
            "speedupVsStd": speedup_std,
            "speedupVsOpt": speedup_opt,
            "status": last_status
        }
        print(f"  --> Average {b}x{b}: Kernel = {k_mean:.4f} ± {k_metrics['std']:.4f} ms | {k_gflops:.2f} GFLOP/s (Speedup vs Std: {speedup_std:.2f}x, vs Opt: {speedup_opt:.2f}x)")

    return results

def benchmark_coarse_grained(repetitions, threads, M, K, N, total_ops, std_mean, opt_mean):
    """Benchmarks Point 4.3 Coarse-Grained Tiled CUDA kernel across (block_size, nt) configurations."""
    results = {}
    for (b, nt) in COARSE_CONFIGS:
        cfg_name = f"CUDA Coarse {b}x{b} (nt={nt})"
        print(f"\n[*] Evaluating {cfg_name} [Tile {b}x{b}, nt={nt} tiles/block, (1+{nt})x{b}x{b} shared floats]...")
        # 1 Warm-up run
        run_single("gpu_coarse", block_size=b, nt=nt, threads=threads, M=M, K=K, N=N)

        k_times, tot_times, alloc_times = [], [], []
        last_status = "PASS"
        grid_dim_str = ""

        for i in range(repetitions):
            r = run_single("gpu_coarse", block_size=b, nt=nt, threads=threads, M=M, K=K, N=N)
            if r:
                k_times.append(r["kernelMs"])
                tot_times.append(r["totalMs"])
                alloc_times.append(r["allocMs"])
                last_status = r["status"]
                grid_dim_str = f"{r['gridX']}x{r['gridY']}"
                print(f"    Run {i+1}: Kernel = {r['kernelMs']:.4f} ms ({r['gflops']:.2f} GFLOP/s) | Total = {r['totalMs']:.4f} ms [{r['status']}]")

        k_metrics = aggregate_metrics(k_times)
        tot_metrics = aggregate_metrics(tot_times)
        alloc_metrics = aggregate_metrics(alloc_times)

        k_mean = k_metrics["mean"]
        k_gflops = (total_ops * 1e-6) / k_mean if k_mean > 0 else 0.0
        tot_mean = tot_metrics["mean"]

        speedup_std = (std_mean / k_mean) if k_mean > 0 else 0.0
        speedup_opt = (opt_mean / k_mean) if k_mean > 0 else 0.0

        results[(b, nt)] = {
            "kernelType": "CUDA Coarse-Grained Tiled",
            "mode": "gpu_coarse",
            "blockSize": b,
            "tileNumber": nt,
            "blockDim": f"{b}x{b}",
            "gridDim": grid_dim_str,
            "threadsPerBlock": b * b,
            "kernelMeanMs": k_mean,
            "kernelStdMs": k_metrics["std"],
            "allocMeanMs": alloc_metrics["mean"],
            "totalMeanMs": tot_mean,
            "totalStdMs": tot_metrics["std"],
            "gflops": k_gflops,
            "speedupVsStd": speedup_std,
            "speedupVsOpt": speedup_opt,
            "status": last_status
        }
        print(f"  --> Average {b}x{b} (nt={nt}): Kernel = {k_mean:.4f} ± {k_metrics['std']:.4f} ms | {k_gflops:.2f} GFLOP/s (Speedup vs Std: {speedup_std:.2f}x, vs Opt: {speedup_opt:.2f}x)")

    return results

def run_benchmarks(repetitions=REPETITIONS, threads=8, M=DEFAULT_M, K=DEFAULT_K, N=DEFAULT_N):
    ensure_binary()

    total_ops = 2.0 * M * N * K
    print("=" * 100)
    print(" Lab 3: Automated Multi-Run Benchmark Suite (Naive, Tiled & Coarse-Grained)")
    print(f" Matrix Dimensions             : M={M}, K={K}, N={N} ({total_ops * 1e-9:.3f} GFLOP)")
    print(f" Repetitions per Configuration : {repetitions} (after 1 warm-up run)")
    print(f" CPU Baseline Threads          : {threads} (OpenMP)")
    print(f" Tested CUDA Block Sizes       : 8x8, 16x16, 32x32")
    print(f" Tested Coarsening Factors (nt): 2, 4")
    print("=" * 100)

    # -------------------------------------------------------------------------
    # 1. CPU Standard Baseline (i-j-k loop)
    # -------------------------------------------------------------------------
    print(f"\n[*] Evaluating CPU OpenMP Standard Baseline ({threads} threads)...")
    run_single("cpu", threads=threads, M=M, K=K, N=N)
    
    cpu_std_times = []
    for i in range(repetitions):
        r = run_single("cpu", threads=threads, M=M, K=K, N=N)
        if r:
            cpu_std_times.append(r["computeMs"])
            print(f"    Run {i+1}: {r['computeMs']:.4f} ms ({r['gflops']:.2f} GFLOP/s) [{r['status']}]")

    std_metrics = aggregate_metrics(cpu_std_times)
    std_mean = std_metrics["mean"]
    std_gflops = (total_ops * 1e-6) / std_mean if std_mean > 0 else 0.0
    print(f"  --> CPU Standard Mean: {std_mean:.4f} ± {std_metrics['std']:.4f} ms | {std_gflops:.2f} GFLOP/s")

    # -------------------------------------------------------------------------
    # 2. CPU Optimized Baseline (i-k-j loop interchange)
    # -------------------------------------------------------------------------
    print(f"\n[*] Evaluating CPU OpenMP Optimized Baseline (i-k-j loop, {threads} threads)...")
    run_single("cpu_opt", threads=threads, M=M, K=K, N=N)

    cpu_opt_times = []
    for i in range(repetitions):
        r = run_single("cpu_opt", threads=threads, M=M, K=K, N=N)
        if r:
            cpu_opt_times.append(r["computeMs"])
            print(f"    Run {i+1}: {r['computeMs']:.4f} ms ({r['gflops']:.2f} GFLOP/s) [{r['status']}]")

    opt_metrics = aggregate_metrics(cpu_opt_times)
    opt_mean = opt_metrics["mean"]
    opt_gflops = (total_ops * 1e-6) / opt_mean if opt_mean > 0 else 0.0
    speedup_opt_vs_std = (std_mean / opt_mean) if opt_mean > 0 else 0.0
    print(f"  --> CPU Optimized Mean: {opt_mean:.4f} ± {opt_metrics['std']:.4f} ms | {opt_gflops:.2f} GFLOP/s (Speedup: {speedup_opt_vs_std:.2f}x)")

    # -------------------------------------------------------------------------
    # 3. Point 4.1: CUDA Naive Kernel (Global Memory Only)
    # -------------------------------------------------------------------------
    print(f"\n" + "=" * 100)
    print(" POINT 4.1: CUDA NAIVE MATRIX MULTIPLICATION (GLOBAL MEMORY ONLY)")
    print("=" * 100)
    naive_results = benchmark_cuda_kernel(
        "gpu_naive", "CUDA Naive", repetitions, threads, M, K, N, total_ops, std_mean, opt_mean
    )

    # -------------------------------------------------------------------------
    # 4. Point 4.2: CUDA Tiled Kernel (Global & Shared Memory)
    # -------------------------------------------------------------------------
    print(f"\n" + "=" * 100)
    print(" POINT 4.2: CUDA TILED MATRIX MULTIPLICATION (GLOBAL & SHARED MEMORY)")
    print("=" * 100)
    tiled_results = benchmark_cuda_kernel(
        "gpu_tiled", "CUDA Tiled", repetitions, threads, M, K, N, total_ops, std_mean, opt_mean
    )

    # -------------------------------------------------------------------------
    # 5. Point 4.3: CUDA Coarse-Grained Tiled Kernel
    # -------------------------------------------------------------------------
    print(f"\n" + "=" * 100)
    print(" POINT 4.3: CUDA COARSE-GRAINED TILED MATRIX MULTIPLICATION (SHARED MEM + COARSENING)")
    print("=" * 100)
    coarse_results = benchmark_coarse_grained(
        repetitions, threads, M, K, N, total_ops, std_mean, opt_mean
    )

    # -------------------------------------------------------------------------
    # 6. Save Consolidated CSV
    # -------------------------------------------------------------------------
    csv_rows = []
    # CPU Standard
    csv_rows.append({
        "Configuration": "CPU OpenMP (Standard)",
        "Type": "CPU",
        "KernelType": "CPU Standard",
        "BlockSize": "-",
        "TileNumber": 0,
        "Threads": threads,
        "M": M, "K": K, "N": N,
        "KernelMeanMs": round(std_mean, 4),
        "KernelStdMs": round(std_metrics["std"], 4),
        "AllocMeanMs": 0.0,
        "TotalMeanMs": round(std_mean, 4),
        "TotalStdMs": round(std_metrics["std"], 4),
        "GFlops": round(std_gflops, 2),
        "SpeedupVsStd": 1.0,
        "SpeedupVsOpt": round(std_mean / opt_mean, 2) if opt_mean > 0 else 0.0,
        "SpeedupVsNaive": 0.0,
        "Status": "PASS"
    })
    # CPU Optimized
    csv_rows.append({
        "Configuration": "CPU OpenMP (Optimized I-K-J)",
        "Type": "CPU",
        "KernelType": "CPU Optimized",
        "BlockSize": "-",
        "TileNumber": 0,
        "Threads": threads,
        "M": M, "K": K, "N": N,
        "KernelMeanMs": round(opt_mean, 4),
        "KernelStdMs": round(opt_metrics["std"], 4),
        "AllocMeanMs": 0.0,
        "TotalMeanMs": round(opt_mean, 4),
        "TotalStdMs": round(opt_metrics["std"], 4),
        "GFlops": round(opt_gflops, 2),
        "SpeedupVsStd": round(speedup_opt_vs_std, 2),
        "SpeedupVsOpt": 1.0,
        "SpeedupVsNaive": 0.0,
        "Status": "PASS"
    })
    # CUDA Naive Configurations
    for b, res in naive_results.items():
        csv_rows.append({
            "Configuration": f"CUDA Naive {b}x{b}",
            "Type": "GPU",
            "KernelType": "CUDA Naive (Global)",
            "BlockSize": f"{b}x{b}",
            "TileNumber": 1,
            "Threads": res["threadsPerBlock"],
            "M": M, "K": K, "N": N,
            "KernelMeanMs": round(res["kernelMeanMs"], 4),
            "KernelStdMs": round(res["kernelStdMs"], 4),
            "AllocMeanMs": round(res["allocMeanMs"], 4),
            "TotalMeanMs": round(res["totalMeanMs"], 4),
            "TotalStdMs": round(res["totalStdMs"], 4),
            "GFlops": round(res["gflops"], 2),
            "SpeedupVsStd": round(res["speedupVsStd"], 2),
            "SpeedupVsOpt": round(res["speedupVsOpt"], 2),
            "SpeedupVsNaive": 1.0,
            "Status": res["status"]
        })
    # CUDA Tiled Configurations
    for b, res in tiled_results.items():
        naive_k_mean = naive_results[b]["kernelMeanMs"] if b in naive_results else res["kernelMeanMs"]
        speedup_vs_naive = (naive_k_mean / res["kernelMeanMs"]) if res["kernelMeanMs"] > 0 else 0.0
        csv_rows.append({
            "Configuration": f"CUDA Tiled {b}x{b}",
            "Type": "GPU",
            "KernelType": "CUDA Tiled (Shared)",
            "BlockSize": f"{b}x{b}",
            "TileNumber": 1,
            "Threads": res["threadsPerBlock"],
            "M": M, "K": K, "N": N,
            "KernelMeanMs": round(res["kernelMeanMs"], 4),
            "KernelStdMs": round(res["kernelStdMs"], 4),
            "AllocMeanMs": round(res["allocMeanMs"], 4),
            "TotalMeanMs": round(res["totalMeanMs"], 4),
            "TotalStdMs": round(res["totalStdMs"], 4),
            "GFlops": round(res["gflops"], 2),
            "SpeedupVsStd": round(res["speedupVsStd"], 2),
            "SpeedupVsOpt": round(res["speedupVsOpt"], 2),
            "SpeedupVsNaive": round(speedup_vs_naive, 2),
            "Status": res["status"]
        })
    # CUDA Coarse-Grained Tiled Configurations
    for (b, nt), res in coarse_results.items():
        naive_k_mean = naive_results[b]["kernelMeanMs"] if b in naive_results else res["kernelMeanMs"]
        speedup_vs_naive = (naive_k_mean / res["kernelMeanMs"]) if res["kernelMeanMs"] > 0 else 0.0
        csv_rows.append({
            "Configuration": f"CUDA Coarse {b}x{b} (nt={nt})",
            "Type": "GPU",
            "KernelType": "CUDA Coarse (Shared)",
            "BlockSize": f"{b}x{b}",
            "TileNumber": nt,
            "Threads": res["threadsPerBlock"],
            "M": M, "K": K, "N": N,
            "KernelMeanMs": round(res["kernelMeanMs"], 4),
            "KernelStdMs": round(res["kernelStdMs"], 4),
            "AllocMeanMs": round(res["allocMeanMs"], 4),
            "TotalMeanMs": round(res["totalMeanMs"], 4),
            "TotalStdMs": round(res["totalStdMs"], 4),
            "GFlops": round(res["gflops"], 2),
            "SpeedupVsStd": round(res["speedupVsStd"], 2),
            "SpeedupVsOpt": round(res["speedupVsOpt"], 2),
            "SpeedupVsNaive": round(speedup_vs_naive, 2),
            "Status": res["status"]
        })

    csv_path = os.path.join(REPORT_DIR, "benchmark_results_averaged.csv")
    save_csv(csv_path, csv_rows)
    print(f"\n[+] Saved consolidated benchmark data to '{csv_path}'")

    # -------------------------------------------------------------------------
    # 7. Generate Markdown Summary Report
    # -------------------------------------------------------------------------
    md_path = os.path.join(REPORT_DIR, "benchmark_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Lab 3: Matrix Multiplication Performance Benchmark Report\n\n")
        f.write(f"- **Course**: Massively Parallel Programming on GPUs\n")
        f.write(f"- **Teacher**: Donato D'Ambrosio\n")
        f.write(f"- **Dimensions**: $M={M}, K={K}, N={N}$ ($2.147\\text{{ GFLOP}}$)\n")
        f.write(f"- **Repetitions**: {repetitions} runs (following 1 warm-up run)\n")
        f.write(f"- **CPU Threads**: {threads} OpenMP threads\n\n")
        f.write("## Consolidated Benchmark Results\n\n")
        f.write("| Implementation | Kernel Type | Block / Threads | nt | Kernel Time (ms) | Total Time (ms) | Throughput (GFLOP/s) | Speedup vs CPU Std | Speedup vs CPU Opt | Speedup vs Naive | Status |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for row in csv_rows:
            f.write(f"| {row['Configuration']} | {row['KernelType']} | {row['BlockSize']} ({row['Threads']}) | {row['TileNumber']} | "
                    f"{row['KernelMeanMs']:.4f} ± {row['KernelStdMs']:.4f} | "
                    f"{row['TotalMeanMs']:.4f} ± {row['TotalStdMs']:.4f} | "
                    f"{row['GFlops']:.2f} | "
                    f"{row['SpeedupVsStd']:.2f}x | "
                    f"{row['SpeedupVsOpt']:.2f}x | "
                    f"{row['SpeedupVsNaive']:.2f}x | "
                    f"{row['Status']} |\n")
        f.write("\n\n## Key Observations & Analysis\n\n")
        f.write(f"1. **CPU Memory Optimization**: Loop interchange (i-k-j) improves CPU performance by **{speedup_opt_vs_std:.2f}x** over the standard i-j-k baseline due to stride-1 continuous cache-line streaming across matrix $B$.\n")
        best_naive_b = min(naive_results.keys(), key=lambda b: naive_results[b]["kernelMeanMs"])
        best_naive = naive_results[best_naive_b]
        best_tiled_b = min(tiled_results.keys(), key=lambda b: tiled_results[b]["kernelMeanMs"])
        best_tiled = tiled_results[best_tiled_b]
        best_coarse_key = min(coarse_results.keys(), key=lambda k: coarse_results[k]["kernelMeanMs"])
        best_coarse = coarse_results[best_coarse_key]

        f.write(f"2. **Point 4.1 (CUDA Naive)**: Block size **{best_naive_b}x{best_naive_b}** achieved the fastest Naive GPU time of **{best_naive['kernelMeanMs']:.4f} ms** ({best_naive['gflops']:.2f} GFLOP/s).\n")
        f.write(f"3. **Point 4.2 (CUDA Tiled)**: Block size **{best_tiled_b}x{best_tiled_b}** achieved **{best_tiled['kernelMeanMs']:.4f} ms** ({best_tiled['gflops']:.2f} GFLOP/s), outperforming Naive by **{best_naive['kernelMeanMs'] / best_tiled['kernelMeanMs']:.2f}x** due to $T$-fold shared memory reuse.\n")
        f.write(f"4. **Point 4.3 (CUDA Coarse-Grained Tiled)**: Configuration **{best_coarse_key[0]}x{best_coarse_key[0]} with nt={best_coarse_key[1]}** achieved the overall fastest GPU time of **{best_coarse['kernelMeanMs']:.4f} ms** ({best_coarse['gflops']:.2f} GFLOP/s), providing a **{best_naive['kernelMeanMs'] / best_coarse['kernelMeanMs']:.2f}x** speedup over Naive and **{best_coarse['speedupVsStd']:.2f}x** over the CPU baseline.\n")
        f.write(f"5. **Data Reuse with Coarsening**: By assigning $n_t$ tiles of $B$ to each block, each cooperatively loaded tile of $A$ is reused $n_t$ times across the $n_t$ partial products, cutting global memory reads for $A$ by an additional factor of $n_t$.\n")

    print(f"[+] Saved Markdown summary report to '{md_path}'")

    # -------------------------------------------------------------------------
    # 8. Trigger Analytical Plot Generation
    # -------------------------------------------------------------------------
    plot_script = os.path.join(SCRIPT_DIR, "generate_plots.py")
    if os.path.exists(plot_script):
        print("\n[*] Triggering automated chart generation via generate_plots.py...")
        subprocess.run([sys.executable, plot_script], check=False)

    print("\n" + "=" * 100)
    print(" Benchmark Suite Completed Successfully!")
    print("=" * 100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lab 3 Multi-Run Matrix Multiplication Benchmark Harness")
    parser.add_argument("-r", "--repetitions", type=int, default=REPETITIONS, help="Number of timed repetitions per configuration (default: 4)")
    parser.add_argument("-t", "--threads", type=int, default=8, help="Number of OpenMP CPU threads (default: 8)")
    parser.add_argument("-M", type=int, default=DEFAULT_M, help="Matrix dimension M (default: 2048)")
    parser.add_argument("-K", type=int, default=DEFAULT_K, help="Matrix dimension K (default: 1024)")
    parser.add_argument("-N", type=int, default=DEFAULT_N, help="Matrix dimension N (default: 512)")
    args = parser.parse_args()

    run_benchmarks(repetitions=args.repetitions, threads=args.threads, M=args.M, K=args.K, N=args.N)
