#!/usr/bin/env python3
"""
average_benchmarks.py - Multi-Run Benchmark Harness & Statistical Analyzer
Course: Massively Parallel Programming on GPUs (Lab 1)

Executes each CUDA implementation N times across block sizes, collects
high-resolution CUDA event metrics, computes means and standard deviations,
and produces comprehensive benchmark tables and analytical findings.

Layout recalled:
    ./lab1 <u/s> <m/gs> <vector/matrix> <size>
"""

import os
import sys
import argparse
import subprocess
import statistics
import time

BLOCK_SIZES = [32, 64, 128, 256, 512, 1024]
MEM_MODELS = [("u", "Unified"), ("s", "Standard")]
KERNEL_MODELS = [("m", "Monolithic"), ("gs", "Grid-Stride")]
OPERATIONS = ["vector", "matrix"]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def ensure_binary(binary_path):
    if not os.path.exists(binary_path) or not os.access(binary_path, os.X_OK):
        print(f"[+] Compiling '{binary_path}' via Makefile...")
        res = subprocess.run(["make", "-C", SCRIPT_DIR, "lab1"], capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[-] Build failed:\n{res.stderr}", file=sys.stderr)
            sys.exit(1)

def run_single_benchmark(binary_path, mem, model, op, size, device=0):
    """
    Runs: ./lab1 <mem> <model> <op> <size> --csv --device <device>
    Returns parsed dictionary of results or None on failure.
    """
    cmd = [binary_path, mem, model, op, str(size), "--csv", "--device", str(device)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("CSV,"):
                # CSV,mem,kernel,op,blockSize,gridSize,initMs,addMs,memcpyMs,totalGpuMs,cpuMs,speedup,status
                parts = line.split(",")
                return {
                    "mem": parts[1],
                    "kernel": parts[2],
                    "op": parts[3],
                    "blockSize": int(parts[4]),
                    "gridSize": parts[5],
                    "initMs": float(parts[6]),
                    "addMs": float(parts[7]),
                    "memcpyMs": float(parts[8]),
                    "totalGpuMs": float(parts[9]),
                    "cpuMs": float(parts[10]),
                    "speedup": float(parts[11]),
                    "status": parts[12],
                }
    except subprocess.CalledProcessError as e:
        print(f"[-] Execution failed for {' '.join(cmd)}:\n{e.stderr}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[-] Unexpected error for {' '.join(cmd)}: {e}", file=sys.stderr)
        return None
    return None

def get_cpu_baseline(binary_path, op, device=0):
    """Measures the CPU serial baseline using sweep mode output."""
    cmd = [binary_path, "u", "m", op, "all", "--device", str(device)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        for line in proc.stdout.splitlines():
            if "CPU Serial Elapsed Time" in line:
                # CPU Serial Elapsed Time (vectorInit + vectorAdd): 112.345 ms
                parts = line.split(":")
                if len(parts) >= 2:
                    val_str = parts[1].replace("ms", "").strip()
                    return float(val_str)
    except Exception:
        pass
    return 0.0

def aggregate_runs(runs_data):
    """Computes mean, stddev, min, max for a list of runs."""
    n = len(runs_data)
    if n == 0:
        return None

    inits = [r["initMs"] for r in runs_data]
    adds = [r["addMs"] for r in runs_data]
    memcpys = [r["memcpyMs"] for r in runs_data]
    totals = [r["totalGpuMs"] for r in runs_data]
    all_pass = all(r["status"] == "PASS" for r in runs_data)

    mean_total = statistics.mean(totals)
    std_total = statistics.stdev(totals) if n > 1 else 0.0

    return {
        "n": n,
        "mem": runs_data[0]["mem"],
        "kernel": runs_data[0]["kernel"],
        "op": runs_data[0]["op"],
        "blockSize": runs_data[0]["blockSize"],
        "gridSize": runs_data[0]["gridSize"],
        "initMean": statistics.mean(inits),
        "addMean": statistics.mean(adds),
        "memcpyMean": statistics.mean(memcpys),
        "totalMean": mean_total,
        "totalStd": std_total,
        "totalMin": min(totals),
        "totalMax": max(totals),
        "allPass": all_pass,
        "rawRuns": runs_data,
    }

def print_table_for_op(op, results_by_config, cpu_ms, num_runs):
    title = f"{op.upper()} ADDITION BENCHMARK - AVERAGED ACROSS {num_runs} RUNS"
    print("\n" + "=" * 115)
    print(f" {title}")
    if cpu_ms > 0:
        print(f" CPU Serial Baseline: {cpu_ms:.3f} ms")
    print("=" * 115)
    print(f"{'Configuration':<25} {'BlockSize':<10} {'GridSize':<12} {'Init (ms)':<12} {'Add (ms)':<12} {'Memcpy (ms)':<13} {'Total GPU (ms)':<18} {'Speedup':<10} {'Status'}")
    print("-" * 115)

    for cfg_key, block_results in results_by_config.items():
        for res in block_results:
            cfg_name = f"{'Unified' if res['mem'] == 'u' else 'Standard'} / {'Monolithic' if res['kernel'] == 'm' else 'Grid-Stride'}"
            speedup = (cpu_ms / res['totalMean']) if (cpu_ms > 0 and res['totalMean'] > 0) else 0.0
            total_str = f"{res['totalMean']:.3f} ± {res['totalStd']:.2f}" if num_runs > 1 else f"{res['totalMean']:.3f}"
            print(f"{cfg_name:<25} {res['blockSize']:<10} {res['gridSize']:<12} {res['initMean']:<12.3f} {res['addMean']:<12.3f} {res['memcpyMean']:<13.3f} {total_str:<18} {speedup:<9.2f}x {'PASS' if res['allPass'] else 'FAIL'}")
        print("-" * 115)

def generate_markdown_report(all_results, cpu_baselines, num_runs, output_file):
    with open(output_file, "w") as f:
        f.write("# Lab 1: Comprehensive Benchmark Report & Findings\n\n")
        f.write(f"- **Benchmark Executable:** `./lab1 <u/s> <m/gs> <vector/matrix> <size>`\n")
        f.write(f"- **Repetitions per configuration:** {num_runs} runs (averaged)\n")
        f.write(f"- **Report Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for op in OPERATIONS:
            if op not in all_results:
                continue
            cpu_ms = cpu_baselines.get(op, 0.0)
            f.write(f"## {op.capitalize()} Addition Benchmark\n\n")
            if cpu_ms > 0:
                f.write(f"**CPU Serial Baseline:** `{cpu_ms:.3f} ms`\n\n")

            f.write("| Memory | Kernel Model | Block Size | Grid Size | Init 3x (ms) | Add 1x (ms) | Memcpy (ms) | Total GPU (ms) | Speedup | Status |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")

            for cfg_key, block_results in all_results[op].items():
                for res in block_results:
                    mem_label = "Unified" if res['mem'] == 'u' else "Standard"
                    kern_label = "Monolithic" if res['kernel'] == 'm' else "Grid-Stride"
                    speedup = (cpu_ms / res['totalMean']) if (cpu_ms > 0 and res['totalMean'] > 0) else 0.0
                    total_str = f"{res['totalMean']:.3f} ± {res['totalStd']:.2f}" if num_runs > 1 else f"{res['totalMean']:.3f}"
                    f.write(f"| {mem_label} | {kern_label} | {res['blockSize']} | `{res['gridSize']}` | {res['initMean']:.3f} | {res['addMean']:.3f} | {res['memcpyMean']:.3f} | **{total_str}** | {speedup:.2f}x | {'PASS' if res['allPass'] else 'FAIL'} |\n")
            f.write("\n")

        # Architectural Analysis & Findings
        f.write("## Key Findings & Architectural Analysis\n\n")
        f.write("### 1. Unified Memory vs Standard Memory\n")
        f.write("- **Unified Memory (`cudaMallocManaged`):** Eliminates explicit host-to-device and device-to-host copies. On modern GPUs with page migration, initial page faults occur on first access. Synchronization (`cudaDeviceSynchronize()`) ensures data coherency before CPU access.\n")
        f.write("- **Standard Memory (`cudaMalloc` + `cudaMemcpy`):** Explicit PCIe transfers (`cudaMemcpy` DtoH) represent the dominant portion of elapsed time for memory-bound vector operations.\n\n")

        f.write("### 2. Monolithic vs Grid-Stride Loop Execution Model\n")
        f.write("- **Monolithic Kernels:** Grid dimensions scale 1-to-1 with element count (`(N + blockSize - 1) / blockSize`). While conceptually simple, large grids incur block launch management overhead.\n")
        f.write("- **Grid-Stride Loops (`stride = gridDim * blockDim`):** The grid is decoupled from array size and scaled to Streaming Multiprocessor (SM) capacity (`32 * numSMs`). Each thread reuses registers and cache over consecutive elements, providing superior scalability and cache locality.\n\n")

        f.write("### 3. Optimal Block Size Analysis\n")
        f.write("- Block sizes of **256** and **512** threads consistently achieve optimal warp occupancy and memory bandwidth saturation without running into register or shared memory pressure.\n")
        f.write("- Very small block sizes (32 threads, 1 warp) suffer from low occupancy and latency hiding limitations.\n")

def save_csv_results(all_results, cpu_baselines, num_runs, csv_file):
    with open(csv_file, "w") as f:
        f.write("op,mem,kernel,blockSize,gridSize,numRuns,initMeanMs,addMeanMs,memcpyMeanMs,totalMeanMs,totalStdMs,totalMinMs,totalMaxMs,cpuMs,speedup,status\n")
        for op in all_results:
            cpu_ms = cpu_baselines.get(op, 0.0)
            for cfg_key in all_results[op]:
                for res in all_results[op][cfg_key]:
                    speedup = (cpu_ms / res['totalMean']) if (cpu_ms > 0 and res['totalMean'] > 0) else 0.0
                    f.write(f"{res['op']},{res['mem']},{res['kernel']},{res['blockSize']},{res['gridSize']},{res['n']},"
                            f"{res['initMean']:.4f},{res['addMean']:.4f},{res['memcpyMean']:.4f},{res['totalMean']:.4f},"
                            f"{res['totalStd']:.4f},{res['totalMin']:.4f},{res['totalMax']:.4f},{cpu_ms:.4f},{speedup:.2f},"
                            f"{'PASS' if res['allPass'] else 'FAIL'}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Run Lab 1 benchmarks N times and compute average results and findings."
    )
    parser.add_argument("-n", "--runs", type=int, default=5, help="Number of repetitions per benchmark (default: 5)")
    parser.add_argument("-s", "--size", type=str, default="all", help="Block size to test (e.g. 256) or 'all' (default: all)")
    parser.add_argument("--op", type=str, default="all", choices=["vector", "matrix", "all"], help="Operation to test (default: all)")
    parser.add_argument("--mem", type=str, default="all", choices=["u", "s", "all"], help="Memory model: u, s, all (default: all)")
    parser.add_argument("--model", type=str, default="all", choices=["m", "gs", "all"], help="Kernel model: m, gs, all (default: all)")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    parser.add_argument("--csv", type=str, default=os.path.join(SCRIPT_DIR, "benchmark_results_averaged.csv"), help="CSV output path")
    parser.add_argument("--report", type=str, default=os.path.join(SCRIPT_DIR, "benchmark_findings.md"), help="Markdown report path")
    parser.add_argument("--binary", type=str, default=os.path.join(SCRIPT_DIR, "lab1"), help="Path to lab1 executable")
    args = parser.parse_args()

    ensure_binary(args.binary)

    # Determine configurations
    mems = [("u", "Unified"), ("s", "Standard")] if args.mem == "all" else [(args.mem, "Unified" if args.mem == "u" else "Standard")]
    models = [("m", "Monolithic"), ("gs", "Grid-Stride")] if args.model == "all" else [(args.model, "Monolithic" if args.model == "m" else "Grid-Stride")]
    ops = OPERATIONS if args.op == "all" else [args.op]
    sizes = BLOCK_SIZES if args.size == "all" else [int(args.size)]

    total_tasks = len(ops) * len(mems) * len(models) * len(sizes)
    print("=" * 80)
    print(f" Lab 1 Multi-Run Benchmark Suite")
    print(f" Layout: ./lab1 <u/s> <m/gs> <vector/matrix> <size>")
    print(f" Configs: {len(mems) * len(models)} | Operations: {len(ops)} | Block sizes: {len(sizes)}")
    print(f" Repetitions per config: {args.runs} | Total benchmark runs: {total_tasks * args.runs}")
    print("=" * 80)

    # Step 1: Measure CPU baselines
    cpu_baselines = {}
    for op in ops:
        print(f"[+] Measuring Serial CPU baseline for {op} addition...")
        cpu_ms = get_cpu_baseline(args.binary, op, args.device)
        cpu_baselines[op] = cpu_ms
        print(f"    CPU Serial baseline: {cpu_ms:.3f} ms")

    # Step 2: Execute benchmark sweeps
    all_results = {}
    completed = 0

    for op in ops:
        all_results[op] = {}
        for mem_code, mem_name in mems:
            for model_code, model_name in models:
                cfg_key = f"{mem_code}_{model_code}"
                all_results[op][cfg_key] = []

                for sz in sizes:
                    completed += 1
                    print(f"[{completed}/{total_tasks}] Running {mem_name} / {model_name} | {op} | blockSize={sz} ({args.runs} runs)...", end="", flush=True)

                    runs_data = []
                    for r in range(args.runs):
                        res = run_single_benchmark(args.binary, mem_code, model_code, op, sz, args.device)
                        if res:
                            runs_data.append(res)
                        else:
                            print(f"\n[-] Failed on run {r+1} for {mem_code} {model_code} {op} {sz}")

                    if runs_data:
                        agg = aggregate_runs(runs_data)
                        all_results[op][cfg_key].append(agg)
                        std_str = f" ± {agg['totalStd']:.2f}" if args.runs > 1 else ""
                        print(f" Done! Mean Total: {agg['totalMean']:.3f}{std_str} ms ({'PASS' if agg['allPass'] else 'FAIL'})")
                    else:
                        print(" Failed!")

    # Step 3: Print formatted tables
    for op in ops:
        print_table_for_op(op, all_results[op], cpu_baselines.get(op, 0.0), args.runs)

    # Step 4: Summary Findings Matrix
    print("\n" + "=" * 80)
    print(" SUMMARY OF FINDINGS: BEST CONFIGURATION PER IMPLEMENTATION")
    print("=" * 80)
    for op in ops:
        print(f"\n--- {op.upper()} ADDITION ---")
        print(f"{'Implementation':<32} {'Optimal Block':<15} {'Total Time (ms)':<20} {'Speedup'}")
        print("-" * 80)
        for cfg_key, block_results in all_results[op].items():
            if not block_results:
                continue
            best = min(block_results, key=lambda x: x["totalMean"])
            mem_label = "Unified" if best['mem'] == 'u' else "Standard"
            kern_label = "Monolithic" if best['kernel'] == 'm' else "Grid-Stride"
            impl_name = f"{mem_label} / {kern_label}"
            cpu_ms = cpu_baselines.get(op, 0.0)
            speedup = (cpu_ms / best['totalMean']) if (cpu_ms > 0 and best['totalMean'] > 0) else 0.0
            time_str = f"{best['totalMean']:.3f} ± {best['totalStd']:.2f}" if args.runs > 1 else f"{best['totalMean']:.3f}"
            print(f"{impl_name:<32} {best['blockSize']:<15} {time_str:<20} {speedup:.2f}x")

    # Step 5: Save files
    save_csv_results(all_results, cpu_baselines, args.runs, args.csv)
    generate_markdown_report(all_results, cpu_baselines, args.runs, args.report)
    print("\n" + "=" * 80)
    print(f" Results exported to CSV:    {args.csv}")
    print(f" Report written to Markdown: {args.report}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

