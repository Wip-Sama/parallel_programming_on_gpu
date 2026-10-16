#!/usr/bin/env bash
# ==============================================================================
# run_comparisons.sh
# 
# Comprehensive benchmark runner for Lab 1: First CUDA Applications
# Course: Massively Parallel Programming on GPUs
# 
# Usage:
#   ./run_comparisons.sh             # Runs full comparison sweep across all 4 configurations
#   ./run_comparisons.sh 5           # Runs each configuration 5 times and averages findings
#   ./run_comparisons.sh 5 256       # Runs each configuration 5 times with blockSize=256
#   ./run_comparisons.sh 5 256 vector# Runs each configuration 5 times for vector addition
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

make lab1

RUNS=${1:-5}
SIZE=${2:-all}
OP=${3:-all}

python3 average_benchmarks.py -n "$RUNS" -s "$SIZE" --op "$OP"
