#!/usr/bin/env bash
# ==============================================================================
# run_averages.sh
# 
# Runs each CUDA implementation N times across block sizes and averages findings.
# 
# Usage:
#   ./run_averages.sh             # 5 runs per configuration (all block sizes)
#   ./run_averages.sh 10          # 10 runs per configuration (all block sizes)
#   ./run_averages.sh 5 256       # 5 runs specifically for block size 256
#   ./run_averages.sh 5 all vector# 5 runs specifically for vector addition
# ==============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=${1:-5}
SIZE=${2:-all}
OP=${3:-all}

make lab1

python3 average_benchmarks.py -n "$RUNS" -s "$SIZE" --op "$OP"

