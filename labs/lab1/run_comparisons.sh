#!/usr/bin/env bash
# ==============================================================================
# run_comparisons.sh
# 
# Comprehensive benchmark runner for Lab 1: First CUDA Applications
# Course: Massively Parallel Programming on GPUs
# 
# Runs all 4 implementations across all block sizes:
#   1. Unified-Memory / Monolithic-Kernels
#   2. Unified-Memory / Grid-Stride-Loop-Kernels
#   3. Standard-Memory / Monolithic-Kernels
#   4. Standard-Memory / Grid-Stride-Loop-Kernels
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo " Building Lab 1 Implementations..."
echo "================================================================================"
make all

echo ""
echo "================================================================================"
echo " [1/4] Running Unified-Memory / Monolithic-Kernels"
echo "================================================================================"
./unified_monolithic "$@"

echo ""
echo "================================================================================"
echo " [2/4] Running Unified-Memory / Grid-Stride-Loop-Kernels"
echo "================================================================================"
./unified_grid_stride "$@"

echo ""
echo "================================================================================"
echo " [3/4] Running Standard-Memory / Monolithic-Kernels"
echo "================================================================================"
./standard_monolithic "$@"

echo ""
echo "================================================================================"
echo " [4/4] Running Standard-Memory / Grid-Stride-Loop-Kernels"
echo "================================================================================"
./standard_grid_stride "$@"

echo ""
echo "================================================================================"
echo " All 4 implementations completed successfully!"
echo "================================================================================"

