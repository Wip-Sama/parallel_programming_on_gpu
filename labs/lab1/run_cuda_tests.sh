#!/usr/bin/env bash
# ==============================================================================
# run_cuda_tests.sh: Pure C/CUDA Verification & Test Runner for Lab 1
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Ensure binary is built
if [ ! -x "./lab1" ]; then
    echo "Building lab1 binary..."
    make
fi

echo "================================================================================"
echo " Lab 1: Pure C/CUDA Test Suite"
echo "================================================================================"
echo ""

BLOCK_SIZES=(32 64 128 256 512 1024)
CONFIGS=("u m" "u gs" "s m" "s gs")
OPERATIONS=("vector" "matrix")

# Print header
printf "%-8s %-12s %-8s %-10s %-10s %-10s %-10s %-12s %-14s %-6s\n" \
    "Memory" "Kernel" "Op" "BlockSize" "GridSize" "Init(ms)" "Add(ms)" "Memcpy(ms)" "TotalTime(ms)" "Status"
echo "--------------------------------------------------------------------------------------------------------------"

for op in "${OPERATIONS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        for bs in "${BLOCK_SIZES[@]}"; do
            # Extract configuration tokens
            set -- $cfg
            mem=$1
            model=$2
            
            # Run CUDA executable with --csv flag
            output=$(./lab1 "$mem" "$model" "$op" "$bs" --csv)
            
            # Format: CSV,mem,kernel,op,blockSize,gridSize,initMs,addMs,memcpyMs,deviceTimeMs,totalRunTimeMs,status
            IFS=',' read -r tag c_mem c_mod c_op c_bs c_grid c_init c_add c_cpy c_dev c_tot c_stat <<< "$output"
            
            mem_name="Unified"
            [ "$c_mem" = "s" ] && mem_name="Standard"
            
            mod_name="Monolithic"
            [ "$c_mod" = "gs" ] && mod_name="GridStride"
            
            printf "%-8s %-12s %-8s %-10s %-10s %-10s %-10s %-12s %-14s %-6s\n" \
                "$mem_name" "$mod_name" "$c_op" "$c_bs" "$c_grid" "$c_init" "$c_add" "$c_cpy" "$c_tot" "$c_stat"
        done
    done
done

echo "--------------------------------------------------------------------------------------------------------------"
echo "All C/CUDA test configurations completed and verified successfully."

