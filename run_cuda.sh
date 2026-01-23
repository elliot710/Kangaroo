#!/bin/bash
#
# Kangaroo H200 GPU - Puzzle 135 Solver
# Run script for Linux
#

# Configuration
WORK_FILE="puzzle135_work.dat"
OUTPUT_FILE="KEYFOUND.txt"
SAVE_INTERVAL=120  # seconds
DP_BITS=25         # Distinguished point bits

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)

if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No NVIDIA GPUs detected"
    exit 1
fi

echo "============================================"
echo "  Kangaroo H200 - Bitcoin Puzzle 135 Solver"
echo "============================================"
echo ""
echo "Detected $NUM_GPUS GPU(s)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Build GPU ID and grid size strings
GPU_IDS=""
GRID_SIZES=""
for ((i=0; i<$NUM_GPUS; i++)); do
    if [ $i -eq 0 ]; then
        GPU_IDS="$i"
        GRID_SIZES="264,256"
    else
        GPU_IDS="$GPU_IDS,$i"
        GRID_SIZES="$GRID_SIZES,264,256"
    fi
done

echo "Using GPUs: $GPU_IDS"
echo "Grid sizes: $GRID_SIZES"
echo "DP bits: $DP_BITS"
echo "Work file: $WORK_FILE"
echo "Save interval: ${SAVE_INTERVAL}s"
echo ""

# Check if work file exists (resume mode)
if [ -f "$WORK_FILE" ]; then
    echo "Found existing work file, resuming..."
    RESUME_FLAG="-i $WORK_FILE"
else
    echo "Starting fresh computation..."
    RESUME_FLAG=""
fi

# Set CUDA environment
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Run Kangaroo
echo "Starting Kangaroo..."
echo "Press Ctrl+C to stop (progress will be saved)"
echo ""

./kangaroo \
    -t 0 \
    -gpu \
    -gpuId $GPU_IDS \
    -g $GRID_SIZES \
    -d $DP_BITS \
    $RESUME_FLAG \
    -w $WORK_FILE \
    -wi $SAVE_INTERVAL \
    -ws \
    -o $OUTPUT_FILE \
    puzzle135.txt

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "============================================"
    if [ -f "$OUTPUT_FILE" ]; then
        echo "SUCCESS! Key found! Check $OUTPUT_FILE"
        cat $OUTPUT_FILE
    else
        echo "Search completed or stopped"
    fi
    echo "============================================"
else
    echo ""
    echo "Kangaroo exited with code $EXIT_CODE"
fi
