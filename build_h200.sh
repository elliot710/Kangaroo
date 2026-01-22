#!/bin/bash
#
# Build script for Kangaroo H200 GPU version
# Optimized for Bitcoin Puzzle 135
#

set -e

echo "============================================"
echo "  Kangaroo H200 GPU Build Script"
echo "============================================"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please install CUDA toolkit."
    echo "Try: export PATH=/usr/local/cuda/bin:\$PATH"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
echo "CUDA version: $CUDA_VERSION"

# Check for g++
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ not found. Please install gcc/g++."
    exit 1
fi

GCC_VERSION=$(g++ --version | head -1)
echo "GCC: $GCC_VERSION"
echo ""

# Clean previous build
echo "Cleaning previous build..."
make -f Makefile.h200 clean 2>/dev/null || true

# Create object directories
mkdir -p obj obj/GPU obj/SECPK1

# Build
echo ""
echo "Building Kangaroo for H200..."
echo ""
make -f Makefile.h200 all

if [ -f "./kangaroo" ]; then
    echo ""
    echo "============================================"
    echo "  Build successful!"
    echo "============================================"
    echo ""
    echo "Binary: ./kangaroo"
    ls -lh ./kangaroo
    echo ""
    echo "To run:"
    echo "  ./run_h200.sh"
    echo ""
    echo "Or manually:"
    echo "  ./kangaroo -gpu -gpuId 0 puzzle135.txt"
    echo ""
else
    echo ""
    echo "Build failed!"
    exit 1
fi
