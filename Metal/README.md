# Kangaroo Metal GPU Implementation

High-performance Metal shader implementation of Pollard's Kangaroo algorithm for solving the SECP256K1 discrete logarithm problem. Based on [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo) CUDA implementation, optimized for Apple Silicon (M1/M2/M3/M4) GPUs.

## Overview

This implementation ports the CUDA-based Pollard's Kangaroo algorithm to Apple's Metal framework, enabling native GPU acceleration on macOS devices. The algorithm is used to solve the Bitcoin Puzzle challenge by searching for private keys in a constrained key space.

## Features

- **Native Metal Shaders**: Full implementation of SECP256K1 elliptic curve operations in Metal Shading Language
- **Batch Modular Inverse**: Montgomery's trick for efficient batch inversions
- **Distinguished Points**: DP-based collision detection for reduced memory usage
- **Symmetry Mode**: Optional optimization using the symmetry of the elliptic curve
- **Unified Memory**: Takes advantage of Apple Silicon's unified memory architecture
- **High Performance**: Optimized 256-bit arithmetic for GPU execution

## Requirements

- macOS 11.0 (Big Sur) or later
- Apple Silicon (M1/M2/M3/M4) or AMD GPU with Metal support
- Xcode Command Line Tools

## Building

```bash
# Clone or navigate to the Kangaroo directory
cd Kangaroo-master

# Build with Metal support
make -f Makefile.metal

# Or with symmetry optimization enabled
make -f Makefile.metal CXXFLAGS="-O3 -std=c++17 -Wall -DWITHMETAL -DUSE_SYMMETRY"
```

## Usage

```bash
# Run with Metal GPU acceleration
./KangarooMetal -gpu -t 4 -d 16 in.txt

# Options:
#   -gpu        Enable GPU acceleration
#   -t <n>      Number of CPU threads
#   -d <n>      Distinguished point size (bits)
#   -w <file>   Work file for checkpointing
```

## Algorithm

The Pollard's Kangaroo (also known as Lambda) algorithm works by:

1. **Tame Kangaroos**: Start at known positions within the search range
2. **Wild Kangaroos**: Start at the target public key
3. **Random Jumps**: Both types make deterministic jumps based on their x-coordinate
4. **Collision Detection**: When a tame and wild kangaroo land on the same point, we can compute the private key

### Complexity

Expected operations: **O(√n)** where n is the size of the search range

For puzzle #71 (70-bit key space):
- Expected operations: ~2^35 ≈ 34 billion
- With GPU: Several hours on Apple Silicon

## Metal Shader Architecture

```
Metal/
├── KangarooKernel.metal   # Main compute shader
├── MetalEngine.h          # C++ interface header
└── MetalEngine.mm         # Objective-C++ implementation
```

### Key Components

1. **256-bit Arithmetic**: Full modular arithmetic for SECP256K1 field
2. **Point Addition**: Elliptic curve point addition in affine coordinates
3. **Batch Inversion**: Grouped modular inverse using Montgomery's trick
4. **DP Detection**: Atomic operations for distinguished point collection

## Performance Optimization

The Metal implementation includes several optimizations:

1. **Coalesced Memory Access**: Thread layout optimized for memory bandwidth
2. **Threadgroup Size**: Tuned for Apple GPU architecture (128 threads/group)
3. **Register Pressure**: Careful management of per-thread storage
4. **Batch Processing**: Multiple kangaroos per thread (GPU_GRP_SIZE = 128)
5. **Loop Unrolling**: NB_RUN = 64 iterations per kernel invocation

## Configuration

Key constants in `Constants.h`:

```cpp
#define NB_JUMP 32       // Number of precomputed jump points
#define GPU_GRP_SIZE 128 // Kangaroos per thread
#define NB_RUN 64        // Iterations per kernel call
```

## Bitcoin Puzzle Context

This tool is designed for the Bitcoin Puzzle challenge:
- First #1–70, #75, #80, #85, #90, #95, #100, #105, #110, #115, #120, #125, #130 have been solved
- Next target: #71 (71-bit key space, ~7.1 BTC prize)
- Or #135 if you have the computational resources for baby-step giant-step

## Comparison with CUDA

| Feature | CUDA | Metal |
|---------|------|-------|
| Platform | NVIDIA GPUs | Apple GPUs |
| Memory Model | Explicit | Unified |
| Inline Assembly | PTX | Not available |
| Atomic Operations | Full support | Full support |
| Thread Synchronization | __syncthreads() | threadgroup_barrier() |

## License

GNU General Public License v3.0

Based on work by Jean Luc PONS.

## References

- [Original Kangaroo Repository](https://github.com/JeanLucPons/Kangaroo)
- [Bitcoin Puzzle Transaction](https://privatekeys.pw/puzzles/bitcoin-puzzle-tx)
- [Pollard's Kangaroo Algorithm](https://en.wikipedia.org/wiki/Pollard%27s_kangaroo_algorithm)
- [SECP256K1 Curve Parameters](https://en.bitcoin.it/wiki/Secp256k1)
