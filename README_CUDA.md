# Kangaroo for NVIDIA H200 GPU - Puzzle 135 Solver

This is an optimized version of Kangaroo ECDLP solver specifically configured for solving Bitcoin Puzzle #135 using NVIDIA H200 GPUs on Linux.

## Puzzle 135 Details

- **Range**: [2^134, 2^135-1] = `4000000000000000000000000000000000` to `7fffffffffffffffffffffffffffffffff`
- **Public Key**: `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`
- **Address**: `16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v`
- **Prize**: 13.5 BTC

## H200 GPU Specifications

- Compute Capability: 9.0 (Hopper)
- SMs: 132
- CUDA Cores per SM: 128
- Total CUDA Cores: 16,896
- Memory: 80GB/141GB HBM3/HBM3e
- Memory Bandwidth: ~4.8 TB/s

## Optimizations Applied

1. **Compute Capability 9.0**: Added support for Hopper architecture
2. **Increased GPU Group Size**: 256 (vs 128) for better occupancy
3. **Increased Jump Table**: 64 jumps (vs 32) for better cache utilization
4. **Increased Runs per Kernel**: 256 (vs 128) for better throughput
5. **Optimized Grid Size**: Auto-tuned for H200's 132 SMs

## Building

### Prerequisites

- CUDA Toolkit 12.0 or newer (for H200 support)
- GCC 7.0 or newer
- Linux x86_64

### Build Commands

```bash
# Clean previous build
make -f Makefile.h200 clean

# Build optimized for H200
make -f Makefile.h200 all
```

### Custom CUDA Path

If CUDA is installed in a non-standard location, edit `Makefile.h200`:

```makefile
CUDA = /path/to/your/cuda
```

## Running

### Basic Usage

```bash
# Single H200 GPU
./kangaroo -gpu -gpuId 0 puzzle135.txt

# Multiple H200 GPUs
./kangaroo -gpu -gpuId 0,1,2,3 puzzle135.txt

# With work file saving every 60 seconds
./kangaroo -gpu -gpuId 0 -w work135.dat -wi 60 -ws puzzle135.txt

# Resume from work file
./kangaroo -gpu -gpuId 0 -i work135.dat -w work135.dat -wi 60 -ws
```

### Recommended Settings for H200

```bash
# Optimal configuration for single H200
./kangaroo -t 0 -gpu -gpuId 0 -g 264,256 -d 25 -w puzzle135_work.dat -wi 120 -ws -o found_key.txt puzzle135.txt

# Explanation:
# -t 0          : No CPU threads (GPU only)
# -gpu          : Enable GPU
# -gpuId 0      : Use GPU 0
# -g 264,256    : Grid 264 blocks x 256 threads (2 * 132 SMs)
# -d 25         : Distinguished point bits (auto-calculated optimal is ~25 for 134-bit range)
# -w            : Work file for saving progress
# -wi 120       : Save work every 120 seconds
# -ws           : Save kangaroos in work file
# -o            : Output file for found key
```

### Multi-GPU Configuration

```bash
# 4x H200 GPUs
./kangaroo -t 0 -gpu -gpuId 0,1,2,3 -g 264,256,264,256,264,256,264,256 -d 27 -w puzzle135_work.dat -wi 120 -ws -o found_key.txt puzzle135.txt
```

## Expected Performance

Based on H200 specifications and similar workloads:

| Configuration | Est. Keys/sec | Est. Time to Solve |
|---------------|--------------|-------------------|
| 1x H200       | ~15-20 GKey/s | ~2-4 years |
| 4x H200       | ~60-80 GKey/s | ~6-12 months |
| 8x H200       | ~120-160 GKey/s | ~3-6 months |

**Note**: Actual performance depends on memory bandwidth, system configuration, and driver version.

## Expected Operations

For a 134-bit search range:
- Expected operations: 2^68.05 ≈ 3 × 10^20
- With symmetry optimization: ~2^67.5

## Work File Management

### Save work periodically

```bash
./kangaroo -gpu -w work.dat -wi 300 -ws puzzle135.txt
```

### Resume from saved work

```bash
./kangaroo -gpu -i work.dat -w work.dat -wi 300 -ws
```

### Merge multiple work files

```bash
./kangaroo -wm work1.dat work2.dat merged.dat
```

### Check work file info

```bash
./kangaroo -winfo work.dat
```

## Distributed Computing (Client/Server)

### Server

```bash
./kangaroo -s -d 25 -w server_work.dat -wi 300 -wsplit puzzle135.txt
```

### Clients

```bash
./kangaroo -t 0 -gpu -gpuId 0 -c server_ip -w client_kang.dat -wss -wi 120
```

## Files

- `puzzle135.txt` - Configuration file for puzzle 135
- `Makefile.h200` - Makefile optimized for H200
- `Constants.h` - Updated with H200-optimized parameters
- `GPU/GPUEngine.cu` - Updated with Hopper architecture support

## Important Notes

1. **Power Consumption**: H200 can draw up to 700W. Ensure adequate power supply and cooling.
2. **Memory**: The program requires significant RAM for the hash table. For 134-bit range, expect ~10-50GB RAM usage.
3. **ECC**: H200 uses ECC memory by default. This is recommended for long-running computations.
4. **Persistence**: Use work file saving (`-w -wi -ws`) to prevent loss of progress.

## Troubleshooting

### CUDA not found
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Out of Memory
- Reduce grid size: `-g 132,256` instead of `-g 264,256`
- Increase DP bits: `-d 30` (reduces RAM but increases operations)

### Low Performance
- Check GPU utilization: `nvidia-smi`
- Ensure no thermal throttling
- Verify CUDA version compatibility

## License

GNU General Public License v3.0

## Credits

Original Kangaroo by Jean-Luc PONS
H200 optimizations for Puzzle 135
