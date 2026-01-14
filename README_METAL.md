# Kangaroo Metal - GPU Bitcoin Puzzle Solver for Apple Silicon

A high-performance Metal GPU implementation of Pollard's Kangaroo algorithm for solving the Bitcoin Puzzle Challenge, optimized for Apple Silicon (M1/M2/M3/M4) Macs.

Based on [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo) CUDA implementation.

## Performance

| Device | Threads | Speed |
|--------|---------|-------|
| Apple M4 Pro | 32x512 (16384) | ~4800 MK/s |
| Apple M4 Pro | 16x256 (4096)  | ~2500 MK/s |
| Apple M4 Pro | 8×128 (1024)   | ~1200 MK/s |


## Building

```bash
# Build the Metal GPU version
make -f Makefile.metal -j8

# This creates: KangarooMetal
```

## Quick Start

### 1. Create Input File

Create a file (e.g., `puzzle135.txt`) with 3 lines:
```
4000000000000000000000000000000000
7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16
```

- Line 1: Start of search range (hex)
- Line 2: End of search range (hex)
- Line 3: Public key to find (compressed, hex)

### 2. Run

```bash
# Basic GPU-only run
./KangarooMetal -t 0 -gpu puzzle135.txt

# With work file saving (recommended for long runs)
./KangarooMetal -t 0 -gpu -g 16,128 -w puzzle135_work.dat puzzle135.txt

# Resume from saved work
./KangarooMetal -t 0 -gpu -g 16,128 -i puzzle135_work.dat -w puzzle135_work.dat puzzle135.txt

# Save found key to file
./KangarooMetal -t 0 -gpu -g 16,128 -w puzzle135_work.dat -o found_key.txt puzzle135.txt
```

## Command Line Arguments

### Essential

| Argument | Description |
|----------|-------------|
| `inFile` | Input configuration file (required) |
| `-t N` | Number of CPU threads (`-t 0` for GPU only) |
| `-gpu` | Enable GPU (Metal) |
| `-gpuId N` | GPU device ID (default: 0) |
| `-g X,Y` | Grid size: X thread groups × Y threads per group |
| `-d N` | Distinguished point bits (default: auto) |

### Work Files

| Argument | Description |
|----------|-------------|
| `-w file` | Save work to file (on exit and periodically) |
| `-i file` | Load/resume work from file |
| `-wi N` | Save interval in seconds (default: 60) |
| `-ws` | Also save kangaroo positions (larger files, exact resume) |
| `-o file` | Output found private key to file |

### Info/Debug

| Argument | Description |
|----------|-------------|
| `-v` | Print version |
| `-h` | Print help |
| `-l` | List available GPUs |
| `-check` | Verify GPU kernel vs CPU |
| `-winfo file` | Show work file info |

## Understanding the Output

```
[761.45 MK/s][GPU 761.45 MK/s][Count 2^32.92][Dead 0][12s (Avg 13272.6y)][2.0/4.0MB]
```

| Field | Meaning |
|-------|---------|
| `761.45 MK/s` | Overall speed (million keys/second) |
| `GPU 761.45 MK/s` | GPU speed |
| `Count 2^32.92` | Total operations performed (2^32.92 ≈ 8 billion) |
| `Dead 0` | Collisions within same herd (should be 0 or low) |
| `12s` | Time elapsed |
| `Avg 13272.6y` | Estimated average time to solve (statistical) |
| `2.0/4.0MB` | Hash table usage (used/allocated) |

## Bitcoin Puzzle #135

- **Range**: 2^134 to 2^135 (134-bit key space)
- **Public Key**: `02145d2611c823a396ef6712ce0f712f09b9b4f3135e3e0aa3230fb9b6d08d1e16`
- **Address**: `16RGFo6hjq9ym6Pj7N5H7L1NR1rVPJyw2v`
- **Prize**: 13.5 BTC
- **Expected Operations**: 2^68.11 (using Pollard's Kangaroo)
- **Estimated Time at 760 MK/s**: ~13,000 years (solo)

## How It Works

Pollard's Kangaroo algorithm finds the discrete logarithm (private key) in O(√n) operations:

1. **Tame Kangaroos**: Start from known points, jump randomly
2. **Wild Kangaroos**: Start from the target public key, jump randomly  
3. **Distinguished Points**: When a kangaroo lands on a "special" point (leading zero bits), record it
4. **Collision**: When tame and wild kangaroos land on the same point, the private key can be computed

The algorithm is probabilistic - on average it finds the key in 2^(n/2) operations for an n-bit range.

## Tips for Best Performance

1. **Use GPU only**: `-t 0 -gpu` (CPU is much slower)
2. **Maximize threads**: `-g 16,128` or higher if Mac stays responsive
3. **Always save work**: `-w work.dat` prevents losing progress on crashes
4. **Lower DP for speed**: `-d 44` uses more RAM but finds collisions faster

## Troubleshooting

### Mac freezes during GPU computation
- Reduce grid size: `-g 8,64` instead of `-g 16,128`
- The implementation includes 500μs delays between kernel dispatches to prevent display freezing

### Work file not saved
- Work is saved on graceful shutdown (Ctrl+C) and periodically (default 60s)
- Use `-wi 30` for more frequent saves

### GPU not detected
- Ensure you're on macOS with Metal support
- Run `./KangarooMetal -l` to list GPUs

## Files

| File | Purpose |
|------|---------|
| `KangarooMetal` | Main executable |
| `KangarooKernel.metallib` | Compiled Metal shader |
| `puzzle135.txt` | Input config for puzzle #135 |
| `puzzle135_work.dat` | Saved work/progress |

## License

GNU General Public License v3.0

## Credits

- Original CUDA implementation: [JeanLucPons/Kangaroo](https://github.com/JeanLucPons/Kangaroo)
- Metal port: Adapted for Apple Silicon

## How It Works

**Pollard's Kangaroo Algorithm** solves the Elliptic Curve Discrete Logarithm Problem (ECDLP) in O(√n) time:

1. **Tame kangaroos** start at random points within the search range (known private keys)
2. **Wild kangaroos** start at the target public key (unknown private key)
3. **Random walks** - each kangaroo jumps by adding pre-computed points based on current position
4. **Collision detection** - when tame and wild land on the same point, we can compute the private key
5. **Distinguished Points** - only report points with N leading zero bits to reduce storage

```
Range: [------------------------------------]
        ^         ^    ^        ^
        |         |    |        |
      Tame1    Wild1  Tame2   Wild2
        |         |    |        |
        v         v    v        v
       jump      jump jump     jump
        ...collision!...
```

## Example Usage

```bash
# Puzzle 135 (recommended settings for M4 Pro)
./KangarooMetal -t 0 -gpu -gpuId 0 -g 32,512 -w puzzle135_work.dat puzzle135.txt

# Resume from saved work
./KangarooMetal -t 0 -gpu -gpuId 0 -g 32,512 -i puzzle135_work.dat -w puzzle135_work.dat puzzle135.txt
```