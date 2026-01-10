/*
 * Metal Shader for Pollard's Kangaroo Algorithm
 * Based on JeanLucPons/Kangaroo CUDA implementation
 * 
 * Copyright (c) 2024
 * Licensed under GNU General Public License v3.0
 *
 * Optimized for Apple Silicon (M1/M2/M3) GPUs
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------

#define NB_JUMP 32
#define GPU_GRP_SIZE 128
#define NB_RUN 8  // Work iterations per kernel dispatch
// #define USE_SYMMETRY  // Disabled - causes GPU timeout on Apple Silicon

#ifdef USE_SYMMETRY
#define KSIZE 11
#else
#define KSIZE 10
#endif

#define ITEM_SIZE 56
#define ITEM_SIZE32 (ITEM_SIZE / 4)

// SECP256K1 field prime: p = 2^256 - 2^32 - 977
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
constant uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
constant uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
constant uint64_t P2 = 0xFFFFFFFFFFFFFFFFULL;
constant uint64_t P3 = 0xFFFFFFFFFFFFFFFFULL;

// Order constant for symmetry mode
constant uint64_t ORDER0 = 0xBFD25E8CD0364141ULL;
constant uint64_t ORDER1 = 0xBAAEDCE6AF48A03BULL;
constant uint64_t ORDER2 = 0xFFFFFFFFFFFFFFFEULL;
constant uint64_t ORDER3 = 0xFFFFFFFFFFFFFFFFULL;

// 64-bit LSB negative inverse of P (mod 2^64)
constant uint64_t MM64 = 0xD838091DD2253531ULL;
constant uint64_t MSK62 = 0x3FFFFFFFFFFFFFFFULL;

// ---------------------------------------------------------------------------------
// 256-bit integer type for SECP256K1 operations
// ---------------------------------------------------------------------------------

struct uint256_t {
    uint64_t d[4];
    
    uint256_t() {
        d[0] = d[1] = d[2] = d[3] = 0;
    }
    
    uint256_t(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
        d[0] = v0; d[1] = v1; d[2] = v2; d[3] = v3;
    }
};

// Extended 320-bit type for modular inversion
struct uint320_t {
    uint64_t d[5];
};

// ---------------------------------------------------------------------------------
// Kangaroo data structure
// ---------------------------------------------------------------------------------

struct Kangaroo {
    uint256_t px;        // Point X coordinate
    uint256_t py;        // Point Y coordinate
    uint64_t dist[2];    // Distance (128-bit)
#ifdef USE_SYMMETRY
    uint64_t lastJump;   // Last jump for symmetry
#endif
};

// ---------------------------------------------------------------------------------
// Output structure for distinguished points
// ---------------------------------------------------------------------------------

struct DPOutput {
    uint32_t x[8];       // Point X (256-bit)
    uint32_t dist[4];    // Distance (128-bit)
    uint64_t kIdx;       // Kangaroo index
};

// ---------------------------------------------------------------------------------
// Arithmetic helper functions with carry propagation
// ---------------------------------------------------------------------------------

// Add with carry
inline uint64_t addcc(uint64_t a, uint64_t b, thread bool& carry) {
    uint64_t result = a + b;
    carry = result < a;
    return result;
}

inline uint64_t addc(uint64_t a, uint64_t b, thread bool& carry) {
    uint64_t result = a + b + (carry ? 1ULL : 0ULL);
    carry = (result < a) || (carry && result == a);
    return result;
}

// Subtract with borrow
inline uint64_t subcc(uint64_t a, uint64_t b, thread bool& borrow) {
    borrow = a < b;
    return a - b;
}

inline uint64_t subc(uint64_t a, uint64_t b, thread bool& borrow) {
    uint64_t c = borrow ? 1ULL : 0ULL;
    borrow = (a < b) || (a == b && borrow);
    return a - b - c;
}

// 64x64 -> 128 multiplication
inline void mul64(uint64_t a, uint64_t b, thread uint64_t& hi, thread uint64_t& lo) {
    uint64_t al = a & 0xFFFFFFFFULL;
    uint64_t ah = a >> 32;
    uint64_t bl = b & 0xFFFFFFFFULL;
    uint64_t bh = b >> 32;
    
    uint64_t p0 = al * bl;
    uint64_t p1 = al * bh;
    uint64_t p2 = ah * bl;
    uint64_t p3 = ah * bh;
    
    uint64_t mid = p1 + p2;
    bool carry = mid < p1;
    
    uint64_t mid_lo = mid << 32;
    uint64_t mid_hi = (mid >> 32) | (carry ? 0x100000000ULL : 0ULL);
    
    lo = p0 + mid_lo;
    hi = p3 + mid_hi + (lo < p0 ? 1ULL : 0ULL);
}

// ---------------------------------------------------------------------------------
// 256-bit modular arithmetic for SECP256K1
// ---------------------------------------------------------------------------------

// Load 256-bit value
inline void load256(thread uint256_t& r, thread const uint256_t& a) {
    r.d[0] = a.d[0];
    r.d[1] = a.d[1];
    r.d[2] = a.d[2];
    r.d[3] = a.d[3];
}

// Check if 256-bit value is zero
inline bool isZero256(thread const uint256_t& a) {
    return (a.d[0] | a.d[1] | a.d[2] | a.d[3]) == 0ULL;
}

// Compare two 256-bit values (returns -1, 0, 1)
inline int cmp256(thread const uint256_t& a, thread const uint256_t& b) {
    for (int i = 3; i >= 0; i--) {
        if (a.d[i] > b.d[i]) return 1;
        if (a.d[i] < b.d[i]) return -1;
    }
    return 0;
}

// Add P to 256-bit value (for underflow correction)
inline void addP256(thread uint256_t& r) {
    bool carry;
    r.d[0] = addcc(r.d[0], P0, carry);
    r.d[1] = addc(r.d[1], P1, carry);
    r.d[2] = addc(r.d[2], P2, carry);
    r.d[3] = addc(r.d[3], P3, carry);
}

// Subtract P from 256-bit value (for overflow correction)
inline void subP256(thread uint256_t& r) {
    bool borrow;
    r.d[0] = subcc(r.d[0], P0, borrow);
    r.d[1] = subc(r.d[1], P1, borrow);
    r.d[2] = subc(r.d[2], P2, borrow);
    r.d[3] = subc(r.d[3], P3, borrow);
}

// Modular subtraction: r = a - b (mod P)
inline void modSub256(thread uint256_t& r, thread const uint256_t& a, thread const uint256_t& b) {
    bool borrow;
    r.d[0] = subcc(a.d[0], b.d[0], borrow);
    r.d[1] = subc(a.d[1], b.d[1], borrow);
    r.d[2] = subc(a.d[2], b.d[2], borrow);
    r.d[3] = subc(a.d[3], b.d[3], borrow);
    
    // If borrow, add P back
    if (borrow) {
        addP256(r);
    }
}

// Modular subtraction in place: r = r - b (mod P)
inline void modSub256(thread uint256_t& r, thread const uint256_t& b) {
    bool borrow;
    r.d[0] = subcc(r.d[0], b.d[0], borrow);
    r.d[1] = subc(r.d[1], b.d[1], borrow);
    r.d[2] = subc(r.d[2], b.d[2], borrow);
    r.d[3] = subc(r.d[3], b.d[3], borrow);
    
    if (borrow) {
        addP256(r);
    }
}

// Modular negation: r = -a (mod P)
inline void modNeg256(thread uint256_t& r, thread const uint256_t& a) {
    bool borrow;
    uint64_t t0 = subcc(0ULL, a.d[0], borrow);
    uint64_t t1 = subc(0ULL, a.d[1], borrow);
    uint64_t t2 = subc(0ULL, a.d[2], borrow);
    uint64_t t3 = subc(0ULL, a.d[3], borrow);
    
    bool carry;
    r.d[0] = addcc(t0, P0, carry);
    r.d[1] = addc(t1, P1, carry);
    r.d[2] = addc(t2, P2, carry);
    r.d[3] = addc(t3, P3, carry);
}

// Modular negation in place
inline void modNeg256(thread uint256_t& r) {
    uint256_t tmp;
    modNeg256(tmp, r);
    load256(r, tmp);
}

// 128-bit addition for distance
inline void add128(thread uint64_t* r, constant uint64_t* a) {
    bool carry;
    r[0] = addcc(r[0], a[0], carry);
    r[1] = addc(r[1], a[1], carry);
}

// ---------------------------------------------------------------------------------
// Modular multiplication for SECP256K1
// Uses the special form of P for efficient reduction
// ---------------------------------------------------------------------------------

inline void modMult256(thread uint256_t& r, thread const uint256_t& a, thread const uint256_t& b) {
    // 512-bit product
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Schoolbook multiplication
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            mul64(a.d[i], b.d[j], hi, lo);
            
            // Add to result with carry propagation
            bool c1, c2;
            p[i + j] = addcc(p[i + j], lo, c1);
            p[i + j + 1] = addc(p[i + j + 1], hi, c2);
            
            if (c1) {
                bool tmpCarry;
                for (int k = i + j + 1; k < 8 && c1; k++) {
                    p[k] = addcc(p[k], 1ULL, tmpCarry);
                    c1 = tmpCarry;
                }
            }
        }
    }
    
    // Reduction using P = 2^256 - 2^32 - 977
    // c = 0x1000003D1 = 2^32 + 977
    const uint64_t c = 0x1000003D1ULL;
    
    // First reduction: p[0:3] += p[4:7] * c
    uint64_t t[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo;
        mul64(p[4 + i], c, hi, lo);
        
        bool carry;
        t[i] = addcc(t[i], lo, carry);
        t[i + 1] = addc(t[i + 1], hi, carry);
    }
    
    bool carry;
    p[0] = addcc(p[0], t[0], carry);
    p[1] = addc(p[1], t[1], carry);
    p[2] = addc(p[2], t[2], carry);
    p[3] = addc(p[3], t[3], carry);
    uint64_t overflow = t[4] + (carry ? 1ULL : 0ULL);
    
    // Second reduction for overflow
    if (overflow > 0) {
        uint64_t hi, lo;
        mul64(overflow, c, hi, lo);
        
        p[0] = addcc(p[0], lo, carry);
        p[1] = addc(p[1], hi, carry);
        p[2] = addc(p[2], 0ULL, carry);
        p[3] = addc(p[3], 0ULL, carry);
    }
    
    r.d[0] = p[0];
    r.d[1] = p[1];
    r.d[2] = p[2];
    r.d[3] = p[3];
    
    // Final reduction if >= P
    uint256_t P = {P0, P1, P2, P3};
    if (cmp256(r, P) >= 0) {
        subP256(r);
    }
}

// Modular multiplication in place: r = r * a (mod P)
inline void modMult256(thread uint256_t& r, thread const uint256_t& a) {
    uint256_t tmp;
    load256(tmp, r);
    modMult256(r, tmp, a);
}

// Modular squaring
inline void modSqr256(thread uint256_t& r, thread const uint256_t& a) {
    // Optimized squaring with Karatsuba-like approach
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Diagonal terms
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo;
        mul64(a.d[i], a.d[i], hi, lo);
        
        bool carry;
        p[2*i] = addcc(p[2*i], lo, carry);
        p[2*i + 1] = addc(p[2*i + 1], hi, carry);
    }
    
    // Off-diagonal terms (doubled)
    for (int i = 0; i < 4; i++) {
        for (int j = i + 1; j < 4; j++) {
            uint64_t hi, lo;
            mul64(a.d[i], a.d[j], hi, lo);
            
            // Double the product
            uint64_t carry_bit = hi >> 63;
            hi = (hi << 1) | (lo >> 63);
            lo <<= 1;
            
            bool carry;
            p[i + j] = addcc(p[i + j], lo, carry);
            p[i + j + 1] = addc(p[i + j + 1], hi, carry);
            if (carry_bit || carry) {
                for (int k = i + j + 2; k < 8; k++) {
                    bool tmpCarry;
                    p[k] = addcc(p[k], carry_bit || carry ? 1ULL : 0ULL, tmpCarry);
                    if (!tmpCarry) break;
                }
            }
        }
    }
    
    // Reduction using P = 2^256 - 2^32 - 977
    const uint64_t c = 0x1000003D1ULL;
    
    uint64_t t[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo;
        mul64(p[4 + i], c, hi, lo);
        
        bool carry;
        t[i] = addcc(t[i], lo, carry);
        t[i + 1] = addc(t[i + 1], hi, carry);
    }
    
    bool carry;
    p[0] = addcc(p[0], t[0], carry);
    p[1] = addc(p[1], t[1], carry);
    p[2] = addc(p[2], t[2], carry);
    p[3] = addc(p[3], t[3], carry);
    uint64_t overflow = t[4] + (carry ? 1ULL : 0ULL);
    
    if (overflow > 0) {
        uint64_t hi, lo;
        mul64(overflow, c, hi, lo);
        
        p[0] = addcc(p[0], lo, carry);
        p[1] = addc(p[1], hi, carry);
        p[2] = addc(p[2], 0ULL, carry);
        p[3] = addc(p[3], 0ULL, carry);
    }
    
    r.d[0] = p[0];
    r.d[1] = p[1];
    r.d[2] = p[2];
    r.d[3] = p[3];
    
    uint256_t P = {P0, P1, P2, P3};
    if (cmp256(r, P) >= 0) {
        subP256(r);
    }
}

// ---------------------------------------------------------------------------------
// Modular inverse using optimized addition chain for secp256k1
// Uses special structure of p = 2^256 - 2^32 - 977
// Only ~260 multiplications instead of ~512 with Fermat
// Based on: https://briansmith.org/ecc-inversion-addition-chains-01
// ---------------------------------------------------------------------------------

inline void modInv256(thread uint256_t& r) {
    uint256_t x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223;
    uint256_t t1;
    
    // x2 = r^2 * r = r^3... wait, x2 = r^(2^1) * r = r^(2+1) = r^3? No.
    // x2 = r^2
    modSqr256(x2, r);
    // x2 = r^2 * r = r^3
    modMult256(x2, r);
    
    // x3 = x2^2 * r = r^(3*2+1) = r^7
    modSqr256(x3, x2);
    modMult256(x3, r);
    
    // x6 = x3^(2^3) * x3 = r^(7*8 + 7) = r^63
    modSqr256(x6, x3);
    modSqr256(x6, x6);
    modSqr256(x6, x6);
    modMult256(x6, x3);
    
    // x9 = x6^(2^3) * x3 = r^(63*8 + 7) = r^511
    modSqr256(x9, x6);
    modSqr256(x9, x9);
    modSqr256(x9, x9);
    modMult256(x9, x3);
    
    // x11 = x9^(2^2) * x2 = r^(511*4 + 3) = r^2047
    modSqr256(x11, x9);
    modSqr256(x11, x11);
    modMult256(x11, x2);
    
    // x22 = x11^(2^11) * x11
    load256(x22, x11);
    for (int i = 0; i < 11; i++) modSqr256(x22, x22);
    modMult256(x22, x11);
    
    // x44 = x22^(2^22) * x22
    load256(x44, x22);
    for (int i = 0; i < 22; i++) modSqr256(x44, x44);
    modMult256(x44, x22);
    
    // x88 = x44^(2^44) * x44
    load256(x88, x44);
    for (int i = 0; i < 44; i++) modSqr256(x88, x88);
    modMult256(x88, x44);
    
    // x176 = x88^(2^88) * x88
    load256(x176, x88);
    for (int i = 0; i < 88; i++) modSqr256(x176, x176);
    modMult256(x176, x88);
    
    // x220 = x176^(2^44) * x44
    load256(x220, x176);
    for (int i = 0; i < 44; i++) modSqr256(x220, x220);
    modMult256(x220, x44);
    
    // x223 = x220^(2^3) * x3
    modSqr256(x223, x220);
    modSqr256(x223, x223);
    modSqr256(x223, x223);
    modMult256(x223, x3);
    
    // Final: t1 = x223^(2^23) * x22
    load256(t1, x223);
    for (int i = 0; i < 23; i++) modSqr256(t1, t1);
    modMult256(t1, x22);
    
    // t1 = t1^(2^5) * r
    for (int i = 0; i < 5; i++) modSqr256(t1, t1);
    modMult256(t1, r);
    
    // t1 = t1^(2^3) * x2
    for (int i = 0; i < 3; i++) modSqr256(t1, t1);
    modMult256(t1, x2);
    
    // t1 = t1^(2^2) * r
    modSqr256(t1, t1);
    modSqr256(t1, t1);
    modMult256(t1, r);
    
    load256(r, t1);
}

// ---------------------------------------------------------------------------------
// Batch modular inverse using Montgomery's trick
// Much more efficient than individual inversions
// ---------------------------------------------------------------------------------

inline void modInvGrouped(thread uint256_t* dx, int groupSize) {
    uint256_t subProducts[GPU_GRP_SIZE];
    uint256_t inverse;
    
    // Compute cumulative products
    load256(subProducts[0], dx[0]);
    for (int i = 1; i < groupSize; i++) {
        modMult256(subProducts[i], subProducts[i - 1], dx[i]);
    }
    
    // Invert the final product
    load256(inverse, subProducts[groupSize - 1]);
    modInv256(inverse);
    
    // Recover individual inverses
    for (int i = groupSize - 1; i > 0; i--) {
        uint256_t newValue;
        modMult256(newValue, subProducts[i - 1], inverse);
        modMult256(inverse, dx[i]);
        load256(dx[i], newValue);
    }
    
    load256(dx[0], inverse);
}

// ---------------------------------------------------------------------------------
// Symmetry mode helpers
// ---------------------------------------------------------------------------------

#ifdef USE_SYMMETRY
inline bool modPositive256(thread uint256_t& py) {
    // Check if y > (P-1)/2 (probability ~1/2^192)
    if (py.d[3] > 0x7FFFFFFFFFFFFFFFULL) {
        modNeg256(py);
        return true;
    }
    return false;
}

inline void modNeg256Order(thread uint64_t* dist) {
    bool borrow;
    uint64_t t0 = subcc(0ULL, dist[0], borrow);
    uint64_t t1 = subc(0ULL, dist[1], borrow);
    
    bool carry;
    dist[0] = addcc(t0, ORDER0, carry);
    dist[1] = addc(t1, ORDER1, carry);
}
#endif

// ---------------------------------------------------------------------------------
// Main Kangaroo compute kernel
// ---------------------------------------------------------------------------------

kernel void computeKangaroos(
    device uint64_t* kangaroos [[buffer(0)]],          // Kangaroo data (px, py, dist)
    constant uint64_t* jumpDistances [[buffer(1)]],     // Jump distances [NB_JUMP][2]
    constant uint64_t* jumpPointsX [[buffer(2)]],       // Jump point X coords [NB_JUMP][4]
    constant uint64_t* jumpPointsY [[buffer(3)]],       // Jump point Y coords [NB_JUMP][4]
    device atomic_uint* foundCount [[buffer(4)]],       // Number of DPs found
    device DPOutput* outputs [[buffer(5)]],             // Output DPs
    constant uint64_t& dpMask [[buffer(6)]],            // DP mask
    constant uint32_t& maxFound [[buffer(7)]],          // Max outputs
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    // Local storage for this thread's kangaroos
    uint256_t px[GPU_GRP_SIZE];
    uint256_t py[GPU_GRP_SIZE];
    uint64_t dist[GPU_GRP_SIZE][2];
#ifdef USE_SYMMETRY
    uint64_t lastJump[GPU_GRP_SIZE];
#endif
    
    uint256_t dx[GPU_GRP_SIZE];
    uint256_t dy;
    uint256_t rx, ry;
    uint256_t s, p;
    
    // Calculate base offset for this thread
    uint32_t xPtr = (gid * threadsPerGroup * GPU_GRP_SIZE) * KSIZE;
    
    // Load kangaroos from global memory
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * threadsPerGroup;
        
        px[g].d[0] = kangaroos[xPtr + tid + 0 * threadsPerGroup + stride];
        px[g].d[1] = kangaroos[xPtr + tid + 1 * threadsPerGroup + stride];
        px[g].d[2] = kangaroos[xPtr + tid + 2 * threadsPerGroup + stride];
        px[g].d[3] = kangaroos[xPtr + tid + 3 * threadsPerGroup + stride];
        
        py[g].d[0] = kangaroos[xPtr + tid + 4 * threadsPerGroup + stride];
        py[g].d[1] = kangaroos[xPtr + tid + 5 * threadsPerGroup + stride];
        py[g].d[2] = kangaroos[xPtr + tid + 6 * threadsPerGroup + stride];
        py[g].d[3] = kangaroos[xPtr + tid + 7 * threadsPerGroup + stride];
        
        dist[g][0] = kangaroos[xPtr + tid + 8 * threadsPerGroup + stride];
        dist[g][1] = kangaroos[xPtr + tid + 9 * threadsPerGroup + stride];
        
#ifdef USE_SYMMETRY
        lastJump[g] = kangaroos[xPtr + tid + 10 * threadsPerGroup + stride];
#endif
    }
    
    // Main computation loop
    for (int run = 0; run < NB_RUN; run++) {
        // Compute dx = px - jPx[jmp] for all kangaroos
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g].d[0] & (NB_JUMP - 1);
            
#ifdef USE_SYMMETRY
            if (jmp == lastJump[g]) {
                jmp = (lastJump[g] + 1) % NB_JUMP;
            }
            lastJump[g] = jmp;
#endif
            
            // Load jump point X
            uint256_t jPx;
            jPx.d[0] = jumpPointsX[jmp * 4 + 0];
            jPx.d[1] = jumpPointsX[jmp * 4 + 1];
            jPx.d[2] = jumpPointsX[jmp * 4 + 2];
            jPx.d[3] = jumpPointsX[jmp * 4 + 3];
            
            modSub256(dx[g], px[g], jPx);
        }
        
        // Batch modular inverse
        modInvGrouped(dx, GPU_GRP_SIZE);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Complete point addition for each kangaroo
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
#ifdef USE_SYMMETRY
            uint32_t jmp = lastJump[g];
#else
            uint32_t jmp = (uint32_t)px[g].d[0] & (NB_JUMP - 1);
#endif
            
            // Load jump point
            uint256_t jPx, jPy;
            jPx.d[0] = jumpPointsX[jmp * 4 + 0];
            jPx.d[1] = jumpPointsX[jmp * 4 + 1];
            jPx.d[2] = jumpPointsX[jmp * 4 + 2];
            jPx.d[3] = jumpPointsX[jmp * 4 + 3];
            
            jPy.d[0] = jumpPointsY[jmp * 4 + 0];
            jPy.d[1] = jumpPointsY[jmp * 4 + 1];
            jPy.d[2] = jumpPointsY[jmp * 4 + 2];
            jPy.d[3] = jumpPointsY[jmp * 4 + 3];
            
            // dy = py - jPy
            modSub256(dy, py[g], jPy);
            
            // s = dy * dx^(-1)
            modMult256(s, dy, dx[g]);
            
            // p = s^2
            modSqr256(p, s);
            
            // rx = p - jPx - px
            modSub256(rx, p, jPx);
            modSub256(rx, px[g]);
            
            // ry = s * (px - rx) - py
            modSub256(ry, px[g], rx);
            modMult256(ry, s);
            modSub256(ry, py[g]);
            
            // Update position
            load256(px[g], rx);
            load256(py[g], ry);
            
            // Update distance
            add128(dist[g], &jumpDistances[jmp * 2]);
            
#ifdef USE_SYMMETRY
            if (modPositive256(py[g])) {
                modNeg256Order(dist[g]);
            }
#endif
            
            // Check for distinguished point
            if ((px[g].d[3] & dpMask) == 0) {
                uint32_t pos = atomic_fetch_add_explicit(foundCount, 1, memory_order_relaxed);
                
                if (pos < maxFound) {
                    uint64_t kIdx = (uint64_t)tid + 
                                   (uint64_t)g * (uint64_t)threadsPerGroup + 
                                   (uint64_t)gid * ((uint64_t)threadsPerGroup * GPU_GRP_SIZE);
                    
                    // Store output
                    outputs[pos].x[0] = (uint32_t)px[g].d[0];
                    outputs[pos].x[1] = (uint32_t)(px[g].d[0] >> 32);
                    outputs[pos].x[2] = (uint32_t)px[g].d[1];
                    outputs[pos].x[3] = (uint32_t)(px[g].d[1] >> 32);
                    outputs[pos].x[4] = (uint32_t)px[g].d[2];
                    outputs[pos].x[5] = (uint32_t)(px[g].d[2] >> 32);
                    outputs[pos].x[6] = (uint32_t)px[g].d[3];
                    outputs[pos].x[7] = (uint32_t)(px[g].d[3] >> 32);
                    
                    outputs[pos].dist[0] = (uint32_t)dist[g][0];
                    outputs[pos].dist[1] = (uint32_t)(dist[g][0] >> 32);
                    outputs[pos].dist[2] = (uint32_t)dist[g][1];
                    outputs[pos].dist[3] = (uint32_t)(dist[g][1] >> 32);
                    
                    outputs[pos].kIdx = kIdx;
                }
            }
        }
    }
    
    // Store kangaroos back to global memory
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * threadsPerGroup;
        
        kangaroos[xPtr + tid + 0 * threadsPerGroup + stride] = px[g].d[0];
        kangaroos[xPtr + tid + 1 * threadsPerGroup + stride] = px[g].d[1];
        kangaroos[xPtr + tid + 2 * threadsPerGroup + stride] = px[g].d[2];
        kangaroos[xPtr + tid + 3 * threadsPerGroup + stride] = px[g].d[3];
        
        kangaroos[xPtr + tid + 4 * threadsPerGroup + stride] = py[g].d[0];
        kangaroos[xPtr + tid + 5 * threadsPerGroup + stride] = py[g].d[1];
        kangaroos[xPtr + tid + 6 * threadsPerGroup + stride] = py[g].d[2];
        kangaroos[xPtr + tid + 7 * threadsPerGroup + stride] = py[g].d[3];
        
        kangaroos[xPtr + tid + 8 * threadsPerGroup + stride] = dist[g][0];
        kangaroos[xPtr + tid + 9 * threadsPerGroup + stride] = dist[g][1];
        
#ifdef USE_SYMMETRY
        kangaroos[xPtr + tid + 10 * threadsPerGroup + stride] = lastJump[g];
#endif
    }
}

// ---------------------------------------------------------------------------------
// Utility kernel for initialization
// ---------------------------------------------------------------------------------

kernel void initializeKangaroos(
    device uint64_t* kangaroos [[buffer(0)]],
    constant uint64_t* initData [[buffer(1)]],       // Initial positions
    constant uint32_t& numKangaroos [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    uint32_t idx = gid * threadsPerGroup + tid;
    if (idx >= numKangaroos) return;
    
    // Copy initial kangaroo data
    for (int i = 0; i < KSIZE; i++) {
        kangaroos[idx * KSIZE + i] = initData[idx * KSIZE + i];
    }
}
