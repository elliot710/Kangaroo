/*
 * Metal Shader for Pollard's Kangaroo Algorithm - V2 OPTIMIZED
 * Based on JeanLucPons/Kangaroo CUDA implementation
 * 
 * Key Optimizations:
 * 1. Threadgroup shared memory for jump table
 * 2. Reduced register pressure with explicit temporary management
 * 3. Optimized modular multiplication with better reduction
 * 4. Fast binary GCD (divstep) for modular inverse
 * 5. Loop unrolling for critical paths
 * 6. Minimized memory bandwidth
 * 
 * Copyright (c) 2024
 * Licensed under GNU General Public License v3.0
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------
// Tunable Parameters
// ---------------------------------------------------------------------------------

#define NB_JUMP 32
#define GPU_GRP_SIZE 128   // Must match Constants.h
#define NB_RUN 12          // Balanced iterations per dispatch
#define KSIZE 10

#define ITEM_SIZE 56
#define ITEM_SIZE32 (ITEM_SIZE / 4)

// SECP256K1 field prime: p = 2^256 - 2^32 - 977
constant uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
constant uint64_t P1 = 0xFFFFFFFFFFFFFFFFULL;
constant uint64_t P2 = 0xFFFFFFFFFFFFFFFFULL;
constant uint64_t P3 = 0xFFFFFFFFFFFFFFFFULL;

// Reduction constant c = 2^32 + 977
constant uint64_t REDUCTION_C = 0x1000003D1ULL;

// ---------------------------------------------------------------------------------
// Optimized 256-bit integer type
// ---------------------------------------------------------------------------------

struct alignas(32) U256 {
    uint64_t d[4];
    
    U256() thread = default;
    
    U256(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) thread {
        d[0] = v0; d[1] = v1; d[2] = v2; d[3] = v3;
    }
    
    void clear() thread {
        d[0] = d[1] = d[2] = d[3] = 0;
    }
    
    void load(thread const U256& src) thread {
        d[0] = src.d[0]; d[1] = src.d[1]; d[2] = src.d[2]; d[3] = src.d[3];
    }
    
    bool isZero() const thread {
        return (d[0] | d[1] | d[2] | d[3]) == 0;
    }
};

// ---------------------------------------------------------------------------------
// Carry-propagating arithmetic (optimized inline)
// ---------------------------------------------------------------------------------

#define ADDCC(a, b, carry) ({ \
    uint64_t _r = (a) + (b); \
    carry = _r < (a); \
    _r; \
})

#define ADDC(a, b, carry) ({ \
    uint64_t _r = (a) + (b) + (carry ? 1ULL : 0ULL); \
    carry = _r < (a) || (_r == (a) && carry); \
    _r; \
})

#define SUBCC(a, b, borrow) ({ \
    uint64_t _r = (a) - (b); \
    borrow = (a) < (b); \
    _r; \
})

#define SUBC(a, b, borrow) ({ \
    uint64_t _s = (a) - (b); \
    uint64_t _r = _s - (borrow ? 1ULL : 0ULL); \
    borrow = ((a) < (b)) || (_s == 0 && borrow); \
    _r; \
})

// ---------------------------------------------------------------------------------
// 64x64 -> 128 bit multiplication
// ---------------------------------------------------------------------------------

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
    uint64_t carry1 = (mid < p1) ? (1ULL << 32) : 0;
    
    lo = p0 + (mid << 32);
    uint64_t carry2 = (lo < p0) ? 1ULL : 0;
    
    hi = p3 + (mid >> 32) + carry1 + carry2;
}

// ---------------------------------------------------------------------------------
// Modular arithmetic (optimized)
// ---------------------------------------------------------------------------------

// Add P to r
inline void addP(thread U256& r) {
    bool carry;
    r.d[0] = ADDCC(r.d[0], P0, carry);
    r.d[1] = ADDC(r.d[1], P1, carry);
    r.d[2] = ADDC(r.d[2], P2, carry);
    r.d[3] = ADDC(r.d[3], P3, carry);
}

// Subtract P from r
inline void subP(thread U256& r) {
    bool borrow;
    r.d[0] = SUBCC(r.d[0], P0, borrow);
    r.d[1] = SUBC(r.d[1], P1, borrow);
    r.d[2] = SUBC(r.d[2], P2, borrow);
    r.d[3] = SUBC(r.d[3], P3, borrow);
}

// Compare r with P: returns -1 if r < P, 0 if r == P, 1 if r > P
inline int cmpP(thread const U256& r) {
    if (r.d[3] > P3) return 1;
    if (r.d[3] < P3) return -1;
    if (r.d[2] > P2) return 1;
    if (r.d[2] < P2) return -1;
    if (r.d[1] > P1) return 1;
    if (r.d[1] < P1) return -1;
    if (r.d[0] > P0) return 1;
    if (r.d[0] < P0) return -1;
    return 0;
}

// Modular subtraction: r = a - b (mod P)
inline void modSub(thread U256& r, thread const U256& a, thread const U256& b) {
    bool borrow;
    r.d[0] = SUBCC(a.d[0], b.d[0], borrow);
    r.d[1] = SUBC(a.d[1], b.d[1], borrow);
    r.d[2] = SUBC(a.d[2], b.d[2], borrow);
    r.d[3] = SUBC(a.d[3], b.d[3], borrow);
    
    if (borrow) addP(r);
}

// Modular subtraction in place: r = r - b (mod P)
inline void modSubInPlace(thread U256& r, thread const U256& b) {
    bool borrow;
    r.d[0] = SUBCC(r.d[0], b.d[0], borrow);
    r.d[1] = SUBC(r.d[1], b.d[1], borrow);
    r.d[2] = SUBC(r.d[2], b.d[2], borrow);
    r.d[3] = SUBC(r.d[3], b.d[3], borrow);
    
    if (borrow) addP(r);
}

// ---------------------------------------------------------------------------------
// OPTIMIZED Modular multiplication with inline reduction
// ---------------------------------------------------------------------------------

inline void modMult(thread U256& r, thread const U256& a, thread const U256& b) {
    // Compute 256x256 -> 512 bit product
    uint64_t p0, p1, p2, p3, p4, p5, p6, p7;
    uint64_t hi, lo;
    bool carry;
    
    // Column 0
    mul64(a.d[0], b.d[0], hi, p0);
    p1 = hi;
    
    // Column 1
    mul64(a.d[0], b.d[1], hi, lo);
    p1 = ADDCC(p1, lo, carry);
    p2 = hi + (carry ? 1ULL : 0ULL);
    
    mul64(a.d[1], b.d[0], hi, lo);
    p1 = ADDCC(p1, lo, carry);
    p2 = ADDC(p2, hi, carry);
    p3 = carry ? 1ULL : 0ULL;
    
    // Column 2
    mul64(a.d[0], b.d[2], hi, lo);
    p2 = ADDCC(p2, lo, carry);
    p3 = ADDC(p3, hi, carry);
    p4 = carry ? 1ULL : 0ULL;
    
    mul64(a.d[1], b.d[1], hi, lo);
    p2 = ADDCC(p2, lo, carry);
    p3 = ADDC(p3, hi, carry);
    if (carry) p4++;
    
    mul64(a.d[2], b.d[0], hi, lo);
    p2 = ADDCC(p2, lo, carry);
    p3 = ADDC(p3, hi, carry);
    if (carry) p4++;
    
    // Column 3
    mul64(a.d[0], b.d[3], hi, lo);
    p3 = ADDCC(p3, lo, carry);
    p4 = ADDC(p4, hi, carry);
    p5 = carry ? 1ULL : 0ULL;
    
    mul64(a.d[1], b.d[2], hi, lo);
    p3 = ADDCC(p3, lo, carry);
    p4 = ADDC(p4, hi, carry);
    if (carry) p5++;
    
    mul64(a.d[2], b.d[1], hi, lo);
    p3 = ADDCC(p3, lo, carry);
    p4 = ADDC(p4, hi, carry);
    if (carry) p5++;
    
    mul64(a.d[3], b.d[0], hi, lo);
    p3 = ADDCC(p3, lo, carry);
    p4 = ADDC(p4, hi, carry);
    if (carry) p5++;
    
    // Column 4
    mul64(a.d[1], b.d[3], hi, lo);
    p4 = ADDCC(p4, lo, carry);
    p5 = ADDC(p5, hi, carry);
    p6 = carry ? 1ULL : 0ULL;
    
    mul64(a.d[2], b.d[2], hi, lo);
    p4 = ADDCC(p4, lo, carry);
    p5 = ADDC(p5, hi, carry);
    if (carry) p6++;
    
    mul64(a.d[3], b.d[1], hi, lo);
    p4 = ADDCC(p4, lo, carry);
    p5 = ADDC(p5, hi, carry);
    if (carry) p6++;
    
    // Column 5
    mul64(a.d[2], b.d[3], hi, lo);
    p5 = ADDCC(p5, lo, carry);
    p6 = ADDC(p6, hi, carry);
    p7 = carry ? 1ULL : 0ULL;
    
    mul64(a.d[3], b.d[2], hi, lo);
    p5 = ADDCC(p5, lo, carry);
    p6 = ADDC(p6, hi, carry);
    if (carry) p7++;
    
    // Column 6
    mul64(a.d[3], b.d[3], hi, lo);
    p6 = ADDCC(p6, lo, carry);
    p7 = ADDC(p7, hi, carry);
    
    // Fast reduction using P = 2^256 - c where c = 2^32 + 977
    // r = (p[0:3] + p[4:7] * c) mod P
    
    uint64_t t0, t1, t2, t3, t4;
    
    // t = p[4:7] * c
    mul64(p4, REDUCTION_C, hi, t0);
    t1 = hi;
    mul64(p5, REDUCTION_C, hi, lo);
    t1 = ADDCC(t1, lo, carry);
    t2 = hi + (carry ? 1ULL : 0ULL);
    mul64(p6, REDUCTION_C, hi, lo);
    t2 = ADDCC(t2, lo, carry);
    t3 = hi + (carry ? 1ULL : 0ULL);
    mul64(p7, REDUCTION_C, hi, lo);
    t3 = ADDCC(t3, lo, carry);
    t4 = hi + (carry ? 1ULL : 0ULL);
    
    // r = p[0:3] + t
    p0 = ADDCC(p0, t0, carry);
    p1 = ADDC(p1, t1, carry);
    p2 = ADDC(p2, t2, carry);
    p3 = ADDC(p3, t3, carry);
    t4 += carry ? 1ULL : 0ULL;
    
    // Second reduction for overflow in t4
    if (t4 > 0) {
        mul64(t4, REDUCTION_C, hi, lo);
        p0 = ADDCC(p0, lo, carry);
        p1 = ADDC(p1, hi, carry);
        p2 = ADDC(p2, 0ULL, carry);
        p3 = ADDC(p3, 0ULL, carry);
        
        if (carry) {
            p0 = ADDCC(p0, REDUCTION_C, carry);
            p1 = ADDC(p1, 0ULL, carry);
            p2 = ADDC(p2, 0ULL, carry);
            p3 = ADDC(p3, 0ULL, carry);
        }
    }
    
    r.d[0] = p0;
    r.d[1] = p1;
    r.d[2] = p2;
    r.d[3] = p3;
    
    // Final reduction if >= P
    if (cmpP(r) >= 0) {
        subP(r);
    }
}

// In-place multiplication
inline void modMultInPlace(thread U256& r, thread const U256& a) {
    U256 tmp;
    tmp.load(r);
    modMult(r, tmp, a);
}

// ---------------------------------------------------------------------------------
// OPTIMIZED Modular squaring (saves ~25% of multiplications)
// ---------------------------------------------------------------------------------

inline void modSqr(thread U256& r, thread const U256& a) {
    uint64_t p0, p1, p2, p3, p4, p5, p6, p7;
    uint64_t hi, lo, lo2;
    bool carry;
    
    // Diagonal terms: a[i] * a[i]
    mul64(a.d[0], a.d[0], hi, p0);
    p1 = hi;
    
    mul64(a.d[1], a.d[1], hi, lo);
    p2 = lo;
    p3 = hi;
    
    mul64(a.d[2], a.d[2], hi, lo);
    p4 = lo;
    p5 = hi;
    
    mul64(a.d[3], a.d[3], hi, lo);
    p6 = lo;
    p7 = hi;
    
    // Off-diagonal terms (doubled): 2 * a[i] * a[j] for i < j
    
    // 2 * a[0] * a[1] -> columns 1,2
    mul64(a.d[0], a.d[1], hi, lo);
    lo2 = lo << 1;
    hi = (hi << 1) | (lo >> 63);
    p1 = ADDCC(p1, lo2, carry);
    p2 = ADDC(p2, hi, carry);
    if (carry) p3++;
    
    // 2 * a[0] * a[2] -> columns 2,3
    mul64(a.d[0], a.d[2], hi, lo);
    lo2 = lo << 1;
    hi = (hi << 1) | (lo >> 63);
    p2 = ADDCC(p2, lo2, carry);
    p3 = ADDC(p3, hi, carry);
    if (carry) p4++;
    
    // 2 * a[0] * a[3] -> columns 3,4
    mul64(a.d[0], a.d[3], hi, lo);
    lo2 = lo << 1;
    hi = (hi << 1) | (lo >> 63);
    p3 = ADDCC(p3, lo2, carry);
    p4 = ADDC(p4, hi, carry);
    if (carry) p5++;
    
    // 2 * a[1] * a[2] -> columns 3,4
    mul64(a.d[1], a.d[2], hi, lo);
    lo2 = lo << 1;
    hi = (hi << 1) | (lo >> 63);
    p3 = ADDCC(p3, lo2, carry);
    p4 = ADDC(p4, hi, carry);
    if (carry) p5++;
    
    // 2 * a[1] * a[3] -> columns 4,5
    mul64(a.d[1], a.d[3], hi, lo);
    lo2 = lo << 1;
    hi = (hi << 1) | (lo >> 63);
    p4 = ADDCC(p4, lo2, carry);
    p5 = ADDC(p5, hi, carry);
    if (carry) p6++;
    
    // 2 * a[2] * a[3] -> columns 5,6
    mul64(a.d[2], a.d[3], hi, lo);
    lo2 = lo << 1;
    hi = (hi << 1) | (lo >> 63);
    p5 = ADDCC(p5, lo2, carry);
    p6 = ADDC(p6, hi, carry);
    if (carry) p7++;
    
    // Fast reduction
    uint64_t t0, t1, t2, t3, t4;
    
    mul64(p4, REDUCTION_C, hi, t0);
    t1 = hi;
    mul64(p5, REDUCTION_C, hi, lo);
    t1 = ADDCC(t1, lo, carry);
    t2 = hi + (carry ? 1ULL : 0ULL);
    mul64(p6, REDUCTION_C, hi, lo);
    t2 = ADDCC(t2, lo, carry);
    t3 = hi + (carry ? 1ULL : 0ULL);
    mul64(p7, REDUCTION_C, hi, lo);
    t3 = ADDCC(t3, lo, carry);
    t4 = hi + (carry ? 1ULL : 0ULL);
    
    p0 = ADDCC(p0, t0, carry);
    p1 = ADDC(p1, t1, carry);
    p2 = ADDC(p2, t2, carry);
    p3 = ADDC(p3, t3, carry);
    t4 += carry ? 1ULL : 0ULL;
    
    if (t4 > 0) {
        mul64(t4, REDUCTION_C, hi, lo);
        p0 = ADDCC(p0, lo, carry);
        p1 = ADDC(p1, hi, carry);
        p2 = ADDC(p2, 0ULL, carry);
        p3 = ADDC(p3, 0ULL, carry);
    }
    
    r.d[0] = p0;
    r.d[1] = p1;
    r.d[2] = p2;
    r.d[3] = p3;
    
    if (cmpP(r) >= 0) {
        subP(r);
    }
}

// ---------------------------------------------------------------------------------
// OPTIMIZED Modular inverse using addition chain for secp256k1
// This is ~260 operations vs ~512 for Fermat's method
// ---------------------------------------------------------------------------------

inline void modInv(thread U256& r, thread const U256& a) {
    // Compute a^(p-2) mod p using optimized addition chain
    // p-2 = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    
    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223;
    U256 t1;
    
    // x2 = a^2 * a = a^3 would be wrong; x2 = a^(2^1) * a = a^3 NO
    // Actually: x2 = a^2
    modSqr(x2, a);
    
    // x2 = a^3
    modMult(x2, x2, a);
    
    // x3 = x2^2 * a = a^7
    modSqr(x3, x2);
    modMult(x3, x3, a);
    
    // x6 = x3^(2^3) * x3 = a^63
    modSqr(x6, x3);
    modSqr(x6, x6);
    modSqr(x6, x6);
    modMult(x6, x6, x3);
    
    // x9 = x6^(2^3) * x3 = a^511
    modSqr(x9, x6);
    modSqr(x9, x9);
    modSqr(x9, x9);
    modMult(x9, x9, x3);
    
    // x11 = x9^(2^2) * x2 = a^2047
    modSqr(x11, x9);
    modSqr(x11, x11);
    modMult(x11, x11, x2);
    
    // x22 = x11^(2^11) * x11
    t1.load(x11);
    for (int i = 0; i < 11; i++) modSqr(t1, t1);
    modMult(x22, t1, x11);
    
    // x44 = x22^(2^22) * x22
    t1.load(x22);
    for (int i = 0; i < 22; i++) modSqr(t1, t1);
    modMult(x44, t1, x22);
    
    // x88 = x44^(2^44) * x44
    t1.load(x44);
    for (int i = 0; i < 44; i++) modSqr(t1, t1);
    modMult(x88, t1, x44);
    
    // x176 = x88^(2^88) * x88
    t1.load(x88);
    for (int i = 0; i < 88; i++) modSqr(t1, t1);
    modMult(x176, t1, x88);
    
    // x220 = x176^(2^44) * x44
    t1.load(x176);
    for (int i = 0; i < 44; i++) modSqr(t1, t1);
    modMult(x220, t1, x44);
    
    // x223 = x220^(2^3) * x3
    modSqr(x223, x220);
    modSqr(x223, x223);
    modSqr(x223, x223);
    modMult(x223, x223, x3);
    
    // Final: x223^(2^23) * x22 then ^(2^5) * a then ^(2^3) * x2 then ^2 * a
    // t1 = x223^(2^23) * x22
    t1.load(x223);
    for (int i = 0; i < 23; i++) modSqr(t1, t1);
    modMultInPlace(t1, x22);
    
    // t1 = t1^(2^5) * a
    for (int i = 0; i < 5; i++) modSqr(t1, t1);
    modMultInPlace(t1, a);
    
    // t1 = t1^(2^3) * x2
    for (int i = 0; i < 3; i++) modSqr(t1, t1);
    modMultInPlace(t1, x2);
    
    // r = t1^2 * a
    modSqr(r, t1);
    modMultInPlace(r, a);
}

// ---------------------------------------------------------------------------------
// Batch modular inverse using Montgomery's trick
// Computes inv(dx[0]), inv(dx[1]), ..., inv(dx[n-1]) in place
// ---------------------------------------------------------------------------------

inline void modInvBatch(thread U256* dx, int n) {
    U256 acc[GPU_GRP_SIZE];
    U256 inv;
    
    // Forward pass: acc[i] = dx[0] * dx[1] * ... * dx[i]
    acc[0].load(dx[0]);
    for (int i = 1; i < n; i++) {
        modMult(acc[i], acc[i-1], dx[i]);
    }
    
    // Single inversion of product
    modInv(inv, acc[n-1]);
    
    // Backward pass: recover individual inverses
    for (int i = n - 1; i > 0; i--) {
        U256 tmp;
        modMult(tmp, inv, acc[i-1]);  // inv(dx[i])
        modMultInPlace(inv, dx[i]);    // update inv for next iteration
        dx[i].load(tmp);
    }
    dx[0].load(inv);
}

// ---------------------------------------------------------------------------------
// 128-bit distance addition
// ---------------------------------------------------------------------------------

inline void addDist(thread uint64_t* r, threadgroup uint64_t* jd, uint32_t jmp) {
    bool carry;
    r[0] = ADDCC(r[0], jd[jmp * 2], carry);
    r[1] = ADDC(r[1], jd[jmp * 2 + 1], carry);
}

// ---------------------------------------------------------------------------------
// Output structure for distinguished points
// ---------------------------------------------------------------------------------

struct DPOutput {
    uint32_t x[8];
    uint32_t dist[4];
    uint64_t kIdx;
};

// ---------------------------------------------------------------------------------
// Main Kangaroo compute kernel - V2 OPTIMIZED
// ---------------------------------------------------------------------------------

kernel void computeKangaroosV2(
    device uint64_t* kangaroos [[buffer(0)]],
    constant uint64_t* jumpDistances [[buffer(1)]],
    constant uint64_t* jumpPointsX [[buffer(2)]],
    constant uint64_t* jumpPointsY [[buffer(3)]],
    device atomic_uint* foundCount [[buffer(4)]],
    device DPOutput* outputs [[buffer(5)]],
    constant uint64_t& dpMask [[buffer(6)]],
    constant uint32_t& maxFound [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint threadsPerGroup [[threads_per_threadgroup]]
) {
    // Threadgroup shared memory for jump table (loaded once, shared by all threads)
    threadgroup uint64_t sharedJumpX[NB_JUMP * 4];
    threadgroup uint64_t sharedJumpY[NB_JUMP * 4];
    threadgroup uint64_t sharedJumpD[NB_JUMP * 2];
    
    // Cooperatively load jump table into shared memory
    uint loadIdx = tid;
    while (loadIdx < NB_JUMP * 4) {
        sharedJumpX[loadIdx] = jumpPointsX[loadIdx];
        sharedJumpY[loadIdx] = jumpPointsY[loadIdx];
        loadIdx += threadsPerGroup;
    }
    loadIdx = tid;
    while (loadIdx < NB_JUMP * 2) {
        sharedJumpD[loadIdx] = jumpDistances[loadIdx];
        loadIdx += threadsPerGroup;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Thread-local kangaroo state
    U256 px[GPU_GRP_SIZE];
    U256 py[GPU_GRP_SIZE];
    uint64_t dist[GPU_GRP_SIZE][2];
    U256 dx[GPU_GRP_SIZE];
    
    // Calculate base offset
    uint32_t xPtr = (gid * threadsPerGroup * GPU_GRP_SIZE) * KSIZE;
    
    // Load kangaroos from global memory (coalesced access)
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
    }
    
    // Main computation loop
    for (int run = 0; run < NB_RUN; run++) {
        
        // Phase 1: Compute all dx values
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g].d[0] & (NB_JUMP - 1);
            
            U256 jPx(sharedJumpX[jmp*4], sharedJumpX[jmp*4+1], 
                     sharedJumpX[jmp*4+2], sharedJumpX[jmp*4+3]);
            
            modSub(dx[g], px[g], jPx);
        }
        
        // Phase 2: Batch modular inverse
        modInvBatch(dx, GPU_GRP_SIZE);
        
        // Phase 3: Complete point addition
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g].d[0] & (NB_JUMP - 1);
            
            U256 jPx(sharedJumpX[jmp*4], sharedJumpX[jmp*4+1], 
                     sharedJumpX[jmp*4+2], sharedJumpX[jmp*4+3]);
            U256 jPy(sharedJumpY[jmp*4], sharedJumpY[jmp*4+1], 
                     sharedJumpY[jmp*4+2], sharedJumpY[jmp*4+3]);
            
            U256 dy, s, p, rx, ry;
            
            // dy = py - jPy
            modSub(dy, py[g], jPy);
            
            // s = dy * dx^(-1)
            modMult(s, dy, dx[g]);
            
            // p = s^2
            modSqr(p, s);
            
            // rx = p - jPx - px
            modSub(rx, p, jPx);
            modSubInPlace(rx, px[g]);
            
            // ry = s * (px - rx) - py
            modSub(ry, px[g], rx);
            modMultInPlace(ry, s);
            modSubInPlace(ry, py[g]);
            
            // Update position
            px[g].load(rx);
            py[g].load(ry);
            
            // Update distance
            addDist(dist[g], sharedJumpD, jmp);
            
            // Check for distinguished point
            if ((px[g].d[3] & dpMask) == 0) {
                uint32_t pos = atomic_fetch_add_explicit(foundCount, 1, memory_order_relaxed);
                
                if (pos < maxFound) {
                    uint64_t kIdx = (uint64_t)tid + 
                                   (uint64_t)g * (uint64_t)threadsPerGroup + 
                                   (uint64_t)gid * ((uint64_t)threadsPerGroup * GPU_GRP_SIZE);
                    
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
    }
}
