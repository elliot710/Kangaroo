/*
 * Metal Shader for Pollard's Kangaroo Algorithm
 * Fixed version with correct 256-bit arithmetic
 *
 * Copyright (c) 2024
 * Licensed under GNU General Public License v3.0
 */

#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------------

#define NB_JUMP 32
#define GPU_GRP_SIZE 128
#define NB_RUN 8
#define KSIZE 10

// SECP256K1 field prime: p = 2^256 - 2^32 - 977
#define P0 0xFFFFFFFEFFFFFC2FULL
#define P1 0xFFFFFFFFFFFFFFFFULL
#define P2 0xFFFFFFFFFFFFFFFFULL
#define P3 0xFFFFFFFFFFFFFFFFULL

// Reduction constant c = 2^32 + 977
#define REDUCTION_C 0x1000003D1ULL

// ---------------------------------------------------------------------------------
// 256-bit integer type
// ---------------------------------------------------------------------------------

struct U256 {
    uint64_t d0, d1, d2, d3;
};

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
    uint64_t carry1 = (mid < p1) ? 0x100000000ULL : 0;
    
    lo = p0 + (mid << 32);
    uint64_t carry2 = (lo < p0) ? 1ULL : 0;
    
    hi = p3 + (mid >> 32) + carry1 + carry2;
}

// ---------------------------------------------------------------------------------
// Modular subtraction with correct borrow handling
// Uses the mask trick from CUDA: after subtraction, borrow flag is used to mask P
// ---------------------------------------------------------------------------------

inline void modSub(thread U256& r, thread const U256& a, thread const U256& b) {
    // First compute a - b
    uint64_t t0 = a.d0 - b.d0;
    uint64_t b0 = (a.d0 < b.d0) ? 1ULL : 0;
    
    uint64_t t1 = a.d1 - b.d1 - b0;
    uint64_t b1 = (a.d1 < b.d1) ? 1ULL : ((a.d1 == b.d1 && b0) ? 1ULL : 0);
    
    uint64_t t2 = a.d2 - b.d2 - b1;
    uint64_t b2 = (a.d2 < b.d2) ? 1ULL : ((a.d2 == b.d2 && b1) ? 1ULL : 0);
    
    uint64_t t3 = a.d3 - b.d3 - b2;
    uint64_t b3 = (a.d3 < b.d3) ? 1ULL : ((a.d3 == b.d3 && b2) ? 1ULL : 0);
    
    // If borrow, add P back using mask
    uint64_t mask = b3 ? 0xFFFFFFFFFFFFFFFFULL : 0;
    
    uint64_t s0 = t0 + (P0 & mask);
    uint64_t c0 = (s0 < t0) ? 1ULL : 0;
    
    uint64_t s1 = t1 + (P1 & mask) + c0;
    uint64_t c1 = (s1 < t1) ? 1ULL : ((s1 == t1 && c0) ? 1ULL : 0);
    
    uint64_t s2 = t2 + (P2 & mask) + c1;
    uint64_t c2 = (s2 < t2) ? 1ULL : ((s2 == t2 && c1) ? 1ULL : 0);
    
    uint64_t s3 = t3 + (P3 & mask) + c2;
    
    r.d0 = s0; r.d1 = s1; r.d2 = s2; r.d3 = s3;
}

inline void modSubInPlace(thread U256& r, thread const U256& b) {
    U256 tmp = r;
    modSub(r, tmp, b);
}

// ---------------------------------------------------------------------------------
// Modular multiplication with 512-bit intermediate
// ---------------------------------------------------------------------------------

inline void modMult(thread U256& r, thread const U256& a, thread const U256& b) {
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t hi, lo;
    
    // Schoolbook multiplication: compute 512-bit product
    // Column 0
    mul64(a.d0, b.d0, hi, p[0]);
    p[1] = hi;
    
    // Column 1
    mul64(a.d0, b.d1, hi, lo);
    p[1] += lo; uint64_t c = (p[1] < lo) ? 1ULL : 0;
    p[2] = hi + c;
    
    mul64(a.d1, b.d0, hi, lo);
    p[1] += lo; c = (p[1] < lo) ? 1ULL : 0;
    p[2] += hi + c; c = (p[2] < hi + c) ? 1ULL : 0;
    p[3] = c;
    
    // Column 2
    mul64(a.d0, b.d2, hi, lo);
    p[2] += lo; c = (p[2] < lo) ? 1ULL : 0;
    p[3] += hi + c; c = (p[3] < hi + c) ? 1ULL : 0;
    p[4] = c;
    
    mul64(a.d1, b.d1, hi, lo);
    p[2] += lo; c = (p[2] < lo) ? 1ULL : 0;
    p[3] += hi + c; c = (p[3] < hi + c) ? 1ULL : 0;
    p[4] += c;
    
    mul64(a.d2, b.d0, hi, lo);
    p[2] += lo; c = (p[2] < lo) ? 1ULL : 0;
    p[3] += hi + c; c = (p[3] < hi + c) ? 1ULL : 0;
    p[4] += c;
    
    // Column 3
    mul64(a.d0, b.d3, hi, lo);
    p[3] += lo; c = (p[3] < lo) ? 1ULL : 0;
    p[4] += hi + c; c = (p[4] < hi + c) ? 1ULL : 0;
    p[5] = c;
    
    mul64(a.d1, b.d2, hi, lo);
    p[3] += lo; c = (p[3] < lo) ? 1ULL : 0;
    p[4] += hi + c; c = (p[4] < hi + c) ? 1ULL : 0;
    p[5] += c;
    
    mul64(a.d2, b.d1, hi, lo);
    p[3] += lo; c = (p[3] < lo) ? 1ULL : 0;
    p[4] += hi + c; c = (p[4] < hi + c) ? 1ULL : 0;
    p[5] += c;
    
    mul64(a.d3, b.d0, hi, lo);
    p[3] += lo; c = (p[3] < lo) ? 1ULL : 0;
    p[4] += hi + c; c = (p[4] < hi + c) ? 1ULL : 0;
    p[5] += c;
    
    // Column 4
    mul64(a.d1, b.d3, hi, lo);
    p[4] += lo; c = (p[4] < lo) ? 1ULL : 0;
    p[5] += hi + c; c = (p[5] < hi + c) ? 1ULL : 0;
    p[6] = c;
    
    mul64(a.d2, b.d2, hi, lo);
    p[4] += lo; c = (p[4] < lo) ? 1ULL : 0;
    p[5] += hi + c; c = (p[5] < hi + c) ? 1ULL : 0;
    p[6] += c;
    
    mul64(a.d3, b.d1, hi, lo);
    p[4] += lo; c = (p[4] < lo) ? 1ULL : 0;
    p[5] += hi + c; c = (p[5] < hi + c) ? 1ULL : 0;
    p[6] += c;
    
    // Column 5
    mul64(a.d2, b.d3, hi, lo);
    p[5] += lo; c = (p[5] < lo) ? 1ULL : 0;
    p[6] += hi + c; c = (p[6] < hi + c) ? 1ULL : 0;
    p[7] = c;
    
    mul64(a.d3, b.d2, hi, lo);
    p[5] += lo; c = (p[5] < lo) ? 1ULL : 0;
    p[6] += hi + c; c = (p[6] < hi + c) ? 1ULL : 0;
    p[7] += c;
    
    // Column 6
    mul64(a.d3, b.d3, hi, lo);
    p[6] += lo; c = (p[6] < lo) ? 1ULL : 0;
    p[7] += hi + c;
    
    // === REDUCTION ===
    // Reduce from 512 to 320: t = p[4:7] * REDUCTION_C
    uint64_t t[5];
    mul64(p[4], REDUCTION_C, hi, t[0]);
    t[1] = hi;
    mul64(p[5], REDUCTION_C, hi, lo);
    t[1] += lo; c = (t[1] < lo) ? 1ULL : 0;
    t[2] = hi + c;
    mul64(p[6], REDUCTION_C, hi, lo);
    t[2] += lo; c = (t[2] < lo) ? 1ULL : 0;
    t[3] = hi + c;
    mul64(p[7], REDUCTION_C, hi, lo);
    t[3] += lo; c = (t[3] < lo) ? 1ULL : 0;
    t[4] = hi + c;
    
    // Add t to p[0:3]
    p[0] += t[0]; c = (p[0] < t[0]) ? 1ULL : 0;
    p[1] += t[1] + c; c = (p[1] < t[1] + c) ? 1ULL : 0;
    p[2] += t[2] + c; c = (p[2] < t[2] + c) ? 1ULL : 0;
    p[3] += t[3] + c; c = (p[3] < t[3] + c) ? 1ULL : 0;
    t[4] += c;
    
    // Reduce from 320 to 256: if t[4] > 0, reduce again
    if (t[4] > 0) {
        mul64(t[4], REDUCTION_C, hi, lo);
        p[0] += lo; c = (p[0] < lo) ? 1ULL : 0;
        p[1] += hi + c; c = (p[1] < hi + c) ? 1ULL : 0;
        p[2] += c; c = (p[2] < c) ? 1ULL : 0;
        p[3] += c;
    }
    
    r.d0 = p[0]; r.d1 = p[1]; r.d2 = p[2]; r.d3 = p[3];
    
    // Final reduction if >= P
    if (r.d3 > P3 || (r.d3 == P3 && (r.d2 > P2 || (r.d2 == P2 && (r.d1 > P1 || (r.d1 == P1 && r.d0 >= P0)))))) {
        U256 P = {P0, P1, P2, P3};
        modSub(r, r, P);
    }
}

// ---------------------------------------------------------------------------------
// Modular squaring (using symmetry)
// ---------------------------------------------------------------------------------

inline void modSqr(thread U256& r, thread const U256& a) {
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint64_t hi, lo, lo2, c;
    
    // Diagonal terms
    mul64(a.d0, a.d0, hi, p[0]); p[1] = hi;
    mul64(a.d1, a.d1, hi, lo); p[2] = lo; p[3] = hi;
    mul64(a.d2, a.d2, hi, lo); p[4] = lo; p[5] = hi;
    mul64(a.d3, a.d3, hi, lo); p[6] = lo; p[7] = hi;
    
    // Off-diagonal (doubled): 2 * a[i] * a[j]
    
    // 2 * a0 * a1 -> p1, p2
    mul64(a.d0, a.d1, hi, lo);
    lo2 = lo << 1; hi = (hi << 1) | (lo >> 63);
    p[1] += lo2; c = (p[1] < lo2) ? 1ULL : 0;
    p[2] += hi + c; c = (p[2] < hi + c) ? 1ULL : 0;
    p[3] += c;
    
    // 2 * a0 * a2 -> p2, p3
    mul64(a.d0, a.d2, hi, lo);
    lo2 = lo << 1; hi = (hi << 1) | (lo >> 63);
    p[2] += lo2; c = (p[2] < lo2) ? 1ULL : 0;
    p[3] += hi + c; c = (p[3] < hi + c) ? 1ULL : 0;
    p[4] += c;
    
    // 2 * a0 * a3 -> p3, p4
    mul64(a.d0, a.d3, hi, lo);
    lo2 = lo << 1; hi = (hi << 1) | (lo >> 63);
    p[3] += lo2; c = (p[3] < lo2) ? 1ULL : 0;
    p[4] += hi + c; c = (p[4] < hi + c) ? 1ULL : 0;
    p[5] += c;
    
    // 2 * a1 * a2 -> p3, p4
    mul64(a.d1, a.d2, hi, lo);
    lo2 = lo << 1; hi = (hi << 1) | (lo >> 63);
    p[3] += lo2; c = (p[3] < lo2) ? 1ULL : 0;
    p[4] += hi + c; c = (p[4] < hi + c) ? 1ULL : 0;
    p[5] += c;
    
    // 2 * a1 * a3 -> p4, p5
    mul64(a.d1, a.d3, hi, lo);
    lo2 = lo << 1; hi = (hi << 1) | (lo >> 63);
    p[4] += lo2; c = (p[4] < lo2) ? 1ULL : 0;
    p[5] += hi + c; c = (p[5] < hi + c) ? 1ULL : 0;
    p[6] += c;
    
    // 2 * a2 * a3 -> p5, p6
    mul64(a.d2, a.d3, hi, lo);
    lo2 = lo << 1; hi = (hi << 1) | (lo >> 63);
    p[5] += lo2; c = (p[5] < lo2) ? 1ULL : 0;
    p[6] += hi + c; c = (p[6] < hi + c) ? 1ULL : 0;
    p[7] += c;
    
    // === REDUCTION ===
    uint64_t t[5];
    mul64(p[4], REDUCTION_C, hi, t[0]); t[1] = hi;
    mul64(p[5], REDUCTION_C, hi, lo);
    t[1] += lo; c = (t[1] < lo) ? 1ULL : 0;
    t[2] = hi + c;
    mul64(p[6], REDUCTION_C, hi, lo);
    t[2] += lo; c = (t[2] < lo) ? 1ULL : 0;
    t[3] = hi + c;
    mul64(p[7], REDUCTION_C, hi, lo);
    t[3] += lo; c = (t[3] < lo) ? 1ULL : 0;
    t[4] = hi + c;
    
    p[0] += t[0]; c = (p[0] < t[0]) ? 1ULL : 0;
    p[1] += t[1] + c; c = (p[1] < t[1] + c) ? 1ULL : 0;
    p[2] += t[2] + c; c = (p[2] < t[2] + c) ? 1ULL : 0;
    p[3] += t[3] + c; c = (p[3] < t[3] + c) ? 1ULL : 0;
    t[4] += c;
    
    if (t[4] > 0) {
        mul64(t[4], REDUCTION_C, hi, lo);
        p[0] += lo; c = (p[0] < lo) ? 1ULL : 0;
        p[1] += hi + c; c = (p[1] < hi + c) ? 1ULL : 0;
        p[2] += c; c = (p[2] < c) ? 1ULL : 0;
        p[3] += c;
    }
    
    r.d0 = p[0]; r.d1 = p[1]; r.d2 = p[2]; r.d3 = p[3];
    
    if (r.d3 > P3 || (r.d3 == P3 && (r.d2 > P2 || (r.d2 == P2 && (r.d1 > P1 || (r.d1 == P1 && r.d0 >= P0)))))) {
        U256 P = {P0, P1, P2, P3};
        modSub(r, r, P);
    }
}

// ---------------------------------------------------------------------------------
// Modular inverse using Fermat's little theorem
// ---------------------------------------------------------------------------------

inline void modInv(thread U256& r, thread const U256& a) {
    U256 x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t1;
    
    modSqr(x2, a);
    modMult(x2, x2, a);  // x2 = a^3
    
    modSqr(x3, x2);
    modMult(x3, x3, a);  // x3 = a^7
    
    modSqr(x6, x3); modSqr(x6, x6); modSqr(x6, x6);
    modMult(x6, x6, x3);  // x6 = a^63
    
    modSqr(x9, x6); modSqr(x9, x9); modSqr(x9, x9);
    modMult(x9, x9, x3);  // x9 = a^511
    
    modSqr(x11, x9); modSqr(x11, x11);
    modMult(x11, x11, x2);  // x11 = a^2047
    
    t1 = x11;
    for (int i = 0; i < 11; i++) modSqr(t1, t1);
    modMult(x22, t1, x11);
    
    t1 = x22;
    for (int i = 0; i < 22; i++) modSqr(t1, t1);
    modMult(x44, t1, x22);
    
    t1 = x44;
    for (int i = 0; i < 44; i++) modSqr(t1, t1);
    modMult(x88, t1, x44);
    
    t1 = x88;
    for (int i = 0; i < 88; i++) modSqr(t1, t1);
    modMult(x176, t1, x88);
    
    t1 = x176;
    for (int i = 0; i < 44; i++) modSqr(t1, t1);
    modMult(x220, t1, x44);
    
    modSqr(x223, x220); modSqr(x223, x223); modSqr(x223, x223);
    modMult(x223, x223, x3);
    
    t1 = x223;
    for (int i = 0; i < 23; i++) modSqr(t1, t1);
    modMult(t1, t1, x22);
    
    for (int i = 0; i < 5; i++) modSqr(t1, t1);
    modMult(t1, t1, a);
    
    for (int i = 0; i < 3; i++) modSqr(t1, t1);
    modMult(t1, t1, x2);
    
    modSqr(r, t1);
    modMult(r, r, a);
}

// ---------------------------------------------------------------------------------
// Batch modular inverse using Montgomery's trick
// ---------------------------------------------------------------------------------

inline void modInvBatch(thread U256* dx, int n) {
    U256 acc[GPU_GRP_SIZE];
    U256 inv;
    
    acc[0] = dx[0];
    for (int i = 1; i < n; i++) {
        modMult(acc[i], acc[i-1], dx[i]);
    }
    
    modInv(inv, acc[n-1]);
    
    for (int i = n - 1; i > 0; i--) {
        U256 tmp;
        modMult(tmp, inv, acc[i-1]);
        modMult(inv, inv, dx[i]);
        dx[i] = tmp;
    }
    dx[0] = inv;
}

// ---------------------------------------------------------------------------------
// 128-bit distance addition
// ---------------------------------------------------------------------------------

inline void addDist(thread uint64_t* r, constant uint64_t* jd, uint32_t jmp) {
    uint64_t lo = jd[jmp * 2];
    uint64_t hi = jd[jmp * 2 + 1];
    r[0] += lo;
    r[1] += hi + ((r[0] < lo) ? 1ULL : 0);
}

// ---------------------------------------------------------------------------------
// Output structure
// ---------------------------------------------------------------------------------

struct DPOutput {
    uint32_t x[8];
    uint32_t dist[4];
    uint64_t kIdx;
};

// ---------------------------------------------------------------------------------
// Main Kangaroo compute kernel
// ---------------------------------------------------------------------------------

kernel void computeKangaroos(
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
    // Thread-local kangaroo state
    U256 px[GPU_GRP_SIZE];
    U256 py[GPU_GRP_SIZE];
    uint64_t dist[GPU_GRP_SIZE][2];
    U256 dx[GPU_GRP_SIZE];
    
    uint32_t xPtr = (gid * threadsPerGroup * GPU_GRP_SIZE) * KSIZE;
    
    // Load kangaroos
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * threadsPerGroup;
        
        px[g].d0 = kangaroos[xPtr + tid + 0 * threadsPerGroup + stride];
        px[g].d1 = kangaroos[xPtr + tid + 1 * threadsPerGroup + stride];
        px[g].d2 = kangaroos[xPtr + tid + 2 * threadsPerGroup + stride];
        px[g].d3 = kangaroos[xPtr + tid + 3 * threadsPerGroup + stride];
        
        py[g].d0 = kangaroos[xPtr + tid + 4 * threadsPerGroup + stride];
        py[g].d1 = kangaroos[xPtr + tid + 5 * threadsPerGroup + stride];
        py[g].d2 = kangaroos[xPtr + tid + 6 * threadsPerGroup + stride];
        py[g].d3 = kangaroos[xPtr + tid + 7 * threadsPerGroup + stride];
        
        dist[g][0] = kangaroos[xPtr + tid + 8 * threadsPerGroup + stride];
        dist[g][1] = kangaroos[xPtr + tid + 9 * threadsPerGroup + stride];
    }
    
    // Main loop
    for (int run = 0; run < NB_RUN; run++) {
        
        // Phase 1: Compute dx = px - jPx for all kangaroos
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g].d0 & (NB_JUMP - 1);
            U256 jPx;
            jPx.d0 = jumpPointsX[jmp * 4];
            jPx.d1 = jumpPointsX[jmp * 4 + 1];
            jPx.d2 = jumpPointsX[jmp * 4 + 2];
            jPx.d3 = jumpPointsX[jmp * 4 + 3];
            modSub(dx[g], px[g], jPx);
        }
        
        // Phase 2: Batch inverse
        modInvBatch(dx, GPU_GRP_SIZE);
        
        // Phase 3: Point addition
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g].d0 & (NB_JUMP - 1);
            
            U256 jPx, jPy;
            jPx.d0 = jumpPointsX[jmp * 4];
            jPx.d1 = jumpPointsX[jmp * 4 + 1];
            jPx.d2 = jumpPointsX[jmp * 4 + 2];
            jPx.d3 = jumpPointsX[jmp * 4 + 3];
            jPy.d0 = jumpPointsY[jmp * 4];
            jPy.d1 = jumpPointsY[jmp * 4 + 1];
            jPy.d2 = jumpPointsY[jmp * 4 + 2];
            jPy.d3 = jumpPointsY[jmp * 4 + 3];
            
            U256 dy, s, p, rx, ry;
            
            modSub(dy, py[g], jPy);
            modMult(s, dy, dx[g]);
            modSqr(p, s);
            modSub(rx, p, jPx);
            modSubInPlace(rx, px[g]);
            modSub(ry, px[g], rx);
            modMult(ry, ry, s);
            modSubInPlace(ry, py[g]);
            
            px[g] = rx;
            py[g] = ry;
            
            addDist(dist[g], jumpDistances, jmp);
            
            // Check for DP
            if ((px[g].d3 & dpMask) == 0) {
                uint32_t pos = atomic_fetch_add_explicit(foundCount, 1, memory_order_relaxed);
                if (pos < maxFound) {
                    uint64_t kIdx = (uint64_t)tid + 
                                   (uint64_t)g * (uint64_t)threadsPerGroup + 
                                   (uint64_t)gid * ((uint64_t)threadsPerGroup * GPU_GRP_SIZE);
                    
                    outputs[pos].x[0] = (uint32_t)px[g].d0;
                    outputs[pos].x[1] = (uint32_t)(px[g].d0 >> 32);
                    outputs[pos].x[2] = (uint32_t)px[g].d1;
                    outputs[pos].x[3] = (uint32_t)(px[g].d1 >> 32);
                    outputs[pos].x[4] = (uint32_t)px[g].d2;
                    outputs[pos].x[5] = (uint32_t)(px[g].d2 >> 32);
                    outputs[pos].x[6] = (uint32_t)px[g].d3;
                    outputs[pos].x[7] = (uint32_t)(px[g].d3 >> 32);
                    
                    outputs[pos].dist[0] = (uint32_t)dist[g][0];
                    outputs[pos].dist[1] = (uint32_t)(dist[g][0] >> 32);
                    outputs[pos].dist[2] = (uint32_t)dist[g][1];
                    outputs[pos].dist[3] = (uint32_t)(dist[g][1] >> 32);
                    
                    outputs[pos].kIdx = kIdx;
                }
            }
        }
    }
    
    // Store kangaroos
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * threadsPerGroup;
        
        kangaroos[xPtr + tid + 0 * threadsPerGroup + stride] = px[g].d0;
        kangaroos[xPtr + tid + 1 * threadsPerGroup + stride] = px[g].d1;
        kangaroos[xPtr + tid + 2 * threadsPerGroup + stride] = px[g].d2;
        kangaroos[xPtr + tid + 3 * threadsPerGroup + stride] = px[g].d3;
        
        kangaroos[xPtr + tid + 4 * threadsPerGroup + stride] = py[g].d0;
        kangaroos[xPtr + tid + 5 * threadsPerGroup + stride] = py[g].d1;
        kangaroos[xPtr + tid + 6 * threadsPerGroup + stride] = py[g].d2;
        kangaroos[xPtr + tid + 7 * threadsPerGroup + stride] = py[g].d3;
        
        kangaroos[xPtr + tid + 8 * threadsPerGroup + stride] = dist[g][0];
        kangaroos[xPtr + tid + 9 * threadsPerGroup + stride] = dist[g][1];
    }
}
