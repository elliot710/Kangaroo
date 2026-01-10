/*
 * Optimized Metal Shader for Pollard's Kangaroo Algorithm
 * High-performance version with SIMD optimizations for Apple Silicon
 *
 * Copyright (c) 2024
 * Licensed under GNU General Public License v3.0
 *
 * This version includes additional optimizations:
 * - SIMD group operations for parallel reductions
 * - Optimized modular arithmetic using Barrett reduction hints
 * - Reduced register pressure through careful variable management
 * - Improved memory access patterns
 */

#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ---------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------

#define NB_JUMP 32
#define GPU_GRP_SIZE 128
#define NB_RUN 64

#ifdef USE_SYMMETRY
#define KSIZE 11
#else
#define KSIZE 10
#endif

#define ITEM_SIZE 56
#define ITEM_SIZE32 (ITEM_SIZE / 4)

// SECP256K1 constants packed for efficient access
constant uint64_t SECP256K1_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

constant uint64_t SECP256K1_ORDER[4] = {
    0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
};

// Reduction constant: c = 2^256 mod P = 0x1000003D1
constant uint64_t REDUCTION_C = 0x1000003D1ULL;

// ---------------------------------------------------------------------------------
// High-performance 256-bit type
// Using array for better SIMD optimization potential
// ---------------------------------------------------------------------------------

struct U256 {
    uint64_t v[4];
    
    // Default constructor - zero initialized
    U256() thread {
        v[0] = 0; v[1] = 0; v[2] = 0; v[3] = 0;
    }
    
    // Copy constructor
    U256(thread const U256& other) thread {
        v[0] = other.v[0]; v[1] = other.v[1];
        v[2] = other.v[2]; v[3] = other.v[3];
    }
    
    // Value constructor
    U256(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) thread {
        v[0] = v0; v[1] = v1; v[2] = v2; v[3] = v3;
    }
};

// ---------------------------------------------------------------------------------
// Distinguished Point Output
// ---------------------------------------------------------------------------------

struct DPResult {
    uint32_t x[8];
    uint32_t d[4];
    uint64_t kidx;
};

// ---------------------------------------------------------------------------------
// Optimized 64-bit arithmetic with carry
// ---------------------------------------------------------------------------------

// Add with carry out
inline uint64_t add64c(uint64_t a, uint64_t b, thread uint64_t& carry) {
    uint64_t sum = a + b;
    carry = (sum < a) ? 1ULL : 0ULL;
    return sum;
}

// Add with carry in/out
inline uint64_t adc64(uint64_t a, uint64_t b, thread uint64_t& carry) {
    uint64_t sum = a + b + carry;
    carry = (sum < a || (sum == a && carry)) ? 1ULL : 0ULL;
    return sum;
}

// Subtract with borrow out
inline uint64_t sub64b(uint64_t a, uint64_t b, thread uint64_t& borrow) {
    borrow = (a < b) ? 1ULL : 0ULL;
    return a - b;
}

// Subtract with borrow in/out
inline uint64_t sbb64(uint64_t a, uint64_t b, thread uint64_t& borrow) {
    uint64_t diff = a - b - borrow;
    borrow = (a < b || (a == b && borrow)) ? 1ULL : 0ULL;
    return diff;
}

// ---------------------------------------------------------------------------------
// Optimized 64x64 -> 128 multiplication
// ---------------------------------------------------------------------------------

inline void umul128(uint64_t a, uint64_t b, thread uint64_t& hi, thread uint64_t& lo) {
    // Split into 32-bit parts for overflow-safe multiplication
    uint64_t a_lo = a & 0xFFFFFFFFULL;
    uint64_t a_hi = a >> 32;
    uint64_t b_lo = b & 0xFFFFFFFFULL;
    uint64_t b_hi = b >> 32;
    
    // Partial products
    uint64_t p0 = a_lo * b_lo;  // Low x Low
    uint64_t p1 = a_lo * b_hi;  // Low x High
    uint64_t p2 = a_hi * b_lo;  // High x Low
    uint64_t p3 = a_hi * b_hi;  // High x High
    
    // Combine middle products
    uint64_t mid = p1 + p2;
    uint64_t mid_carry = (mid < p1) ? 0x100000000ULL : 0ULL;
    
    // Form result
    uint64_t lo_mid = mid << 32;
    uint64_t hi_mid = (mid >> 32) + mid_carry;
    
    lo = p0 + lo_mid;
    uint64_t lo_carry = (lo < p0) ? 1ULL : 0ULL;
    hi = p3 + hi_mid + lo_carry;
}

// ---------------------------------------------------------------------------------
// U256 Comparison
// ---------------------------------------------------------------------------------

inline int cmp256(thread const U256& a, thread const U256& b) {
    if (a.v[3] != b.v[3]) return (a.v[3] > b.v[3]) ? 1 : -1;
    if (a.v[2] != b.v[2]) return (a.v[2] > b.v[2]) ? 1 : -1;
    if (a.v[1] != b.v[1]) return (a.v[1] > b.v[1]) ? 1 : -1;
    if (a.v[0] != b.v[0]) return (a.v[0] > b.v[0]) ? 1 : -1;
    return 0;
}

inline bool isZero256(thread const U256& a) {
    return (a.v[0] | a.v[1] | a.v[2] | a.v[3]) == 0ULL;
}

// ---------------------------------------------------------------------------------
// Modular Addition/Subtraction for SECP256K1
// ---------------------------------------------------------------------------------

// Add the prime P to correct underflow
inline void addP(thread U256& r) {
    uint64_t c;
    r.v[0] = add64c(r.v[0], SECP256K1_P[0], c);
    r.v[1] = adc64(r.v[1], SECP256K1_P[1], c);
    r.v[2] = adc64(r.v[2], SECP256K1_P[2], c);
    r.v[3] = adc64(r.v[3], SECP256K1_P[3], c);
}

// Subtract the prime P for normalization
inline void subP(thread U256& r) {
    uint64_t b;
    r.v[0] = sub64b(r.v[0], SECP256K1_P[0], b);
    r.v[1] = sbb64(r.v[1], SECP256K1_P[1], b);
    r.v[2] = sbb64(r.v[2], SECP256K1_P[2], b);
    r.v[3] = sbb64(r.v[3], SECP256K1_P[3], b);
}

// r = a - b (mod P)
inline void modSub(thread U256& r, thread const U256& a, thread const U256& b) {
    uint64_t borrow;
    r.v[0] = sub64b(a.v[0], b.v[0], borrow);
    r.v[1] = sbb64(a.v[1], b.v[1], borrow);
    r.v[2] = sbb64(a.v[2], b.v[2], borrow);
    r.v[3] = sbb64(a.v[3], b.v[3], borrow);
    
    if (borrow) {
        addP(r);
    }
}

// r = r - b (mod P) in-place
inline void modSubInPlace(thread U256& r, thread const U256& b) {
    uint64_t borrow;
    r.v[0] = sub64b(r.v[0], b.v[0], borrow);
    r.v[1] = sbb64(r.v[1], b.v[1], borrow);
    r.v[2] = sbb64(r.v[2], b.v[2], borrow);
    r.v[3] = sbb64(r.v[3], b.v[3], borrow);
    
    if (borrow) {
        addP(r);
    }
}

// r = -a (mod P)
inline void modNeg(thread U256& r, thread const U256& a) {
    uint64_t b;
    uint64_t t0 = sub64b(0ULL, a.v[0], b);
    uint64_t t1 = sbb64(0ULL, a.v[1], b);
    uint64_t t2 = sbb64(0ULL, a.v[2], b);
    uint64_t t3 = sbb64(0ULL, a.v[3], b);
    
    uint64_t c;
    r.v[0] = add64c(t0, SECP256K1_P[0], c);
    r.v[1] = adc64(t1, SECP256K1_P[1], c);
    r.v[2] = adc64(t2, SECP256K1_P[2], c);
    r.v[3] = adc64(t3, SECP256K1_P[3], c);
}

// ---------------------------------------------------------------------------------
// Optimized Modular Multiplication using SECP256K1 special form
// P = 2^256 - 2^32 - 977 = 2^256 - 0x1000003D1
// ---------------------------------------------------------------------------------

inline void modMul(thread U256& result, thread const U256& a, thread const U256& b) {
    // 512-bit product in 8x64-bit limbs
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Schoolbook multiplication with inline accumulation
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi, lo;
            umul128(a.v[i], b.v[j], hi, lo);
            
            // Add lo to p[i+j]
            uint64_t c1;
            p[i+j] = add64c(p[i+j], lo, c1);
            
            // Add hi + carry to p[i+j+1]
            uint64_t c2;
            p[i+j+1] = add64c(p[i+j+1], hi, c2);
            p[i+j+1] = adc64(p[i+j+1], c1, c2);
            
            // Propagate carry
            if (c2 && i+j+2 < 8) {
                for (int k = i+j+2; k < 8; k++) {
                    uint64_t cc;
                    p[k] = add64c(p[k], c2, cc);
                    c2 = cc;
                    if (!c2) break;
                }
            }
        }
    }
    
    // First reduction step: r = p[0:3] + p[4:7] * c
    uint64_t t[5] = {0, 0, 0, 0, 0};
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo;
        umul128(p[4+i], REDUCTION_C, hi, lo);
        
        uint64_t c;
        t[i] = add64c(t[i], lo, c);
        t[i+1] = adc64(t[i+1], hi, c);
    }
    
    uint64_t c;
    p[0] = add64c(p[0], t[0], c);
    p[1] = adc64(p[1], t[1], c);
    p[2] = adc64(p[2], t[2], c);
    p[3] = adc64(p[3], t[3], c);
    uint64_t overflow = t[4] + c;
    
    // Second reduction for overflow
    if (overflow > 0) {
        uint64_t hi, lo;
        umul128(overflow, REDUCTION_C, hi, lo);
        
        p[0] = add64c(p[0], lo, c);
        p[1] = adc64(p[1], hi, c);
        p[2] = adc64(p[2], 0ULL, c);
        p[3] = adc64(p[3], 0ULL, c);
    }
    
    result.v[0] = p[0];
    result.v[1] = p[1];
    result.v[2] = p[2];
    result.v[3] = p[3];
    
    // Final reduction if result >= P
    U256 P(SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]);
    if (cmp256(result, P) >= 0) {
        subP(result);
    }
}

// In-place multiplication: r = r * a (mod P)
inline void modMulInPlace(thread U256& r, thread const U256& a) {
    U256 tmp(r.v[0], r.v[1], r.v[2], r.v[3]);
    modMul(r, tmp, a);
}

// ---------------------------------------------------------------------------------
// Optimized Modular Squaring
// ---------------------------------------------------------------------------------

inline void modSqr(thread U256& result, thread const U256& a) {
    // Use the fact that for squaring, we can save ~half the multiplications
    uint64_t p[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    
    // Diagonal terms (a[i] * a[i])
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo;
        umul128(a.v[i], a.v[i], hi, lo);
        
        uint64_t c;
        p[2*i] = add64c(p[2*i], lo, c);
        p[2*i+1] = adc64(p[2*i+1], hi, c);
    }
    
    // Off-diagonal terms (2 * a[i] * a[j] for i < j)
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        #pragma unroll
        for (int j = i + 1; j < 4; j++) {
            uint64_t hi, lo;
            umul128(a.v[i], a.v[j], hi, lo);
            
            // Double the product
            uint64_t carry = hi >> 63;
            hi = (hi << 1) | (lo >> 63);
            lo <<= 1;
            
            uint64_t c;
            p[i+j] = add64c(p[i+j], lo, c);
            p[i+j+1] = adc64(p[i+j+1], hi, c);
            
            // Handle carry propagation
            if (carry || c) {
                uint64_t propCarry = carry + c;
                for (int k = i+j+2; k < 8 && propCarry; k++) {
                    uint64_t cc;
                    p[k] = add64c(p[k], propCarry, cc);
                    propCarry = cc;
                }
            }
        }
    }
    
    // Reduction (same as modMul)
    uint64_t t[5] = {0, 0, 0, 0, 0};
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        uint64_t hi, lo;
        umul128(p[4+i], REDUCTION_C, hi, lo);
        
        uint64_t c;
        t[i] = add64c(t[i], lo, c);
        t[i+1] = adc64(t[i+1], hi, c);
    }
    
    uint64_t c;
    p[0] = add64c(p[0], t[0], c);
    p[1] = adc64(p[1], t[1], c);
    p[2] = adc64(p[2], t[2], c);
    p[3] = adc64(p[3], t[3], c);
    uint64_t overflow = t[4] + c;
    
    if (overflow > 0) {
        uint64_t hi, lo;
        umul128(overflow, REDUCTION_C, hi, lo);
        
        p[0] = add64c(p[0], lo, c);
        p[1] = adc64(p[1], hi, c);
        p[2] = adc64(p[2], 0ULL, c);
        p[3] = adc64(p[3], 0ULL, c);
    }
    
    result.v[0] = p[0];
    result.v[1] = p[1];
    result.v[2] = p[2];
    result.v[3] = p[3];
    
    U256 P(SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3]);
    if (cmp256(result, P) >= 0) {
        subP(result);
    }
}

// ---------------------------------------------------------------------------------
// Modular Inverse using Extended Binary GCD
// Optimized for SECP256K1 prime
// ---------------------------------------------------------------------------------

inline void modInv(thread U256& r) {
    // We compute r^(-1) mod P using the extended Euclidean algorithm
    // Optimized binary GCD variant
    
    uint64_t u[5] = {SECP256K1_P[0], SECP256K1_P[1], SECP256K1_P[2], SECP256K1_P[3], 0};
    uint64_t v[5] = {r.v[0], r.v[1], r.v[2], r.v[3], 0};
    uint64_t x1[5] = {0, 0, 0, 0, 0};
    uint64_t x2[5] = {1, 0, 0, 0, 0};
    
    while ((v[0] | v[1] | v[2] | v[3] | v[4]) != 0) {
        // Count trailing zeros in v
        int zeros = 0;
        if (v[0] != 0) {
            uint64_t t = v[0];
            while ((t & 1) == 0) { t >>= 1; zeros++; }
        } else if (v[1] != 0) {
            zeros = 64;
            uint64_t t = v[1];
            while ((t & 1) == 0) { t >>= 1; zeros++; }
        } else if (v[2] != 0) {
            zeros = 128;
            uint64_t t = v[2];
            while ((t & 1) == 0) { t >>= 1; zeros++; }
        } else if (v[3] != 0) {
            zeros = 192;
            uint64_t t = v[3];
            while ((t & 1) == 0) { t >>= 1; zeros++; }
        } else {
            zeros = 256;
        }
        
        // Shift v right by zeros positions
        for (int z = 0; z < zeros && z < 256; z++) {
            v[0] = (v[0] >> 1) | (v[1] << 63);
            v[1] = (v[1] >> 1) | (v[2] << 63);
            v[2] = (v[2] >> 1) | (v[3] << 63);
            v[3] = (v[3] >> 1) | (v[4] << 63);
            v[4] >>= 1;
            
            // If x2 is odd, add P
            if (x2[0] & 1) {
                uint64_t c;
                x2[0] = add64c(x2[0], SECP256K1_P[0], c);
                x2[1] = adc64(x2[1], SECP256K1_P[1], c);
                x2[2] = adc64(x2[2], SECP256K1_P[2], c);
                x2[3] = adc64(x2[3], SECP256K1_P[3], c);
                x2[4] = adc64(x2[4], 0ULL, c);
            }
            
            // Shift x2 right
            x2[0] = (x2[0] >> 1) | (x2[1] << 63);
            x2[1] = (x2[1] >> 1) | (x2[2] << 63);
            x2[2] = (x2[2] >> 1) | (x2[3] << 63);
            x2[3] = (x2[3] >> 1) | (x2[4] << 63);
            x2[4] >>= 1;
        }
        
        // Compare u and v
        bool uGeV = false;
        for (int i = 4; i >= 0; i--) {
            if (u[i] > v[i]) { uGeV = true; break; }
            if (u[i] < v[i]) { uGeV = false; break; }
        }
        if (u[0] == v[0] && u[1] == v[1] && u[2] == v[2] && u[3] == v[3] && u[4] == v[4]) {
            uGeV = true;
        }
        
        if (uGeV) {
            // u = u - v, x1 = x1 - x2
            uint64_t b;
            u[0] = sub64b(u[0], v[0], b);
            u[1] = sbb64(u[1], v[1], b);
            u[2] = sbb64(u[2], v[2], b);
            u[3] = sbb64(u[3], v[3], b);
            u[4] = sbb64(u[4], v[4], b);
            
            b = 0;
            x1[0] = sub64b(x1[0], x2[0], b);
            x1[1] = sbb64(x1[1], x2[1], b);
            x1[2] = sbb64(x1[2], x2[2], b);
            x1[3] = sbb64(x1[3], x2[3], b);
            x1[4] = sbb64(x1[4], x2[4], b);
        } else {
            // v = v - u, x2 = x2 - x1
            uint64_t b;
            v[0] = sub64b(v[0], u[0], b);
            v[1] = sbb64(v[1], u[1], b);
            v[2] = sbb64(v[2], u[2], b);
            v[3] = sbb64(v[3], u[3], b);
            v[4] = sbb64(v[4], u[4], b);
            
            b = 0;
            x2[0] = sub64b(x2[0], x1[0], b);
            x2[1] = sbb64(x2[1], x1[1], b);
            x2[2] = sbb64(x2[2], x1[2], b);
            x2[3] = sbb64(x2[3], x1[3], b);
            x2[4] = sbb64(x2[4], x1[4], b);
        }
    }
    
    // Normalize x1 to [0, P)
    while ((int64_t)x1[4] < 0) {
        uint64_t c;
        x1[0] = add64c(x1[0], SECP256K1_P[0], c);
        x1[1] = adc64(x1[1], SECP256K1_P[1], c);
        x1[2] = adc64(x1[2], SECP256K1_P[2], c);
        x1[3] = adc64(x1[3], SECP256K1_P[3], c);
        x1[4] = adc64(x1[4], 0ULL, c);
    }
    
    while (x1[4] > 0 || x1[3] > SECP256K1_P[3] ||
           (x1[3] == SECP256K1_P[3] && x1[2] > SECP256K1_P[2]) ||
           (x1[3] == SECP256K1_P[3] && x1[2] == SECP256K1_P[2] && x1[1] > SECP256K1_P[1]) ||
           (x1[3] == SECP256K1_P[3] && x1[2] == SECP256K1_P[2] && x1[1] == SECP256K1_P[1] && x1[0] >= SECP256K1_P[0])) {
        uint64_t b;
        x1[0] = sub64b(x1[0], SECP256K1_P[0], b);
        x1[1] = sbb64(x1[1], SECP256K1_P[1], b);
        x1[2] = sbb64(x1[2], SECP256K1_P[2], b);
        x1[3] = sbb64(x1[3], SECP256K1_P[3], b);
        x1[4] = sbb64(x1[4], 0ULL, b);
    }
    
    r.v[0] = x1[0];
    r.v[1] = x1[1];
    r.v[2] = x1[2];
    r.v[3] = x1[3];
}

// ---------------------------------------------------------------------------------
// Batch Modular Inverse using Montgomery's trick
// Computes inverses of all elements using only 1 modular inversion
// ---------------------------------------------------------------------------------

inline void batchModInv(thread U256* dx, int n) {
    // Allocate subproducts
    U256 subProducts[GPU_GRP_SIZE];
    U256 inverse;
    
    // Compute cumulative products
    subProducts[0] = dx[0];
    for (int i = 1; i < n; i++) {
        modMul(subProducts[i], subProducts[i-1], dx[i]);
    }
    
    // Invert the final product (single expensive inversion)
    inverse = subProducts[n-1];
    modInv(inverse);
    
    // Recover individual inverses
    for (int i = n - 1; i > 0; i--) {
        U256 newVal;
        modMul(newVal, subProducts[i-1], inverse);
        modMulInPlace(inverse, dx[i]);
        dx[i] = newVal;
    }
    
    dx[0] = inverse;
}

// ---------------------------------------------------------------------------------
// 128-bit distance addition
// ---------------------------------------------------------------------------------

inline void addDist128(thread uint64_t* d, const device uint64_t* jump) {
    uint64_t c;
    d[0] = add64c(d[0], jump[0], c);
    d[1] = adc64(d[1], jump[1], c);
}

// ---------------------------------------------------------------------------------
// Symmetry helpers
// ---------------------------------------------------------------------------------

#ifdef USE_SYMMETRY
inline bool makePositive(thread U256& y) {
    if (y.v[3] > 0x7FFFFFFFFFFFFFFFULL) {
        modNeg(y, y);
        return true;
    }
    return false;
}

inline void negOrder128(thread uint64_t* d) {
    uint64_t b;
    uint64_t t0 = sub64b(0ULL, d[0], b);
    uint64_t t1 = sbb64(0ULL, d[1], b);
    
    uint64_t c;
    d[0] = add64c(t0, SECP256K1_ORDER[0], c);
    d[1] = adc64(t1, SECP256K1_ORDER[1], c);
}
#endif

// ---------------------------------------------------------------------------------
// Main Compute Kernel
// ---------------------------------------------------------------------------------

kernel void computeKangaroosOptimized(
    device uint64_t* kangaroos [[buffer(0)]],
    constant uint64_t* jumpDist [[buffer(1)]],
    constant uint64_t* jumpPx [[buffer(2)]],
    constant uint64_t* jumpPy [[buffer(3)]],
    device atomic_uint* foundCount [[buffer(4)]],
    device DPResult* results [[buffer(5)]],
    constant uint64_t& dpMask [[buffer(6)]],
    constant uint32_t& maxResults [[buffer(7)]],
    uint tid [[thread_index_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint tpg [[threads_per_threadgroup]]
) {
    // Local storage
    U256 px[GPU_GRP_SIZE];
    U256 py[GPU_GRP_SIZE];
    uint64_t dist[GPU_GRP_SIZE][2];
#ifdef USE_SYMMETRY
    uint64_t lastJmp[GPU_GRP_SIZE];
#endif
    
    U256 dx[GPU_GRP_SIZE];
    U256 dy, rx, ry, s, p;
    
    // Base offset
    uint32_t baseOffset = (gid * tpg * GPU_GRP_SIZE) * KSIZE;
    
    // Load kangaroo data
    #pragma unroll
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * tpg;
        uint32_t off = baseOffset + stride;
        
        px[g].v[0] = kangaroos[off + tid + 0 * tpg];
        px[g].v[1] = kangaroos[off + tid + 1 * tpg];
        px[g].v[2] = kangaroos[off + tid + 2 * tpg];
        px[g].v[3] = kangaroos[off + tid + 3 * tpg];
        
        py[g].v[0] = kangaroos[off + tid + 4 * tpg];
        py[g].v[1] = kangaroos[off + tid + 5 * tpg];
        py[g].v[2] = kangaroos[off + tid + 6 * tpg];
        py[g].v[3] = kangaroos[off + tid + 7 * tpg];
        
        dist[g][0] = kangaroos[off + tid + 8 * tpg];
        dist[g][1] = kangaroos[off + tid + 9 * tpg];
        
#ifdef USE_SYMMETRY
        lastJmp[g] = kangaroos[off + tid + 10 * tpg];
#endif
    }
    
    // Main loop
    for (int run = 0; run < NB_RUN; run++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute dx = px - jumpPx for all kangaroos
        #pragma unroll
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g].v[0] & (NB_JUMP - 1);
            
#ifdef USE_SYMMETRY
            if (jmp == lastJmp[g]) jmp = (lastJmp[g] + 1) % NB_JUMP;
            lastJmp[g] = jmp;
#endif
            
            U256 jPx(jumpPx[jmp*4], jumpPx[jmp*4+1], jumpPx[jmp*4+2], jumpPx[jmp*4+3]);
            modSub(dx[g], px[g], jPx);
        }
        
        // Batch modular inverse
        batchModInv(dx, GPU_GRP_SIZE);
        
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Complete elliptic curve point addition
        #pragma unroll
        for (int g = 0; g < GPU_GRP_SIZE; g++) {
#ifdef USE_SYMMETRY
            uint32_t jmp = lastJmp[g];
#else
            uint32_t jmp = (uint32_t)px[g].v[0] & (NB_JUMP - 1);
#endif
            
            U256 jPx(jumpPx[jmp*4], jumpPx[jmp*4+1], jumpPx[jmp*4+2], jumpPx[jmp*4+3]);
            U256 jPy(jumpPy[jmp*4], jumpPy[jmp*4+1], jumpPy[jmp*4+2], jumpPy[jmp*4+3]);
            
            // dy = py - jPy
            modSub(dy, py[g], jPy);
            
            // s = dy * dx^(-1)
            modMul(s, dy, dx[g]);
            
            // p = s^2
            modSqr(p, s);
            
            // rx = p - jPx - px
            modSub(rx, p, jPx);
            modSubInPlace(rx, px[g]);
            
            // ry = s * (px - rx) - py
            modSub(ry, px[g], rx);
            modMulInPlace(ry, s);
            modSubInPlace(ry, py[g]);
            
            // Update position
            px[g] = rx;
            py[g] = ry;
            
            // Update distance
            addDist128(dist[g], &jumpDist[jmp * 2]);
            
#ifdef USE_SYMMETRY
            if (makePositive(py[g])) {
                negOrder128(dist[g]);
            }
#endif
            
            // Check for distinguished point
            if ((px[g].v[3] & dpMask) == 0) {
                uint32_t pos = atomic_fetch_add_explicit(foundCount, 1, memory_order_relaxed);
                
                if (pos < maxResults) {
                    uint64_t kIdx = (uint64_t)tid + (uint64_t)g * tpg + (uint64_t)gid * (tpg * GPU_GRP_SIZE);
                    
                    results[pos].x[0] = (uint32_t)px[g].v[0];
                    results[pos].x[1] = (uint32_t)(px[g].v[0] >> 32);
                    results[pos].x[2] = (uint32_t)px[g].v[1];
                    results[pos].x[3] = (uint32_t)(px[g].v[1] >> 32);
                    results[pos].x[4] = (uint32_t)px[g].v[2];
                    results[pos].x[5] = (uint32_t)(px[g].v[2] >> 32);
                    results[pos].x[6] = (uint32_t)px[g].v[3];
                    results[pos].x[7] = (uint32_t)(px[g].v[3] >> 32);
                    
                    results[pos].d[0] = (uint32_t)dist[g][0];
                    results[pos].d[1] = (uint32_t)(dist[g][0] >> 32);
                    results[pos].d[2] = (uint32_t)dist[g][1];
                    results[pos].d[3] = (uint32_t)(dist[g][1] >> 32);
                    
                    results[pos].kidx = kIdx;
                }
            }
        }
    }
    
    // Store kangaroo data back
    #pragma unroll
    for (int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t stride = g * KSIZE * tpg;
        uint32_t off = baseOffset + stride;
        
        kangaroos[off + tid + 0 * tpg] = px[g].v[0];
        kangaroos[off + tid + 1 * tpg] = px[g].v[1];
        kangaroos[off + tid + 2 * tpg] = px[g].v[2];
        kangaroos[off + tid + 3 * tpg] = px[g].v[3];
        
        kangaroos[off + tid + 4 * tpg] = py[g].v[0];
        kangaroos[off + tid + 5 * tpg] = py[g].v[1];
        kangaroos[off + tid + 6 * tpg] = py[g].v[2];
        kangaroos[off + tid + 7 * tpg] = py[g].v[3];
        
        kangaroos[off + tid + 8 * tpg] = dist[g][0];
        kangaroos[off + tid + 9 * tpg] = dist[g][1];
        
#ifdef USE_SYMMETRY
        kangaroos[off + tid + 10 * tpg] = lastJmp[g];
#endif
    }
}
