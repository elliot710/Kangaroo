/*
 * Metal Kernel Implementation for Pollard's Kangaroo Algorithm
 * Strictly following Metal Shading Language Specification Version 3.0+
 *
 * Optimized for Apple Silicon (M-series)
 */

#include <metal_stdlib>
using namespace metal;

// ----------------------------------------------------------------------------
// Constants & Configuration
// ----------------------------------------------------------------------------

// These must match Constants.h!
#define NB_JUMP 32
#define NB_RUN 8
#define GPU_GRP_SIZE 128
#ifdef USE_SYMMETRY
#define KSIZE 11
#else
#define KSIZE 10
#endif
#define NBBLOCK 5

// 64-bit unsigned integer type
typedef ulong uint64_t;
typedef long  int64_t;
typedef uint  uint32_t;
typedef int   int32_t;

// P = 2^256 - 2^32 - 977
constant uint64_t P_0 = 0xFFFFFFFEFFFFFC2FULL;
constant uint64_t P_1 = 0xFFFFFFFFFFFFFFFFULL;
constant uint64_t P_2 = 0xFFFFFFFFFFFFFFFFULL;
constant uint64_t P_3 = 0xFFFFFFFFFFFFFFFFULL;

// MM64: -1/P mod 2^64
constant uint64_t MM64 = 0xD838091DD2253531ULL;

// Output structure
struct DPOutput {
    uint32_t x[8];       // 256-bit x
    uint32_t dist[4];    // 128-bit distance
    uint64_t kIdx;       // Kangaroo Index
};

// ----------------------------------------------------------------------------
// Math Helpers (Inline)
// ----------------------------------------------------------------------------

// Unsigned 64-bit addition with carry in/out
[[gnu::always_inline]] inline uint64_t add_carry_out(uint64_t a, uint64_t b, thread uint64_t &carry) {
    uint64_t res = a + b;
    carry = (res < a) ? 1 : 0;
    return res;
}

[[gnu::always_inline]] inline uint64_t add_carry_in_out(uint64_t a, uint64_t b, thread uint64_t &carry) {
    uint64_t sum = a + b;
    uint64_t c1 = (sum < a) ? 1 : 0;
    uint64_t res = sum + carry;
    uint64_t c2 = (res < sum) ? 1 : 0;
    carry = c1 + c2;  // At most 1 since c1 and c2 can't both be 1
    return res;
}

// Unsigned 64-bit subtraction with borrow in/out
[[gnu::always_inline]] inline uint64_t sub_borrow_out(uint64_t a, uint64_t b, thread uint64_t &borrow) {
    uint64_t res = a - b;
    borrow = (a < b) ? 1 : 0;
    return res;
}

[[gnu::always_inline]] inline uint64_t sub_borrow_in_out(uint64_t a, uint64_t b, thread uint64_t &borrow) {
    uint64_t diff = a - b;
    uint64_t b1 = (a < b) ? 1 : 0;
    uint64_t res = diff - borrow;
    uint64_t b2 = (diff < borrow) ? 1 : 0; // if diff < borrow means we wrapped
    
    borrow = b1 + b2;  // At most 1 since b1 and b2 can't both be 1
    return res;
}

// ----------------------------------------------------------------------------
// 256-bit Arithmetic
// ----------------------------------------------------------------------------

// Add P to r (r += P)
[[gnu::always_inline]] inline void AddP(thread uint64_t* r) {
    uint64_t c = 0;
    r[0] = add_carry_out(r[0], P_0, c);
    r[1] = add_carry_in_out(r[1], P_1, c);
    r[2] = add_carry_in_out(r[2], P_2, c);
    r[3] = add_carry_in_out(r[3], P_3, c);
    r[4] = r[4] + c; // no carry check needed for top limb usually
}

// Subtract P from r (r -= P)
[[gnu::always_inline]] inline void SubP(thread uint64_t* r) {
    uint64_t c = 0;
    r[0] = sub_borrow_out(r[0], P_0, c);
    r[1] = sub_borrow_in_out(r[1], P_1, c);
    r[2] = sub_borrow_in_out(r[2], P_2, c);
    r[3] = sub_borrow_in_out(r[3], P_3, c);
    r[4] = r[4] - c;
}

// Copy from constant to thread
[[gnu::always_inline]] inline void Load(thread uint64_t* r, constant uint64_t* a) {
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3]; r[4] = a[4]; 
}

// Copy 5 elements from 5-element array
[[gnu::always_inline]] inline void Load5(thread uint64_t* r, thread uint64_t* a) {
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3]; r[4] = a[4];
}

// Copy 4 elements
[[gnu::always_inline]] inline void Load256(thread uint64_t* r, thread uint64_t* a) {
    r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
}

// Negation (-r)
[[gnu::always_inline]] inline void Neg(thread uint64_t* r) {
    uint64_t c = 0;
    r[0] = sub_borrow_out(0, r[0], c);
    r[1] = sub_borrow_in_out(0, r[1], c);
    r[2] = sub_borrow_in_out(0, r[2], c);
    r[3] = sub_borrow_in_out(0, r[3], c);
    r[4] = sub_borrow_in_out(0, r[4], c);
}

// ShiftR62
[[gnu::always_inline]] inline void ShiftR62(thread uint64_t* r) {
    r[0] = (r[1] << 2) | (r[0] >> 62);
    r[1] = (r[2] << 2) | (r[1] >> 62);
    r[2] = (r[3] << 2) | (r[2] >> 62);
    r[3] = (r[4] << 2) | (r[3] >> 62);
    r[4] = (int64_t)(r[4]) >> 62; // Sign extend
}

[[gnu::always_inline]] inline void ShiftR62_Carry(thread uint64_t* dest, thread uint64_t* r, uint64_t carry) {
    dest[0] = (r[1] << 2) | (r[0] >> 62);
    dest[1] = (r[2] << 2) | (r[1] >> 62);
    dest[2] = (r[3] << 2) | (r[2] >> 62);
    dest[3] = (r[4] << 2) | (r[3] >> 62);
    dest[4] = (carry << 2) | (r[4] >> 62);
}

// Multiply 5-element by scalar (signed)
[[gnu::always_inline]] inline void IMult(thread uint64_t *r, thread uint64_t *a, int64_t b) {
    uint64_t t[5];
    bool negative = (b < 0);
    uint64_t ub = (uint64_t)(negative ? -b : b);
    
    if (negative) {
        uint64_t c = 0;
        t[0] = sub_borrow_out(0, a[0], c);
        t[1] = sub_borrow_in_out(0, a[1], c);
        t[2] = sub_borrow_in_out(0, a[2], c);
        t[3] = sub_borrow_in_out(0, a[3], c);
        t[4] = sub_borrow_in_out(0, a[4], c);
    } else {
        Load5(t, a);
    }
    
    // r[0]
    r[0] = t[0] * ub;
    uint64_t c = mulhi(t[0], ub);
    
    // r[1]
    uint64_t p_lo = t[1] * ub;
    uint64_t p_hi = mulhi(t[1], ub);
    uint64_t r1 = p_lo + c;
    c = p_hi + ((r1 < p_lo) ? 1 : 0);
    r[1] = r1;
    
    // r[2]
    p_lo = t[2] * ub;
    p_hi = mulhi(t[2], ub);
    uint64_t r2 = p_lo + c;
    c = p_hi + ((r2 < p_lo) ? 1 : 0);
    r[2] = r2;
    
    // r[3]
    p_lo = t[3] * ub;
    p_hi = mulhi(t[3], ub);
    uint64_t r3 = p_lo + c;
    c = p_hi + ((r3 < p_lo) ? 1 : 0);
    r[3] = r3;
    
    // r[4]
    p_lo = t[4] * ub;
    p_hi = mulhi(t[4], ub); 
    uint64_t r4 = p_lo + c;
    r[4] = r4;
}

[[gnu::always_inline]] inline uint64_t IMultC(thread uint64_t* r, thread uint64_t* a, int64_t b) {
    uint64_t t[5];
    bool negative = (b < 0);
    uint64_t ub = (uint64_t)(negative ? -b : b);
    
    if (negative) {
        uint64_t c = 0;
        t[0] = sub_borrow_out(0, a[0], c);
        t[1] = sub_borrow_in_out(0, a[1], c);
        t[2] = sub_borrow_in_out(0, a[2], c);
        t[3] = sub_borrow_in_out(0, a[3], c);
        t[4] = sub_borrow_in_out(0, a[4], c);
    } else {
        Load5(t, a);
    }
    
    // r = t * ub
    uint64_t c = mulhi(t[0], ub);
    r[0] = t[0] * ub;
    
    uint64_t p_lo, p_hi;
    
    p_lo = t[1] * ub;
    p_hi = mulhi(t[1], ub);
    r[1] = p_lo + c;
    c = p_hi + ((r[1] < p_lo) ? 1 : 0);
    
    p_lo = t[2] * ub;
    p_hi = mulhi(t[2], ub);
    r[2] = p_lo + c;
    c = p_hi + ((r[2] < p_lo) ? 1 : 0);
    
    p_lo = t[3] * ub;
    p_hi = mulhi(t[3], ub);
    r[3] = p_lo + c;
    c = p_hi + ((r[3] < p_lo) ? 1 : 0);
    
    // For t[4], we need SIGNED multiplication because t[4] is the sign extension
    // CUDA uses madc.hi.s64 for the final carry computation
    // t[4] is treated as signed, b is the positive value ub
    int64_t st4 = (int64_t)t[4];
    int64_t s_product_hi = mulhi(st4, (int64_t)ub);
    
    p_lo = t[4] * ub;  // Low part is same for signed/unsigned
    r[4] = p_lo + c;
    // carryOut = signed high part + carry from addition
    int64_t carry_add = ((r[4] < p_lo) ? 1LL : 0LL);
    uint64_t carryOut = (uint64_t)(s_product_hi + carry_add);
    
    return carryOut;
}

// MulP
[[gnu::always_inline]] inline void MulP(thread uint64_t *r, uint64_t a) {
    // a * P_CONST (0x1000003D1)
    
    uint64_t al = a * 0x1000003D1ULL;
    uint64_t ah = mulhi(a, 0x1000003D1ULL);
    
    uint64_t c = 0;
    r[0] = sub_borrow_out(0, al, c);
    r[1] = sub_borrow_in_out(0, ah, c);
    r[2] = sub_borrow_in_out(0, 0, c);
    r[3] = sub_borrow_in_out(0, 0, c);
    r[4] = sub_borrow_in_out(a, 0, c);
}

// DivStep
[[gnu::always_inline]] inline void _DivStep62(thread uint64_t* u, thread uint64_t* v, 
                       thread int32_t *pos, 
                       thread int64_t* uu, thread int64_t* uv, 
                       thread int64_t* vu, thread int64_t* vv) {
    *uu = 1; *uv = 0;
    *vu = 0; *vv = 1;
    
    uint32_t bitCount = 62;
    uint64_t u0 = u[0];
    uint64_t v0 = v[0];
    uint64_t uh, vh;
    
    // Find active limb
    while(*pos > 0 && (u[*pos] | v[*pos]) == 0) (*pos)--;
    
    if (*pos == 0) {
        uh = u[0];
        vh = v[0];
    } else {
        uint32_t s = clz(u[*pos] | v[*pos]);
        if (s == 0) {
            uh = u[*pos];
            vh = v[*pos];
        } else {
            uh = (u[*pos] << s) | (u[*pos - 1] >> (64 - s));
            vh = (v[*pos] << s) | (v[*pos - 1] >> (64 - s));
        }
    }
    
    for (int i=0; i<62; i++) { // Max 62 steps
        uint32_t zeros = ctz(v0 | (1ULL << bitCount));
        if (zeros > bitCount) zeros = bitCount; // clamp
        
        v0 >>= zeros;
        vh >>= zeros;
        *uu <<= zeros;
        *uv <<= zeros;
        bitCount -= zeros;
        
        if (bitCount == 0) break;
        
        if (vh < uh) {
            uint64_t tmp64 = uh; uh = vh; vh = tmp64;
            tmp64 = u0; u0 = v0; v0 = tmp64;
            int64_t tmpi64 = *uu; *uu = *vu; *vu = tmpi64;
            tmpi64 = *uv; *uv = *vv; *vv = tmpi64;
        }
        
        vh -= uh;
        v0 -= u0;
        *vv -= *uv;
        *vu -= *uu;
    }
}

// Matrix operations
[[gnu::always_inline]] inline void MatrixVecMul(thread uint64_t* u, thread uint64_t* v, int64_t _11, int64_t _12, int64_t _21, int64_t _22) {
    uint64_t t1[5], t2[5], t3[5], t4[5];
    IMult(t1, u, _11);
    IMult(t2, v, _12);
    IMult(t3, u, _21);
    IMult(t4, v, _22);
    
    uint64_t c = 0;
    u[0] = add_carry_out(t1[0], t2[0], c);
    u[1] = add_carry_in_out(t1[1], t2[1], c);
    u[2] = add_carry_in_out(t1[2], t2[2], c);
    u[3] = add_carry_in_out(t1[3], t2[3], c);
    u[4] = add_carry_in_out(t1[4], t2[4], c);
    
    c = 0;
    v[0] = add_carry_out(t3[0], t4[0], c);
    v[1] = add_carry_in_out(t3[1], t4[1], c);
    v[2] = add_carry_in_out(t3[2], t4[2], c);
    v[3] = add_carry_in_out(t3[3], t4[3], c);
    v[4] = add_carry_in_out(t3[4], t4[4], c);
}

[[gnu::always_inline]] inline void MatrixVecMulHalf(thread uint64_t* dest, thread uint64_t* u, thread uint64_t* v, int64_t _11, int64_t _12, thread uint64_t &carry) {
    uint64_t t1[5], t2[5];
    uint64_t c1 = IMultC(t1, u, _11);
    uint64_t c2 = IMultC(t2, v, _12);
    
    uint64_t c = 0;
    dest[0] = add_carry_out(t1[0], t2[0], c);
    dest[1] = add_carry_in_out(t1[1], t2[1], c);
    dest[2] = add_carry_in_out(t1[2], t2[2], c);
    dest[3] = add_carry_in_out(t1[3], t2[3], c);
    dest[4] = add_carry_in_out(t1[4], t2[4], c);
    
    carry = c1 + c2 + c;
}

[[gnu::always_inline]] inline uint64_t AddCh(thread uint64_t* r, thread uint64_t* a, uint64_t carry) {
    uint64_t c = 0;
    r[0] = add_carry_out(r[0], a[0], c);
    r[1] = add_carry_in_out(r[1], a[1], c);
    r[2] = add_carry_in_out(r[2], a[2], c);
    r[3] = add_carry_in_out(r[3], a[3], c);
    r[4] = add_carry_in_out(r[4], a[4], c);
    return carry + c;
}

// ----------------------------------------------------------------------------
// ModInv
// ----------------------------------------------------------------------------

void _ModInv(thread uint64_t *R) {
    int64_t uu, uv, vu, vv;
    uint64_t mr0, ms0;
    int32_t pos = NBBLOCK - 1;
    
    uint64_t u[NBBLOCK];
    uint64_t v[NBBLOCK];
    uint64_t r[NBBLOCK];
    uint64_t s[NBBLOCK];
    uint64_t tr[NBBLOCK];
    uint64_t ts[NBBLOCK];
    uint64_t r0[NBBLOCK];
    uint64_t s0[NBBLOCK];
    uint64_t carryR = 0;
    uint64_t carryS = 0;
    
    u[0] = P_0; u[1] = P_1; u[2] = P_2; u[3] = P_3; u[4] = 0;
    Load5(v, R);
    
    r[0] = 0; r[1] = 0; r[2] = 0; r[3] = 0; r[4] = 0;
    s[0] = 1; s[1] = 0; s[2] = 0; s[3] = 0; s[4] = 0;
    
    // Bounded loop for safety inside kernel
    for(int loop=0; loop<800; loop++) {
        _DivStep62(u, v, &pos, &uu, &uv, &vu, &vv);
        
        MatrixVecMul(u, v, uu, uv, vu, vv);
        
        if (((int64_t)u[4]) < 0) { Neg(u); uu = -uu; uv = -uv; }
        if (((int64_t)v[4]) < 0) { Neg(v); vu = -vu; vv = -vv; }
        
        ShiftR62(u);
        ShiftR62(v);
        
        // Update r
        MatrixVecMulHalf(tr, r, s, uu, uv, carryR);
        mr0 = (tr[0] * MM64) & 0x3FFFFFFFFFFFFFFF;
        MulP(r0, mr0);
        carryR = AddCh(tr, r0, carryR);
        
        bool vIsZero = (v[0] == 0 && v[1] == 0 && v[2] == 0 && v[3] == 0 && v[4] == 0);
        
        if (vIsZero) {
            ShiftR62_Carry(r, tr, carryR);
            break;
        } else {
            // Update s
            MatrixVecMulHalf(ts, r, s, vu, vv, carryS);
            ms0 = (ts[0] * MM64) & 0x3FFFFFFFFFFFFFFF;
            MulP(s0, ms0);
            carryS = AddCh(ts, s0, carryS);
        }
        
        ShiftR62_Carry(r, tr, carryR);
        ShiftR62_Carry(s, ts, carryS);
    }
    
    bool uIsOne = (u[0] == 1 && u[1] == 0 && u[2] == 0 && u[3] == 0 && u[4] == 0);
    
    if (!uIsOne) {
        R[0] = 0; R[1] = 0; R[2] = 0; R[3] = 0; R[4] = 0;
        return;
    }
    
    // Bounded normalization to prevent hangs
    // Need more iterations because r can be quite far from [0, P)
    for(int i=0; i<1000 && ((int64_t)r[4]) < 0; i++) AddP(r);
    for(int i=0; i<1000 && !(((int64_t)r[4]) < 0); i++) SubP(r);
    AddP(r);
    
    Load5(R, r);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Modulus Arithmetic Handlers
// ----------------------------------------------------------------------------

// Multiplies a 4-limb integer 'a' by a 64-bit scalar 'b', producing a 5-limb result 'r'
[[gnu::always_inline]] inline void UMult4(thread uint64_t* r, thread uint64_t* a, uint64_t b) {
    uint64_t carry = 0;
    uint64_t p, h;

    p = a[0] * b;
    h = mulhi(a[0], b);
    r[0] = p;
    carry = h;

    p = a[1] * b;
    h = mulhi(a[1], b);
    r[1] = p + carry;
    carry = h + ((r[1] < p) ? 1 : 0);

    p = a[2] * b;
    h = mulhi(a[2], b);
    r[2] = p + carry;
    carry = h + ((r[2] < p) ? 1 : 0);

    p = a[3] * b;
    h = mulhi(a[3], b);
    r[3] = p + carry;
    carry = h + ((r[3] < p) ? 1 : 0);

    // a[4] is NOT accessed because 'a' is treated as a 256-bit (4-limb) number
    // The carry is the 5th limb
    r[4] = carry;
}

void _ModMultClean(thread uint64_t *r, thread uint64_t *a, thread uint64_t *b) {
    uint64_t r512[8] = {0};
    uint64_t t[5];
    uint64_t c;
    
    // Multiply a[0..3] by b[0]
    UMult4(r512, a, b[0]);
    
    // Multiply a[0..3] by b[1]
    UMult4(t, a, b[1]);
    c = 0;
    r512[1] = add_carry_out(r512[1], t[0], c);
    r512[2] = add_carry_in_out(r512[2], t[1], c);
    r512[3] = add_carry_in_out(r512[3], t[2], c);
    r512[4] = add_carry_in_out(r512[4], t[3], c);
    r512[5] = add_carry_in_out(r512[5], t[4], c);
    
    UMult4(t, a, b[2]);
    c = 0;
    r512[2] = add_carry_out(r512[2], t[0], c);
    r512[3] = add_carry_in_out(r512[3], t[1], c);
    r512[4] = add_carry_in_out(r512[4], t[2], c);
    r512[5] = add_carry_in_out(r512[5], t[3], c);
    r512[6] = add_carry_in_out(r512[6], t[4], c);
    
    UMult4(t, a, b[3]);
    c = 0;
    r512[3] = add_carry_out(r512[3], t[0], c);
    r512[4] = add_carry_in_out(r512[4], t[1], c);
    r512[5] = add_carry_in_out(r512[5], t[2], c);
    r512[6] = add_carry_in_out(r512[6], t[3], c);
    r512[7] = add_carry_in_out(r512[7], t[4], c);
    
    // Reduce
    // Use UMult4 to multiply r512[4..7] by the constant
    UMult4(t, (thread uint64_t*)(r512+4), 0x1000003D1ULL);
    
    c = 0;
    r512[0] = add_carry_out(r512[0], t[0], c);
    r512[1] = add_carry_in_out(r512[1], t[1], c);
    r512[2] = add_carry_in_out(r512[2], t[2], c);
    r512[3] = add_carry_in_out(r512[3], t[3], c);
    
    // Total overflow beyond 256 bits is the carry 'c' + the high part 't[4]'
    // We must multiply this overflow by the constant and add again.
    uint64_t overflow = c + t[4];
    
    uint64_t p_lo = overflow * 0x1000003D1ULL;
    uint64_t p_hi = mulhi(overflow, 0x1000003D1ULL);
    
    c = 0;
    r[0] = add_carry_out(r512[0], p_lo, c);
    r[1] = add_carry_in_out(r512[1], p_hi, c);
    r[2] = add_carry_in_out(r512[2], 0, c);
    r[3] = add_carry_in_out(r512[3], 0, c);
    
    // Final check for carry (very rare)
    if (c) {
        uint64_t cc = 0;
        r[0] = add_carry_out(r[0], 0x1000003D1ULL, cc);
        r[1] = add_carry_in_out(r[1], 0, cc);
        r[2] = add_carry_in_out(r[2], 0, cc);
        r[3] = add_carry_in_out(r[3], 0, cc);
    }
    
    // Final reduction: if r >= P, subtract P
    // Check if r >= P (P = 2^256 - 2^32 - 977)
    // r >= P iff r[3] > P_3 || (r[3] == P_3 && r[2] > P_2) || ...
    // Since P_3,P_2,P_1 = 0xFF..FF, we only need to check if r >= P when r[3] == P_3
    // and that requires r[2] == P_2, r[1] == P_1, and r[0] >= P_0
    if (r[3] == P_3 && r[2] == P_2 && r[1] == P_1 && r[0] >= P_0) {
        uint64_t borrow = 0;
        r[0] = sub_borrow_out(r[0], P_0, borrow);
        r[1] = sub_borrow_in_out(r[1], P_1, borrow);
        r[2] = sub_borrow_in_out(r[2], P_2, borrow);
        r[3] = sub_borrow_in_out(r[3], P_3, borrow);
    }
}

void _ModSqrClean(thread uint64_t *rp, thread uint64_t *up) {
    _ModMultClean(rp, up, up);
}

// ----------------------------------------------------------------------------
// Group Operations
// ----------------------------------------------------------------------------

void _ModInvGrouped(thread uint64_t r[GPU_GRP_SIZE][4]) {
    uint64_t subp[GPU_GRP_SIZE][4];
    uint64_t newValue[4];
    uint64_t inverse[5];

    Load256(subp[0], r[0]);
    for(uint32_t i = 1; i < GPU_GRP_SIZE; i++) {
        _ModMultClean(subp[i], subp[i - 1], r[i]);
    }

    Load256(inverse, subp[GPU_GRP_SIZE - 1]);
    inverse[4] = 0;
    _ModInv(inverse);

    for(uint32_t i = GPU_GRP_SIZE - 1; i > 0; i--) {
        _ModMultClean(newValue, subp[i - 1], inverse);
        _ModMultClean(inverse, inverse, r[i]);
        Load256(r[i], newValue);
    }
    
    Load256(r[0], inverse);
}

// ----------------------------------------------------------------------------
// KERNEL Helpers
// ----------------------------------------------------------------------------

void ModSub256(thread uint64_t* r, thread uint64_t* a, constant uint64_t* b) {
    uint64_t c = 0;
    r[0] = sub_borrow_out(a[0], b[0], c);
    r[1] = sub_borrow_in_out(a[1], b[1], c);
    r[2] = sub_borrow_in_out(a[2], b[2], c);
    r[3] = sub_borrow_in_out(a[3], b[3], c);
    
    if (c) {
        c = 0;
        r[0] = add_carry_out(r[0], P_0, c);
        r[1] = add_carry_in_out(r[1], P_1, c);
        r[2] = add_carry_in_out(r[2], P_2, c);
        r[3] = add_carry_in_out(r[3], P_3, c);
    }
}

void ModSub256Thread(thread uint64_t* r, thread uint64_t* a, thread uint64_t* b) {
    uint64_t c = 0;
    r[0] = sub_borrow_out(a[0], b[0], c);
    r[1] = sub_borrow_in_out(a[1], b[1], c);
    r[2] = sub_borrow_in_out(a[2], b[2], c);
    r[3] = sub_borrow_in_out(a[3], b[3], c);
    
    if (c) {
        c = 0;
        r[0] = add_carry_out(r[0], P_0, c);
        r[1] = add_carry_in_out(r[1], P_1, c);
        r[2] = add_carry_in_out(r[2], P_2, c);
        r[3] = add_carry_in_out(r[3], P_3, c);
    }
}

void Add128(thread uint64_t* r, constant uint64_t* a) {
    uint64_t c = 0;
    r[0] = add_carry_out(r[0], a[0], c);
    r[1] = add_carry_in_out(r[1], a[1], c);
}

// ----------------------------------------------------------------------------
// Main Kernel
// ----------------------------------------------------------------------------
kernel void computeKangaroos(
    device uint64_t* kangaroos [[buffer(0)]],
    constant uint64_t* jumpDist [[buffer(1)]],
    constant uint64_t* jumpPx [[buffer(2)]],
    constant uint64_t* jumpPy [[buffer(3)]],
    device atomic_uint* foundCount [[buffer(4)]],
    device DPOutput* output [[buffer(5)]],
    constant uint64_t& dpMask [[buffer(6)]],
    constant uint32_t& maxFound [[buffer(7)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 threadsPerGroup [[threads_per_threadgroup]]
) {
    uint32_t idx = tid.x;
    uint32_t strideSize = threadsPerGroup.x * KSIZE;
    uint32_t blockOffset = gid.x * threadsPerGroup.x * GPU_GRP_SIZE * KSIZE;
    
    uint64_t px[GPU_GRP_SIZE][4];
    uint64_t py[GPU_GRP_SIZE][4];
    uint64_t dist[GPU_GRP_SIZE][2];
    uint64_t dx[GPU_GRP_SIZE][4];
    uint64_t dy[4], rx[4], ry[4], _s[4], _p[4];
    
    // Load
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t base = blockOffset + g * strideSize + idx;
        px[g][0] = kangaroos[base + 0 * threadsPerGroup.x];
        px[g][1] = kangaroos[base + 1 * threadsPerGroup.x];
        px[g][2] = kangaroos[base + 2 * threadsPerGroup.x];
        px[g][3] = kangaroos[base + 3 * threadsPerGroup.x];
        
        py[g][0] = kangaroos[base + 4 * threadsPerGroup.x];
        py[g][1] = kangaroos[base + 5 * threadsPerGroup.x];
        py[g][2] = kangaroos[base + 6 * threadsPerGroup.x];
        py[g][3] = kangaroos[base + 7 * threadsPerGroup.x];
        
        dist[g][0] = kangaroos[base + 8 * threadsPerGroup.x];
        dist[g][1] = kangaroos[base + 9 * threadsPerGroup.x];
    }
    
    // Run
    for(int run = 0; run < NB_RUN; run++) {
        for(int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g][0] & (NB_JUMP - 1);
            ModSub256(dx[g], px[g], jumpPx + jmp * 4);
        }
        
        _ModInvGrouped(dx);
        
        for(int g = 0; g < GPU_GRP_SIZE; g++) {
            uint32_t jmp = (uint32_t)px[g][0] & (NB_JUMP - 1);
            constant uint64_t* jPxPtr = jumpPx + jmp * 4;
            constant uint64_t* jPyPtr = jumpPy + jmp * 4;
            constant uint64_t* jDPtr = jumpDist + jmp * 2;
            
            ModSub256(dy, py[g], jPyPtr);
            _ModMultClean(_s, dy, dx[g]);
            _ModSqrClean(_p, _s);
            
            ModSub256(rx, _p, jPxPtr);
            ModSub256Thread(rx, rx, px[g]);
            
            ModSub256Thread(ry, px[g], rx);
            _ModMultClean(ry, ry, _s);
            ModSub256Thread(ry, ry, py[g]);
            
            Load256(px[g], rx);
            Load256(py[g], ry);
            
            Add128(dist[g], jDPtr);
            
            if ((px[g][3] & dpMask) == 0) {
                uint32_t pos = atomic_fetch_add_explicit(foundCount, 1, memory_order_relaxed);
                if (pos < maxFound) {
                    uint64_t kIdx = (uint64_t)idx + (uint64_t)g * threadsPerGroup.x + (uint64_t)gid.x * (threadsPerGroup.x * GPU_GRP_SIZE);
                    device DPOutput* out = output + pos;
                    thread uint32_t* x32 = (thread uint32_t*)px[g];
                    out->x[0] = x32[0]; out->x[1] = x32[1]; out->x[2] = x32[2]; out->x[3] = x32[3];
                    out->x[4] = x32[4]; out->x[5] = x32[5]; out->x[6] = x32[6]; out->x[7] = x32[7];
                    thread uint32_t* d32 = (thread uint32_t*)dist[g];
                    out->dist[0] = d32[0]; out->dist[1] = d32[1]; out->dist[2] = d32[2]; out->dist[3] = d32[3];
                    out->kIdx = kIdx;
                }
            }
        }
    }
    
    // Store
    for(int g = 0; g < GPU_GRP_SIZE; g++) {
        uint32_t base = blockOffset + g * strideSize + idx;
        kangaroos[base + 0 * threadsPerGroup.x] = px[g][0];
        kangaroos[base + 1 * threadsPerGroup.x] = px[g][1];
        kangaroos[base + 2 * threadsPerGroup.x] = px[g][2];
        kangaroos[base + 3 * threadsPerGroup.x] = px[g][3];
        kangaroos[base + 4 * threadsPerGroup.x] = py[g][0];
        kangaroos[base + 5 * threadsPerGroup.x] = py[g][1];
        kangaroos[base + 6 * threadsPerGroup.x] = py[g][2];
        kangaroos[base + 7 * threadsPerGroup.x] = py[g][3];
        kangaroos[base + 8 * threadsPerGroup.x] = dist[g][0];
        kangaroos[base + 9 * threadsPerGroup.x] = dist[g][1];
    }
}
