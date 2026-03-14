// =============================================================================
// macro.hpp — Debug macros, depth/shape defines, narrowed types, math functions
// =============================================================================
// Contains all compile-time knobs, narrowed accumulator/activation types,
// quantization scale helpers, and fixed-point math approximations.
// =============================================================================
#ifndef __SSMU_MACRO_HPP__
#define __SSMU_MACRO_HPP__

#include "ssmu_config.hpp"

#ifndef __SYNTHESIS__
#include <cstdio>
#include <cmath>
#endif

// =============================================================
// Debug macros
// =============================================================
#ifndef __SYNTHESIS__
  #define DUT_PRINTF(...) do { std::printf(__VA_ARGS__); std::fflush(stdout); } while(0)
#else
  #define DUT_PRINTF(...) do {} while(0)
#endif

#ifndef __SYNTHESIS__
static inline bool dbg_tok_sel(int idx) {
    return idx == 0 || idx == 312 || idx == 376;
}
static const int DBG_LANE = 3;

static inline void dump_vec_token(FILE* f, const DTYPE_VEC& v) {
    float tmp[VEC_FACTOR];
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
        tmp[l] = (float)vget(v, l);
    }
    std::fwrite(tmp, sizeof(float), VEC_FACTOR, f);
}
#endif

// =============================================================
// Feature toggles
// =============================================================
#ifndef SSMU_ENABLE_LM_MODULES
#define SSMU_ENABLE_LM_MODULES 0
#endif
#ifndef SSMU_ENABLE_ACT_Q_CHAIN
#define SSMU_ENABLE_ACT_Q_CHAIN 0
#endif
#ifndef SSMU_ENABLE_DTADAPT
#define SSMU_ENABLE_DTADAPT 0
#endif
#ifndef SSMU_ENABLE_DTB_QUANT
#define SSMU_ENABLE_DTB_QUANT 0
#endif
#ifndef SSMU_ENABLE_UDYZ_Q
#define SSMU_ENABLE_UDYZ_Q 0
#endif
#ifndef SSMU_ENABLE_GEMM_MUXDEMUX
#define SSMU_ENABLE_GEMM_MUXDEMUX 0
#endif
#ifndef SSMU_ENABLE_TRACE_DDR
#define SSMU_ENABLE_TRACE_DDR 0
#endif
#ifndef SSMU_ENABLE_H1_STREAM_OUT
#define SSMU_ENABLE_H1_STREAM_OUT 1
#endif
#ifndef SSMU_ENABLE_TRACE_STREAMS
#define SSMU_ENABLE_TRACE_STREAMS 0
#endif

// =============================================================
// DT/delta policy
// =============================================================
#ifndef SSMU_ENABLE_DT
#define SSMU_ENABLE_DT 1
#endif
#ifndef SSMU_DELTA_FROM_DT
#define SSMU_DELTA_FROM_DT 1
#endif

// =============================================================
// Math knobs
// =============================================================
#ifndef SSMU_ACCURATE_MATH_CSIM
#define SSMU_ACCURATE_MATH_CSIM 0
#endif
#ifndef SSMU_SOFTPLUS_POS_TH
#define SSMU_SOFTPLUS_POS_TH (8.0f)
#endif
#ifndef SSMU_SOFTPLUS_NEG_TH
#define SSMU_SOFTPLUS_NEG_TH (-8.0f)
#endif
#ifndef SSMU_EXP_CLAMP
#define SSMU_EXP_CLAMP (3.0f)
#endif

// =============================================================
// Local constants
// =============================================================
static const int D_T     = SSMU_D_T;
static const int C2_T    = SSMU_C2_T;
static const int CCONV_T = SSMU_CCONV_T;
static const int CH_T    = SSMU_CH_T;
static const int CIN_T   = SSMU_CIN_T;

static const int STATE_SCALAR = SSMU_STATE;
static const int STATE_V      = SSMU_STATE_T;

static const int CONV_K  = SSMU_K;
static const int J_TILE  = 8;

// =============================================================
// FIFO / STREAM DEPTH POLICY
// =============================================================
#ifndef SSMU_DEPTH_DATA_SHORT
#define SSMU_DEPTH_DATA_SHORT  8
#endif
#ifndef SSMU_DEPTH_DATA_MID
#define SSMU_DEPTH_DATA_MID    16
#endif
#ifndef SSMU_DEPTH_DATA_LONG
#define SSMU_DEPTH_DATA_LONG   32
#endif

#ifndef SSMU_LATENCY_MODE
#define SSMU_LATENCY_MODE 1
#endif

#ifndef SSMU_DEPTH_TILE
  #if SSMU_LATENCY_MODE
    #define SSMU_DEPTH_TILE 2
  #else
    #define SSMU_DEPTH_TILE 4
  #endif
#endif

#ifndef SSMU_DEPTH_TOK
#define SSMU_DEPTH_TOK         2
#endif

#ifndef SSMU_DEPTH_TRACE
  #if SSMU_LATENCY_MODE
    #define SSMU_DEPTH_TRACE       64
  #else
    #define SSMU_DEPTH_TRACE       64
  #endif
#endif

#ifndef SSMU_DEPTH_BULK
#define SSMU_DEPTH_BULK SSMU_DEPTH_DATA_LONG
#endif

// =============================================================
// GEMM/PROJ SHAPE POLICY
// =============================================================
#ifndef SSMU_JJ_UNROLL
#define SSMU_JJ_UNROLL 8
#else
#undef  SSMU_JJ_UNROLL
#define SSMU_JJ_UNROLL 8
#endif

#ifndef SSMU_LANE_UNROLL
#define SSMU_LANE_UNROLL 4
#endif
#ifndef SSMU_JT_II
#define SSMU_JT_II 2
#endif

#ifndef SSMU_I_TILE
#define SSMU_I_TILE 2
#else
#undef  SSMU_I_TILE
#define SSMU_I_TILE 2
#endif

// =============================================================
// Input projection layout constants
// =============================================================
// C2_T + CCONV_T + CH_T = 640+672+10 = 1322 = CIN_T (B/C come from conv, not in-proj)
static const int SSMU_INPROJ_NEED_CIN_BC = (SSMU_C2_T + SSMU_CCONV_T + SSMU_CH_T);
static_assert(SSMU_CIN_T == SSMU_INPROJ_NEED_CIN_BC,
              "CIN_T must equal C2_T + CCONV_T + CH_T (B/C come from conv output)");

static_assert((SSMU_RANK % VEC_FACTOR) == 0, "RANK must be divisible by VEC_FACTOR");
static_assert((SSMU_RANK_T) % J_TILE == 0, "RANK_T must be divisible by J_TILE");

static const int RANK_T = SSMU_RANK_T;   // 128

// In-proj layout (low-rank v3):
//   full-rank out: [ Z: C2_T ] + [ DT/B/C (tail): CH_T + STATE_V + STATE_V ]
//   low-rank X:    [ X_mid: C2_T ] (separate low-rank path)
//   total: Z(C2_T) + CCONV_T(dummy) + CH_T = CIN_T
static const int INP_Z_T      = C2_T;       // 640  (Z gate)
static const int INP_X_T      = C2_T;       // 640  (Low-rank X_mid output)
static const int INP_DT_T     = CH_T;       // 10
static const int INP_B_T      = STATE_V;    // 16
static const int INP_C_T      = STATE_V;    // 16
static const int INP_TAIL_T   = INP_DT_T + INP_B_T + INP_C_T;  // 42
static const int INP_NONLR_T  = INP_Z_T + INP_TAIL_T;          // 682 (Z + DT+B+C)

// CIN_T = Z + CCONV + CH = 640 + 672 + 10 = 1322 (non-LR path covers XBC=CCONV_T)
static_assert((INP_Z_T + CCONV_T + CH_T) == CIN_T,
              "Input projection split must match CIN_T (Z + CCONV + CH = 1322)");

// Depth macros for m_axi
#define SSMU_DEPTH_IN1    (SSMU_D_T  * SSMU_RANK_T)
#define SSMU_DEPTH_IN2_X  (SSMU_RANK_T * SSMU_C2_T)
#define SSMU_DEPTH_IN_NONLR (SSMU_D_T * (SSMU_C2_T + SSMU_CH_T + 2*SSMU_STATE_T))
#define SSMU_DEPTH_OUTA   (SSMU_D_T  * SSMU_RANK_T)
#define SSMU_DEPTH_OUTB   (SSMU_RANK_T * SSMU_C2_T)
#define SSMU_DEPTH_DELTA  (SSMU_C2_T * SSMU_C2_T)

// =============================================================
// R1/R2/R3: Narrowed accumulator / activation types
// =============================================================
#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<24, 8> ACC_T;
#else
typedef float ACC_T;
#endif

typedef ap_fixed<16, 6>  ACT_T;
typedef ap_fixed<16, 8>  EXP_T;
typedef ap_fixed<24,12>  RMS_INV_T;

// =============================================================
// Quantization scales
// =============================================================
#ifndef SSMU_W_SCALE_IN
  #if SSMU_USE_INT8
    #define SSMU_W_SCALE_IN    (1.0f/128.0f)
  #else
    #define SSMU_W_SCALE_IN    (1.0f)
  #endif
#endif
#ifndef SSMU_W_SCALE_DELTA
  #if SSMU_USE_INT8
    #define SSMU_W_SCALE_DELTA (1.0f/128.0f)
  #else
    #define SSMU_W_SCALE_DELTA (1.0f)
  #endif
#endif
#ifndef SSMU_W_SCALE_OUT
  #if SSMU_USE_INT8
    #define SSMU_W_SCALE_OUT   (1.0f/128.0f)
  #else
    #define SSMU_W_SCALE_OUT   (1.0f)
  #endif
#endif

// =============================================================
// Scale / weight helpers
// =============================================================
__attribute__((noinline))
static ACC_T pick_scale_fx(float runtime_scale, float fallback_scale) {
#pragma HLS INLINE off
    float s = (runtime_scale != 0.0f) ? runtime_scale : fallback_scale;
    return (ACC_T)s;
}

static inline ACC_T wget_scaled(const W_VEC &w, unsigned lane, ACC_T scale_fx) {
#pragma HLS INLINE
#if SSMU_USE_INT8
    int wi = (int)vget(w, lane);
    return ((ACC_T)wi) * scale_fx;
#else
    (void)scale_fx;
    return (ACC_T)vget(w, lane);
#endif
}

template<typename T>
static inline T clamp_fx(T x, T lo, T hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// =============================================================
// Zero helpers
// =============================================================
static inline DTYPE_VEC dvec_zero() {
#pragma HLS INLINE
    DTYPE_VEC z;
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        vset(z, l, (DTYPE)0);
    }
    return z;
}

static inline W_VEC wvec_zero() {
#pragma HLS INLINE
    W_VEC z;
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
#if SSMU_USE_INT8
        vset(z, l, (ap_int<8>)0);
#else
        vset(z, l, (DTYPE)0);
#endif
    }
    return z;
}

// =============================================================
// Math functions — v3: use narrowed ACT_T/EXP_T
// =============================================================
static inline ACT_T sigmoid_fx(ACT_T x) {
#pragma HLS INLINE
#if !defined(__SYNTHESIS__) && SSMU_ACCURATE_MATH_CSIM
    float xf = (float)x;
    float sf = 1.0f / (1.0f + std::exp(-xf));
    return (ACT_T)sf;
#else
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fx<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
#endif
}

static inline DTYPE silu_fx(DTYPE a) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_fx(x);
    return (DTYPE)(x * s);
}

static inline DTYPE softplus_fx(ACC_T xin) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)xin;
#if !defined(__SYNTHESIS__) && SSMU_ACCURATE_MATH_CSIM
    float xf = (float)x;
    if (xf > (float)SSMU_SOFTPLUS_POS_TH) return (DTYPE)xf;
    if (xf < (float)SSMU_SOFTPLUS_NEG_TH) return (DTYPE)0;
    float y = std::log1p(std::exp(xf));
    return (DTYPE)y;
#else
    const ACT_T TH  = (ACT_T)SSMU_SOFTPLUS_POS_TH;
    const ACT_T NTH = (ACT_T)SSMU_SOFTPLUS_NEG_TH;
    if (x > TH)  return (DTYPE)x;
    if (x < NTH) return (DTYPE)0;
    const ACT_T half = (ACT_T)0.5;
    const ACT_T one  = (ACT_T)1.0;
    ACT_T y = half * x + one;
    return (DTYPE)y;
#endif
}

static inline EXP_T exp_fx(ACT_T t_in) {
#pragma HLS INLINE
    ACT_T t = clamp_fx<ACT_T>(t_in, (ACT_T)(-SSMU_EXP_CLAMP), (ACT_T)SSMU_EXP_CLAMP);
#if !defined(__SYNTHESIS__) && SSMU_ACCURATE_MATH_CSIM
    float tf = (float)t;
    float yf = std::exp(tf);
    return (EXP_T)yf;
#else
    EXP_T tt;
    EXP_T half_tt;
#pragma HLS BIND_OP variable=tt      op=mul impl=fabric
#pragma HLS BIND_OP variable=half_tt op=mul impl=fabric
    tt      = (EXP_T)t * (EXP_T)t;
    half_tt = (EXP_T)0.5 * tt;
    EXP_T y = (EXP_T)1.0 + (EXP_T)t + half_tt;
    if (y < (EXP_T)0) y = (EXP_T)0;
    return y;
#endif
}

// =============================================================
// Tile tuple struct (shared across GEMM modules)
// =============================================================
struct vec_tuple8 { W_VEC w[J_TILE]; };

static inline ACC_T tree_sum8(ACC_T p0, ACC_T p1, ACC_T p2, ACC_T p3,
                             ACC_T p4, ACC_T p5, ACC_T p6, ACC_T p7) {
#pragma HLS INLINE
    ACC_T s0 = p0 + p1;
    ACC_T s1 = p2 + p3;
    ACC_T s2 = p4 + p5;
    ACC_T s3 = p6 + p7;
    ACC_T s4 = s0 + s1;
    ACC_T s5 = s2 + s3;
    return s4 + s5;
}

// =============================================================
// Stream depth macros
// =============================================================
#ifndef SSMU_STREAM_DEPTH
#define SSMU_STREAM_DEPTH 16
#endif

#ifndef SSMU_TRACE_DEPTH
#define SSMU_TRACE_DEPTH 800
#endif

#ifndef SSMU_AXI_RO_TUNE
#define SSMU_AXI_RO_TUNE max_read_burst_length=32 num_read_outstanding=8
#endif

#endif // __SSMU_MACRO_HPP__
