// ============================ ssm.cpp (INPROJ/OUTPROJ: producer/consumer DATAFLOW + TILE STREAMERS) ============================
// ssm.cpp  (Version C - MAX CHANGE, SSMU.h only)
// [A+B INTEGRATED + FAST DT-DELTA + layers.h-math-aligned + STATE OUT + CONV STATE + RUNTIME SCALES]
//
// IMPORTANT (updated):
// ✅ B/C are ALWAYS produced by input-projection (W_inproj). W_B/W_C ports REMOVED.
// ✅ w_scale_bc REMOVED (no longer used).
//
// Fixes retained:
// (A) HLS pragma parser-safe
// (B) pragma placement safe
// (C) avoid shift-negative warnings
// (D) W_VEC defined for both INT8 and non-INT8 (from SSMU.h)
// (E) depth policy constants
// (F) DATAFLOW-safe split of inproj_packed
//
// FIX #1:
// ✅ Replace preprocessor CIN_T guard with static_assert using SSMU_*_T macros.
//
// FIX #2:
// ✅ Fix STATE-vs-STATE_T mismatch that causes "hls::stream read while empty" in CSIM.
//    The scan/update path consumes B/C/H0 per *vectorized state* count (SSMU_STATE_T),
//    NOT per scalar state (SSMU_STATE). All stage3/stage45 loops use STATE_V = SSMU_STATE_T.
//    H1_OUT_LEN / HUGE_LEN updated accordingly.
//
// FIX #3:
// ✅ pick_scale_fx is forced NOINLINE + INLINE off (prevents INLINE/DATAFLOW conflict)
// ✅ TOP: DEPTH_* use const int (not static const) inside function with DATAFLOW region.
// ✅ Split const AXI bundles: A_fixed/RMS_weight/D_diag use separate bundles (gmemA/gmemRMS/gmemD)
//
// FIX #4 (DT tokens / splitter safety):

//
// FIX #5 (latency / II improvement):
// ✅ in_proj_*: replace ACC_T acc[][] with hls::vector accv[] registers
// ✅ add DEPENDENCE inter false on accv
// ✅ pipeline jt with II=1 (to minimize latency, avoid acc hazard over-conservatism)
//
// FIX #6 (W_inproj/W_out producer/consumer DATAFLOW):
// ✅ in_proj_*: W_inproj AXI read -> tile producer (per-ii streams) + compute consumer
// ✅ out_proj : W_out   AXI read -> tile producer (per-ii streams) + compute consumer
// ✅ padding removes jj_lim/valid_j branches; compute side runs full unrolled loops
//
// NEW FIX #7:
// ✅ ALL read-only m_axi add read_only
// ✅ inproj/outproj jj accumulation switched to TREE reduction (shorter critical path, 0 extra latency)
// ✅ streamer: split AXI load -> wbuf[8] preload then pack tuple
// ✅ larger max_read_burst_length / num_read_outstanding knobs
// ✅ add caller-input predecessor copy_vec_n() for X_in / H0_in / conv_state_in
//
// =============================================================================================================

#include "SSMU.h"

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>
#include <hls_vector.h>

#ifndef __SYNTHESIS__
#include <cstdio>
#include <cmath>
#endif

#ifndef __SYNTHESIS__
  #define DUT_PRINTF(...) do { std::printf(__VA_ARGS__); std::fflush(stdout); } while(0)
#else
  #define DUT_PRINTF(...) do {} while(0)
#endif

// ============================================================
// Safety fallbacks (in case SSMU.h omits these)
// ============================================================
#ifndef SSMU_USE_INT8
#define SSMU_USE_INT8 0
#endif

#ifndef SSMU_K
#define SSMU_K 4
#endif

// ============================================================
// FIFO / STREAM DEPTH POLICY
// ============================================================
#ifndef SSMU_DEPTH_DATA_SHORT
#define SSMU_DEPTH_DATA_SHORT  8
#endif
#ifndef SSMU_DEPTH_DATA_MID
#define SSMU_DEPTH_DATA_MID    16
#endif
#ifndef SSMU_DEPTH_DATA_LONG
#define SSMU_DEPTH_DATA_LONG   32
#endif
#ifndef SSMU_DEPTH_TILE
#define SSMU_DEPTH_TILE        4
#endif
#ifndef SSMU_DEPTH_TOK
#define SSMU_DEPTH_TOK         2
#endif
#ifndef SSMU_DEPTH_TRACE
#define SSMU_DEPTH_TRACE       64
#endif
#ifndef SSMU_DEPTH_BULK
#define SSMU_DEPTH_BULK SSMU_DEPTH_DATA_LONG
#endif

// ============================================================
// GEMM/PROJ SHAPE POLICY
// ============================================================
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

// ============================================================
// (SAFE) Fallback vget/vset if your environment doesn't define them
// ============================================================
#ifndef SSMU_HAVE_VGET_VSET
template<typename V>
static inline auto vget(const V& v, unsigned idx) -> decltype(v[idx]) {
#pragma HLS INLINE
    return v[idx];
}
template<typename V, typename T>
static inline void vset(V& v, unsigned idx, const T& val) {
#pragma HLS INLINE
    v[idx] = (decltype(v[idx]))val;
}
#endif

// ============================================================
// Optional knobs
// ============================================================
#ifndef SSMU_ENABLE_LM_MODULES
#define SSMU_ENABLE_LM_MODULES 0
#endif

#if SSMU_ENABLE_LM_MODULES
#include "../src/common.h"
#include "../src/softplus.h"
#include "../src/uD.h"
#include "../src/yz.h"
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

// ============================================================
// Reference-style knobs
// ============================================================
#ifndef SSMU_ENABLE_TRACE_DDR
#define SSMU_ENABLE_TRACE_DDR 0
#endif

#ifndef SSMU_ENABLE_H1_STREAM_OUT
#define SSMU_ENABLE_H1_STREAM_OUT 1
#endif

// ============================================================
// Enforce: B/C always from inproj (interface removed)
// ============================================================
#ifndef SSMU_STATE_T
#define SSMU_STATE_T (SSMU_STATE / VEC_FACTOR)
#endif

#ifndef SSMU_H1_OUT_LEN
#define SSMU_H1_OUT_LEN (SSMU_STATE_T * SSMU_C2_T)
#endif

// ============================================================
// DT/delta policy
// ============================================================
#ifndef SSMU_ENABLE_DT
#define SSMU_ENABLE_DT 1
#endif
#ifndef SSMU_DELTA_FROM_DT
#define SSMU_DELTA_FROM_DT 1
#endif

// ============================================================
// layers.h-like math knobs
// ============================================================
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

// ============================================================
// Local constants from SSMU.h
// ============================================================
static const int D_T     = SSMU_D_T;
static const int C2_T    = SSMU_C2_T;
static const int CCONV_T = SSMU_CCONV_T;
static const int CH_T    = SSMU_CH_T;
static const int CIN_T   = SSMU_CIN_T;

static const int STATE_SCALAR = SSMU_STATE;
static const int STATE_V      = SSMU_STATE_T;

static const int CONV_K  = SSMU_K;
static const int J_TILE  = 8;

#ifndef HUGE_LEN
static const int HUGE_LEN = STATE_V * C2_T;
#endif

static const int SSMU_INPROJ_NEED_CIN_BC = (SSMU_C2_T + SSMU_CCONV_T + SSMU_CH_T + 2 * SSMU_STATE_T);
static_assert(SSMU_CIN_T >= SSMU_INPROJ_NEED_CIN_BC,
              "CIN_T too small for in-proj layout. Need C2_T + CCONV_T + CH_T + 2*STATE_T");

// ============================================================
// Accumulator / activation types
// ============================================================
#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<32, 10> ACC_T;
#else
typedef float ACC_T;
#endif

typedef ap_fixed<18, 6>  ACT_T;
typedef ap_fixed<20, 8>  EXP_T;

// ============================================================
// Quantization scales (DEFAULTS depend on INT8 path)
// ============================================================
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

// ============================================================
// ✅ Runtime scale helper MUST NOT inline into DATAFLOW
// ============================================================
__attribute__((noinline))
static ACC_T pick_scale_fx(float runtime_scale, float fallback_scale) {
#pragma HLS INLINE off
    float s = (runtime_scale != 0.0f) ? runtime_scale : fallback_scale;
    return (ACC_T)s;
}

// ============================================================
// Weight helpers (use W_VEC from SSMU.h)
// ============================================================
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

// ============================================================
// Zero helpers
// ============================================================
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

// ============================================================
// layers.h-aligned math (CSIM accurate, SYNTH approx)
// ============================================================
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
    ap_fixed<24, 8> tt = (ap_fixed<24,8>)t * (ap_fixed<24,8>)t;
    ap_fixed<24, 8> y  = (ap_fixed<24,8>)1.0 + (ap_fixed<24,8>)t + (ap_fixed<24,8>)0.5 * tt;
    if (y < 0) y = 0;
    return (EXP_T)y;
#endif
}

// ============================================================
// Minimal dynamic int8 quant per vector + scale stream
// ============================================================
typedef ap_int<8> Q8_LANE_T;
typedef hls::vector<Q8_LANE_T, VEC_FACTOR> Q8_VEC_T;
typedef ap_fixed<16, 4> SCALE_FX_T;

static inline Q8_LANE_T sat_q8(int x) {
#pragma HLS INLINE
    if (x > 127) return (Q8_LANE_T)127;
    if (x < -128) return (Q8_LANE_T)(-128);
    return (Q8_LANE_T)x;
}

static void dynamic_int8_quant_stream_local(
    hls::stream<DTYPE_VEC>&  in,
    hls::stream<Q8_VEC_T>&   q_out,
    hls::stream<SCALE_FX_T>& s1_out,
    int n_vec
) {
#pragma HLS INLINE off
    for (int i = 0; i < n_vec; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();

        ACC_T maxa = (ACC_T)0;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T a = (ACC_T)vget(v, l);
            if (a < (ACC_T)0) a = (ACC_T)(-a);
            if (a > maxa) maxa = a;
        }

        ACC_T scale;
        if (maxa > (ACC_T)0) {
            const ACC_T denom = (ACC_T)127;
            scale = (ACC_T)(maxa / denom);
        } else {
            scale = (ACC_T)1;
        }
        if (!(scale > (ACC_T)0)) scale = (ACC_T)1;

        Q8_VEC_T qv;
#pragma HLS ARRAY_PARTITION variable=qv complete dim=1
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T x  = (ACC_T)vget(v, l);
            ACC_T qf = (ACC_T)(x / scale);

            const ACC_T half_pos = (ACC_T)0.5;
            const ACC_T half_neg = (ACC_T)(-0.5);
            ACC_T r = (qf >= (ACC_T)0) ? half_pos : half_neg;
            ACC_T q_round = (ACC_T)(qf + r);

            int qi = (int)q_round;
            qv[l] = sat_q8(qi);
        }

        q_out.write(qv);
        s1_out.write((SCALE_FX_T)scale);
    }
}

static void int8_dequant_stream_local(
    hls::stream<Q8_VEC_T>&   q_in,
    hls::stream<SCALE_FX_T>& s1_in,
    hls::stream<DTYPE_VEC>&  out,
    int n_vec
) {
#pragma HLS INLINE off
    for (int i = 0; i < n_vec; ++i) {
#pragma HLS PIPELINE II=1
        Q8_VEC_T qv = q_in.read();
        SCALE_FX_T s = s1_in.read();

        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            int qi = (int)qv[l];
            ACC_T xf = (ACC_T)qi * (ACC_T)s;
            vset(o, l, (DTYPE)xf);
        }
        out.write(o);
    }
}

// ============================================================
// generic helpers (bounded loops)
// ============================================================
static void copy_kernel_k(hls::stream<DTYPE>& in, hls::stream<DTYPE>& out) {
#pragma HLS INLINE off
    for (int i = 0; i < CONV_K; ++i) {
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }
}

static void copy_vec_n(hls::stream<DTYPE_VEC>& in, hls::stream<DTYPE_VEC>& out, int count) {
#pragma HLS INLINE off
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }
}

static void drain_vec_n(hls::stream<DTYPE_VEC>& in, int count) {
#pragma HLS INLINE off
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        (void)in.read();
    }
}

static void tee_vecDT_stream2_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2
) {
#pragma HLS INLINE off
    for (int i = 0; i < D_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

static void dup_vecC2_stream3_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2,
    hls::stream<DTYPE_VEC>& out3
) {
#pragma HLS INLINE off
    for (int i = 0; i < C2_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
        out3.write(v);
    }
}

static void dup_vecC2_stream2_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2
) {
#pragma HLS INLINE off
    for (int i = 0; i < C2_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

static void add_residual_local_D(
    hls::stream<DTYPE_VEC>& y_in,
    hls::stream<DTYPE_VEC>& x_res_in,
    hls::stream<DTYPE_VEC>& y_out
) {
#pragma HLS INLINE off
    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC y = y_in.read();
        DTYPE_VEC x = x_res_in.read();
        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(o, l, (DTYPE)((ACC_T)vget(y, l) + (ACC_T)vget(x, l)));
        }
        y_out.write(o);
    }
}

// ============================================================
// Preload small constant tables from gmem into on-chip buffers
// ============================================================
template<int N>
static void preload_vec_table_local(
    const DTYPE_VEC in_gmem[N],
    DTYPE_VEC       out_local[N]
) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        out_local[i] = in_gmem[i];
    }
}

static void preload_vec_table_local_dyn(
    const DTYPE_VEC* in_gmem,
    DTYPE_VEC*       out_local,
    int              n
) {
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        out_local[i] = in_gmem[i];
    }
}

// ============================================================
// RMSNorm
// ============================================================
static void rmsnorm_vecDT_stream_local(
    hls::stream<DTYPE_VEC>& x_in,
    const DTYPE_VEC RMS_weight[D_T],
    hls::stream<DTYPE_VEC>& y_out
) {
#pragma HLS INLINE off
    const float eps = 1e-5f;

    DTYPE_VEC xbuf[D_T];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_s2p impl=lutram

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        xbuf[j] = x_in.read();
    }

    ACC_T lane_sumsq[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=lane_sumsq complete dim=1
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        lane_sumsq[l] = 0;
    }

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = xbuf[j];
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T vv = (ACC_T)vget(xv, l);
            lane_sumsq[l] += vv * vv;
        }
    }

    ACC_T sumsq = 0;
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        sumsq += lane_sumsq[l];
    }

    const float denom = (float)(D_T * VEC_FACTOR);
    float ms = (float)sumsq / denom;
    if (!(ms >= 0.0f)) ms = 0.0f;
    if (ms < 0.0f) ms = 0.0f;

    float inv = 1.0f / hls::sqrtf(ms + eps);

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC w = RMS_weight[j];
        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            float xv = (float)(ACC_T)vget(xbuf[j], l);
            float ww = (float)(ACC_T)vget(w, l);
            float yv = xv * inv * ww;
            vset(o, l, (DTYPE)yv);
        }
        y_out.write(o);
    }
}

// ============================================================
// Tile tuple (J_TILE=8)
// ============================================================
struct vec_tuple8 { W_VEC w[J_TILE]; };

// ============================================================
// ✅ TREE reduction helper (8-term sum)
// ============================================================
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

// ============================================================
// ✅ W_inproj tile streamer: preload 8 -> pack tuple
// ============================================================
static void stream_Winproj_tiles_local(
    const W_VEC W_inproj[D_T][CIN_T],
    hls::stream<vec_tuple8> Win_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_inproj cyclic factor=8 dim=1

    for (int it = 0; it < CIN_T; it += SSMU_I_TILE) {
        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int jidx = jt + jj;
                    if ((i < CIN_T) && (jidx < D_T)) wbuf[jj] = W_inproj[jidx][i];
                    else                              wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                Win_tiles[ii].write(tup);
            }
        }
    }
}

// ============================================================
// ✅ W_out tile streamer: preload 8 contiguous -> pack tuple
// ============================================================
static void stream_Wout_tiles_local(
    const W_VEC W_out[D_T][C2_T],
    hls::stream<vec_tuple8> Wo_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_out cyclic factor=8 dim=2

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {
        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int jidx = jt + jj;
                    if ((i < D_T) && (jidx < C2_T)) wbuf[jj] = W_out[i][jidx];
                    else                             wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                Wo_tiles[ii].write(tup);
            }
        }
    }
}

// ============================================================
// ✅ in_proj consumer (non-packed): TREE reduction in jj
// DT: always emit CH_T tokens (DT disabled -> write zeros)
// ============================================================
static void inproj_consume_tiles_local(
    const DTYPE_VEC X_buf[D_T],
    hls::stream<vec_tuple8> Win_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& XBC_out,
    hls::stream<DTYPE_VEC>& DT_out,
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    const int baseZ  = 0;
    const int baseX  = C2_T;
    const int baseDT = C2_T + CCONV_T;
    const int baseB  = C2_T + CCONV_T + CH_T;
    const int baseC  = C2_T + CCONV_T + CH_T + SSMU_STATE_T;

    for (int it = 0; it < CIN_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
#pragma HLS ARRAY_PARTITION variable=z complete dim=1
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = Win_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(X_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_in_fx);
                    ACC_T p1 = (ACC_T)vget(X_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_in_fx);
                    ACC_T p2 = (ACC_T)vget(X_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_in_fx);
                    ACC_T p3 = (ACC_T)vget(X_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_in_fx);
                    ACC_T p4 = (ACC_T)vget(X_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_in_fx);
                    ACC_T p5 = (ACC_T)vget(X_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_in_fx);
                    ACC_T p6 = (ACC_T)vget(X_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_in_fx);
                    ACC_T p7 = (ACC_T)vget(X_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_in_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < CIN_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }

                if (i >= baseZ && i < baseZ + C2_T) {
                    Z_out.write(outv);
                } else if (i >= baseX && i < baseX + CCONV_T) {
                    XBC_out.write(outv);
                } else if (i >= baseDT && i < baseDT + CH_T) {
#if SSMU_ENABLE_DT
                    DT_out.write(outv);
#else
                    (void)outv;
                    DT_out.write(dvec_zero());
#endif
                } else if (i >= baseB && i < baseB + SSMU_STATE_T) {
                    B_out.write(outv);
                } else if (i >= baseC && i < baseC + SSMU_STATE_T) {
                    C_out.write(outv);
                } else {
                    // ignore extras
                }
            }
        }
    }
}

// ============================================================
// ✅ in_proj consumer (packed): TREE reduction in jj
// ============================================================
static void inproj_consume_tiles_packed_local(
    const DTYPE_VEC X_buf[D_T],
    hls::stream<vec_tuple8> Win_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& packed_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    const int baseZ  = 0;
    const int baseX  = C2_T;
    const int baseDT = C2_T + CCONV_T;
    const int baseB  = C2_T + CCONV_T + CH_T;
    const int baseC  = C2_T + CCONV_T + CH_T + SSMU_STATE_T;

    for (int it = 0; it < CIN_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
#pragma HLS ARRAY_PARTITION variable=z complete dim=1
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = Win_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(X_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_in_fx);
                    ACC_T p1 = (ACC_T)vget(X_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_in_fx);
                    ACC_T p2 = (ACC_T)vget(X_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_in_fx);
                    ACC_T p3 = (ACC_T)vget(X_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_in_fx);
                    ACC_T p4 = (ACC_T)vget(X_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_in_fx);
                    ACC_T p5 = (ACC_T)vget(X_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_in_fx);
                    ACC_T p6 = (ACC_T)vget(X_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_in_fx);
                    ACC_T p7 = (ACC_T)vget(X_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_in_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < CIN_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }

                if (i >= baseZ && i < baseZ + C2_T) {
                    packed_out.write(outv);
                } else if (i >= baseX && i < baseX + CCONV_T) {
                    packed_out.write(outv);
                } else if (i >= baseDT && i < baseDT + CH_T) {
#if SSMU_ENABLE_DT
                    packed_out.write(outv);
#else
                    (void)outv;
                    packed_out.write(dvec_zero());
#endif
                } else if (i >= baseB && i < baseB + SSMU_STATE_T) {
                    packed_out.write(outv);
                } else if (i >= baseC && i < baseC + SSMU_STATE_T) {
                    packed_out.write(outv);
                } else {
                    // ignore extras
                }
            }
        }
    }
}

// ============================================================
// IN_PROJ pack (Z + XBC + DT + B/C ALWAYS)
// ============================================================
static void in_proj_pack_stream_local(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_inproj[D_T][CIN_T],
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& XBC_out,
    hls::stream<DTYPE_VEC>& DT_out,
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_d.read();
    }

    hls::stream<vec_tuple8> Win_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Win_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    stream_Winproj_tiles_local(W_inproj, Win_tiles);
    inproj_consume_tiles_local(X_buf, Win_tiles, Z_out, XBC_out, DT_out, B_out, C_out, wscale_in_fx);
}

// ============================================================
// Packed emitter
// ============================================================
static void in_proj_pack_stream_local_packed(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_inproj[D_T][CIN_T],
    hls::stream<DTYPE_VEC>& packed_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_d.read();
    }

    hls::stream<vec_tuple8> Win_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Win_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    stream_Winproj_tiles_local(W_inproj, Win_tiles);
    inproj_consume_tiles_packed_local(X_buf, Win_tiles, packed_out, wscale_in_fx);
}

// ============================================================
// FIX(F): DATAFLOW-safe splitter for inproj_packed
// ✅ always consume CH_T token; DT disabled -> drop
// ============================================================
static void split_inproj_packed_local(
    hls::stream<DTYPE_VEC>& packed_in,
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& XBC_out,
    hls::stream<DTYPE_VEC>& DT_out,
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out
) {
#pragma HLS INLINE off

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        Z_out.write(packed_in.read());
    }
    for (int j = 0; j < CCONV_T; ++j) {
#pragma HLS PIPELINE II=1
        XBC_out.write(packed_in.read());
    }

    for (int j = 0; j < CH_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = packed_in.read();
#if SSMU_ENABLE_DT
        DT_out.write(v);
#else
        (void)v;
#endif
    }

    for (int j = 0; j < SSMU_STATE_T; ++j) {
#pragma HLS PIPELINE II=1
        B_out.write(packed_in.read());
    }
    for (int j = 0; j < SSMU_STATE_T; ++j) {
#pragma HLS PIPELINE II=1
        C_out.write(packed_in.read());
    }
}

// ============================================================
// conv+silu (K=4 taps) WITH conv_state in/out
// ============================================================
static void conv1d_silu_stream_local_with_state(
    hls::stream<DTYPE_VEC>& XBC_in,
    hls::stream<DTYPE_VEC>& Z_in,
    hls::stream<DTYPE>&     kernel_in,
    hls::stream<DTYPE_VEC>& conv_state_in,
    hls::stream<DTYPE_VEC>& conv_state_out,
    hls::stream<DTYPE_VEC>& X_gate_out,
    hls::stream<DTYPE_VEC>& X_ssm_out
) {
#pragma HLS INLINE off

    DTYPE line_buffer[CONV_K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=lutram

    for (int k = 0; k < CONV_K-1; ++k) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC sv = conv_state_in.read();
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            line_buffer[k][l] = vget(sv, (unsigned)l);
        }
    }

    DTYPE kernel_buffer[CONV_K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
    for (int i = 0; i < CONV_K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }

    for (int i = 0; i < CCONV_T; ++i) {
#pragma HLS PIPELINE II=1
        const bool do_c2 = (i < C2_T);

        if (do_c2) {
            DTYPE_VEC z_in = Z_in.read();
            DTYPE_VEC gate_out;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                vset(gate_out, l, silu_fx(vget(z_in, l)));
            }
            X_gate_out.write(gate_out);
        }

        DTYPE_VEC x_in = XBC_in.read();

        DTYPE window0[VEC_FACTOR], window1[VEC_FACTOR], window2[VEC_FACTOR], window3[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window0 complete
#pragma HLS ARRAY_PARTITION variable=window1 complete
#pragma HLS ARRAY_PARTITION variable=window2 complete
#pragma HLS ARRAY_PARTITION variable=window3 complete

        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            window0[l] = line_buffer[2][l];
            window1[l] = line_buffer[1][l];
            window2[l] = line_buffer[0][l];
            window3[l] = vget(x_in, (unsigned)l);
        }

        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            line_buffer[2][l] = line_buffer[1][l];
            line_buffer[1][l] = line_buffer[0][l];
            line_buffer[0][l] = vget(x_in, (unsigned)l);
        }

        if (do_c2) {
            DTYPE_VEC ssm_out;
            for (unsigned lane = 0; lane < (unsigned)VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
                ACC_T sum = 0;
                sum += (ACC_T)kernel_buffer[0] * (ACC_T)window0[lane];
                sum += (ACC_T)kernel_buffer[1] * (ACC_T)window1[lane];
                sum += (ACC_T)kernel_buffer[2] * (ACC_T)window2[lane];
                sum += (ACC_T)kernel_buffer[3] * (ACC_T)window3[lane];
                vset(ssm_out, lane, silu_fx((DTYPE)sum));
            }
            X_ssm_out.write(ssm_out);
        }
    }

    for (int k = 0; k < CONV_K-1; ++k) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC sv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(sv, (unsigned)l, line_buffer[k][l]);
        }
        conv_state_out.write(sv);
    }
}

// ============================================================
// dtadapt helper (always consumes CH_T, outputs C2_T)
// ============================================================
static void dtadapt_stream_local(
    hls::stream<DTYPE_VEC>& dt_in,
    hls::stream<DTYPE_VEC>& dt_out
) {
#pragma HLS INLINE off
    DTYPE_VEC dt_buf[CH_T];
#pragma HLS BIND_STORAGE variable=dt_buf type=ram_s2p impl=lutram
    for (int j = 0; j < CH_T; ++j) {
#pragma HLS PIPELINE II=1
        dt_buf[j] = dt_in.read();
    }
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        int src = j % CH_T;
        dt_out.write(dt_buf[src]);
    }
}

// Not using DT for delta: drain CH_T and output zeros for C2_T (if needed elsewhere)
static void drain_dt_and_pad_zero_c2_local(
    hls::stream<DTYPE_VEC>& dt_in,
    hls::stream<DTYPE_VEC>& dt_c2_out
) {
#pragma HLS INLINE off
    for (int j = 0; j < CH_T; ++j) {
#pragma HLS PIPELINE II=1
        (void)dt_in.read();
    }
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        dt_c2_out.write(dvec_zero());
    }
}

// ============================================================
// DT -> delta (softplus)
// ============================================================
static void dt_to_delta_stream_local(
    hls::stream<DTYPE_VEC>& DT_in_C2,
    hls::stream<DTYPE_VEC>& delta_out
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC dtv = DT_in_C2.read();
        DTYPE_VEC dv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T x = (ACC_T)vget(dtv, l);
            vset(dv, l, softplus_fx(x));
        }
        delta_out.write(dv);
    }
}

// ============================================================
// Weight tile streamers (W_delta)
// ============================================================
static void stream_Wdelta_tiles_local(
    const W_VEC W_delta[C2_T][C2_T],
    hls::stream<vec_tuple8>& Wd_tiles
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_delta cyclic factor=8 dim=2
    for (int i = 0; i < C2_T; ++i) {
        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                tup.w[jj] = (jidx < C2_T) ? W_delta[i][jidx] : wvec_zero();
            }
            Wd_tiles.write(tup);
        }
    }
}

static void write_token1_local(hls::stream<ap_uint<1> >& tok) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    tok.write((ap_uint<1>)1);
}

static void stream_Wdelta_tiles_gated_local(
    const W_VEC W_delta[C2_T][C2_T],
    hls::stream<ap_uint<1> >& start_tok,
    hls::stream<vec_tuple8>& Wd_tiles
) {
#pragma HLS INLINE off
    (void)start_tok.read();
    stream_Wdelta_tiles_local(W_delta, Wd_tiles);
}

// ============================================================
// Delta-only projection: X_ssm -> delta = softplus(W_delta * X)
// ============================================================
static void projection_delta_only_local(
    hls::stream<DTYPE_VEC>&  X_ssm_in,
    hls::stream<vec_tuple8>& Wd_tiles,
    hls::stream<DTYPE_VEC>&  delta_out,
    ACC_T wscale_delta_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int i = 0; i < C2_T; ++i) {
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS DEPENDENCE variable=acc inter false
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = 0;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=SSMU_JT_II
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            vec_tuple8 wd = Wd_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wd.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC w = wd.w[jj];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                        ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                        ACC_T wv = wget_scaled(w, l, wscale_delta_fx);
                        acc[l] += xv * wv;
                    }
                }
            }
        }

        DTYPE_VEC delta_vec;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
            vset(delta_vec, l, softplus_fx(acc[l]));
        }
        delta_out.write(delta_vec);
    }
}

// ============================================================
// stage3, stage45(+H1 state out), (optional) ddr_writer, stage6
// ============================================================
static void stage3_dA_stream_local(
    hls::stream<DTYPE_VEC>& delta_in,
    const DTYPE_VEC A_fixed_local[STATE_V],
    hls::stream<DTYPE_VEC>& dA_out
) {
#pragma HLS INLINE off

    DTYPE_VEC delta_buf[C2_T];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    for (int i = 0; i < STATE_V; ++i) {
        DTYPE_VEC Avec = A_fixed_local[i];
        for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC dlt = delta_buf[j];
            DTYPE_VEC dA_vec;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACT_T a  = (ACT_T)vget(Avec, l);
                ACT_T dl = (ACT_T)vget(dlt,  l);
                EXP_T e  = exp_fx(a * dl);
                vset(dA_vec, l, (DTYPE)e);
            }
            dA_out.write(dA_vec);
        }
    }
}

static void stage45_update_reduce_local(
    hls::stream<DTYPE_VEC>& X_ssm_in,
    hls::stream<DTYPE_VEC>& delta_in,
    hls::stream<DTYPE_VEC>& dA_in,
    hls::stream<DTYPE_VEC>& B_in,
    hls::stream<DTYPE_VEC>& C_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& htC_out,
    hls::stream<DTYPE_VEC>& C_trace_out,
    hls::stream<DTYPE_VEC>& H1_trace_out,
    hls::stream<DTYPE_VEC>& H1_state_out
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC delta_buf[C2_T];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    DTYPE_VEC acc[C2_T];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        acc[j] = dvec_zero();
    }

    for (int i = 0; i < STATE_V; ++i) {
        DTYPE_VEC B_vec = B_in.read();
        DTYPE_VEC C_vec = C_in.read();

        for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC H0v  = H0_in.read();
            DTYPE_VEC dlt  = delta_buf[j];
            DTYPE_VEC xssm = X_buf[j];
            DTYPE_VEC dA   = dA_in.read();

            DTYPE_VEC H1v;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T H0  = (ACC_T)vget(H0v,   l);
                ACC_T ddA = (ACC_T)vget(dA,    l);
                ACC_T Bx  = (ACC_T)vget(B_vec, l);
                ACC_T dl  = (ACC_T)vget(dlt,   l);
                ACC_T Xs  = (ACC_T)vget(xssm,  l);
                ACC_T H1  = H0 * ddA + (Bx * dl) * Xs;
                vset(H1v, l, (DTYPE)H1);
            }

#if SSMU_ENABLE_TRACE_DDR
            C_trace_out.write(C_vec);
            H1_trace_out.write(H1v);
#endif
            H1_state_out.write(H1v);

            DTYPE_VEC prev = acc[j];
            DTYPE_VEC next;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T base = (ACC_T)vget(prev, l);
                ACC_T addt = (ACC_T)vget(H1v, l) * (ACC_T)vget(C_vec, l);
                vset(next, l, (DTYPE)(base + addt));
            }
            acc[j] = next;
        }
    }

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        htC_out.write(acc[j]);
    }
}

static void ddr_writer_local(
    hls::stream<DTYPE_VEC>& C_trace_in,
    hls::stream<DTYPE_VEC>& H1_trace_in,
    DTYPE_VEC* C_ddr,
    DTYPE_VEC* H1_ddr
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    for (int idx = 0; idx < HUGE_LEN; ++idx) {
#pragma HLS PIPELINE II=1
        (void)C_trace_in.read();
        (void)H1_trace_in.read();
    }
#else
    for (int idx = 0; idx < HUGE_LEN; ++idx) {
#pragma HLS PIPELINE II=1
        C_ddr[idx]  = C_trace_in.read();
        H1_ddr[idx] = H1_trace_in.read();
    }
#endif
}

static void stage6_out_combine_local(
    hls::stream<DTYPE_VEC>& htC_in,
    const DTYPE_VEC D_diag[C2_T],
    hls::stream<DTYPE_VEC>& X_ssm_in,
    hls::stream<DTYPE_VEC>& X_gate_in,
    hls::stream<DTYPE_VEC>& out
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC htC  = htC_in.read();
        DTYPE_VEC xssm = X_ssm_in.read();
        DTYPE_VEC zvec = X_gate_in.read();
        DTYPE_VEC dvec = D_diag[j];

        DTYPE_VEC outv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T htC_l = (ACC_T)vget(htC,  l);
            ACC_T x_l   = (ACC_T)vget(xssm, l);
            ACC_T d_l   = (ACC_T)vget(dvec, l);
            ACC_T z_l   = (ACC_T)vget(zvec, l);

            ACC_T y_l  = htC_l + d_l * x_l;
            ACC_T yz_l = y_l * z_l;
            vset(outv, l, (DTYPE)yz_l);
        }
        out.write(outv);
    }
}

static void stage6_out_combine_local_q(
    hls::stream<DTYPE_VEC>&  htC_in,
    const DTYPE_VEC          D_diag[C2_T],
    hls::stream<Q8_VEC_T>&   Xq_in,
    hls::stream<SCALE_FX_T>& Xs_in,
    hls::stream<Q8_VEC_T>&   Zq_in,
    hls::stream<SCALE_FX_T>& Zs_in,
    hls::stream<DTYPE_VEC>&  out
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC htC  = htC_in.read();
        DTYPE_VEC dvec = D_diag[j];

        Q8_VEC_T    Xq = Xq_in.read();
        SCALE_FX_T  Xs = Xs_in.read();

        Q8_VEC_T    Zq = Zq_in.read();
        SCALE_FX_T  Zs = Zs_in.read();

        DTYPE_VEC outv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T x = (ACC_T)((int)Xq[l]) * (ACC_T)Xs;
            ACC_T z = (ACC_T)((int)Zq[l]) * (ACC_T)Zs;

            ACC_T ht = (ACC_T)vget(htC, l);
            ACC_T d  = (ACC_T)vget(dvec, l);

            ACC_T y  = ht + d * x;
            ACC_T yz = y * z;
            vset(outv, l, (DTYPE)yz);
        }
        out.write(outv);
    }
}

// ============================================================
// ✅ out_proj consumer: TREE reduction in jj
// ============================================================
static void outproj_consume_tiles_local(
    const DTYPE_VEC X_buf[C2_T],
    hls::stream<vec_tuple8> Wo_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
#pragma HLS ARRAY_PARTITION variable=z complete dim=1
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = Wo_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(X_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_out_fx);
                    ACC_T p1 = (ACC_T)vget(X_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_out_fx);
                    ACC_T p2 = (ACC_T)vget(X_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_out_fx);
                    ACC_T p3 = (ACC_T)vget(X_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_out_fx);
                    ACC_T p4 = (ACC_T)vget(X_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_out_fx);
                    ACC_T p5 = (ACC_T)vget(X_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_out_fx);
                    ACC_T p6 = (ACC_T)vget(X_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_out_fx);
                    ACC_T p7 = (ACC_T)vget(X_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_out_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < D_T) {
                DTYPE_VEC y;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    vset(y, (unsigned)l, (DTYPE)accv[ii][l]);
                }
                Y_out.write(y);
            }
        }
    }
}

// ============================================================
// ✅ out_proj with producer/consumer DATAFLOW
// ============================================================
static void out_proj_stream_local_rect(
    hls::stream<DTYPE_VEC>& X_in,
    const W_VEC W_out[D_T][C2_T],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in.read();
    }

    hls::stream<vec_tuple8> Wo_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Wo_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    stream_Wout_tiles_local(W_out, Wo_tiles);
    outproj_consume_tiles_local(X_buf, Wo_tiles, Y_out, wscale_out_fx);
}

// ============================================================
// TOP  ---- SSMU KERNEL (W_B/W_C removed)
// ============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,

    const DTYPE_VEC A_fixed[STATE_V],
    const DTYPE_VEC RMS_weight[D_T],
    const W_VEC W_inproj[D_T][CIN_T],
    const W_VEC W_delta[C2_T][C2_T],
    const W_VEC W_out[D_T][C2_T],
    const DTYPE_VEC D_diag[C2_T],

    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,

    hls::stream<DTYPE_VEC>& conv_state_in,
    hls::stream<DTYPE_VEC>& conv_state_out,

    DTYPE_VEC* C_ddr,
    DTYPE_VEC* H1_ddr,

    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out,

    float w_scale_in,
    float w_scale_delta,
    float w_scale_out
) {
#pragma HLS INTERFACE ap_fifo port=kernel_in
#pragma HLS INTERFACE ap_fifo port=X_in
#pragma HLS INTERFACE ap_fifo port=H0_in
#pragma HLS INTERFACE ap_fifo port=conv_state_in
#pragma HLS INTERFACE ap_fifo port=conv_state_out
#pragma HLS INTERFACE ap_fifo port=H1_out
#pragma HLS INTERFACE ap_fifo port=out

    // ✅ read_only on all read-only m_axi
#pragma HLS INTERFACE m_axi port=A_fixed     offset=slave bundle=gmemA   depth=STATE_V  max_read_burst_length=128 num_read_outstanding=32 read_only
#pragma HLS INTERFACE m_axi port=RMS_weight  offset=slave bundle=gmemRMS depth=D_T      max_read_burst_length=128 num_read_outstanding=32 read_only
#pragma HLS INTERFACE m_axi port=D_diag      offset=slave bundle=gmemD   depth=C2_T     max_read_burst_length=128 num_read_outstanding=32 read_only

    const int DEPTH_INPROJ = D_T * CIN_T;
    const int DEPTH_DELTA  = C2_T * C2_T;
    const int DEPTH_OUT    = D_T * C2_T;

#pragma HLS INTERFACE m_axi port=W_inproj    offset=slave bundle=gmemInproj depth=DEPTH_INPROJ max_read_burst_length=128 num_read_outstanding=32 read_only
#pragma HLS INTERFACE m_axi port=W_delta     offset=slave bundle=gmemDelta  depth=DEPTH_DELTA  max_read_burst_length=128 num_read_outstanding=32 read_only
#pragma HLS INTERFACE m_axi port=W_out       offset=slave bundle=gmemOut    depth=DEPTH_OUT    max_read_burst_length=128 num_read_outstanding=32 read_only

#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control

#pragma HLS INTERFACE s_axilite port=w_scale_in    bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_delta bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_out   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const ACC_T wscale_in_fx    = pick_scale_fx(w_scale_in,    (float)SSMU_W_SCALE_IN);
    const ACC_T wscale_delta_fx = pick_scale_fx(w_scale_delta, (float)SSMU_W_SCALE_DELTA);
    const ACC_T wscale_out_fx   = pick_scale_fx(w_scale_out,   (float)SSMU_W_SCALE_OUT);

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] SSMU_USE_INT8=%d\n", (int)SSMU_USE_INT8);
    DUT_PRINTF("[DUT] TRACE_DDR=%d H1_OUT=%d H1_OUT_LEN=%d\n",
               (int)SSMU_ENABLE_TRACE_DDR, (int)SSMU_ENABLE_H1_STREAM_OUT, (int)SSMU_H1_OUT_LEN);
    DUT_PRINTF("[DUT] TREE reduction enabled for inproj/outproj jj (J_TILE=8)\n");
    DUT_PRINTF("[DUT] ALL read-only m_axi marked read_only; burst=128, outstanding=32\n");
    DUT_PRINTF("[DUT] STATE_SCALAR=%d STATE_V(STATE_T)=%d\n", (int)STATE_SCALAR, (int)STATE_V);
#endif

    // preload constants
    DTYPE_VEC A_local[STATE_V];
    DTYPE_VEC RMS_local[D_T];
    DTYPE_VEC D_local[C2_T];
#pragma HLS BIND_STORAGE variable=A_local   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=RMS_local type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=D_local   type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=A_local   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=RMS_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=D_local   cyclic factor=8 dim=1

    preload_vec_table_local_dyn(A_fixed,      A_local,   STATE_V);
    preload_vec_table_local<D_T>(RMS_weight,  RMS_local);
    preload_vec_table_local<C2_T>(D_diag,     D_local);

    // =========================
    // ✅ predecessor copies (break ap_fifo backpressure hazards)
    // =========================
    hls::stream<DTYPE_VEC> X_in_pre("X_in_pre");
    hls::stream<DTYPE_VEC> H0_in_pre("H0_in_pre");
    hls::stream<DTYPE_VEC> conv_state_pre("conv_state_pre");
#pragma HLS STREAM variable=X_in_pre        depth=2500
#pragma HLS STREAM variable=H0_in_pre       depth=2500
#pragma HLS STREAM variable=conv_state_pre  depth=2500

    // main streams
    hls::stream<DTYPE> kernel_local("kernel_local");
#pragma HLS STREAM variable=kernel_local depth=2500

    hls::stream<DTYPE_VEC> X_local("X_local");
    hls::stream<DTYPE_VEC> X_for_norm("X_for_norm");
    hls::stream<DTYPE_VEC> X_residual("X_residual");
    hls::stream<DTYPE_VEC> X_normed("X_normed");
#pragma HLS STREAM variable=X_local     depth=2500
#pragma HLS STREAM variable=X_for_norm  depth=2500
#pragma HLS STREAM variable=X_residual  depth=2500
#pragma HLS STREAM variable=X_normed    depth=2500

    hls::stream<DTYPE_VEC> inproj_packed("inproj_packed");
#pragma HLS STREAM variable=inproj_packed depth=2500

    hls::stream<DTYPE_VEC> Z_stream("Z_stream");
    hls::stream<DTYPE_VEC> XBC_stream("XBC_stream");
#pragma HLS STREAM variable=Z_stream   depth=2500
#pragma HLS STREAM variable=XBC_stream depth=2500

    hls::stream<DTYPE_VEC> DT_stream("DT_stream");
#pragma HLS STREAM variable=DT_stream depth=2500

    hls::stream<DTYPE_VEC> B_from_inproj("B_from_inproj");
    hls::stream<DTYPE_VEC> C_from_inproj("C_from_inproj");
#pragma HLS STREAM variable=B_from_inproj depth=2500
#pragma HLS STREAM variable=C_from_inproj depth=2500

    hls::stream<DTYPE_VEC> DT_C2_stream("DT_C2_stream");
#pragma HLS STREAM variable=DT_C2_stream depth=2500

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
#pragma HLS STREAM variable=X_gate_stream depth=2500
#pragma HLS STREAM variable=X_ssm_stream  depth=2500

    hls::stream<DTYPE_VEC> conv_state_local_in("conv_state_local_in");
    hls::stream<DTYPE_VEC> conv_state_local_out("conv_state_local_out");
#pragma HLS STREAM variable=conv_state_local_in  depth=2500
#pragma HLS STREAM variable=conv_state_local_out depth=2500

    hls::stream<vec_tuple8> Wd_tiles("Wd_tiles");
    hls::stream<ap_uint<1> > start_wd("start_wd");
#pragma HLS STREAM variable=Wd_tiles depth=2500
#pragma HLS STREAM variable=start_wd depth=2500

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_scan_stream("X_ssm_scan_stream");
    hls::stream<DTYPE_VEC> X_ssm_out_stream ("X_ssm_out_stream");
#pragma HLS STREAM variable=X_ssm_proj_stream depth=2500
#pragma HLS STREAM variable=X_ssm_scan_stream depth=2500
#pragma HLS STREAM variable=X_ssm_out_stream  depth=2500

    hls::stream<DTYPE_VEC> B_stream_S("B_stream_S");
    hls::stream<DTYPE_VEC> C_stream_S("C_stream_S");
#pragma HLS STREAM variable=B_stream_S depth=2500
#pragma HLS STREAM variable=C_stream_S depth=2500

    hls::stream<DTYPE_VEC> delta_selected("delta_selected");
    hls::stream<DTYPE_VEC> delta_for_dA("delta_for_dA");
    hls::stream<DTYPE_VEC> delta_for_scan("delta_for_scan");
#pragma HLS STREAM variable=delta_selected depth=2500
#pragma HLS STREAM variable=delta_for_dA   depth=2500
#pragma HLS STREAM variable=delta_for_scan depth=2500

    hls::stream<DTYPE_VEC> dA_stream("dA_stream");
#pragma HLS STREAM variable=dA_stream depth=2500

    hls::stream<DTYPE_VEC> htC_stream("htC_stream");
#pragma HLS STREAM variable=htC_stream depth=2500

    hls::stream<DTYPE_VEC> C_trace_stream("C_trace_stream");
    hls::stream<DTYPE_VEC> H1_trace_stream("H1_trace_stream");
#pragma HLS STREAM variable=C_trace_stream  depth=2500
#pragma HLS STREAM variable=H1_trace_stream depth=2500

    hls::stream<DTYPE_VEC> H1_state_stream("H1_state_stream");
#pragma HLS STREAM variable=H1_state_stream depth=2500

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
#pragma HLS STREAM variable=ssm_core_out_stream depth=2500

    hls::stream<DTYPE_VEC> out_proj_stream_s("out_proj_stream_s");
    hls::stream<DTYPE_VEC> out_local("out_local");
#pragma HLS STREAM variable=out_proj_stream_s depth=2500
#pragma HLS STREAM variable=out_local         depth=2500

    // quant chain streams
    hls::stream<Q8_VEC_T>   Xssm_q("Xssm_q");
    hls::stream<SCALE_FX_T> Xssm_s1("Xssm_s1");
    hls::stream<Q8_VEC_T>   Z_q("Z_q");
    hls::stream<SCALE_FX_T> Z_s1("Z_s1");
    hls::stream<Q8_VEC_T>   delta_q("delta_q");
    hls::stream<SCALE_FX_T> delta_s1("delta_s1");
#pragma HLS STREAM variable=Xssm_q   depth=2500
#pragma HLS STREAM variable=Xssm_s1  depth=2500
#pragma HLS STREAM variable=Z_q      depth=2500
#pragma HLS STREAM variable=Z_s1     depth=2500
#pragma HLS STREAM variable=delta_q  depth=2500
#pragma HLS STREAM variable=delta_s1 depth=2500

    hls::stream<Q8_VEC_T>   B_q("B_q");
    hls::stream<SCALE_FX_T> B_s1("B_s1");
    hls::stream<Q8_VEC_T>   C_q("C_q");
    hls::stream<SCALE_FX_T> C_s1("C_s1");
#pragma HLS STREAM variable=B_q  depth=2500
#pragma HLS STREAM variable=B_s1 depth=2500
#pragma HLS STREAM variable=C_q  depth=2500
#pragma HLS STREAM variable=C_s1 depth=2500

    hls::stream<DTYPE_VEC> Xssm_deq("Xssm_deq");
    hls::stream<DTYPE_VEC> delta_deq("delta_deq");
    hls::stream<DTYPE_VEC> B_deq("B_deq");
    hls::stream<DTYPE_VEC> C_deq("C_deq");
    hls::stream<DTYPE_VEC> Z_deq("Z_deq");
#pragma HLS STREAM variable=Xssm_deq  depth=2500
#pragma HLS STREAM variable=delta_deq depth=2500
#pragma HLS STREAM variable=B_deq     depth=2500
#pragma HLS STREAM variable=C_deq     depth=2500
#pragma HLS STREAM variable=Z_deq     depth=2500

    hls::stream<Q8_VEC_T>   Xout_q("Xout_q");
    hls::stream<SCALE_FX_T> Xout_s("Xout_s");
#pragma HLS STREAM variable=Xout_q depth=2500
#pragma HLS STREAM variable=Xout_s depth=2500

#pragma HLS DATAFLOW

    // predecessor copies
    copy_vec_n(X_in,          X_in_pre,       D_T);
    copy_vec_n(H0_in,         H0_in_pre,      SSMU_H1_OUT_LEN);   // STATE_V * C2_T
    copy_vec_n(conv_state_in, conv_state_pre, CONV_K-1);

    // main flow
    copy_kernel_k(kernel_in, kernel_local);
    copy_vec_n(X_in_pre, X_local, D_T);

    copy_vec_n(conv_state_pre, conv_state_local_in, CONV_K-1);

    tee_vecDT_stream2_local(X_local, X_for_norm, X_residual);
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_local, X_normed);

#if SSMU_ENABLE_GEMM_MUXDEMUX
    in_proj_pack_stream_local_packed(X_normed, W_inproj, inproj_packed, wscale_in_fx);
    split_inproj_packed_local(inproj_packed, Z_stream, XBC_stream, DT_stream, B_from_inproj, C_from_inproj);
#else
    in_proj_pack_stream_local(X_normed, W_inproj, Z_stream, XBC_stream, DT_stream, B_from_inproj, C_from_inproj, wscale_in_fx);
#endif

    copy_vec_n(B_from_inproj, B_stream_S, SSMU_STATE_T);
    copy_vec_n(C_from_inproj, C_stream_S, SSMU_STATE_T);

    conv1d_silu_stream_local_with_state(
        XBC_stream, Z_stream, kernel_local,
        conv_state_local_in, conv_state_local_out,
        X_gate_stream, X_ssm_stream
    );

    copy_vec_n(conv_state_local_out, conv_state_out, CONV_K-1);

    dup_vecC2_stream3_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_scan_stream, X_ssm_out_stream);

#if (SSMU_ENABLE_DT && SSMU_DELTA_FROM_DT)
    // DT_stream always has CH_T tokens (DT disabled path would have written zeros already, but still tokens exist)
    dtadapt_stream_local(DT_stream, DT_C2_stream);
    dt_to_delta_stream_local(DT_C2_stream, delta_selected);

    // X_ssm_proj_stream not used in this branch
    drain_vec_n(X_ssm_proj_stream, C2_T);
#else
    // Not using DT for delta. Still MUST drain DT_stream (CH_T tokens) because we ALWAYS emit them.
    drain_vec_n(DT_stream, CH_T);

    write_token1_local(start_wd);
    stream_Wdelta_tiles_gated_local(W_delta, start_wd, Wd_tiles);
    projection_delta_only_local(X_ssm_proj_stream, Wd_tiles, delta_selected, wscale_delta_fx);
#endif

    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan);

    stage3_dA_stream_local(delta_for_dA, A_local, dA_stream);

#if SSMU_ENABLE_ACT_Q_CHAIN
    dynamic_int8_quant_stream_local(X_ssm_scan_stream, Xssm_q, Xssm_s1, C2_T);
    dynamic_int8_quant_stream_local(X_gate_stream,     Z_q,    Z_s1,    C2_T);
    dynamic_int8_quant_stream_local(delta_for_scan,    delta_q,delta_s1,C2_T);

    dynamic_int8_quant_stream_local(B_stream_S, B_q, B_s1, SSMU_STATE_T);
    dynamic_int8_quant_stream_local(C_stream_S, C_q, C_s1, SSMU_STATE_T);

    int8_dequant_stream_local(Xssm_q,  Xssm_s1,  Xssm_deq,  C2_T);
    int8_dequant_stream_local(delta_q, delta_s1, delta_deq, C2_T);
    int8_dequant_stream_local(B_q,     B_s1,     B_deq,     SSMU_STATE_T);
    int8_dequant_stream_local(C_q,     C_s1,     C_deq,     SSMU_STATE_T);

    stage45_update_reduce_local(
        Xssm_deq,
        delta_deq,
        dA_stream,
        B_deq,
        C_deq,
        H0_in_pre,
        htC_stream,
        C_trace_stream,
        H1_trace_stream,
        H1_state_stream
    );
#else
    stage45_update_reduce_local(
        X_ssm_scan_stream,
        delta_for_scan,
        dA_stream,
        B_stream_S,
        C_stream_S,
        H0_in_pre,
        htC_stream,
        C_trace_stream,
        H1_trace_stream,
        H1_state_stream
    );
#endif

#if SSMU_ENABLE_TRACE_DDR
    ddr_writer_local(C_trace_stream, H1_trace_stream, C_ddr, H1_ddr);
#endif

#if SSMU_ENABLE_H1_STREAM_OUT
    copy_vec_n(H1_state_stream, H1_out, SSMU_H1_OUT_LEN);
#else
    drain_vec_n(H1_state_stream, SSMU_H1_OUT_LEN);
#endif

#if SSMU_ENABLE_ACT_Q_CHAIN && SSMU_ENABLE_UDYZ_Q
    dynamic_int8_quant_stream_local(X_ssm_out_stream, Xout_q, Xout_s, C2_T);

    stage6_out_combine_local_q(
        htC_stream,
        D_local,
        Xout_q, Xout_s,
        Z_q, Z_s1,
        ssm_core_out_stream
    );
#elif SSMU_ENABLE_ACT_Q_CHAIN
    int8_dequant_stream_local(Z_q, Z_s1, Z_deq, C2_T);

    stage6_out_combine_local(
        htC_stream,
        D_local,
        X_ssm_out_stream,
        Z_deq,
        ssm_core_out_stream
    );
#else
    stage6_out_combine_local(
        htC_stream,
        D_local,
        X_ssm_out_stream,
        X_gate_stream,
        ssm_core_out_stream
    );
#endif

    out_proj_stream_local_rect(ssm_core_out_stream, W_out, out_proj_stream_s, wscale_out_fx);
    add_residual_local_D(out_proj_stream_s, X_residual, out_local);

    copy_vec_n(out_local, out, D_T);
}
// ============================ end of ssm.cpp ============================
