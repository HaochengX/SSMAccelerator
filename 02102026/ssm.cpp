// ssm.cpp  (Version C - MAX CHANGE, SSMU.h only)
// [A+B INTEGRATED + FAST DT-DELTA + layers.h-math-aligned + STATE OUT + CONV STATE + RUNTIME SCALES]
// + (Optional) Stream MUX/SILU/QUANT/DEMUX modules integrated in this same file (NO testbench / NO main).
// + (Optional) LightMamba-style SOFTPLUS / uD / YZ reusable stream modules integrated in this same file (NO tb / NO main).
// + (NEW) Optional LightMamba/Spinal-alignment "activation quant/dequant + scale streams (s1/s2)"
// + (NEW) Optional dtadapt + dtB_quant-style fused quant-domain multiply path
// + (NEW) Optional UD + YZ quant-domain-ish staging (without changing SSMU() interface)
//
// IMPORTANT (match reference behavior):
// - TRACE (C_ddr / H1_ddr) is OPTIONAL and OFF by default so it does NOT extend critical-path latency.
// - H1_out (state out) is OPTIONAL; if disabled we internally drain to avoid backpressure deadlock.
// - SSMU() interface unchanged.
//
// FIXES APPLIED:
// (A) HLS pragma parser-safe: NO depth=(...), use named DEPTH constants
// (B) HLS pragma placement: NEVER put #pragma on same line inside { ... } one-liners
// (C) Avoid ap_fixed to_int() shift-negative warnings in some configs
// (D) FIX: ensure W_VEC is defined for BOTH INT8 and non-INT8 builds (previously missing in non-INT8)
// (E) FIX: avoid huge literal FIFO depths; use policy constants (SSMU_DEPTH_*)
// (F) FIX: DATAFLOW-safe split of inproj_packed (no raw for-loops in DATAFLOW body)  <<< NEW
//
// (PATCH APPLIED NOW):
// ✅ Change 1 (MUST): out_proj remove W_out dim=1 partition
// ✅ Change 2 (MUST): SSMU_JJ_UNROLL -> 8
// ✅ Change 3 (SUGGEST): SSMU_I_TILE -> 2
//
// NOTE: This file assumes SSMU.h provides:
//   - DTYPE, DTYPE_VEC, VEC_FACTOR, and shape macros SSMU_*_T.

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
// FIFO / STREAM DEPTH POLICY  (Scala-like, bounded)
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

// A single knob to scale “bulk” FIFOs (keep bounded, avoid 1000+ depth surprises)
#ifndef SSMU_DEPTH_BULK
#define SSMU_DEPTH_BULK SSMU_DEPTH_DATA_LONG
#endif

// ============================================================
// GEMM/PROJ SHAPE POLICY (avoid DSP blow-up / fix timing)
// ============================================================
#ifndef SSMU_JJ_UNROLL
#define SSMU_JJ_UNROLL 4   // legacy default
#endif
// ✅ Change 2 (MUST): force JJ_UNROLL = 8
#undef  SSMU_JJ_UNROLL
#define SSMU_JJ_UNROLL 8

#ifndef SSMU_LANE_UNROLL
#define SSMU_LANE_UNROLL 4
#endif
#ifndef SSMU_JT_II
#define SSMU_JT_II 2
#endif

// ============================================================
// Output-channel tiling (Spinal-like PE block)
// ============================================================
#ifndef SSMU_I_TILE
#define SSMU_I_TILE 4      // legacy default
#endif
// ✅ Change 3 (SUGGEST): force I_TILE = 2
#undef  SSMU_I_TILE
#define SSMU_I_TILE 2

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
// (OPTIONAL) LightMamba modules integration (NO tb / NO main)
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

// ============================================================
// (NEW) Reference-like quant/scale chain knobs (Spinal alignment)
// ============================================================
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
// (NEW) "B/C from in_proj" knob (LightMamba/Spinal style)
// ============================================================
#ifndef SSMU_ENABLE_BC_FROM_INPROJ
#define SSMU_ENABLE_BC_FROM_INPROJ 1
#endif

#ifndef SSMU_INPROJ_NEED_CIN_BC
#define SSMU_INPROJ_NEED_CIN_BC (SSMU_C2_T + SSMU_CCONV_T + SSMU_CH_T + 2*SSMU_STATE)
#endif

#if (SSMU_ENABLE_BC_FROM_INPROJ != 0) && (SSMU_CIN_T >= SSMU_INPROJ_NEED_CIN_BC)
#define SSMU_BC_FROM_INPROJ_OK 1
#else
#define SSMU_BC_FROM_INPROJ_OK 0
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

#ifndef SSMU_H1_OUT_LEN
#define SSMU_H1_OUT_LEN (SSMU_STATE * SSMU_C2_T)
#endif

// ============================================================
// B-mode feature knobs (compile-time)
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
#define SSMU_ACCURATE_MATH_CSIM 1
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
static const int STATE   = SSMU_STATE;

static const int CONV_K  = SSMU_K;
static const int J_TILE  = 8;

#ifndef HUGE_LEN
static const int HUGE_LEN = STATE * C2_T;
#endif

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

#ifndef SSMU_W_SCALE_BC
  #if SSMU_USE_INT8
    #define SSMU_W_SCALE_BC    (1.0f/128.0f)
  #else
    #define SSMU_W_SCALE_BC    (1.0f)
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
// Runtime scale helper
// ============================================================
static inline ACC_T pick_scale_fx(float runtime_scale, float fallback_scale) {
#pragma HLS INLINE
    float s = (runtime_scale != 0.0f) ? runtime_scale : fallback_scale;
    return (ACC_T)s;
}

// ============================================================
// Weight helpers (int8 path vs DTYPE path)
//   FIX(D): define W_VEC for BOTH paths.
// ============================================================
#if SSMU_USE_INT8
typedef ap_int<8> Q8_T;
typedef hls::vector<Q8_T, VEC_FACTOR> W_VEC;
static inline ACC_T wget_scaled(const W_VEC &w, unsigned idx, ACC_T scale_fx) {
#pragma HLS INLINE
    int wi = (int)vget(w, idx);
    return ((ACC_T)wi) * scale_fx;
}
#else
typedef hls::vector<DTYPE, VEC_FACTOR> W_VEC;
static inline ACC_T wget_scaled(const W_VEC &w, unsigned idx, ACC_T /*scale_fx*/) {
#pragma HLS INLINE
    return (ACC_T)vget(w, idx);
}
#endif

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
        vset(z, l, (Q8_T)0);
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
// Front-end modules (MUX / SILU / QUANT / DEMUX)
// ============================================================
static void silu_mux_stream_local(
    hls::stream<DTYPE_VEC>& xbc_in,
    hls::stream<DTYPE_VEC>& z_in,
    hls::stream<DTYPE_VEC>& packed_out,
    int n_vec
) {
#pragma HLS INLINE off
    for (int i = 0; i < n_vec; ++i) {
#pragma HLS PIPELINE II=1
        packed_out.write(xbc_in.read());
        packed_out.write(z_in.read());
    }
}

static void silu_vec_stream_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out,
    int n_vec
) {
#pragma HLS INLINE off
    for (int i = 0; i < n_vec; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(o, l, silu_fx(vget(v, l)));
        }
        out.write(o);
    }
}

// 3) Minimal dynamic int8 quant per vector + scale stream
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

            // FIX(C): avoid to_int() shift-negative warning in some ap_fixed configs
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

static void scale_forward_stream_local(
    hls::stream<SCALE_FX_T>& s_in,
    hls::stream<SCALE_FX_T>& s_out,
    int n
) {
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        s_out.write(s_in.read());
    }
}

// DEMUX helper (generic)
static void demux_packed_stream_local(
    hls::stream<DTYPE_VEC>& packed_in,
    hls::stream<DTYPE_VEC>& x_out,
    hls::stream<DTYPE_VEC>& x2_out,
    hls::stream<DTYPE_VEC>& b_out,
    hls::stream<DTYPE_VEC>& c_out,
    hls::stream<DTYPE_VEC>& z_out,
    int x_cnt, int x2_cnt, int b_cnt, int c_cnt, int z_cnt
) {
#pragma HLS INLINE off
    for (int i = 0; i < x_cnt;  ++i) {
#pragma HLS PIPELINE II=1
        x_out.write(packed_in.read());
    }
    for (int i = 0; i < x2_cnt; ++i) {
#pragma HLS PIPELINE II=1
        x2_out.write(packed_in.read());
    }
    for (int i = 0; i < b_cnt;  ++i) {
#pragma HLS PIPELINE II=1
        b_out.write(packed_in.read());
    }
    for (int i = 0; i < c_cnt;  ++i) {
#pragma HLS PIPELINE II=1
        c_out.write(packed_in.read());
    }
    for (int i = 0; i < z_cnt;  ++i) {
#pragma HLS PIPELINE II=1
        z_out.write(packed_in.read());
    }
}

// ============================================================
// dtadapt helper
// ============================================================
static void dtadapt_stream_local(
    hls::stream<DTYPE_VEC>& dt_in,
    hls::stream<DTYPE_VEC>& dt_out
) {
#pragma HLS INLINE off
#if (SSMU_ENABLE_DTADAPT)
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
#else
    // NOTE: caller decides how many items are in dt_in.
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        dt_out.write(dt_in.read());
    }
#endif
}

// ============================================================
// generic helpers
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

// single-token producer for DATAFLOW gating
static void write_token1_local(hls::stream<ap_uint<1> >& tok) {
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    tok.write((ap_uint<1>)1);
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

// RMSNorm
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
// Tile-stream weight helpers (for WB/WC/Wdelta paths)
// ============================================================
struct vec_tuple8 { W_VEC w[J_TILE]; };

// ============================================================
// IN_PROJ pack (Z + XBC + DT + optional B/C)
//   ✅ UPDATED: I_TILE version (like out_proj) while preserving output routing order
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

    // Buffer X on-chip
    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_d.read();
    }

    // Weight partition (only along input-j dimension)
#pragma HLS ARRAY_PARTITION variable=W_inproj cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    const int baseZ  = 0;
    const int baseX  = C2_T;
    const int baseDT = C2_T + CCONV_T;
    const int baseB  = C2_T + CCONV_T + CH_T;
    const int baseC  = C2_T + CCONV_T + CH_T + STATE;

    // Output-tiling over i to reduce overhead (like out_proj)
    for (int it = 0; it < CIN_T; it += SSMU_I_TILE) {

        ACC_T acc[SSMU_I_TILE][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

        // init
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                acc[ii][l] = 0;
            }
        }

        // accumulate over j (D_T)
        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=SSMU_JT_II

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < D_T) {
                    for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                        int i = it + ii;
                        if (i < CIN_T) {
                            W_VEC w = W_inproj[jidx][i];
                            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                                ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                                ACC_T wv = wget_scaled(w, l, wscale_in_fx);
                                acc[ii][l] = acc[ii][l] + xv * wv;
                            }
                        }
                    }
                }
            }
        }

        // emit in strictly increasing i order, preserving routing behavior
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < CIN_T) {
                DTYPE_VEC outv;
                for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    vset(outv, l, (DTYPE)acc[ii][l]);
                }

                if (i >= baseZ && i < baseZ + C2_T) {
                    Z_out.write(outv);
                } else if (i >= baseX && i < baseX + CCONV_T) {
                    XBC_out.write(outv);
                } else if (i >= baseDT && i < baseDT + CH_T) {
#if SSMU_ENABLE_DT
                    DT_out.write(outv);
#endif
                } else if (i >= baseB && i < baseB + STATE) {
                    B_out.write(outv);
                } else if (i >= baseC && i < baseC + STATE) {
                    C_out.write(outv);
                } else {
                    // ignore extras
                }
            }
        }
    }
}

// Packed emitter (for mux/demux style in dataflow)
static void in_proj_pack_stream_local_packed(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_inproj[D_T][CIN_T],
    hls::stream<DTYPE_VEC>& packed_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    // Buffer X on-chip
    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_d.read();
    }

    // Weight partition (only along input-j dimension)
#pragma HLS ARRAY_PARTITION variable=W_inproj cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    const int baseZ  = 0;
    const int baseX  = C2_T;
    const int baseDT = C2_T + CCONV_T;
    const int baseB  = C2_T + CCONV_T + CH_T;
    const int baseC  = C2_T + CCONV_T + CH_T + STATE;

    // Output-tiling over i, but preserve exact packed ordering and conditional writes
    for (int it = 0; it < CIN_T; it += SSMU_I_TILE) {

        ACC_T acc[SSMU_I_TILE][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                acc[ii][l] = 0;
            }
        }

        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=SSMU_JT_II

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < D_T) {
                    for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                        int i = it + ii;
                        if (i < CIN_T) {
                            W_VEC w = W_inproj[jidx][i];
                            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                                ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                                ACC_T wv = wget_scaled(w, l, wscale_in_fx);
                                acc[ii][l] = acc[ii][l] + xv * wv;
                            }
                        }
                    }
                }
            }
        }

        // emit packed stream with SAME condition logic as original
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < CIN_T) {
                DTYPE_VEC outv;
                for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    vset(outv, l, (DTYPE)acc[ii][l]);
                }

                if (i >= baseZ && i < baseZ + C2_T) {
                    packed_out.write(outv);
                } else if (i >= baseX && i < baseX + CCONV_T) {
                    packed_out.write(outv);
                } else if (i >= baseDT && i < baseDT + CH_T) {
#if SSMU_ENABLE_DT
                    packed_out.write(outv);
#endif
                } else if (i >= baseB && i < baseB + STATE) {
#if SSMU_BC_FROM_INPROJ_OK
                    packed_out.write(outv);
#endif
                } else if (i >= baseC && i < baseC + STATE) {
#if SSMU_BC_FROM_INPROJ_OK
                    packed_out.write(outv);
#endif
                } else {
                    // ignore extras
                }
            }
        }
    }
}

// ============================================================
// FIX(F): DATAFLOW-safe splitter for inproj_packed
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

#if SSMU_ENABLE_DT
    for (int j = 0; j < CH_T; ++j) {
#pragma HLS PIPELINE II=1
        DT_out.write(packed_in.read());
    }
#endif

#if SSMU_BC_FROM_INPROJ_OK
    for (int j = 0; j < STATE; ++j) {
#pragma HLS PIPELINE II=1
        B_out.write(packed_in.read());
    }
    for (int j = 0; j < STATE; ++j) {
#pragma HLS PIPELINE II=1
        C_out.write(packed_in.read());
    }
#endif
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
// Weight tile streamers (+ gated start)  [WB/WC/Wdelta paths]
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

static void stream_WBWC_tiles_local(
    const W_VEC W_B[STATE][C2_T],
    const W_VEC W_C[STATE][C2_T],
    hls::stream<vec_tuple8>& WB_tiles,
    hls::stream<vec_tuple8>& WC_tiles
) {
#pragma HLS INLINE off

    for (int i = 0; i < STATE; ++i) {
        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            vec_tuple8 tb, tc;
#pragma HLS ARRAY_PARTITION variable=tb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=tc.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    tb.w[jj] = W_B[i][jidx];
                    tc.w[jj] = W_C[i][jidx];
                } else {
                    tb.w[jj] = wvec_zero();
                    tc.w[jj] = wvec_zero();
                }
            }
            WB_tiles.write(tb);
            WC_tiles.write(tc);
        }
    }
}

static void stream_WBWC_tiles_gated_local(
    const W_VEC W_B[STATE][C2_T],
    const W_VEC W_C[STATE][C2_T],
    hls::stream<ap_uint<1> >& start_tok,
    hls::stream<vec_tuple8>& WB_tiles,
    hls::stream<vec_tuple8>& WC_tiles
) {
#pragma HLS INLINE off
    (void)start_tok.read();
    stream_WBWC_tiles_local(W_B, W_C, WB_tiles, WC_tiles);
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
// Full projection: X_ssm -> delta + B/C   (A-mode, legacy)
// ============================================================
static void projection_streams_local(
    hls::stream<DTYPE_VEC>&  X_ssm_in,
    hls::stream<vec_tuple8>& Wd_tiles,
    hls::stream<vec_tuple8>& WB_tiles,
    hls::stream<vec_tuple8>& WC_tiles,
    hls::stream<DTYPE_VEC>&  B_out_S,
    hls::stream<DTYPE_VEC>&  C_out_S,
    hls::stream<DTYPE_VEC>&  delta_out,
    ACC_T wscale_delta_fx,
    ACC_T wscale_bc_fx
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

    // ----- delta = softplus(W_delta * X) -----
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

    // ----- B/C = W_B/W_C * X -----
    for (int i = 0; i < STATE; ++i) {
        ACC_T accB[VEC_FACTOR], accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
#pragma HLS DEPENDENCE variable=accB inter false
#pragma HLS DEPENDENCE variable=accC inter false
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accB[l] = 0;
            accC[l] = 0;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=SSMU_JT_II
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            vec_tuple8 wb = WB_tiles.read();
            vec_tuple8 wc = WC_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC wB = wb.w[jj];
                    W_VEC wC = wc.w[jj];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                        ACC_T x  = (ACC_T)vget(X_tile[jj], l);
                        ACC_T b  = wget_scaled(wB, l, wscale_bc_fx);
                        ACC_T c  = wget_scaled(wC, l, wscale_bc_fx);
                        accB[l] += x * b;
                        accC[l] += x * c;
                    }
                }
            }
        }

        DTYPE_VEC outB, outC;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
            vset(outB, l, (DTYPE)accB[l]);
            vset(outC, l, (DTYPE)accC[l]);
        }
        B_out_S.write(outB);
        C_out_S.write(outC);
    }
}

static void projection_BC_only_local(
    hls::stream<DTYPE_VEC>&  X_ssm_in,
    hls::stream<vec_tuple8>& WB_tiles,
    hls::stream<vec_tuple8>& WC_tiles,
    hls::stream<DTYPE_VEC>&  B_out_S,
    hls::stream<DTYPE_VEC>&  C_out_S,
    ACC_T wscale_bc_fx
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

    for (int i = 0; i < STATE; ++i) {
        ACC_T accB[VEC_FACTOR], accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
#pragma HLS DEPENDENCE variable=accB inter false
#pragma HLS DEPENDENCE variable=accC inter false
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accB[l] = 0;
            accC[l] = 0;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=SSMU_JT_II
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            vec_tuple8 wb = WB_tiles.read();
            vec_tuple8 wc = WC_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC wB = wb.w[jj];
                    W_VEC wC = wc.w[jj];

                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        const ACC_T x  = (ACC_T)vget(X_tile[jj], l);
                        const ACC_T b  = wget_scaled(wB, l, wscale_bc_fx);
                        const ACC_T c  = wget_scaled(wC, l, wscale_bc_fx);

                        accB[l] = accB[l] + x * b;
                        accC[l] = accC[l] + x * c;
                    }
                }
            }
        }

        DTYPE_VEC outB, outC;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outB, l, (DTYPE)accB[l]);
            vset(outC, l, (DTYPE)accC[l]);
        }
        B_out_S.write(outB);
        C_out_S.write(outC);
    }
}

// ============================================================
// stage3, stage45(+H1 state out), (optional) ddr_writer, stage6, out_proj
// ============================================================
static void stage3_dA_stream_local(
    hls::stream<DTYPE_VEC>& delta_in,
    const DTYPE_VEC A_fixed[STATE],
    hls::stream<DTYPE_VEC>& dA_out
) {
#pragma HLS INLINE off

    DTYPE_VEC delta_buf[C2_T];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    for (int i = 0; i < STATE; ++i) {
        DTYPE_VEC Avec = A_fixed[i];
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

    for (int i = 0; i < STATE; ++i) {
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

// -------- Scan update: quant-ish dtB_quant path (optional) --------
static void stage45_update_reduce_local_q(
    hls::stream<Q8_VEC_T>&   Xq_in,
    hls::stream<SCALE_FX_T>& Xs_in,
    hls::stream<Q8_VEC_T>&   dq_in,
    hls::stream<SCALE_FX_T>& ds_in,
    hls::stream<DTYPE_VEC>&  dA_in,
    hls::stream<Q8_VEC_T>&   Bq_in,
    hls::stream<SCALE_FX_T>& Bs_in,
    hls::stream<Q8_VEC_T>&   Cq_in,
    hls::stream<SCALE_FX_T>& Cs_in,
    hls::stream<DTYPE_VEC>&  H0_in,
    hls::stream<DTYPE_VEC>&  htC_out,
    hls::stream<DTYPE_VEC>&  C_trace_out,
    hls::stream<DTYPE_VEC>&  H1_trace_out,
    hls::stream<DTYPE_VEC>&  H1_state_out
) {
#pragma HLS INLINE off

    Q8_VEC_T   Xq_buf[C2_T];
    SCALE_FX_T Xs_buf[C2_T];
#pragma HLS BIND_STORAGE variable=Xq_buf type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=Xs_buf type=ram_s2p impl=lutram

    Q8_VEC_T   dq_buf[C2_T];
    SCALE_FX_T ds_buf[C2_T];
#pragma HLS BIND_STORAGE variable=dq_buf type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=ds_buf type=ram_s2p impl=lutram

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        Xq_buf[j] = Xq_in.read();
        Xs_buf[j] = Xs_in.read();
    }
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        dq_buf[j] = dq_in.read();
        ds_buf[j] = ds_in.read();
    }

    DTYPE_VEC acc[C2_T];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        acc[j] = dvec_zero();
    }

    for (int i = 0; i < STATE; ++i) {
        Q8_VEC_T   Bq = Bq_in.read();
        SCALE_FX_T Bs = Bs_in.read();
        Q8_VEC_T   Cq = Cq_in.read();
        SCALE_FX_T Cs = Cs_in.read();

        DTYPE_VEC C_vec_f;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            int ci = (int)Cq[l];
            ACC_T cf = (ACC_T)ci * (ACC_T)Cs;
            vset(C_vec_f, l, (DTYPE)cf);
        }

        for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC H0v = H0_in.read();
            DTYPE_VEC dA  = dA_in.read();

            Q8_VEC_T   Xq = Xq_buf[j];
            SCALE_FX_T Xs = Xs_buf[j];
            Q8_VEC_T   dq = dq_buf[j];
            SCALE_FX_T ds = ds_buf[j];

            DTYPE_VEC H1v;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T H0  = (ACC_T)vget(H0v, l);
                ACC_T ddA = (ACC_T)vget(dA,  l);

                int bi = (int)Bq[l];
                int di = (int)dq[l];
                int xi = (int)Xq[l];

                ap_int<32> prod = (ap_int<32>)bi * (ap_int<32>)di;
                prod = prod * (ap_int<32>)xi;

                ACC_T scale = (ACC_T)Bs * (ACC_T)ds * (ACC_T)Xs;
                ACC_T term  = (ACC_T)prod * scale;

                ACC_T H1 = H0 * ddA + term;
                vset(H1v, l, (DTYPE)H1);
            }

#if SSMU_ENABLE_TRACE_DDR
            C_trace_out.write(C_vec_f);
            H1_trace_out.write(H1v);
#endif
            H1_state_out.write(H1v);

            DTYPE_VEC prev = acc[j];
            DTYPE_VEC next;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T base = (ACC_T)vget(prev, l);
                ACC_T addt = (ACC_T)vget(H1v, l) * (ACC_T)vget(C_vec_f, l);
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
        DTYPE_VEC htC = htC_in.read();
        DTYPE_VEC dvec = D_diag[j];

        Q8_VEC_T Xq = Xq_in.read();
        SCALE_FX_T Xs = Xs_in.read();

        Q8_VEC_T Zq = Zq_in.read();
        SCALE_FX_T Zs = Zs_in.read();

        DTYPE_VEC outv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T x = (ACC_T)((int)Xq[l]) * (ACC_T)Xs;
            ACC_T z = (ACC_T)((int)Zq[l]) * (ACC_T)Zs;

            ACC_T ht = (ACC_T)vget(htC, l);
            ACC_T d  = (ACC_T)vget(dvec, l);

            ACC_T y = ht + d * x;
            ACC_T yz = y * z;
            vset(outv, l, (DTYPE)yz);
        }
        out.write(outv);
    }
}

// ============================================================
// (PATCH #4): out_proj with output-channel tiling
// ============================================================
static void out_proj_stream_local_rect(
    hls::stream<DTYPE_VEC>& X_in,
    const W_VEC W_out[D_T][C2_T],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off

    // Buffer X on-chip
    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in.read();
    }

    // ✅ Change 1 (MUST): remove W_out dim=1 partition
    // (keep dim=2 partition to help j-tile bandwidth; do NOT partition dim=1)
#pragma HLS ARRAY_PARTITION variable=W_out cyclic factor=8 dim=2

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {
        ACC_T acc[SSMU_I_TILE][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

        // init
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                acc[ii][l] = 0;
            }
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                        int i = it + ii;
                        if (i < D_T) {
                            W_VEC w = W_out[i][jidx];
                            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                                ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                                ACC_T wv = wget_scaled(w, l, wscale_out_fx);
                                acc[ii][l] = acc[ii][l] + xv * wv;
                            }
                        }
                    }
                }
            }
        }

        // emit
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < D_T) {
                DTYPE_VEC y;
                for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    vset(y, l, (DTYPE)acc[ii][l]);
                }
                Y_out.write(y);
            }
        }
    }
}

// ============================================================
// TOP  ---- YOUR SSMU KERNEL (interface unchanged)
// ============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,

    const DTYPE_VEC A_fixed[STATE],
    const DTYPE_VEC RMS_weight[D_T],
    const W_VEC W_inproj[D_T][CIN_T],
    const W_VEC W_B[STATE][C2_T],
    const W_VEC W_C[STATE][C2_T],
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
    float w_scale_bc,
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

// -------- Split bundles to avoid gmemW arbitration stalls --------
#pragma HLS INTERFACE m_axi port=A_fixed     offset=slave bundle=gmemConst depth=STATE          max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=RMS_weight  offset=slave bundle=gmemConst depth=D_T            max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=D_diag      offset=slave bundle=gmemConst depth=C2_T           max_read_burst_length=64 num_read_outstanding=16

    static const int DEPTH_INPROJ = D_T * CIN_T;
    static const int DEPTH_BC     = STATE * C2_T;
    static const int DEPTH_DELTA  = C2_T * C2_T;
    static const int DEPTH_OUT    = D_T * C2_T;

#pragma HLS INTERFACE m_axi port=W_inproj    offset=slave bundle=gmemInproj depth=DEPTH_INPROJ  max_read_burst_length=64 num_read_outstanding=16

#pragma HLS INTERFACE m_axi port=W_B         offset=slave bundle=gmemB      depth=DEPTH_BC      max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=W_C         offset=slave bundle=gmemC      depth=DEPTH_BC      max_read_burst_length=64 num_read_outstanding=16

#pragma HLS INTERFACE m_axi port=W_delta     offset=slave bundle=gmemDelta  depth=DEPTH_DELTA   max_read_burst_length=64 num_read_outstanding=16
#pragma HLS INTERFACE m_axi port=W_out       offset=slave bundle=gmemOut    depth=DEPTH_OUT     max_read_burst_length=64 num_read_outstanding=16

#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control

#pragma HLS INTERFACE s_axilite port=w_scale_in    bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_bc    bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_delta bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_out   bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    const ACC_T wscale_in_fx    = pick_scale_fx(w_scale_in,    (float)SSMU_W_SCALE_IN);
    const ACC_T wscale_bc_fx    = pick_scale_fx(w_scale_bc,    (float)SSMU_W_SCALE_BC);
    const ACC_T wscale_delta_fx = pick_scale_fx(w_scale_delta, (float)SSMU_W_SCALE_DELTA);
    const ACC_T wscale_out_fx   = pick_scale_fx(w_scale_out,   (float)SSMU_W_SCALE_OUT);

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] SSMU_USE_INT8=%d\n", (int)SSMU_USE_INT8);
    DUT_PRINTF("[DUT] TRACE_DDR=%d H1_OUT=%d H1_OUT_LEN=%d\n",
               (int)SSMU_ENABLE_TRACE_DDR, (int)SSMU_ENABLE_H1_STREAM_OUT, (int)SSMU_H1_OUT_LEN);
    DUT_PRINTF("[DUT] ACT_Q_CHAIN=%d DTADAPT=%d DTB_QUANT=%d UDYZ_Q=%d GEMM_MUXDEMUX=%d\n",
               (int)SSMU_ENABLE_ACT_Q_CHAIN, (int)SSMU_ENABLE_DTADAPT, (int)SSMU_ENABLE_DTB_QUANT,
               (int)SSMU_ENABLE_UDYZ_Q, (int)SSMU_ENABLE_GEMM_MUXDEMUX);
    DUT_PRINTF("[DUT] BC_FROM_INPROJ_OK=%d (CIN_T=%d need>=%d)\n",
               (int)SSMU_BC_FROM_INPROJ_OK, (int)CIN_T, (int)SSMU_INPROJ_NEED_CIN_BC);
    DUT_PRINTF("[DUT] GEMM policy: JJ_UNROLL=%d LANE_UNROLL=%d JT_II=%d I_TILE=%d\n",
               (int)SSMU_JJ_UNROLL, (int)SSMU_LANE_UNROLL, (int)SSMU_JT_II, (int)SSMU_I_TILE);
#endif

    // ========================================================
    // On-chip cached tables (cut repeated DRAM reads)
    // ========================================================
    DTYPE_VEC A_local[STATE];
    DTYPE_VEC RMS_local[D_T];
    DTYPE_VEC D_local[C2_T];
#pragma HLS BIND_STORAGE variable=A_local   type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=RMS_local type=ram_2p impl=bram
#pragma HLS BIND_STORAGE variable=D_local   type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=A_local   cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=RMS_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=D_local   cyclic factor=8 dim=1

    preload_vec_table_local<STATE>(A_fixed,   A_local);
    preload_vec_table_local<D_T>(RMS_weight,  RMS_local);
    preload_vec_table_local<C2_T>(D_diag,     D_local);

    // ========================================================
    // Streams (ALL declared BEFORE DATAFLOW)
    // ========================================================
    hls::stream<DTYPE> kernel_local("kernel_local");
#pragma HLS STREAM variable=kernel_local depth=2250
    hls::stream<DTYPE_VEC> X_local("X_local");
#pragma HLS STREAM variable=X_local depth=2250

    hls::stream<DTYPE_VEC> X_for_norm("X_for_norm");
    hls::stream<DTYPE_VEC> X_residual("X_residual");
#pragma HLS STREAM variable=X_for_norm depth=2250
#pragma HLS STREAM variable=X_residual depth=2250

    hls::stream<DTYPE_VEC> X_normed("X_normed");
#pragma HLS STREAM variable=X_normed depth=2250

    hls::stream<DTYPE_VEC> inproj_packed("inproj_packed");
#pragma HLS STREAM variable=inproj_packed depth=2250

    hls::stream<DTYPE_VEC> Z_stream("Z_stream");
    hls::stream<DTYPE_VEC> XBC_stream("XBC_stream");
#pragma HLS STREAM variable=Z_stream depth=2250
#pragma HLS STREAM variable=XBC_stream depth=2250

    hls::stream<DTYPE_VEC> DT_stream("DT_stream");
#pragma HLS STREAM variable=DT_stream depth=2250

    hls::stream<DTYPE_VEC> B_from_inproj("B_from_inproj");
    hls::stream<DTYPE_VEC> C_from_inproj("C_from_inproj");
#pragma HLS STREAM variable=B_from_inproj depth=2250
#pragma HLS STREAM variable=C_from_inproj depth=2250

    hls::stream<DTYPE_VEC> DT_C2_stream("DT_C2_stream");
#pragma HLS STREAM variable=DT_C2_stream depth=2250

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
#pragma HLS STREAM variable=X_gate_stream depth=2250
#pragma HLS STREAM variable=X_ssm_stream  depth=2250

    hls::stream<DTYPE_VEC> conv_state_local_in("conv_state_local_in");
    hls::stream<DTYPE_VEC> conv_state_local_out("conv_state_local_out");
#pragma HLS STREAM variable=conv_state_local_in  depth=2250
#pragma HLS STREAM variable=conv_state_local_out depth=2250

    hls::stream<vec_tuple8> WB_tiles("WB_tiles");
    hls::stream<vec_tuple8> WC_tiles("WC_tiles");
#pragma HLS STREAM variable=WB_tiles depth=2250
#pragma HLS STREAM variable=WC_tiles depth=2250

    hls::stream<vec_tuple8> Wd_tiles("Wd_tiles");
#pragma HLS STREAM variable=Wd_tiles depth=2250

    hls::stream<ap_uint<1> > start_bc("start_bc");
    hls::stream<ap_uint<1> > start_wd("start_wd");
#pragma HLS STREAM variable=start_bc depth=2250
#pragma HLS STREAM variable=start_wd depth=2250

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_scan_stream("X_ssm_scan_stream");
    hls::stream<DTYPE_VEC> X_ssm_out_stream ("X_ssm_out_stream");
#pragma HLS STREAM variable=X_ssm_proj_stream depth=2250
#pragma HLS STREAM variable=X_ssm_scan_stream depth=2250
#pragma HLS STREAM variable=X_ssm_out_stream  depth=2250

    hls::stream<DTYPE_VEC> B_stream_S("B_stream_S");
    hls::stream<DTYPE_VEC> C_stream_S("C_stream_S");
#pragma HLS STREAM variable=B_stream_S depth=2250
#pragma HLS STREAM variable=C_stream_S depth=2250

    hls::stream<DTYPE_VEC> delta_selected("delta_selected");
#pragma HLS STREAM variable=delta_selected depth=2250

    hls::stream<DTYPE_VEC> delta_for_dA("delta_for_dA");
    hls::stream<DTYPE_VEC> delta_for_scan("delta_for_scan");
#pragma HLS STREAM variable=delta_for_dA   depth=2250
#pragma HLS STREAM variable=delta_for_scan depth=2250

    hls::stream<DTYPE_VEC> dA_stream("dA_stream");
#pragma HLS STREAM variable=dA_stream depth=2250

    hls::stream<DTYPE_VEC> htC_stream("htC_stream");
#pragma HLS STREAM variable=htC_stream depth=2250

    hls::stream<DTYPE_VEC> C_trace_stream("C_trace_stream");
    hls::stream<DTYPE_VEC> H1_trace_stream("H1_trace_stream");
#pragma HLS STREAM variable=C_trace_stream  depth=2250
#pragma HLS STREAM variable=H1_trace_stream depth=2250

    hls::stream<DTYPE_VEC> H1_state_stream("H1_state_stream");
#pragma HLS STREAM variable=H1_state_stream depth=2250

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
#pragma HLS STREAM variable=ssm_core_out_stream depth=2250

    hls::stream<DTYPE_VEC> out_proj_stream_s("out_proj_stream_s");
    hls::stream<DTYPE_VEC> out_local("out_local");
#pragma HLS STREAM variable=out_proj_stream_s depth=2250
#pragma HLS STREAM variable=out_local depth=2250

    hls::stream<DTYPE_VEC> delta_from_proj("delta_from_proj");
#pragma HLS STREAM variable=delta_from_proj depth=2250

    hls::stream<DTYPE_VEC> B_throw("B_throw");
    hls::stream<DTYPE_VEC> C_throw("C_throw");
#pragma HLS STREAM variable=B_throw depth=2250
#pragma HLS STREAM variable=C_throw depth=2250

    hls::stream<DTYPE_VEC> Xssm_deq("Xssm_deq");
    hls::stream<DTYPE_VEC> delta_deq("delta_deq");
    hls::stream<DTYPE_VEC> B_deq("B_deq");
    hls::stream<DTYPE_VEC> C_deq("C_deq");
#pragma HLS STREAM variable=Xssm_deq depth=2250
#pragma HLS STREAM variable=delta_deq depth=2250
#pragma HLS STREAM variable=B_deq depth=2250
#pragma HLS STREAM variable=C_deq depth=2250

    hls::stream<DTYPE_VEC> Z_deq("Z_deq");
#pragma HLS STREAM variable=Z_deq depth=2250

    // Quant/scale streams (internal)
    hls::stream<Q8_VEC_T>   Xssm_q("Xssm_q");
    hls::stream<SCALE_FX_T> Xssm_s1("Xssm_s1");
#pragma HLS STREAM variable=Xssm_q  depth=2250
#pragma HLS STREAM variable=Xssm_s1 depth=2250

    hls::stream<Q8_VEC_T>   Z_q("Z_q");
    hls::stream<SCALE_FX_T> Z_s1("Z_s1");
#pragma HLS STREAM variable=Z_q  depth=2250
#pragma HLS STREAM variable=Z_s1 depth=2250

    hls::stream<Q8_VEC_T>   delta_q("delta_q");
    hls::stream<SCALE_FX_T> delta_s1("delta_s1");
#pragma HLS STREAM variable=delta_q  depth=2250
#pragma HLS STREAM variable=delta_s1 depth=2250

    hls::stream<Q8_VEC_T>   B_q("B_q");
    hls::stream<SCALE_FX_T> B_s1("B_s1");
    hls::stream<Q8_VEC_T>   C_q("C_q");
    hls::stream<SCALE_FX_T> C_s1("C_s1");
#pragma HLS STREAM variable=B_q  depth=2250
#pragma HLS STREAM variable=B_s1 depth=2250
#pragma HLS STREAM variable=C_q  depth=2250
#pragma HLS STREAM variable=C_s1 depth=2250

    hls::stream<Q8_VEC_T>   Xout_q("Xout_q");
    hls::stream<SCALE_FX_T> Xout_s("Xout_s");
#pragma HLS STREAM variable=Xout_q depth=2250
#pragma HLS STREAM variable=Xout_s depth=2250

#pragma HLS DATAFLOW

    // kernel, X
    copy_kernel_k(kernel_in, kernel_local);
    copy_vec_n(X_in, X_local, D_T);

    // conv_state pass-through
    copy_vec_n(conv_state_in, conv_state_local_in, CONV_K-1);

    // X dup + norm
    tee_vecDT_stream2_local(X_local, X_for_norm, X_residual);
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_local, X_normed);

    // in_proj -> Z, XBC, DT (+ optional B/C)
#if SSMU_ENABLE_GEMM_MUXDEMUX
    in_proj_pack_stream_local_packed(X_normed, W_inproj, inproj_packed, wscale_in_fx);
    split_inproj_packed_local(inproj_packed, Z_stream, XBC_stream, DT_stream, B_from_inproj, C_from_inproj);
#else
    in_proj_pack_stream_local(X_normed, W_inproj, Z_stream, XBC_stream, DT_stream, B_from_inproj, C_from_inproj, wscale_in_fx);
#endif

    // conv (with conv_state)
    conv1d_silu_stream_local_with_state(
        XBC_stream, Z_stream, kernel_local,
        conv_state_local_in, conv_state_local_out,
        X_gate_stream, X_ssm_stream
    );

    // return conv_state
    copy_vec_n(conv_state_local_out, conv_state_out, CONV_K-1);

    // dup X_ssm
    dup_vecC2_stream3_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_scan_stream, X_ssm_out_stream);

    // B/C source selection
#if SSMU_BC_FROM_INPROJ_OK
    copy_vec_n(B_from_inproj, B_stream_S, STATE);
    copy_vec_n(C_from_inproj, C_stream_S, STATE);
#else
    write_token1_local(start_bc);
    stream_WBWC_tiles_gated_local(W_B, W_C, start_bc, WB_tiles, WC_tiles);
#endif

    // DELTA source selection
#if SSMU_ENABLE_DT && SSMU_DELTA_FROM_DT

  #if SSMU_ENABLE_DTADAPT
    dtadapt_stream_local(DT_stream, DT_C2_stream);
  #else
    #if (SSMU_CH_T == SSMU_C2_T)
      copy_vec_n(DT_stream, DT_C2_stream, C2_T);
    #else
      drain_vec_n(DT_stream, CH_T);
      for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
          DT_C2_stream.write(dvec_zero());
      }
    #endif
  #endif

    dt_to_delta_stream_local(DT_C2_stream, delta_selected);

#if SSMU_BC_FROM_INPROJ_OK
    drain_vec_n(X_ssm_proj_stream, C2_T);
#else
    projection_BC_only_local(
        X_ssm_proj_stream,
        WB_tiles, WC_tiles,
        B_stream_S, C_stream_S,
        wscale_bc_fx
    );
#endif

#else
    write_token1_local(start_wd);
    stream_Wdelta_tiles_gated_local(W_delta, start_wd, Wd_tiles);

#if SSMU_BC_FROM_INPROJ_OK
    write_token1_local(start_bc);
    stream_WBWC_tiles_gated_local(W_B, W_C, start_bc, WB_tiles, WC_tiles);

    hls::stream<DTYPE_VEC> delta_from_proj_local("delta_from_proj_local");
#pragma HLS STREAM variable=delta_from_proj_local depth=SSMU_DEPTH_DATA_MID

    projection_streams_local(
        X_ssm_proj_stream,
        Wd_tiles, WB_tiles, WC_tiles,
        B_throw, C_throw,
        delta_from_proj_local,
        wscale_delta_fx,
        wscale_bc_fx
    );
    drain_vec_n(B_throw, STATE);
    drain_vec_n(C_throw, STATE);
    copy_vec_n(delta_from_proj_local, delta_selected, C2_T);
#else
    projection_streams_local(
        X_ssm_proj_stream,
        Wd_tiles, WB_tiles, WC_tiles,
        B_stream_S, C_stream_S,
        delta_from_proj,
        wscale_delta_fx,
        wscale_bc_fx
    );
    copy_vec_n(delta_from_proj, delta_selected, C2_T);
#endif

#if SSMU_ENABLE_DT
    drain_vec_n(DT_stream, CH_T);
#endif
#endif

    // dup delta
    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan);

    // stage3 (dA)
    stage3_dA_stream_local(delta_for_dA, A_local, dA_stream);

    // activation q chain (optional)
#if SSMU_ENABLE_ACT_Q_CHAIN
    dynamic_int8_quant_stream_local(X_ssm_scan_stream, Xssm_q, Xssm_s1, C2_T);
    dynamic_int8_quant_stream_local(X_gate_stream,     Z_q,    Z_s1,    C2_T);
    dynamic_int8_quant_stream_local(delta_for_scan,    delta_q,delta_s1,C2_T);

    dynamic_int8_quant_stream_local(B_stream_S, B_q, B_s1, STATE);
    dynamic_int8_quant_stream_local(C_stream_S, C_q, C_s1, STATE);

#if SSMU_ENABLE_DTB_QUANT
    stage45_update_reduce_local_q(
        Xssm_q, Xssm_s1,
        delta_q, delta_s1,
        dA_stream,
        B_q, B_s1,
        C_q, C_s1,
        H0_in,
        htC_stream,
        C_trace_stream,
        H1_trace_stream,
        H1_state_stream
    );
#else
    int8_dequant_stream_local(Xssm_q,  Xssm_s1,  Xssm_deq,  C2_T);
    int8_dequant_stream_local(delta_q, delta_s1, delta_deq, C2_T);
    int8_dequant_stream_local(B_q,     B_s1,     B_deq,     STATE);
    int8_dequant_stream_local(C_q,     C_s1,     C_deq,     STATE);

    stage45_update_reduce_local(
        Xssm_deq,
        delta_deq,
        dA_stream,
        B_deq,
        C_deq,
        H0_in,
        htC_stream,
        C_trace_stream,
        H1_trace_stream,
        H1_state_stream
    );
#endif

#else
    stage45_update_reduce_local(
        X_ssm_scan_stream,
        delta_for_scan,
        dA_stream,
        B_stream_S,
        C_stream_S,
        H0_in,
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

    // stage6
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

    // out_proj + residual
    out_proj_stream_local_rect(ssm_core_out_stream, W_out, out_proj_stream_s, wscale_out_fx);
    add_residual_local_D(out_proj_stream_s, X_residual, out_local);

    // output
    copy_vec_n(out_local, out, D_T);
}
