// ============================ ssm_lowrank.cpp ============================
// LOW-RANK FACTORIZATION variant of ssm.cpp
//
// Changes from original ssm.cpp:
//   W_inproj[D_T][CIN_T]  →  W_in_1[D_T][RANK_T] + W_in_2[RANK_T][CIN_T]
//   W_out[D_T][C2_T]      →  W_out_A[D_T][RANK_T] + W_out_B[RANK_T][C2_T]
//
// Math:
//   Original inproj:  Y[i] = Σ_j  X[j] · W_inproj[j][i]
//   Low-rank inproj:  temp[r] = Σ_j  X[j] · W_in_1[j][r]   (stage 1)
//                     Y[i]    = Σ_r  temp[r] · W_in_2[r][i]  (stage 2)
//
//   Original outproj: Y[i] = Σ_j  W_out[i][j] · X[j]
//   Low-rank outproj: temp[r] = Σ_j  W_out_B[r][j] · X[j]   (stage 1)
//                     Y[i]    = Σ_r  W_out_A[i][r] · temp[r]  (stage 2)
//
// RANK = 1024 (configurable via SSMU_RANK), RANK_T = RANK / VEC_FACTOR = 128
//
// Memory savings:
//   inproj: D_T*CIN_T = 433,280 → D_T*RANK_T + RANK_T*CIN_T = 214,272 (−51%)
//   outproj: D_T*C2_T = 204,800 → D_T*RANK_T + RANK_T*C2_T  = 122,880 (−40%)
//
// NOTE: W_delta is NOT factored (kept as [C2_T][C2_T]).
// NOTE: For SSMU_USE_INT8=1, each sub-matrix may need its own quantization
//       scale. This POC applies the same scale convention as the original.

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
// Safety fallbacks
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
// vget/vset fallback
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
// Optional knobs (keep defaults)
// ============================================================
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
// Math knobs
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
// Local constants
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
              "CIN_T too small for in-proj layout.");

// ============================================================
// ★ LOW-RANK CONSTANTS
// ============================================================
#ifndef SSMU_RANK
#define SSMU_RANK 1024
#endif

static_assert((SSMU_RANK % VEC_FACTOR) == 0, "RANK must be divisible by VEC_FACTOR");
static_assert((SSMU_RANK / VEC_FACTOR) % J_TILE == 0, "RANK_T must be divisible by J_TILE");

#define SSMU_RANK_T (SSMU_RANK / VEC_FACTOR)

static const int RANK_T = SSMU_RANK_T;   // 128

// Depth macros for m_axi
#define SSMU_DEPTH_IN1    (SSMU_D_T  * SSMU_RANK_T)
#define SSMU_DEPTH_IN2    (SSMU_RANK_T * SSMU_CIN_T)
#define SSMU_DEPTH_OUTA   (SSMU_D_T  * SSMU_RANK_T)
#define SSMU_DEPTH_OUTB   (SSMU_RANK_T * SSMU_C2_T)
#define SSMU_DEPTH_DELTA  (SSMU_C2_T * SSMU_C2_T)

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
// Quantization scales
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
// Math functions (unchanged)
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
// Generic helpers (unchanged)
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

static void tee_vec_n_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out_main,
    hls::stream<DTYPE_VEC>& out_trace,
    int n
) {
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out_main.write(v);
#if SSMU_ENABLE_TRACE_STREAMS
        out_trace.write(v);
#else
        (void)out_trace;
#endif
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
// Preload (unchanged)
// ============================================================
template<int N>
static void preload_vec_table_local(const DTYPE_VEC in_gmem[N], DTYPE_VEC out_local[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        out_local[i] = in_gmem[i];
    }
}

static void preload_vec_table_local_dyn(const DTYPE_VEC* in_gmem, DTYPE_VEC* out_local, int n) {
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        out_local[i] = in_gmem[i];
    }
}

// ============================================================
// RMSNorm (unchanged)
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
// Tile tuple (J_TILE=8) + tree reduction
// ============================================================
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

// ############################################################
// ★★★ LOW-RANK INPUT PROJECTION ★★★
// ############################################################

// ---- Stage 1 tile streamer: W_in_1[D_T][RANK_T] ----
// Produces tiles for: temp[r] = Σ_j X[j] · W_in_1[j][r]
// Outer: RANK_T (output), Inner: D_T (reduction)
static void stream_Win1_tiles_local(
    const W_VEC W_in_1[D_T][RANK_T],
    hls::stream<vec_tuple8> Win1_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_in_1 cyclic factor=8 dim=1

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {
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
                    if ((i < RANK_T) && (jidx < D_T)) wbuf[jj] = W_in_1[jidx][i];
                    else                                wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                Win1_tiles[ii].write(tup);
            }
        }
    }
}

// ============================================================
// Helper: read stream → on-chip buffer
// ============================================================
static void read_x_buf_D_local(
    hls::stream<DTYPE_VEC>& x_in,
    DTYPE_VEC x_buf[D_T]
) {
#pragma HLS INLINE off
    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        x_buf[j] = x_in.read();
    }
}

static void read_temp_buf_RANK_local(
    hls::stream<DTYPE_VEC>& t_in,
    DTYPE_VEC t_buf[RANK_T]
) {
#pragma HLS INLINE off
    for (int j = 0; j < RANK_T; ++j) {
#pragma HLS PIPELINE II=1
        t_buf[j] = t_in.read();
    }
}

static void read_x_buf_C2_local(
    hls::stream<DTYPE_VEC>& x_in,
    DTYPE_VEC x_buf[C2_T]
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        x_buf[j] = x_in.read();
    }
}

// ---- Stage 1 consumer: X @ W_in_1 → temp stream[RANK_T] ----
static void inproj_stage1_consume_local(
    const DTYPE_VEC X_buf[D_T],
    hls::stream<vec_tuple8> Win1_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
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
                vec_tuple8 wt = Win1_tiles[ii].read();
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
            if (i < RANK_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }
                temp_out.write(outv);
            }
        }
    }
}

// ---- Stage 2 tile streamer: W_in_2[RANK_T][CIN_T] ----
// Produces tiles for: Y[i] = Σ_r temp[r] · W_in_2[r][i]
// Outer: CIN_T (output), Inner: RANK_T (reduction)
static void stream_Win2_tiles_local(
    const W_VEC W_in_2[RANK_T][CIN_T],
    hls::stream<vec_tuple8> Win2_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_in_2 cyclic factor=8 dim=1

    for (int it = 0; it < CIN_T; it += SSMU_I_TILE) {
        for (int rt = 0; rt < RANK_T; rt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int ridx = rt + jj;
                    if ((i < CIN_T) && (ridx < RANK_T)) wbuf[jj] = W_in_2[ridx][i];
                    else                                   wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                Win2_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 2 consumer: temp @ W_in_2 → Z/XBC/DT/B/C streams ----
static void inproj_stage2_consume_local(
    const DTYPE_VEC temp_buf[RANK_T],
    hls::stream<vec_tuple8> Win2_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& XBC_out,
    hls::stream<DTYPE_VEC>& DT_out,
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    DTYPE_VEC T_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=T_tile complete dim=1

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
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int rt = 0; rt < RANK_T; rt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int ridx = rt + jj;
                T_tile[jj] = (ridx < RANK_T) ? temp_buf[ridx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = Win2_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    // NOTE: For USE_INT8=0, wget_scaled is identity.
                    // For USE_INT8=1, each sub-matrix would ideally have its own scale.
                    ACC_T p0 = (ACC_T)vget(T_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_in_fx);
                    ACC_T p1 = (ACC_T)vget(T_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_in_fx);
                    ACC_T p2 = (ACC_T)vget(T_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_in_fx);
                    ACC_T p3 = (ACC_T)vget(T_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_in_fx);
                    ACC_T p4 = (ACC_T)vget(T_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_in_fx);
                    ACC_T p5 = (ACC_T)vget(T_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_in_fx);
                    ACC_T p6 = (ACC_T)vget(T_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_in_fx);
                    ACC_T p7 = (ACC_T)vget(T_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_in_fx);

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
                }
            }
        }
    }
}

// ---- Stage 1: X_normed × W_in_1 → temp_stream ----
// Top-level DATAFLOW node.  Reads X from stream, does tile-streamed GEMM,
// writes RANK_T temporaries to output stream.
static void in_proj_lr_stage1(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_in_1[D_T][RANK_T],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> Win1_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Win1_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_x_buf_D_local(X_in_d, X_buf);
    stream_Win1_tiles_local(W_in_1, Win1_tiles);
    inproj_stage1_consume_local(X_buf, Win1_tiles, temp_out, wscale_in_fx);
}

// ---- Stage 2: temp × W_in_2 → Z/XBC/DT/B/C streams ----
// Top-level DATAFLOW node.  Reads RANK_T temps from stream, does tile-streamed
// GEMM, dispatches rows to Z/XBC/DT/B/C output streams.
static void in_proj_lr_stage2(
    hls::stream<DTYPE_VEC>& temp_in,
    const W_VEC W_in_2[RANK_T][CIN_T],
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& XBC_out,
    hls::stream<DTYPE_VEC>& DT_out,
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC temp_buf[RANK_T];
#pragma HLS BIND_STORAGE variable=temp_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> Win2_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Win2_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_temp_buf_RANK_local(temp_in, temp_buf);
    stream_Win2_tiles_local(W_in_2, Win2_tiles);
    inproj_stage2_consume_local(temp_buf, Win2_tiles, Z_out, XBC_out, DT_out, B_out, C_out, wscale_in_fx);
}

// ============================================================
// conv+silu with state (unchanged)
// ============================================================
static void conv1d_silu_stream_local_with_state(
    hls::stream<DTYPE_VEC>& XBC_in,
    hls::stream<DTYPE_VEC>& Z_in,
    hls::stream<DTYPE>&     kernel_in,
    hls::stream<DTYPE_VEC>& conv_state_in,
    hls::stream<DTYPE_VEC>& conv_state_out,
    hls::stream<DTYPE_VEC>& G_out,
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
            G_out.write(gate_out);
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
// dtadapt + dt_to_delta (unchanged)
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
// W_delta tile streamers + delta projection (unchanged)
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
// stage3, stage45, ddr_writer, stage6 (all unchanged)
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
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC delta_buf[C2_T];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=delta_buf cyclic factor=8 dim=1
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    DTYPE_VEC acc[C2_T];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=acc cyclic factor=8 dim=1
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        acc[j] = dvec_zero();
    }

    for (int i = 0; i < STATE_V; ++i) {
        DTYPE_VEC B_vec = B_in.read();
        DTYPE_VEC C_vec = C_in.read();

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1

            DTYPE_VEC H0_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=H0_tile complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int j = jt + jj;
                if (j < C2_T) H0_tile[jj] = H0_in.read();
                else          H0_tile[jj] = dvec_zero();
            }

            DTYPE_VEC dA_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=dA_tile complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int j = jt + jj;
                if (j < C2_T) dA_tile[jj] = dA_in.read();
                else          dA_tile[jj] = dvec_zero();
            }

            DTYPE_VEC acc_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=acc_tile complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int j = jt + jj;
                acc_tile[jj] = (j < C2_T) ? acc[j] : dvec_zero();
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int j = jt + jj;
                if (j < C2_T) {
                    DTYPE_VEC dlt  = delta_buf[j];
                    DTYPE_VEC xssm = X_buf[j];
                    DTYPE_VEC H1v;

                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T H0  = (ACC_T)vget(H0_tile[jj], l);
                        ACC_T ddA = (ACC_T)vget(dA_tile[jj], l);
                        ACC_T Bx  = (ACC_T)vget(B_vec, l);
                        ACC_T dl  = (ACC_T)vget(dlt,  l);
                        ACC_T Xs  = (ACC_T)vget(xssm, l);
                        ACC_T H1  = H0 * ddA + (Bx * dl) * Xs;
                        vset(H1v, l, (DTYPE)H1);
                    }

#if SSMU_ENABLE_TRACE_DDR
                    C_trace_out.write(C_vec);
                    H1_trace_out.write(H1v);
#endif
                    H1_state_out.write(H1v);

                    DTYPE_VEC prev = acc_tile[jj];
                    DTYPE_VEC next;
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T base = (ACC_T)vget(prev, l);
                        ACC_T addt = (ACC_T)vget(H1v,  l) * (ACC_T)vget(C_vec, l);
                        vset(next, l, (DTYPE)(base + addt));
                    }
                    acc_tile[jj] = next;
                }
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int j = jt + jj;
                if (j < C2_T) acc[j] = acc_tile[jj];
            }
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

// stage6 uD/yz nodes (unchanged)
static inline DTYPE uD_node_lane_fx(DTYPE y) {
#pragma HLS INLINE
    return silu_fx(y);
}
static inline DTYPE yz_node_lane_fx(DTYPE u, DTYPE gate) {
#pragma HLS INLINE
    return (DTYPE)((ACC_T)u * (ACC_T)gate);
}
static inline void uD_node_vec_local(const DTYPE_VEC& y_in, DTYPE_VEC& u_out) {
#pragma HLS INLINE
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        DTYPE y = (DTYPE)vget(y_in, l);
        vset(u_out, l, uD_node_lane_fx(y));
    }
}
static inline void yz_node_vec_local(const DTYPE_VEC& u_in, const DTYPE_VEC& gate_in, DTYPE_VEC& outv) {
#pragma HLS INLINE
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        DTYPE u = (DTYPE)vget(u_in, l);
        DTYPE g = (DTYPE)vget(gate_in, l);
        vset(outv, l, yz_node_lane_fx(u, g));
    }
}

static void stage6_out_udyz_vec_local(
    hls::stream<DTYPE_VEC>& htC_in,
    const DTYPE_VEC         D_diag[C2_T],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& G_in,
    hls::stream<DTYPE_VEC>& out
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC htC  = htC_in.read();
        DTYPE_VEC xvec = X_in.read();
        DTYPE_VEC gvec = G_in.read();
        DTYPE_VEC dvec = D_diag[j];

        DTYPE_VEC yvec;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T ht = (ACC_T)vget(htC,  l);
            ACC_T x  = (ACC_T)vget(xvec, l);
            ACC_T d  = (ACC_T)vget(dvec, l);
            ACC_T y  = ht + d * x;
            vset(yvec, l, (DTYPE)y);
        }

        DTYPE_VEC uvec;
        uD_node_vec_local(yvec, uvec);

        DTYPE_VEC outv;
        yz_node_vec_local(uvec, gvec, outv);

        out.write(outv);
    }
}

// ############################################################
// ★★★ LOW-RANK OUTPUT PROJECTION ★★★
// ############################################################

// ---- Stage 1 tile streamer: W_out_B[RANK_T][C2_T] ----
// Produces tiles for: temp[r] = Σ_j W_out_B[r][j] · X[j]
// Outer: RANK_T (output rows), Inner: C2_T (reduction cols)
static void stream_WoutB_tiles_local(
    const W_VEC W_out_B[RANK_T][C2_T],
    hls::stream<vec_tuple8> WoB_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_out_B cyclic factor=8 dim=2

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {
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
                    if ((i < RANK_T) && (jidx < C2_T)) wbuf[jj] = W_out_B[i][jidx];
                    else                                 wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                WoB_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 1 consumer: W_out_B @ X → temp stream[RANK_T] ----
static void outproj_stage1_consume_local(
    const DTYPE_VEC X_buf[C2_T],
    hls::stream<vec_tuple8> WoB_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
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
                vec_tuple8 wt = WoB_tiles[ii].read();
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
            if (i < RANK_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }
                temp_out.write(outv);
            }
        }
    }
}

// ---- Stage 2 tile streamer: W_out_A[D_T][RANK_T] ----
// Produces tiles for: Y[i] = Σ_r W_out_A[i][r] · temp[r]
// Outer: D_T (output), Inner: RANK_T (reduction)
static void stream_WoutA_tiles_local(
    const W_VEC W_out_A[D_T][RANK_T],
    hls::stream<vec_tuple8> WoA_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_out_A cyclic factor=8 dim=2

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {
        for (int rt = 0; rt < RANK_T; rt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int ridx = rt + jj;
                    if ((i < D_T) && (ridx < RANK_T)) wbuf[jj] = W_out_A[i][ridx];
                    else                                wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                WoA_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 2 consumer: W_out_A @ temp → Y[D_T] stream ----
static void outproj_stage2_consume_local(
    const DTYPE_VEC temp_buf[RANK_T],
    hls::stream<vec_tuple8> WoA_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    DTYPE_VEC T_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=T_tile complete dim=1

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int rt = 0; rt < RANK_T; rt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int ridx = rt + jj;
                T_tile[jj] = (ridx < RANK_T) ? temp_buf[ridx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = WoA_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(T_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_out_fx);
                    ACC_T p1 = (ACC_T)vget(T_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_out_fx);
                    ACC_T p2 = (ACC_T)vget(T_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_out_fx);
                    ACC_T p3 = (ACC_T)vget(T_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_out_fx);
                    ACC_T p4 = (ACC_T)vget(T_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_out_fx);
                    ACC_T p5 = (ACC_T)vget(T_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_out_fx);
                    ACC_T p6 = (ACC_T)vget(T_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_out_fx);
                    ACC_T p7 = (ACC_T)vget(T_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_out_fx);

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

// ---- Stage 1: stage6_out × W_out_B → temp_stream ----
// Top-level DATAFLOW node.  Reads X from stream, does tile-streamed GEMM,
// writes RANK_T temporaries to output stream.
static void out_proj_lr_stage1(
    hls::stream<DTYPE_VEC>& X_in,
    const W_VEC W_out_B[RANK_T][C2_T],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> WoB_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=WoB_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_x_buf_C2_local(X_in, X_buf);
    stream_WoutB_tiles_local(W_out_B, WoB_tiles);
    outproj_stage1_consume_local(X_buf, WoB_tiles, temp_out, wscale_out_fx);
}

// ---- Stage 2: temp × W_out_A → Y[D_T] stream ----
// Top-level DATAFLOW node.  Reads RANK_T temps from stream, does tile-streamed
// GEMM, writes D_T results to Y output stream.
static void out_proj_lr_stage2(
    hls::stream<DTYPE_VEC>& temp_in,
    const W_VEC W_out_A[D_T][RANK_T],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC temp_buf[RANK_T];
#pragma HLS BIND_STORAGE variable=temp_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> WoA_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=WoA_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_temp_buf_RANK_local(temp_in, temp_buf);
    stream_WoutA_tiles_local(W_out_A, WoA_tiles);
    outproj_stage2_consume_local(temp_buf, WoA_tiles, Y_out, wscale_out_fx);
}

// ============================================================
// ★ TOP: SSMU KERNEL (LOW-RANK INTERFACE)
// ============================================================

#ifndef SSMU_AXI_RO_TUNE
#define SSMU_AXI_RO_TUNE max_read_burst_length=128 num_read_outstanding=32
#endif

#ifndef SSMU_STREAM_DEPTH
  #if SSMU_LATENCY_MODE
    #define SSMU_STREAM_DEPTH 650
  #else
    #define SSMU_STREAM_DEPTH 650
  #endif
#endif

#ifndef SSMU_TRACE_DEPTH
  #if SSMU_LATENCY_MODE
    #define SSMU_TRACE_DEPTH 800
  #else
    #define SSMU_TRACE_DEPTH 800
  #endif
#endif

void SSMU(
    hls::stream<DTYPE>&      kernel_in,

    const DTYPE_VEC          A_fixed[STATE_V],
    const DTYPE_VEC          RMS_weight[D_T],

    // ★ LOW-RANK: W_inproj factored into W_in_1 × W_in_2
    const W_VEC              W_in_1[D_T][RANK_T],
    const W_VEC              W_in_2[RANK_T][CIN_T],

    const W_VEC              W_delta[C2_T][C2_T],

    // ★ LOW-RANK: W_out factored into W_out_A × W_out_B
    const W_VEC              W_out_A[D_T][RANK_T],
    const W_VEC              W_out_B[RANK_T][C2_T],

    const DTYPE_VEC          D_diag[C2_T],

    hls::stream<DTYPE_VEC>&  X_in,
    hls::stream<DTYPE_VEC>&  H0_in,

    hls::stream<DTYPE_VEC>&  conv_state_in,
    hls::stream<DTYPE_VEC>&  conv_state_out,

    DTYPE_VEC*               C_ddr,
    DTYPE_VEC*               H1_ddr,

    hls::stream<DTYPE_VEC>&  H1_out,
    hls::stream<DTYPE_VEC>&  out,

    float                    w_scale_in,
    float                    w_scale_delta,
    float                    w_scale_out
) {
    // =========================================================================
    // 1) Interfaces
    // =========================================================================
#pragma HLS INTERFACE ap_fifo   port=kernel_in
#pragma HLS INTERFACE ap_fifo   port=X_in
#pragma HLS INTERFACE ap_fifo   port=H0_in
#pragma HLS INTERFACE ap_fifo   port=conv_state_in
#pragma HLS INTERFACE ap_fifo   port=conv_state_out
#pragma HLS INTERFACE ap_fifo   port=H1_out
#pragma HLS INTERFACE ap_fifo   port=out

#pragma HLS INTERFACE m_axi port=A_fixed     offset=slave bundle=gmemA     depth=STATE_V            SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=RMS_weight  offset=slave bundle=gmemRMS   depth=D_T               SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=D_diag      offset=slave bundle=gmemD     depth=C2_T              SSMU_AXI_RO_TUNE

// ★ LOW-RANK weight interfaces (4 sub-matrices instead of 2 full matrices)
#pragma HLS INTERFACE m_axi port=W_in_1      offset=slave bundle=gmemIn1   depth=SSMU_DEPTH_IN1     SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_in_2      offset=slave bundle=gmemIn2   depth=SSMU_DEPTH_IN2     SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_delta     offset=slave bundle=gmemDelta depth=SSMU_DEPTH_DELTA   SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_out_A     offset=slave bundle=gmemOutA  depth=SSMU_DEPTH_OUTA    SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_out_B     offset=slave bundle=gmemOutB  depth=SSMU_DEPTH_OUTB    SSMU_AXI_RO_TUNE

#pragma HLS INTERFACE m_axi     port=C_ddr    offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi     port=H1_ddr   offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr    bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr   bundle=control

#pragma HLS INTERFACE s_axilite port=w_scale_in     bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_delta  bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_out    bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    // =========================================================================
    // 2) Runtime scales
    // =========================================================================
    const ACC_T wscale_in_fx    = pick_scale_fx(w_scale_in,    (float)SSMU_W_SCALE_IN);
    const ACC_T wscale_delta_fx = pick_scale_fx(w_scale_delta, (float)SSMU_W_SCALE_DELTA);
    const ACC_T wscale_out_fx   = pick_scale_fx(w_scale_out,   (float)SSMU_W_SCALE_OUT);

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] SSMU LOW-RANK variant: RANK=%d RANK_T=%d\n", (int)SSMU_RANK, (int)RANK_T);
    DUT_PRINTF("[DUT] SSMU_USE_INT8=%d\n", (int)SSMU_USE_INT8);
    DUT_PRINTF("[DUT] TRACE_DDR=%d H1_OUT=%d H1_OUT_LEN=%d\n",
               (int)SSMU_ENABLE_TRACE_DDR, (int)SSMU_ENABLE_H1_STREAM_OUT, (int)SSMU_H1_OUT_LEN);
    DUT_PRINTF("[DUT] STATE_SCALAR=%d STATE_V(STATE_T)=%d\n", (int)STATE_SCALAR, (int)STATE_V);
    DUT_PRINTF("[DUT] W_in_1: [%d][%d], W_in_2: [%d][%d]\n", D_T, RANK_T, RANK_T, CIN_T);
    DUT_PRINTF("[DUT] W_out_A: [%d][%d], W_out_B: [%d][%d]\n", D_T, RANK_T, RANK_T, C2_T);
#endif

    // =========================================================================
    // 3) Preload constants
    // =========================================================================
    DTYPE_VEC A_local[STATE_V];
    DTYPE_VEC RMS_local[D_T];
    DTYPE_VEC D_local[C2_T];

#pragma HLS BIND_STORAGE    variable=A_local    type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE    variable=RMS_local  type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE    variable=D_local    type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=A_local    cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=RMS_local  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=D_local    cyclic factor=8 dim=1

    preload_vec_table_local_dyn(A_fixed,      A_local,   STATE_V);
    preload_vec_table_local<D_T>(RMS_weight,  RMS_local);
    preload_vec_table_local<C2_T>(D_diag,     D_local);

    // =========================================================================
    // 4) Streams
    // =========================================================================
    hls::stream<DTYPE_VEC> X_in_pre("X_in_pre");
    hls::stream<DTYPE_VEC> H0_in_pre("H0_in_pre");
    hls::stream<DTYPE_VEC> conv_state_pre("conv_state_pre");

    hls::stream<DTYPE>     kernel_local("kernel_local");
    hls::stream<DTYPE_VEC> X_local("X_local");
    hls::stream<DTYPE_VEC> X_for_norm("X_for_norm");
    hls::stream<DTYPE_VEC> X_residual("X_residual");
    hls::stream<DTYPE_VEC> X_normed("X_normed");

#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> trace_rms("trace_rms");
    hls::stream<DTYPE_VEC> trace_delta("trace_delta");
    hls::stream<DTYPE_VEC> trace_htC("trace_htC");
    hls::stream<DTYPE_VEC> trace_core("trace_core");
#endif

    hls::stream<DTYPE_VEC> Z_stream("Z_stream");
    hls::stream<DTYPE_VEC> XBC_stream("XBC_stream");
    hls::stream<DTYPE_VEC> DT_stream("DT_stream");
    hls::stream<DTYPE_VEC> B_from_inproj("B_from_inproj");
    hls::stream<DTYPE_VEC> C_from_inproj("C_from_inproj");

    hls::stream<DTYPE_VEC> DT_C2_stream("DT_C2_stream");
    hls::stream<DTYPE_VEC> G_stream("G_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream("X_ssm_stream");
    hls::stream<DTYPE_VEC> conv_state_local_in("conv_state_local_in");
    hls::stream<DTYPE_VEC> conv_state_local_out("conv_state_local_out");

    hls::stream<vec_tuple8>  Wd_tiles("Wd_tiles");
    hls::stream<ap_uint<1> > start_wd("start_wd");

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_scan_stream("X_ssm_scan_stream");
    hls::stream<DTYPE_VEC> X_ssm_out_stream("X_ssm_out_stream");

    hls::stream<DTYPE_VEC> B_stream_S("B_stream_S");
    hls::stream<DTYPE_VEC> C_stream_S("C_stream_S");

    hls::stream<DTYPE_VEC> delta_selected("delta_selected");
    hls::stream<DTYPE_VEC> delta_for_dA("delta_for_dA");
    hls::stream<DTYPE_VEC> delta_for_scan("delta_for_scan");
    hls::stream<DTYPE_VEC> dA_stream("dA_stream");

    hls::stream<DTYPE_VEC> htC_stream("htC_stream");
    hls::stream<DTYPE_VEC> C_trace_stream("C_trace_stream");
    hls::stream<DTYPE_VEC> H1_trace_stream("H1_trace_stream");
    hls::stream<DTYPE_VEC> H1_state_stream("H1_state_stream");

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
    hls::stream<DTYPE_VEC> out_proj_stream_s("out_proj_stream_s");
    hls::stream<DTYPE_VEC> out_local("out_local");

    // ★ LOW-RANK intermediate temp streams
    hls::stream<DTYPE_VEC> inproj_temp_stream("inproj_temp_stream");
    hls::stream<DTYPE_VEC> outproj_temp_stream("outproj_temp_stream");

    // =========================================================================
    // 5) Stream depths
    // =========================================================================
#pragma HLS STREAM variable=X_in_pre             depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=H0_in_pre            depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=conv_state_pre       depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=kernel_local         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_local              depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_for_norm           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_residual           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_normed             depth=SSMU_STREAM_DEPTH

#if SSMU_ENABLE_TRACE_STREAMS
#pragma HLS STREAM variable=trace_rms            depth=SSMU_TRACE_DEPTH
#pragma HLS STREAM variable=trace_delta          depth=SSMU_TRACE_DEPTH
#pragma HLS STREAM variable=trace_htC            depth=SSMU_TRACE_DEPTH
#pragma HLS STREAM variable=trace_core           depth=SSMU_TRACE_DEPTH
#endif

#pragma HLS STREAM variable=Z_stream             depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=XBC_stream           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=DT_stream            depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=B_from_inproj        depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_from_inproj        depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=DT_C2_stream         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=G_stream             depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_ssm_stream         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=conv_state_local_in  depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=conv_state_local_out depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=Wd_tiles             depth=SSMU_DEPTH_TILE
#pragma HLS STREAM variable=start_wd             depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=X_ssm_proj_stream    depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_ssm_scan_stream    depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_ssm_out_stream     depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=B_stream_S           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_stream_S           depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=delta_selected       depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=delta_for_dA         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=delta_for_scan       depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=dA_stream            depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=htC_stream           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_trace_stream       depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=H1_trace_stream      depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=H1_state_stream      depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=ssm_core_out_stream  depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=out_proj_stream_s    depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=out_local            depth=SSMU_STREAM_DEPTH

    // ★ LOW-RANK intermediate temp stream depths
#pragma HLS STREAM variable=inproj_temp_stream   depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=outproj_temp_stream  depth=SSMU_STREAM_DEPTH

    // =========================================================================
    // 6) Dataflow graph
    // =========================================================================
#pragma HLS DATAFLOW

    // predecessor copies
    copy_vec_n(X_in,          X_in_pre,       D_T);
    copy_vec_n(H0_in,         H0_in_pre,      SSMU_H1_OUT_LEN);
    copy_vec_n(conv_state_in, conv_state_pre, CONV_K-1);

    // main flow
    copy_kernel_k(kernel_in, kernel_local);
    copy_vec_n(X_in_pre, X_local, D_T);
    copy_vec_n(conv_state_pre, conv_state_local_in, CONV_K-1);

    // residual split
    tee_vecDT_stream2_local(X_local, X_for_norm, X_residual);

    // RMSNorm
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> X_normed_raw("X_normed_raw");
#pragma HLS STREAM variable=X_normed_raw depth=SSMU_STREAM_DEPTH
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_local, X_normed_raw);
    tee_vec_n_local(X_normed_raw, X_normed, trace_rms, D_T);
#else
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_local, X_normed);
#endif

    // ★ LOW-RANK in-projection: two top-level DATAFLOW stages
    in_proj_lr_stage1(X_normed, W_in_1, inproj_temp_stream, wscale_in_fx);
    in_proj_lr_stage2(inproj_temp_stream, W_in_2,
        Z_stream, XBC_stream, DT_stream, B_from_inproj, C_from_inproj,
        wscale_in_fx);

    copy_vec_n(B_from_inproj, B_stream_S, SSMU_STATE_T);
    copy_vec_n(C_from_inproj, C_stream_S, SSMU_STATE_T);

    // conv + state
    conv1d_silu_stream_local_with_state(
        XBC_stream, Z_stream, kernel_local,
        conv_state_local_in, conv_state_local_out,
        G_stream, X_ssm_stream
    );
    copy_vec_n(conv_state_local_out, conv_state_out, CONV_K-1);

    // duplicate X_ssm
    dup_vecC2_stream3_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_scan_stream, X_ssm_out_stream);

    // delta selection
#if (SSMU_ENABLE_DT && SSMU_DELTA_FROM_DT)
    dtadapt_stream_local(DT_stream, DT_C2_stream);
    dt_to_delta_stream_local(DT_C2_stream, delta_selected);
    drain_vec_n(X_ssm_proj_stream, C2_T);
#else
    drain_vec_n(DT_stream, CH_T);
    write_token1_local(start_wd);
    stream_Wdelta_tiles_gated_local(W_delta, start_wd, Wd_tiles);
    projection_delta_only_local(X_ssm_proj_stream, Wd_tiles, delta_selected, wscale_delta_fx);
#endif

    // delta tee
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> delta_for_scan_raw("delta_for_scan_raw");
#pragma HLS STREAM variable=delta_for_scan_raw depth=SSMU_STREAM_DEPTH
    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan_raw);
    tee_vec_n_local(delta_for_scan_raw, delta_for_scan, trace_delta, C2_T);
#else
    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan);
#endif

    // stage3
    stage3_dA_stream_local(delta_for_dA, A_local, dA_stream);

    // stage45
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> htC_raw("htC_raw");
#pragma HLS STREAM variable=htC_raw depth=SSMU_STREAM_DEPTH
    stage45_update_reduce_local(
        X_ssm_scan_stream, delta_for_scan, dA_stream, B_stream_S, C_stream_S, H0_in_pre,
        htC_raw, C_trace_stream, H1_trace_stream, H1_state_stream
    );
    tee_vec_n_local(htC_raw, htC_stream, trace_htC, C2_T);
#else
    stage45_update_reduce_local(
        X_ssm_scan_stream, delta_for_scan, dA_stream, B_stream_S, C_stream_S, H0_in_pre,
        htC_stream, C_trace_stream, H1_trace_stream, H1_state_stream
    );
#endif

    // optional DDR trace
#if SSMU_ENABLE_TRACE_DDR
    ddr_writer_local(C_trace_stream, H1_trace_stream, C_ddr, H1_ddr);
#endif

    // H1 stream out
#if SSMU_ENABLE_H1_STREAM_OUT
    copy_vec_n(H1_state_stream, H1_out, SSMU_H1_OUT_LEN);
#else
    drain_vec_n(H1_state_stream, SSMU_H1_OUT_LEN);
#endif

    // stage6
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> core_raw("core_raw");
#pragma HLS STREAM variable=core_raw depth=SSMU_STREAM_DEPTH
    stage6_out_udyz_vec_local(htC_stream, D_local, X_ssm_out_stream, G_stream, core_raw);
    tee_vec_n_local(core_raw, ssm_core_out_stream, trace_core, C2_T);
#else
    stage6_out_udyz_vec_local(htC_stream, D_local, X_ssm_out_stream, G_stream, ssm_core_out_stream);
#endif

    // ★ LOW-RANK out-proj: two top-level DATAFLOW stages
    out_proj_lr_stage1(ssm_core_out_stream, W_out_B, outproj_temp_stream, wscale_out_fx);
    out_proj_lr_stage2(outproj_temp_stream, W_out_A, out_proj_stream_s, wscale_out_fx);

    add_residual_local_D(out_proj_stream_s, X_residual, out_local);
    copy_vec_n(out_local, out, D_T);
}

// ============================================================
// Compatibility wrapper: SSMU_STACK64 (LOW-RANK interface)
// ============================================================
extern "C" void SSMU_STACK64(
    hls::stream<DTYPE>&      kernel_in,

    const DTYPE_VEC          A_fixed[STATE_V],
    const DTYPE_VEC          RMS_weight[D_T],

    const W_VEC              W_in_1[D_T][RANK_T],
    const W_VEC              W_in_2[RANK_T][CIN_T],

    const W_VEC              W_delta[C2_T][C2_T],

    const W_VEC              W_out_A[D_T][RANK_T],
    const W_VEC              W_out_B[RANK_T][C2_T],

    const DTYPE_VEC          D_diag[C2_T],

    hls::stream<DTYPE_VEC>&  X_in,
    hls::stream<DTYPE_VEC>&  H0_in,

    hls::stream<DTYPE_VEC>&  conv_state_in,
    hls::stream<DTYPE_VEC>&  conv_state_out,

    DTYPE_VEC*               C_ddr,
    DTYPE_VEC*               H1_ddr,

    hls::stream<DTYPE_VEC>&  H1_out,
    hls::stream<DTYPE_VEC>&  out,

    float                    w_scale_in,
    float                    w_scale_delta,
    float                    w_scale_out
) {
#pragma HLS INLINE off

    SSMU(kernel_in,
         A_fixed, RMS_weight,
         W_in_1, W_in_2,
         W_delta,
         W_out_A, W_out_B,
         D_diag,
         X_in, H0_in,
         conv_state_in, conv_state_out,
         C_ddr, H1_ddr,
         H1_out, out,
         w_scale_in, w_scale_delta, w_scale_out);
}
