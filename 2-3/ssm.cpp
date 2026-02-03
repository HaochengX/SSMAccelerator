// ssm.cpp  (Version C - MAX CHANGE, SSMU.h only)  [A+B INTEGRATED + FAST DT-DELTA]
// COSIM latency optimization:
// - When using DT as delta source (CH_T==C2_T), SKIP W_delta matmul entirely.
// - Compute ONLY B/C in projection stage.
// - Do not stream Wd_tiles in DT-delta mode.

#include "SSMU.h"

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#ifndef __SYNTHESIS__
#include <cstdio>
#endif

#ifndef __SYNTHESIS__
  #define DUT_PRINTF(...) do { std::printf(__VA_ARGS__); std::fflush(stdout); } while(0)
#else
  #define DUT_PRINTF(...) do {} while(0)
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
// Local constants from SSMU.h
// ============================================================
static const int D_T     = SSMU_D_T;
static const int C2_T    = SSMU_C2_T;
static const int CCONV_T = SSMU_CCONV_T;
static const int CH_T    = SSMU_CH_T;
static const int CIN_T   = SSMU_CIN_T;
static const int STATE   = SSMU_STATE;

static const int CONV_K  = SSMU_K;     // = 4
static const int J_TILE  = 8;

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
//   - INT8 : default 1/128
//   - non-INT8 : default 1.0
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
// Weight helpers (int8 path vs DTYPE path)
// ============================================================
#if SSMU_USE_INT8
static inline ACC_T wget_scaled(const W_VEC &w, unsigned idx, ACC_T scale_fx) {
#pragma HLS INLINE
    int wi = (int)vget(w, idx);
    return ((ACC_T)wi) * scale_fx;
}
#else
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
// Activation approximations
// ============================================================
static inline ACT_T sigmoid_pwl_fx(ACT_T x) {
#pragma HLS INLINE
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fx<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

static inline DTYPE silu_elem(DTYPE a) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_pwl_fx(x);
    ACT_T y = x * s;
    return (DTYPE)y;
}

static inline DTYPE softplus_pwl_fx(ACC_T xin) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)xin;
    const ACT_T TH  = (ACT_T)8.0;
    const ACT_T NTH = (ACT_T)(-8.0);

    if (x > TH)  return (DTYPE)x;
    if (x < NTH) return (DTYPE)0;

    const ACT_T half = (ACT_T)0.5;
    const ACT_T one  = (ACT_T)1.0;
    ACT_T y = half * x + one;
    return (DTYPE)y;
}

static inline EXP_T exp_poly_fx(ACT_T t) {
#pragma HLS INLINE
    const ACT_T TH  = (ACT_T)3.0;
    const ACT_T NTH = (ACT_T)(-3.0);

    const EXP_T EXP3  = (EXP_T)20.0855369;
    const EXP_T EXPN3 = (EXP_T)0.0497871;

    if (t > TH)  return EXP3;
    if (t < NTH) return EXPN3;

    ap_fixed<24, 8> tt = (ap_fixed<24,8>)t * (ap_fixed<24,8>)t;
    ap_fixed<24, 8> y  = (ap_fixed<24,8>)1.0 + (ap_fixed<24,8>)t + (ap_fixed<24,8>)0.5 * tt;
    if (y < 0) y = 0;
    return (EXP_T)y;
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

static void copy_vec_n(hls::stream<DTYPE_VEC>& in,
                       hls::stream<DTYPE_VEC>& out,
                       int count) {
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

static void tee_vecDT_stream2_local(hls::stream<DTYPE_VEC>& in,
                                    hls::stream<DTYPE_VEC>& out1,
                                    hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < D_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

static void dup_vecC2_stream3_local(hls::stream<DTYPE_VEC>& in,
                                    hls::stream<DTYPE_VEC>& out1,
                                    hls::stream<DTYPE_VEC>& out2,
                                    hls::stream<DTYPE_VEC>& out3) {
#pragma HLS INLINE off
    for (int i = 0; i < C2_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
        out3.write(v);
    }
}

static void dup_vecC2_stream2_local(hls::stream<DTYPE_VEC>& in,
                                    hls::stream<DTYPE_VEC>& out1,
                                    hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < C2_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

static void add_residual_local_D(hls::stream<DTYPE_VEC>& y_in,
                                hls::stream<DTYPE_VEC>& x_res_in,
                                hls::stream<DTYPE_VEC>& y_out) {
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

// RMSNorm (unchanged)
static void rmsnorm_vecDT_stream_local(hls::stream<DTYPE_VEC>& x_in,
                                       const DTYPE_VEC RMS_weight[D_T],
                                       hls::stream<DTYPE_VEC>& y_out) {
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
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        lane_sumsq[l] = 0;
    }

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = xbuf[j];
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T v = (ACC_T)vget(xv, l);
            lane_sumsq[l] += v * v;
        }
    }

    ACC_T sumsq = 0;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
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
// Tile-stream weight helpers
// ============================================================
struct vec_tuple8 { W_VEC w[J_TILE]; };

// ============================================================
// IN_PROJ pack (Z + XBC + DT)
// ============================================================
static void in_proj_pack_stream_local(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_inproj[D_T][CIN_T],
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& XBC_out,
    hls::stream<DTYPE_VEC>& DT_out
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1
    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_d.read();
    }

    const ACC_T wscale = (ACC_T)SSMU_W_SCALE_IN;
#pragma HLS ARRAY_PARTITION variable=W_inproj cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int i = 0; i < CIN_T; ++i) {
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = 0;
        }

        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                if (jidx < D_T) {
                    W_VEC w = W_inproj[jidx][i];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                        ACC_T wv = wget_scaled(w, l, wscale);
                        acc[l] += xv * wv;
                    }
                }
            }
        }

        DTYPE_VEC outv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outv, l, (DTYPE)acc[l]);
        }

        if (i < C2_T) {
            Z_out.write(outv);
        } else if (i < (C2_T + CCONV_T)) {
            XBC_out.write(outv);
        } else {
#if SSMU_ENABLE_DT
            int dt_idx = i - (C2_T + CCONV_T);
            if (dt_idx < CH_T) DT_out.write(outv);
#endif
        }
    }
}

// ============================================================
// conv+silu (K=4 taps)  (unchanged)
// ============================================================
static void conv1d_silu_stream_local(
    hls::stream<DTYPE_VEC>& XBC_in,
    hls::stream<DTYPE_VEC>& Z_in,
    hls::stream<DTYPE>&     kernel_in,
    hls::stream<DTYPE_VEC>& X_gate_out,
    hls::stream<DTYPE_VEC>& X_ssm_out
) {
#pragma HLS INLINE off

    DTYPE line_buffer[CONV_K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=lutram

    DTYPE kernel_buffer[CONV_K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
    for (int i = 0; i < CONV_K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }

    for (int i = 0; i < CONV_K-1; ++i) {
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS PIPELINE II=1
            line_buffer[i][l] = (DTYPE)0;
        }
    }

    for (int i = 0; i < CCONV_T; ++i) {
#pragma HLS PIPELINE II=1
        const bool do_c2 = (i < C2_T);

        if (do_c2) {
            DTYPE_VEC z_in = Z_in.read();
            DTYPE_VEC gate_out;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                vset(gate_out, l, silu_elem(vget(z_in, l)));
            }
            X_gate_out.write(gate_out);
        }

        DTYPE_VEC x_in = XBC_in.read();

        DTYPE window0[VEC_FACTOR];
        DTYPE window1[VEC_FACTOR];
        DTYPE window2[VEC_FACTOR];
        DTYPE window3[VEC_FACTOR];
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
                vset(ssm_out, lane, silu_elem((DTYPE)sum));
            }
            X_ssm_out.write(ssm_out);
        }
    }
}

// ============================================================
// DT -> delta (softplus)  [only safe when CH_T == C2_T]
// ============================================================
static void dt_to_delta_stream_local(
    hls::stream<DTYPE_VEC>& DT_in,
    hls::stream<DTYPE_VEC>& delta_out
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC dtv = DT_in.read();
        DTYPE_VEC dv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T x = (ACC_T)vget(dtv, l);
            vset(dv, l, softplus_pwl_fx(x));
        }
        delta_out.write(dv);
    }
}

// ============================================================
// Weight tile streamers (unchanged, but Wdelta will be skipped in DT-delta mode)
// ============================================================
static void stream_Wdelta_tiles_local(
        const W_VEC W_delta[C2_T][C2_T],
        hls::stream<vec_tuple8>& Wd_tiles) {
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
        hls::stream<vec_tuple8>& WC_tiles) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_B cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=W_C cyclic factor=8 dim=2

    for (int i = 0; i < STATE; ++i) {
        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            vec_tuple8 tb, tc;
#pragma HLS ARRAY_PARTITION variable=tb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=tc.w complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
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

// ============================================================
// Full projection: X_ssm -> delta + B/C   (A-mode)
// ============================================================
static void projection_streams_local(
        hls::stream<DTYPE_VEC>& X_ssm_in,
        hls::stream<vec_tuple8>& Wd_tiles,
        hls::stream<vec_tuple8>& WB_tiles,
        hls::stream<vec_tuple8>& WC_tiles,
        hls::stream<DTYPE_VEC>& B_out_S,
        hls::stream<DTYPE_VEC>& C_out_S,
        hls::stream<DTYPE_VEC>& delta_out
) {
#pragma HLS INLINE off

    const ACC_T wscale_delta = (ACC_T)SSMU_W_SCALE_DELTA;
    const ACC_T wscale_bc    = (ACC_T)SSMU_W_SCALE_BC;

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    // stage1: delta
    for (int i = 0; i < C2_T; ++i) {
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS DEPENDENCE variable=acc inter false
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = 0;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            vec_tuple8 wd = Wd_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wd.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC w = wd.w[jj];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                        ACC_T wv = wget_scaled(w, l, wscale_delta);
                        acc[l] += xv * wv;
                    }
                }
            }
        }

        DTYPE_VEC delta_vec;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(delta_vec, l, softplus_pwl_fx(acc[l]));
        }
        delta_out.write(delta_vec);
    }

    // stage2: B/C
    for (int i = 0; i < STATE; ++i) {
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
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
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            vec_tuple8 wb = WB_tiles.read();
            vec_tuple8 wc = WC_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC wB = wb.w[jj];
                    W_VEC wC = wc.w[jj];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T x  = (ACC_T)vget(X_tile[jj], l);
                        ACC_T b  = wget_scaled(wB, l, wscale_bc);
                        ACC_T c  = wget_scaled(wC, l, wscale_bc);
                        accB[l] += x * b;
                        accC[l] += x * c;
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
// FAST projection for DT-delta mode: ONLY B/C (skip delta matmul)
// ============================================================
static void projection_BC_only_local(
        hls::stream<DTYPE_VEC>& X_ssm_in,     // len=C2_T
        hls::stream<vec_tuple8>& WB_tiles,
        hls::stream<vec_tuple8>& WC_tiles,
        hls::stream<DTYPE_VEC>& B_out_S,      // len=STATE
        hls::stream<DTYPE_VEC>& C_out_S       // len=STATE
) {
#pragma HLS INLINE off

    const ACC_T wscale_bc = (ACC_T)SSMU_W_SCALE_BC;

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int i = 0; i < STATE; ++i) {
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
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
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            vec_tuple8 wb = WB_tiles.read();
            vec_tuple8 wc = WC_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc.w complete dim=1

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC wB = wb.w[jj];
                    W_VEC wC = wc.w[jj];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T x  = (ACC_T)vget(X_tile[jj], l);
                        ACC_T b  = wget_scaled(wB, l, wscale_bc);
                        ACC_T c  = wget_scaled(wC, l, wscale_bc);
                        accB[l] += x * b;
                        accC[l] += x * c;
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
// stage3, stage45, ddr_writer, stage6, out_proj  (UNCHANGED from your current version)
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

                ACT_T adl = a * dl;
                adl = clamp_fx<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);

                EXP_T e = exp_poly_fx(adl);
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
        hls::stream<DTYPE_VEC>& H1_trace_out
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
                ACC_T H0  = (ACC_T)vget(H0v, l);
                ACC_T ddA = (ACC_T)vget(dA,  l);
                ACC_T Bx  = (ACC_T)vget(B_vec, l);
                ACC_T dl  = (ACC_T)vget(dlt,  l);
                ACC_T Xs  = (ACC_T)vget(xssm, l);

                ACC_T H1 = H0 * ddA + (Bx * dl) * Xs;
                vset(H1v, l, (DTYPE)H1);
            }

            C_trace_out.write(C_vec);
            H1_trace_out.write(H1v);

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

static void out_proj_stream_local_rect(
    hls::stream<DTYPE_VEC>& X_in,
    const W_VEC W_out[D_T][C2_T],
    hls::stream<DTYPE_VEC>& Y_out
) {
#pragma HLS INLINE off
    const ACC_T wscale = (ACC_T)SSMU_W_SCALE_OUT;

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in.read();
    }

#pragma HLS ARRAY_PARTITION variable=W_out cyclic factor=8 dim=2

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int i = 0; i < D_T; ++i) {
        ACC_T accY[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accY complete dim=1
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accY[l] = 0;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                int jidx = jt + jj;
                if (jidx < C2_T) {
                    W_VEC w = W_out[i][jidx];
                    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                        ACC_T xv = (ACC_T)vget(X_tile[jj], l);
                        ACC_T wv = wget_scaled(w, l, wscale);
                        accY[l] += xv * wv;
                    }
                }
            }
        }

        DTYPE_VEC y;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(y, l, (DTYPE)accY[l]);
        }
        Y_out.write(y);
    }
}

// ============================================================
// TOP
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
    DTYPE_VEC* C_ddr,
    DTYPE_VEC* H1_ddr,
    hls::stream<DTYPE_VEC>& out
) {
#pragma HLS INTERFACE ap_fifo port=kernel_in
#pragma HLS INTERFACE ap_fifo port=X_in
#pragma HLS INTERFACE ap_fifo port=H0_in
#pragma HLS INTERFACE ap_fifo port=out

#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] SSMU_USE_INT8=%d\n", (int)SSMU_USE_INT8);
    DUT_PRINTF("[DUT] W_SCALE_IN=%f BC=%f DELTA=%f OUT=%f\n",
               (float)SSMU_W_SCALE_IN, (float)SSMU_W_SCALE_BC,
               (float)SSMU_W_SCALE_DELTA, (float)SSMU_W_SCALE_OUT);
    DUT_PRINTF("[DUT] ENABLE_DT=%d DELTA_FROM_DT=%d (CH_T=%d C2_T=%d)\n",
               (int)SSMU_ENABLE_DT, (int)SSMU_DELTA_FROM_DT, (int)CH_T, (int)C2_T);
#endif

    // Streams
    hls::stream<DTYPE> kernel_local("kernel_local");
#pragma HLS STREAM variable=kernel_local depth=CONV_K

    hls::stream<DTYPE_VEC> X_local("X_local");
#pragma HLS STREAM variable=X_local depth=1024

    hls::stream<DTYPE_VEC> X_for_norm("X_for_norm");
    hls::stream<DTYPE_VEC> X_residual("X_residual");
#pragma HLS STREAM variable=X_for_norm depth=1024
#pragma HLS STREAM variable=X_residual depth=D_T

    hls::stream<DTYPE_VEC> X_normed("X_normed");
#pragma HLS STREAM variable=X_normed depth=1024

    hls::stream<DTYPE_VEC> Z_stream("Z_stream");
    hls::stream<DTYPE_VEC> XBC_stream("XBC_stream");
#pragma HLS STREAM variable=Z_stream depth=1024
#pragma HLS STREAM variable=XBC_stream depth=1024

    hls::stream<DTYPE_VEC> DT_stream("DT_stream");
#pragma HLS STREAM variable=DT_stream depth=1024

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
#pragma HLS STREAM variable=X_gate_stream depth=1024
#pragma HLS STREAM variable=X_ssm_stream  depth=1024

    hls::stream<DTYPE_VEC> H0_local("H0_local");
#pragma HLS STREAM variable=H0_local depth=1024

    hls::stream<vec_tuple8> WB_tiles("WB_tiles");
    hls::stream<vec_tuple8> WC_tiles("WC_tiles");
#pragma HLS STREAM variable=WB_tiles depth=1024
#pragma HLS STREAM variable=WC_tiles depth=1024

    // Only needed in A-mode (projection delta)
    hls::stream<vec_tuple8> Wd_tiles("Wd_tiles");
#pragma HLS STREAM variable=Wd_tiles depth=1024

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_scan_stream("X_ssm_scan_stream");
    hls::stream<DTYPE_VEC> X_ssm_out_stream ("X_ssm_out_stream");
#pragma HLS STREAM variable=X_ssm_proj_stream depth=1024
#pragma HLS STREAM variable=X_ssm_scan_stream depth=1024
#pragma HLS STREAM variable=X_ssm_out_stream  depth=1024

    hls::stream<DTYPE_VEC> B_stream_S("B_stream_S");
    hls::stream<DTYPE_VEC> C_stream_S("C_stream_S");
#pragma HLS STREAM variable=B_stream_S depth=1024
#pragma HLS STREAM variable=C_stream_S depth=1024

    hls::stream<DTYPE_VEC> delta_selected("delta_selected");
#pragma HLS STREAM variable=delta_selected depth=1024

    hls::stream<DTYPE_VEC> delta_for_dA("delta_for_dA");
    hls::stream<DTYPE_VEC> delta_for_scan("delta_for_scan");
#pragma HLS STREAM variable=delta_for_dA   depth=1024
#pragma HLS STREAM variable=delta_for_scan depth=1024

    hls::stream<DTYPE_VEC> dA_stream("dA_stream");
#pragma HLS STREAM variable=dA_stream depth=1024

    hls::stream<DTYPE_VEC> htC_stream("htC_stream");
#pragma HLS STREAM variable=htC_stream depth=1024

    hls::stream<DTYPE_VEC> C_trace_stream("C_trace_stream");
    hls::stream<DTYPE_VEC> H1_trace_stream("H1_trace_stream");
#pragma HLS STREAM variable=C_trace_stream  depth=1024
#pragma HLS STREAM variable=H1_trace_stream depth=1024

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
#pragma HLS STREAM variable=ssm_core_out_stream depth=1024

    hls::stream<DTYPE_VEC> out_proj_stream_s("out_proj_stream");
    hls::stream<DTYPE_VEC> out_local("out_local");
#pragma HLS STREAM variable=out_proj_stream_s depth=1024
#pragma HLS STREAM variable=out_local depth=1024

#pragma HLS DATAFLOW

    // kernel, X, H0
    copy_kernel_k(kernel_in, kernel_local);
    copy_vec_n(X_in, X_local, D_T);
    copy_vec_n(H0_in, H0_local, HUGE_LEN);

    // weight tiles B/C always needed
    stream_WBWC_tiles_local(W_B, W_C, WB_tiles, WC_tiles);

    // X dup + norm
    tee_vecDT_stream2_local(X_local, X_for_norm, X_residual);
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_weight, X_normed);

    // in_proj -> Z, XBC, DT
    in_proj_pack_stream_local(X_normed, W_inproj, Z_stream, XBC_stream, DT_stream);

    // conv
    conv1d_silu_stream_local(XBC_stream, Z_stream, kernel_local, X_gate_stream, X_ssm_stream);

    // dup X_ssm
    dup_vecC2_stream3_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_scan_stream, X_ssm_out_stream);

    // ----------------------------------------------------------
    // DELTA source selection + Projection
    //   DT-delta mode: SKIP W_delta matmul entirely (FAST PATH)
    // ----------------------------------------------------------
#if SSMU_ENABLE_DT && SSMU_DELTA_FROM_DT && (SSMU_CH_T == SSMU_C2_T)

    // 1) delta from DT
    dt_to_delta_stream_local(DT_stream, delta_selected);

    // 2) ONLY compute B/C (no delta matmul)
    projection_BC_only_local(
        X_ssm_proj_stream,
        WB_tiles, WC_tiles,
        B_stream_S, C_stream_S
    );

#else
    // A-mode: delta from projection, DT drained if produced

    // stream W_delta tiles (only needed here)
    stream_Wdelta_tiles_local(W_delta, Wd_tiles);

    hls::stream<DTYPE_VEC> delta_from_proj("delta_from_proj");
#pragma HLS STREAM variable=delta_from_proj depth=1024

    projection_streams_local(
        X_ssm_proj_stream,
        Wd_tiles, WB_tiles, WC_tiles,
        B_stream_S, C_stream_S,
        delta_from_proj
    );

    copy_vec_n(delta_from_proj, delta_selected, C2_T);

#if SSMU_ENABLE_DT
    drain_vec_n(DT_stream, CH_T);
#endif

#endif

    // dup delta
    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan);

    // stage3
    stage3_dA_stream_local(delta_for_dA, A_fixed, dA_stream);

    // stage45
    stage45_update_reduce_local(
        X_ssm_scan_stream,
        delta_for_scan,
        dA_stream,
        B_stream_S,
        C_stream_S,
        H0_local,
        htC_stream,
        C_trace_stream,
        H1_trace_stream
    );

    // DDR writer (csim drains)
    ddr_writer_local(C_trace_stream, H1_trace_stream, C_ddr, H1_ddr);

    // stage6
    stage6_out_combine_local(
        htC_stream,
        D_diag,
        X_ssm_out_stream,
        X_gate_stream,
        ssm_core_out_stream
    );

    // stage7 + residual
    out_proj_stream_local_rect(ssm_core_out_stream, W_out, out_proj_stream_s);
    add_residual_local_D(out_proj_stream_s, X_residual, out_local);

    // output
    copy_vec_n(out_local, out, D_T);
}
