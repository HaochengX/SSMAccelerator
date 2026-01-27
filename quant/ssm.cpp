// quant.cpp  (FULL, DATAFLOW + "stream vec_tuple W[8]" + II=1-focused)
// NOW: TRUE INT8 WEIGHT QUANTIZATION PATH (SSMU_USE_INT8=1 in SSMU.h)
//
// - Weights (W_in_x/W_in_z/W_B/W_C/W_delta/W_out) use W_VEC (Q8VEC when quant enabled)
// - Activations/streams stay DTYPE_VEC
// - Includes scale knobs (defaults) so int8 weights map back to real domain
//
// Goals you asked for:
// 1) Use "// stream vec_tuple W[8]" (tile-stream) for W_B / W_C / W_delta
// 2) Push PIPELINE II=1 in projection hot loops (avoid 200-885 port II violations)
// 3) Remove DATAFLOW 200-1449 warnings by copying caller ports into internal streams
// 4) Mitigate 200-887 add-chain timing by using narrower projection accumulators + partial sums
// 5) Fix 214-366 by unifying vget/vset index type to unsigned
// 6) Fix 207-949 include case: use "SSMU.h" (match disk)
//
// NOTE on "true quantization":
// - This file truly uses int8 weights (Q8VEC) when SSMU_USE_INT8=1.
// - You MUST ensure your host/tb actually provides int8-packed weights matching W_VEC.
// - Scales: if you have per-tensor/per-channel scales, extend by passing scale arrays
//   or baking them into weights. Here we provide default per-tensor scale macros.

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

#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<32, 10> ACC_T;
#else
typedef float ACC_T;
#endif

// ============================================================
// Quantization scale knobs (default: symmetric int8 with scale=1/128)
// Replace these with your real scales if available.
// ============================================================
#ifndef SSMU_W_SCALE_IN
#define SSMU_W_SCALE_IN  (1.0f/128.0f)
#endif

#ifndef SSMU_W_SCALE_BC
#define SSMU_W_SCALE_BC  (1.0f/128.0f)
#endif

#ifndef SSMU_W_SCALE_DELTA
#define SSMU_W_SCALE_DELTA (1.0f/128.0f)
#endif

#ifndef SSMU_W_SCALE_OUT
#define SSMU_W_SCALE_OUT (1.0f/128.0f)
#endif

// ============================================================
// Vector accessors (UNIFY INDEX TYPE to avoid HLS 214-366)
// ============================================================
static inline DTYPE vget(const DTYPE_VEC &v, unsigned idx) {
#pragma HLS INLINE
    return v[idx];
}
static inline void vset(DTYPE_VEC &v, unsigned idx, DTYPE val) {
#pragma HLS INLINE
    v[idx] = val;
}

// ---- Weight accessors (DTYPE_VEC or Q8VEC) -> ACC_T
#if SSMU_USE_INT8
static inline ap_int<8> wraw_get(const W_VEC &w, unsigned idx) {
#pragma HLS INLINE
    return w[idx];
}
static inline ACC_T wget_scaled(const W_VEC &w, unsigned idx, float scale) {
#pragma HLS INLINE
    ACC_T wi = (ACC_T)((int)wraw_get(w, idx));
    return wi * (ACC_T)scale;
}
#else
static inline DTYPE wraw_get(const W_VEC &w, unsigned idx) {
#pragma HLS INLINE
    return w[idx];
}
static inline ACC_T wget_scaled(const W_VEC &w, unsigned idx, float /*scale*/) {
#pragma HLS INLINE
    return (ACC_T)wraw_get(w, idx);
}
#endif

// ============================================================
// NO exp/log approximations (fixed-point)
// ============================================================
typedef ap_fixed<18, 6>  ACT_T;   // activations / inputs
typedef ap_fixed<20, 8>  EXP_T;   // exp approx output

template<typename T>
static inline T clamp_fx(T x, T lo, T hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// sigmoid(x) ≈ clamp(0.5 + x/4, 0, 1)
static inline ACT_T sigmoid_pwl_fx(ACT_T x) {
#pragma HLS INLINE
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fx<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

// SiLU(x)=x*sigmoid(x)
static inline DTYPE silu_elem(DTYPE a) {
#pragma HLS INLINE
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_pwl_fx(x);
    ACT_T y = x * s;
    return (DTYPE)y;
}

// softplus PWL:
//   x > 8:  ≈ x
//   x < -8: ≈ 0
//   else:   ≈ 0.5*x + 1
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

// exp(t) approx: clamp + 2nd order poly in [-3,3]
//   t > 3  -> exp(3)
//   t < -3 -> exp(-3)
//   else   -> 1 + t + 0.5 t^2
static inline EXP_T exp2_poly_fx(ACT_T t) {
#pragma HLS INLINE
    const ACT_T TH  = (ACT_T)3.0;
    const ACT_T NTH = (ACT_T)(-3.0);

    const EXP_T EXP3  = (EXP_T)20.0855369; // exp(3)
    const EXP_T EXPN3 = (EXP_T)0.0497871;  // exp(-3)

    if (t > TH)  return EXP3;
    if (t < NTH) return EXPN3;

    ap_fixed<24, 8> tt = (ap_fixed<24,8>)t * (ap_fixed<24,8>)t;
    ap_fixed<24, 8> y  = (ap_fixed<24,8>)1.0 + (ap_fixed<24,8>)t + (ap_fixed<24,8>)0.5 * tt;

    if (y < 0) y = 0;
    return (EXP_T)y;
}

// ============================================================
// DATAFLOW adapter copies (avoid HLS 200-1449)
// ============================================================
static void copy_kernel_k(hls::stream<DTYPE>& in,
                          hls::stream<DTYPE>& out) {
#pragma HLS INLINE off
    for (int i = 0; i < K; ++i) {
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }
}

static void copy_vecd_n(hls::stream<DTYPE_VEC>& in,
                        hls::stream<DTYPE_VEC>& out,
                        int count) {
#pragma HLS INLINE off
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }
}

// ============================================================
// tee for VEC_D tokens (3-way): main / D-skip / residual
// ============================================================
static void tee_vecD_stream3_local(hls::stream<DTYPE_VEC>& in,
                                   hls::stream<DTYPE_VEC>& out1,
                                   hls::stream<DTYPE_VEC>& out2,
                                   hls::stream<DTYPE_VEC>& out3) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
        out3.write(v);
    }
}

// ============================================================
// dup for VEC_D tokens (safe small)
// ============================================================
static void dup_vecD_stream_local(hls::stream<DTYPE_VEC>& in,
                                  hls::stream<DTYPE_VEC>& out1,
                                  hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

// ============================================================
// D skip add (diagonal): y += D_diag[j] * x
// ============================================================
static void add_D_skip_local(hls::stream<DTYPE_VEC>& y_in,
                             hls::stream<DTYPE_VEC>& x_in,
                             const DTYPE_VEC D_diag[VEC_D],
                             hls::stream<DTYPE_VEC>& y_out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC y = y_in.read();
        DTYPE_VEC x = x_in.read();
        DTYPE_VEC d = D_diag[j];

        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T yy = (ACC_T)vget(y, l);
            ACC_T xx = (ACC_T)vget(x, l);
            ACC_T dd = (ACC_T)vget(d, l);
            vset(o, l, (DTYPE)(yy + dd * xx));
        }
        y_out.write(o);
    }
}

// ============================================================
// Residual add: y += x_res
// ============================================================
static void add_residual_local(hls::stream<DTYPE_VEC>& y_in,
                               hls::stream<DTYPE_VEC>& x_res_in,
                               hls::stream<DTYPE_VEC>& y_out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
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
// RMSNorm (pre-norm): y = x * rsqrt(mean(x^2) + eps) * weight
// ============================================================
static void rmsnorm_vecD_stream_local(hls::stream<DTYPE_VEC>& x_in,
                                      const DTYPE_VEC RMS_weight[VEC_D],
                                      hls::stream<DTYPE_VEC>& y_out) {
#pragma HLS INLINE off
    const float eps = 1e-5f;

    DTYPE_VEC xbuf[VEC_D];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_s2p impl=lutram

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        xbuf[j] = x_in.read();
    }

    ACC_T sumsq = 0;
    for (int j = 0; j < VEC_D; ++j) {
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS PIPELINE II=1
            ACC_T v = (ACC_T)vget(xbuf[j], l);
            sumsq += v * v;
        }
    }

    const float denom = (float)(VEC_D * VEC_FACTOR);
    float ms = (float)sumsq / denom;

    if (!(ms >= 0.0f)) ms = 0.0f;
    if (ms < 0.0f) ms = 0.0f;
    float inv = 1.0f / hls::sqrtf(ms + eps);

    for (int j = 0; j < VEC_D; ++j) {
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
// Input projection (in_proj): produce X_for_conv and Z_for_gate
// NOW: W_in_x / W_in_z are W_VEC (Q8VEC when quant enabled)
// ============================================================
static void in_proj_stream_local(hls::stream<DTYPE_VEC>& X_in_raw,
                                 const W_VEC W_in_x[VEC_D][VEC_D],
                                 const W_VEC W_in_z[VEC_D][VEC_D],
                                 hls::stream<DTYPE_VEC>& X_for_conv,
                                 hls::stream<DTYPE_VEC>& Z_for_gate) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_raw.read();
    }

    const float wscale = (float)SSMU_W_SCALE_IN;

    for (int i = 0; i < VEC_D; ++i) {
        ACC_T accX[VEC_FACTOR];
        ACC_T accZ[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accX complete dim=1
#pragma HLS ARRAY_PARTITION variable=accZ complete dim=1
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accX[l] = (ACC_T)0;
            accZ[l] = (ACC_T)0;
        }

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC x  = X_buf[j];
            W_VEC wx = W_in_x[i][j];
            W_VEC wz = W_in_z[i][j];

            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T xv  = (ACC_T)vget(x, l);
                ACC_T wvx = wget_scaled(wx, l, wscale);
                ACC_T wvz = wget_scaled(wz, l, wscale);
                accX[l] += xv * wvx;
                accZ[l] += xv * wvz;
            }
        }

        DTYPE_VEC outX, outZ;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outX, l, (DTYPE)accX[l]);
            vset(outZ, l, (DTYPE)accZ[l]);
        }
        X_for_conv.write(outX);
        Z_for_gate.write(outZ);
    }
}

// ============================================================
// conv1d + SiLU (unchanged; kernel is still DTYPE)
// ============================================================
static void conv1d_silu_stream_local(hls::stream<DTYPE_VEC>& X_for_conv_in,
                                     hls::stream<DTYPE_VEC>& Z_for_gate_in,
                                     hls::stream<DTYPE>& kernel_in,
                                     hls::stream<DTYPE_VEC>& X_gate_out,
                                     hls::stream<DTYPE_VEC>& X_ssm_out) {
#pragma HLS INLINE off

    static DTYPE line_buffer[K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=lutram

    DTYPE kernel_buffer[K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
    for (int i = 0; i < K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }

    for (int i = 0; i < K-1; ++i)
        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[i][k] = 0;

    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=2
        DTYPE_VEC x_in = X_for_conv_in.read();
        DTYPE_VEC z_in = Z_for_gate_in.read();

        DTYPE_VEC gate_out;
        for (unsigned k = 0; k < (unsigned)VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(gate_out, k, silu_elem(vget(z_in, k)));
        }
        X_gate_out.write(gate_out);

        DTYPE window[K][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window complete dim=2

        for (int j = 0; j < K-1; ++j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                window[j][k] = line_buffer[j][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            window[K-1][k] = vget(x_in, (unsigned)k);

        for (int j = K-2; j > 0; --j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                line_buffer[j][k] = line_buffer[j-1][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[0][k] = vget(x_in, (unsigned)k);

        DTYPE_VEC conv_out;
        for (unsigned lane = 0; lane < (unsigned)VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
            ACC_T sum = 0;
            for (int kk = 0; kk < K; ++kk) {
#pragma HLS UNROLL
                sum += (ACC_T)kernel_buffer[kk] * (ACC_T)window[kk][lane];
            }
            vset(conv_out, lane, (DTYPE)sum);
        }

        DTYPE_VEC ssm_out;
        for (unsigned k = 0; k < (unsigned)VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(ssm_out, k, silu_elem(vget(conv_out, k)));
        }
        X_ssm_out.write(ssm_out);
    }
}

// ============================================================
// stream vec_tuple W[8] (tile stream) definitions
// NOW: tile holds W_VEC (Q8VEC when quant enabled)
// ============================================================
static const int J_TILE = 8;

struct vec_tuple8 {
    W_VEC w[J_TILE];
};

// Produce W_delta tiles: order = (i=0..VEC_D-1), (jt=0..VEC_D-1 step 8)
static void stream_Wdelta_tiles_local(
        const W_VEC W_delta[VEC_D][VEC_D],
        hls::stream<vec_tuple8>& Wd_tiles) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_delta cyclic factor=8 dim=2

    for (int i = 0; i < VEC_D; ++i) {
        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                tup.w[jj] = W_delta[i][jt + jj];
            }
            Wd_tiles.write(tup);
        }
    }
}

// Produce W_B / W_C tiles: order = (i=0..N-1), (jt=0..VEC_D-1 step 8)
static void stream_WBWC_tiles_local(
        const W_VEC W_B[N][VEC_D],
        const W_VEC W_C[N][VEC_D],
        hls::stream<vec_tuple8>& WB_tiles,
        hls::stream<vec_tuple8>& WC_tiles) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_B cyclic factor=8 dim=2
#pragma HLS ARRAY_PARTITION variable=W_C cyclic factor=8 dim=2

    for (int i = 0; i < N; ++i) {
        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            vec_tuple8 tb, tc;
#pragma HLS ARRAY_PARTITION variable=tb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=tc.w complete dim=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                tb.w[jj] = W_B[i][jt + jj];
                tc.w[jj] = W_C[i][jt + jj];
            }
            WB_tiles.write(tb);
            WC_tiles.write(tc);
        }
    }
}

// ============================================================
// projections (delta) then (B,C) using tile streams W[8]
// NOW: weights are W_VEC (Q8VEC when quant enabled), scaled by macros.
// ============================================================
static void projection_streams_local(
        hls::stream<DTYPE_VEC>& X_ssm_in,
        hls::stream<vec_tuple8>& Wd_tiles,
        hls::stream<vec_tuple8>& WB_tiles,
        hls::stream<vec_tuple8>& WC_tiles,
        hls::stream<DTYPE_VEC>& B_out_N,
        hls::stream<DTYPE_VEC>& C_out_N,
        hls::stream<DTYPE_VEC>& delta_out) {
#pragma HLS INLINE off

    typedef ap_fixed<24, 8> PROJ_ACC_T;

    const float wscale_delta = (float)SSMU_W_SCALE_DELTA;
    const float wscale_bc    = (float)SSMU_W_SCALE_BC;

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    // ---- delta ----
    for (int i = 0; i < VEC_D; ++i) {
        PROJ_ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
#pragma HLS DEPENDENCE variable=acc inter false
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = (PROJ_ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }

            vec_tuple8 wd = Wd_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wd.w complete dim=1

            PROJ_ACC_T partial[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                partial[l] = (PROJ_ACC_T)0;
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                W_VEC w = wd.w[jj];
                for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    PROJ_ACC_T xv = (PROJ_ACC_T)(ACC_T)vget(X_tile[jj], l);
                    PROJ_ACC_T wv = (PROJ_ACC_T)(ACC_T)wget_scaled(w, l, wscale_delta);
                    partial[l] += xv * wv;
                }
            }

            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                acc[l] += partial[l];
            }
        }

        DTYPE_VEC delta_vec;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(delta_vec, l, softplus_pwl_fx((ACC_T)acc[l]));
        }
        delta_out.write(delta_vec);
    }

    // ---- B/C ----
    for (int i = 0; i < N; ++i) {
        PROJ_ACC_T accB[VEC_FACTOR];
        PROJ_ACC_T accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
#pragma HLS DEPENDENCE variable=accB inter false
#pragma HLS DEPENDENCE variable=accC inter false

        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accB[l] = (PROJ_ACC_T)0;
            accC[l] = (PROJ_ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }

            vec_tuple8 wb = WB_tiles.read();
            vec_tuple8 wc = WC_tiles.read();
#pragma HLS ARRAY_PARTITION variable=wb.w complete dim=1
#pragma HLS ARRAY_PARTITION variable=wc.w complete dim=1

            PROJ_ACC_T pB[VEC_FACTOR];
            PROJ_ACC_T pC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=pB complete dim=1
#pragma HLS ARRAY_PARTITION variable=pC complete dim=1
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                pB[l] = (PROJ_ACC_T)0;
                pC[l] = (PROJ_ACC_T)0;
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                W_VEC wB = wb.w[jj];
                W_VEC wC = wc.w[jj];
                for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    PROJ_ACC_T x  = (PROJ_ACC_T)(ACC_T)vget(X_tile[jj], l);
                    PROJ_ACC_T b  = (PROJ_ACC_T)(ACC_T)wget_scaled(wB, l, wscale_bc);
                    PROJ_ACC_T c  = (PROJ_ACC_T)(ACC_T)wget_scaled(wC, l, wscale_bc);
                    pB[l] += x * b;
                    pC[l] += x * c;
                }
            }

            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                accB[l] += pB[l];
                accC[l] += pC[l];
            }
        }

        DTYPE_VEC outB, outC;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outB, l, (DTYPE)(ACC_T)accB[l]);
            vset(outC, l, (DTYPE)(ACC_T)accC[l]);
        }
        B_out_N.write(outB);
        C_out_N.write(outC);
    }
}

// ============================================================
// FUSED stage (Mamba-aligned) (unchanged)
// ============================================================
static void fused_update_write_accum_output_mamba(
        hls::stream<DTYPE_VEC>& X_gate_in,
        hls::stream<DTYPE_VEC>& X_ssm_in,
        hls::stream<DTYPE_VEC>& delta_in,
        const DTYPE_VEC A_fixed[N],
        hls::stream<DTYPE_VEC>& B_in,
        hls::stream<DTYPE_VEC>& C_in,
        hls::stream<DTYPE_VEC>& H0_in,
        DTYPE_VEC* C_ddr,
        DTYPE_VEC* H1_ddr,
        hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off

    DTYPE_VEC X_gate[VEC_D];
#pragma HLS BIND_STORAGE variable=X_gate type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_gate[j] = X_gate_in.read();
    }

    DTYPE_VEC X_ssm_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_ssm_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_ssm_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        delta_buf[j] = delta_in.read();
    }

    DTYPE_VEC acc[VEC_D];
#pragma HLS BIND_STORAGE variable=acc type=ram_2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC z;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(z, l, (DTYPE)0);
        }
        acc[j] = z;
    }

    for (int i = 0; i < N; ++i) {
        DTYPE_VEC A_vec = A_fixed[i];
        DTYPE_VEC B_vec = B_in.read();
        DTYPE_VEC C_vec = C_in.read();

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            long long idx_ll = (long long)i * (long long)VEC_D + (long long)j;
            int idx = (int)idx_ll;

            DTYPE_VEC H0v  = H0_in.read();
            DTYPE_VEC dlt  = delta_buf[j];
            DTYPE_VEC xssm = X_ssm_buf[j];

            DTYPE_VEC H1v;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACT_T a   = (ACT_T)vget(A_vec, l);
                ACT_T dl  = (ACT_T)vget(dlt,  l);
                ACT_T adl = a * dl;
                adl = clamp_fx<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);
                EXP_T ddA_fx = exp2_poly_fx(adl);

                ACC_T H0  = (ACC_T)vget(H0v, l);
                ACC_T Bx  = (ACC_T)vget(B_vec, l);
                ACC_T Xs  = (ACC_T)vget(xssm, l);

                ACC_T H1  = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * Xs;
                vset(H1v, l, (DTYPE)H1);
            }

            // element-wise DDR store avoids class operator= issues
            DTYPE* cptr = (DTYPE*)(&C_ddr[idx]);
            DTYPE* hptr = (DTYPE*)(&H1_ddr[idx]);
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                cptr[l] = vget(C_vec, l);
                hptr[l] = vget(H1v,  l);
            }

            DTYPE_VEC aold = acc[j];
            DTYPE_VEC anew;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T base = (ACC_T)vget(aold, l);
                ACC_T addt = (ACC_T)vget(H1v,  l) * (ACC_T)vget(C_vec, l);
                vset(anew, l, (DTYPE)(base + addt));
            }
            acc[j] = anew;
        }
    }

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC yssm = acc[j];
        DTYPE_VEC gate = X_gate[j];
        DTYPE_VEC outv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T g = (ACC_T)vget(gate, l);
            ACC_T y = (ACC_T)vget(yssm, l);
            vset(outv, l, (DTYPE)(g * y));
        }
        out.write(outv);
    }
}

// ============================================================
// Output projection (out_proj): y = W_out * x
// NOW: W_out is W_VEC (Q8VEC when quant enabled)
// ============================================================
static void out_proj_stream_local(hls::stream<DTYPE_VEC>& X_in,
                                  const W_VEC W_out[VEC_D][VEC_D],
                                  hls::stream<DTYPE_VEC>& Y_out) {
#pragma HLS INLINE off

    const float wscale = (float)SSMU_W_SCALE_OUT;

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in.read();
    }

    for (int i = 0; i < VEC_D; ++i) {
        ACC_T accY[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accY complete dim=1
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accY[l] = (ACC_T)0;
        }

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC x = X_buf[j];
            W_VEC w = W_out[i][j];
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T xv = (ACC_T)vget(x, l);
                ACC_T wv = wget_scaled(w, l, wscale);
                accY[l] += xv * wv;
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
// TOP: Full Mamba Block (DATAFLOW + adapter copies + W[8] tile streams)
// NOW: weights are W_VEC (Q8VEC when quant enabled)
// ============================================================
void SSMU(hls::stream<DTYPE>& kernel_in,
          const DTYPE_VEC A_fixed[N],
          const DTYPE_VEC RMS_weight[VEC_D],

          // --- quantized weights (W_VEC) ---
          const W_VEC W_in_x[VEC_D][VEC_D],
          const W_VEC W_in_z[VEC_D][VEC_D],
          const W_VEC W_B[N][VEC_D],
          const W_VEC W_C[N][VEC_D],
          const W_VEC W_delta[VEC_D][VEC_D],
          const W_VEC W_out[VEC_D][VEC_D],

          // --- fp activations/params ---
          const DTYPE_VEC D_diag[VEC_D],

          hls::stream<DTYPE_VEC>& X_in,
          hls::stream<DTYPE_VEC>& H0_in,
          DTYPE_VEC* C_ddr,
          DTYPE_VEC* H1_ddr,
          hls::stream<DTYPE_VEC>& out) {

#pragma HLS INTERFACE ap_fifo port=kernel_in
#pragma HLS INTERFACE ap_fifo port=X_in
#pragma HLS INTERFACE ap_fifo port=H0_in
#pragma HLS INTERFACE ap_fifo port=out

#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // INTERNALIZED streams (legal to apply STREAM depth)
    hls::stream<DTYPE> kernel_local("kernel_local");
#pragma HLS STREAM variable=kernel_local depth=512

    hls::stream<DTYPE_VEC> X_local("X_local");
#pragma HLS STREAM variable=X_local depth=512

    hls::stream<DTYPE_VEC> H0_local("H0_local");
#pragma HLS STREAM variable=H0_local depth=512

    hls::stream<DTYPE_VEC> out_local("out_local");
#pragma HLS STREAM variable=out_local depth=512

    // W tile streams (// stream vec_tuple W[8])
    hls::stream<vec_tuple8> Wd_tiles("Wd_tiles");
    hls::stream<vec_tuple8> WB_tiles("WB_tiles");
    hls::stream<vec_tuple8> WC_tiles("WC_tiles");
#pragma HLS STREAM variable=Wd_tiles depth=64
#pragma HLS STREAM variable=WB_tiles depth=64
#pragma HLS STREAM variable=WC_tiles depth=64

    // ---- stage streams
    hls::stream<DTYPE_VEC> X_in_norm("X_in_norm");
    hls::stream<DTYPE_VEC> X_in_D   ("X_in_D");
    hls::stream<DTYPE_VEC> X_in_res ("X_in_res");
#pragma HLS STREAM variable=X_in_norm depth=512
#pragma HLS STREAM variable=X_in_D    depth=512
#pragma HLS STREAM variable=X_in_res  depth=512

    hls::stream<DTYPE_VEC> X_normed("X_normed");
#pragma HLS STREAM variable=X_normed depth=512

    hls::stream<DTYPE_VEC> X_for_conv_stream("X_for_conv_stream");
    hls::stream<DTYPE_VEC> Z_for_gate_stream("Z_for_gate_stream");
#pragma HLS STREAM variable=X_for_conv_stream depth=512
#pragma HLS STREAM variable=Z_for_gate_stream depth=512

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_upd_stream ("X_ssm_upd_stream");
#pragma HLS STREAM variable=X_gate_stream      depth=512
#pragma HLS STREAM variable=X_ssm_stream       depth=512
#pragma HLS STREAM variable=X_ssm_proj_stream  depth=512
#pragma HLS STREAM variable=X_ssm_upd_stream   depth=512

    hls::stream<DTYPE_VEC> B_stream_N("B_stream_N");
    hls::stream<DTYPE_VEC> C_stream_N("C_stream_N");
    hls::stream<DTYPE_VEC> delta_stream("delta_stream");
#pragma HLS STREAM variable=delta_stream depth=512
#pragma HLS STREAM variable=B_stream_N   depth=512
#pragma HLS STREAM variable=C_stream_N   depth=512

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
#pragma HLS STREAM variable=ssm_core_out_stream depth=512

    hls::stream<DTYPE_VEC> out_proj_stream("out_proj_stream");
#pragma HLS STREAM variable=out_proj_stream depth=512

    hls::stream<DTYPE_VEC> after_D_stream("after_D_stream");
#pragma HLS STREAM variable=after_D_stream depth=512

#pragma HLS DATAFLOW

    // ---- adapters: copy caller ports into internal streams
    copy_kernel_k(kernel_in, kernel_local);
    copy_vecd_n(X_in, X_local, VEC_D);
    copy_vecd_n(H0_in, H0_local, N * VEC_D);

    // ---- produce weight tiles (W[8]) into streams
    stream_Wdelta_tiles_local(W_delta, Wd_tiles);
    stream_WBWC_tiles_local(W_B, W_C, WB_tiles, WC_tiles);

    // ---- compute chain (all read internal streams only)
    tee_vecD_stream3_local(X_local, X_in_norm, X_in_D, X_in_res);
    rmsnorm_vecD_stream_local(X_in_norm, RMS_weight, X_normed);
    in_proj_stream_local(X_normed, W_in_x, W_in_z, X_for_conv_stream, Z_for_gate_stream);
    conv1d_silu_stream_local(X_for_conv_stream, Z_for_gate_stream, kernel_local, X_gate_stream, X_ssm_stream);
    dup_vecD_stream_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_upd_stream);

    // projection uses W[8] tile streams
    projection_streams_local(X_ssm_proj_stream, Wd_tiles, WB_tiles, WC_tiles,
                             B_stream_N, C_stream_N, delta_stream);

    fused_update_write_accum_output_mamba(
        X_gate_stream,
        X_ssm_upd_stream,
        delta_stream,
        A_fixed,
        B_stream_N,
        C_stream_N,
        H0_local,
        C_ddr,
        H1_ddr,
        ssm_core_out_stream
    );

    out_proj_stream_local(ssm_core_out_stream, W_out, out_proj_stream);
    add_D_skip_local(out_proj_stream, X_in_D, D_diag, after_D_stream);
    add_residual_local(after_D_stream, X_in_res, out_local);

    // ---- adapter: copy internal out back to top-level out port
    copy_vecd_n(out_local, out, VEC_D);
}
