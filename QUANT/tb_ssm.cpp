// tb_ssm.cpp  (FULL, quant-matched golden + matches latest DUT interfaces)
// NOTE: Include must match your DUT header name exactly: "SSMU.h"
#include "SSMU.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hls_stream.h>
#include <ap_fixed.h>
#include <ap_int.h>

// ============================================================
// TB-side typedefs & helpers (MATCH DUT)
// ============================================================
#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<32, 10> ACC_T;   // MATCH DUT
#else
typedef float ACC_T;
#endif

typedef ap_fixed<18, 6>  ACT_T;   // MATCH DUT
typedef ap_fixed<20, 8>  EXP_T;   // MATCH DUT

static inline DTYPE vget_u(const DTYPE_VEC &v, unsigned idx) { return v[idx]; }
static inline void  vset_u(DTYPE_VEC &v, unsigned idx, DTYPE val) { v[idx] = val; }

template<typename T>
static inline T clamp_fixed(T x, T lo, T hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// sigmoid(x) ≈ clamp(0.5 + x/4, 0, 1)  (MATCH DUT)
static inline ACT_T sigmoid_pwl_fx(ACT_T x) {
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fixed<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

// SiLU(x)=x*sigmoid(x)  (MATCH DUT)
static inline DTYPE silu_elem(DTYPE a) {
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_pwl_fx(x);
    ACT_T y = x * s;
    return (DTYPE)y;
}

// softplus PWL (MATCH DUT)
static inline DTYPE softplus_pwl_fx(ACC_T xin) {
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

// exp(t) approx: clamp + 2nd order poly in [-3,3] (MATCH DUT)
static inline EXP_T exp2_poly_fx(ACT_T t) {
    const ACT_T TH  = (ACT_T)3.0;
    const ACT_T NTH = (ACT_T)(-3.0);

    const EXP_T EXP3  = (EXP_T)20.0855369; // exp(3)
    const EXP_T EXPN3 = (EXP_T)0.0497871;  // exp(-3)

    if (t > TH)  return EXP3;
    if (t < NTH) return EXPN3;

    ap_fixed<24, 8> tt = (ap_fixed<24,8>)t * (ap_fixed<24,8>)t;
    ap_fixed<24, 8> y  = (ap_fixed<24,8>)1.0
                       + (ap_fixed<24,8>)t
                       + (ap_fixed<24,8>)0.5 * tt;

    if (y < 0) y = 0;
    return (EXP_T)y;
}

// ============================================================
// Quant insertion points (MATCH DUT defaults)
// ============================================================
#ifndef INS_Q_AFTER_RMSNORM
#define INS_Q_AFTER_RMSNORM 1
#endif
#ifndef INS_Q_AFTER_INPROJ
#define INS_Q_AFTER_INPROJ  1
#endif
#ifndef INS_Q_AFTER_CONV
#define INS_Q_AFTER_CONV    1
#endif

// ============================================================
// BFP Quant Toolkit (MATCH DUT)
// ============================================================
#ifndef Q_BITS
#define Q_BITS 8
#endif
#ifndef Q_S_MAX
#define Q_S_MAX 15
#endif

typedef ap_int<Q_BITS>  QINT_T;
typedef ap_uint<5>      QSCALE_T;

namespace bfpq {

static inline QINT_T clamp_qint(ap_int<32> x32) {
    const int QMAX_I = (int)((1u << (Q_BITS - 1)) - 1u);
    const int QMIN_I = (int)(-(1 << (Q_BITS - 1)));
    if (x32 > QMAX_I) return (QINT_T)QMAX_I;
    if (x32 < QMIN_I) return (QINT_T)QMIN_I;
    return (QINT_T)x32;
}

static inline ap_uint<32> abs_i32(ap_int<32> x) {
    ap_int<33> xe = (ap_int<33>)x;
    ap_int<33> me = (xe < 0) ? (ap_int<33>)(-xe) : xe;
    return (ap_uint<32>)me;
}

static inline ap_int<32> rshift_round_i32(ap_int<32> x, QSCALE_T s) {
    int si = (int)s;
    if (si <= 0) return x;

    ap_int<32> add = (ap_int<32>)1;
    add <<= (si - 1);

    ap_int<32> y = x;
    if (y >= 0) y = y + add;
    else        y = y - add;

    return (y >> si);
}

static inline QSCALE_T calc_shared_scale_u32(ap_uint<32> max_abs) {
    const ap_uint<32> QMAX_U = (ap_uint<32>)((1u << (Q_BITS - 1)) - 1u);

    QSCALE_T s = 0;
    ap_uint<32> v = max_abs;
    while (v > QMAX_U && s < (QSCALE_T)Q_S_MAX) {
        v >>= 1;
        s++;
    }
    return s;
}

static inline void quant_vec(const DTYPE_VEC &in, DTYPE_VEC &q_out, QSCALE_T &s_out) {
    ap_uint<32> max_abs = 0;

    for (int l = 0; l < VEC_FACTOR; ++l) {
        ap_int<32> v = (ap_int<32>)(ACC_T)vget_u(in, (unsigned)l);
        ap_uint<32> a = abs_i32(v);
        if (a > max_abs) max_abs = a;
    }

    QSCALE_T s = calc_shared_scale_u32(max_abs);
    s_out = s;

    for (int l = 0; l < VEC_FACTOR; ++l) {
        ap_int<32> v   = (ap_int<32>)(ACC_T)vget_u(in, (unsigned)l);
        ap_int<32> q32 = rshift_round_i32(v, s);
        QINT_T q = clamp_qint(q32);
        vset_u(q_out, (unsigned)l, (DTYPE)(ACC_T)q); // store q as DTYPE (MATCH DUT)
    }
}

static inline ACC_T deq_lane(DTYPE q, QSCALE_T s) {
    ap_int<32> qi = (ap_int<32>)(ACC_T)q;
    ap_int<32> xi = (s == 0) ? qi : (ap_int<32>)(qi << (int)s);
    return (ACC_T)xi;
}

} // namespace bfpq

// ============================================================
// Deterministic RNG
// ============================================================
static inline unsigned lcg_next(unsigned &s) {
    s = 1664525u * s + 1013904223u;
    return s;
}
static inline float frand(unsigned &s, float lo=-1.0f, float hi=1.0f) {
    unsigned r = lcg_next(s);
    float u = (float)(r & 0x00FFFFFFu) / (float)0x01000000u;
    return lo + (hi - lo) * u;
}
static inline DTYPE_VEC rand_vec(unsigned &seed, float lo=-1.0f, float hi=1.0f) {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
        vset_u(v, l, (DTYPE)frand(seed, lo, hi));
    return v;
}

static inline int check_close(const DTYPE_VEC &dut, const DTYPE_VEC &ref,
                              float tol, unsigned j) {
    int bad = 0;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        float dv = (float)(ACC_T)vget_u(dut, l);
        float rv = (float)(ACC_T)vget_u(ref, l);
        float err = std::fabs(dv - rv);
        if (err > tol) {
            std::printf("[TB] FAIL j=%u lane=%u dut=%f ref=%f err=%f tol=%f\n",
                        j, l, dv, rv, err, tol);
            bad = 1;
        }
    }
    return bad;
}

// ============================================================
// Golden model (MATCHES your posted quant.cpp behavior)
// IMPORTANT MATCH POINT:
// - After conv, DUT does drop_scale (scale thrown away). So later stages
//   use the quantized DTYPE values directly WITHOUT dequant.
// ============================================================
static void golden_model_quant_matched(
    const DTYPE kernel[K],
    const DTYPE_VEC X_in_raw[VEC_D],
    const DTYPE_VEC A_fixed[N],
    const DTYPE_VEC RMS_weight[VEC_D],
    const DTYPE_VEC H0_in[HUGE_LEN],
    const DTYPE_VEC W_in_x[VEC_D][VEC_D],
    const DTYPE_VEC W_in_z[VEC_D][VEC_D],
    const DTYPE_VEC W_B[N][VEC_D],
    const DTYPE_VEC W_C[N][VEC_D],
    const DTYPE_VEC W_delta[VEC_D][VEC_D],
    const DTYPE_VEC W_out[VEC_D][VEC_D],
    const DTYPE_VEC D_diag[VEC_D],
    DTYPE_VEC out_golden[VEC_D]
) {
    // ----------------------------------------------------------
    // 0) tee: X_in_norm, X_in_D, X_in_res are all the same
    // ----------------------------------------------------------
    static DTYPE_VEC X_in_norm[VEC_D];
    static DTYPE_VEC X_in_D   [VEC_D];
    static DTYPE_VEC X_in_res [VEC_D];
    for (int j=0; j<VEC_D; ++j) {
        X_in_norm[j] = X_in_raw[j];
        X_in_D[j]    = X_in_raw[j];
        X_in_res[j]  = X_in_raw[j];
    }

    // ----------------------------------------------------------
    // 1) RMSNorm
    // ----------------------------------------------------------
    static DTYPE_VEC X_normed[VEC_D];

    static DTYPE_VEC xbuf[VEC_D];
    for (int j=0; j<VEC_D; ++j) xbuf[j] = X_in_norm[j];

    ACC_T sumsq_lane[VEC_FACTOR];
    for (int l=0; l<VEC_FACTOR; ++l) sumsq_lane[l] = (ACC_T)0;

    for (int j=0; j<VEC_D; ++j) {
        DTYPE_VEC xv = xbuf[j];
        for (int l=0; l<VEC_FACTOR; ++l) {
            ACC_T v = (ACC_T)vget_u(xv, (unsigned)l);
            sumsq_lane[l] += v * v;
        }
    }

    ACC_T sumsq = 0;
    for (int l=0; l<VEC_FACTOR; ++l) sumsq += sumsq_lane[l];

    const float eps = 1e-5f;
    const float denom = (float)(VEC_D * VEC_FACTOR);
    float ms = (float)sumsq / denom;
    if (!(ms >= 0.0f)) ms = 0.0f;
    if (ms < 0.0f) ms = 0.0f;
    float inv = 1.0f / std::sqrt(ms + eps);

    for (int j=0; j<VEC_D; ++j) {
        DTYPE_VEC w = RMS_weight[j];
        DTYPE_VEC o;
        for (int l=0; l<VEC_FACTOR; ++l) {
            float xv = (float)(ACC_T)vget_u(xbuf[j], (unsigned)l);
            float ww = (float)(ACC_T)vget_u(w,     (unsigned)l);
            float yv = xv * inv * ww;
            vset_u(o, (unsigned)l, (DTYPE)yv);
        }
        X_normed[j] = o;
    }

    // ----------------------------------------------------------
    // 2) Optional quant after RMSNorm
    // ----------------------------------------------------------
    static DTYPE_VEC X_norm_q[VEC_D];
    static QSCALE_T  X_norm_s[VEC_D];

#if INS_Q_AFTER_RMSNORM
    for (int j=0; j<VEC_D; ++j) {
        DTYPE_VEC qv; QSCALE_T s;
        bfpq::quant_vec(X_normed[j], qv, s);
        X_norm_q[j] = qv;
        X_norm_s[j] = s;
    }
#else
    for (int j=0; j<VEC_D; ++j) {
        X_norm_q[j] = X_normed[j];
        X_norm_s[j] = (QSCALE_T)0;
    }
#endif

    // ----------------------------------------------------------
    // 3) in_proj_stream_local_q
    // ----------------------------------------------------------
    static DTYPE_VEC X_for_conv_q[VEC_D];
    static QSCALE_T  X_for_conv_s[VEC_D];
    static DTYPE_VEC Z_for_gate_q[VEC_D];
    static QSCALE_T  Z_for_gate_s[VEC_D];

    for (int i=0; i<VEC_D; ++i) {
        ACC_T accX[VEC_FACTOR];
        ACC_T accZ[VEC_FACTOR];
        for (int l=0; l<VEC_FACTOR; ++l) { accX[l]=0; accZ[l]=0; }

        for (int j=0; j<VEC_D; ++j) {
            DTYPE_VEC xq = X_norm_q[j];
            QSCALE_T  xs = X_norm_s[j];
            DTYPE_VEC wx = W_in_x[i][j];
            DTYPE_VEC wz = W_in_z[i][j];

            for (int l=0; l<VEC_FACTOR; ++l) {
                ACC_T xv = bfpq::deq_lane(vget_u(xq, (unsigned)l), xs);
                accX[l] += xv * (ACC_T)vget_u(wx, (unsigned)l);
                accZ[l] += xv * (ACC_T)vget_u(wz, (unsigned)l);
            }
        }

        DTYPE_VEC outX_raw, outZ_raw;
        for (int l=0; l<VEC_FACTOR; ++l) {
            vset_u(outX_raw, (unsigned)l, (DTYPE)accX[l]);
            vset_u(outZ_raw, (unsigned)l, (DTYPE)accZ[l]);
        }

#if INS_Q_AFTER_INPROJ
        DTYPE_VEC outX_q, outZ_q;
        QSCALE_T sx, sz;
        bfpq::quant_vec(outX_raw, outX_q, sx);
        bfpq::quant_vec(outZ_raw, outZ_q, sz);
        X_for_conv_q[i] = outX_q;  X_for_conv_s[i] = sx;
        Z_for_gate_q[i] = outZ_q;  Z_for_gate_s[i] = sz;
#else
        X_for_conv_q[i] = outX_raw; X_for_conv_s[i] = (QSCALE_T)0;
        Z_for_gate_q[i] = outZ_raw; Z_for_gate_s[i] = (QSCALE_T)0;
#endif
    }

    // ----------------------------------------------------------
    // 4) conv1d_silu_stream_local_q
    // ----------------------------------------------------------
    static DTYPE_VEC X_gate_q[VEC_D];
    static QSCALE_T  X_gate_s[VEC_D];
    static DTYPE_VEC X_ssm_q[VEC_D];
    static QSCALE_T  X_ssm_s[VEC_D];

    static DTYPE line_buffer[K-1][VEC_FACTOR];
    for (int kk=0; kk<K-1; ++kk)
        for (int lane=0; lane<VEC_FACTOR; ++lane)
            line_buffer[kk][lane] = (DTYPE)0;

    for (int i=0; i<VEC_D; ++i) {
        DTYPE_VEC xq = X_for_conv_q[i];
        QSCALE_T  xs = X_for_conv_s[i];
        DTYPE_VEC zq = Z_for_gate_q[i];
        QSCALE_T  zs = Z_for_gate_s[i];

        // gate = SiLU(dequant(z))
        DTYPE_VEC gate_raw;
        for (int k=0; k<VEC_FACTOR; ++k) {
            ACC_T z_deq = bfpq::deq_lane(vget_u(zq, (unsigned)k), zs);
            vset_u(gate_raw, (unsigned)k, silu_elem((DTYPE)z_deq));
        }

#if INS_Q_AFTER_CONV
        {
            DTYPE_VEC gate_q; QSCALE_T gs;
            bfpq::quant_vec(gate_raw, gate_q, gs);
            X_gate_q[i] = gate_q;
            X_gate_s[i] = gs;
        }
#else
        X_gate_q[i] = gate_raw;
        X_gate_s[i] = (QSCALE_T)0;
#endif

        // x_new (dequant)
        DTYPE x_new[VEC_FACTOR];
        for (int k=0; k<VEC_FACTOR; ++k) {
            ACC_T x_deq = bfpq::deq_lane(vget_u(xq, (unsigned)k), xs);
            x_new[k] = (DTYPE)x_deq;
        }

        // conv
        DTYPE_VEC conv_raw;
        for (int lane=0; lane<VEC_FACTOR; ++lane) {
            ACC_T sum = 0;
            for (int kk=0; kk<K-1; ++kk) sum += (ACC_T)kernel[kk] * (ACC_T)line_buffer[kk][lane];
            sum += (ACC_T)kernel[K-1] * (ACC_T)x_new[lane];
            vset_u(conv_raw, (unsigned)lane, (DTYPE)sum);
        }

        // shift
        for (int kk=K-2; kk>0; --kk)
            for (int lane=0; lane<VEC_FACTOR; ++lane)
                line_buffer[kk][lane] = line_buffer[kk-1][lane];
        for (int lane=0; lane<VEC_FACTOR; ++lane)
            line_buffer[0][lane] = x_new[lane];

        // ssm input = SiLU(conv)
        DTYPE_VEC ssm_raw;
        for (int k=0; k<VEC_FACTOR; ++k) {
            vset_u(ssm_raw, (unsigned)k, silu_elem(vget_u(conv_raw, (unsigned)k)));
        }

#if INS_Q_AFTER_CONV
        {
            DTYPE_VEC ssm_q; QSCALE_T ss;
            bfpq::quant_vec(ssm_raw, ssm_q, ss);
            X_ssm_q[i] = ssm_q;
            X_ssm_s[i] = ss;
        }
#else
        X_ssm_q[i] = ssm_raw;
        X_ssm_s[i] = (QSCALE_T)0;
#endif
    }

    // ----------------------------------------------------------
    // 5) drop_scale (MATCH DUT)
    // ----------------------------------------------------------
    static DTYPE_VEC X_gate_stream[VEC_D];
    static DTYPE_VEC X_ssm_stream [VEC_D];
    for (int j=0; j<VEC_D; ++j) {
        (void)X_gate_s[j];
        (void)X_ssm_s[j];
        X_gate_stream[j] = X_gate_q[j];
        X_ssm_stream[j]  = X_ssm_q[j];
    }

    // ----------------------------------------------------------
    // 6) projection_streams_local
    // ----------------------------------------------------------
    static DTYPE_VEC delta_buf[VEC_D];
    static DTYPE_VEC B_stream_N[N];
    static DTYPE_VEC C_stream_N[N];

    for (int i=0; i<VEC_D; ++i) {
        ACC_T acc[VEC_FACTOR];
        for (int l=0; l<VEC_FACTOR; ++l) acc[l]=0;

        for (int j=0; j<VEC_D; ++j) {
            DTYPE_VEC x = X_ssm_stream[j];
            DTYPE_VEC w = W_delta[i][j];
            for (int l=0; l<VEC_FACTOR; ++l) {
                acc[l] += (ACC_T)vget_u(x, (unsigned)l) * (ACC_T)vget_u(w, (unsigned)l);
            }
        }

        DTYPE_VEC dv;
        for (int l=0; l<VEC_FACTOR; ++l) vset_u(dv, (unsigned)l, softplus_pwl_fx(acc[l]));
        delta_buf[i] = dv;
    }

    for (int i=0; i<N; ++i) {
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
        for (int l=0; l<VEC_FACTOR; ++l) { accB[l]=0; accC[l]=0; }

        for (int j=0; j<VEC_D; ++j) {
            DTYPE_VEC x  = X_ssm_stream[j];
            DTYPE_VEC wB = W_B[i][j];
            DTYPE_VEC wC = W_C[i][j];
            for (int l=0; l<VEC_FACTOR; ++l) {
                ACC_T xv = (ACC_T)vget_u(x,  (unsigned)l);
                accB[l] += xv * (ACC_T)vget_u(wB, (unsigned)l);
                accC[l] += xv * (ACC_T)vget_u(wC, (unsigned)l);
            }
        }

        DTYPE_VEC outB, outC;
        for (int l=0; l<VEC_FACTOR; ++l) {
            vset_u(outB, (unsigned)l, (DTYPE)accB[l]);
            vset_u(outC, (unsigned)l, (DTYPE)accC[l]);
        }
        B_stream_N[i] = outB;
        C_stream_N[i] = outC;
    }

    // ----------------------------------------------------------
    // 7) fused_update_write_accum_output_mamba (MATCH DUT CAST POINT)
    //    ✅ FIXED: match DUT doing (DTYPE)H1 BEFORE multiply+accumulate
    // ----------------------------------------------------------
    static ACC_T acc_s[VEC_D][VEC_FACTOR];
    for (int j=0; j<VEC_D; ++j)
        for (int l=0; l<VEC_FACTOR; ++l)
            acc_s[j][l] = (ACC_T)0;

    for (int i=0; i<N; ++i) {
        DTYPE_VEC A_vec = A_fixed[i];
        DTYPE_VEC B_vec = B_stream_N[i];
        DTYPE_VEC C_vec = C_stream_N[i];

        for (int j=0; j<VEC_D; ++j) {
            unsigned idx = (unsigned)i * (unsigned)VEC_D + (unsigned)j;
            DTYPE_VEC H0v  = H0_in[idx];
            DTYPE_VEC dlt  = delta_buf[j];
            DTYPE_VEC xssm = X_ssm_stream[j];

            for (int l=0; l<VEC_FACTOR; ++l) {
                ACT_T a   = (ACT_T)vget_u(A_vec, (unsigned)l);
                ACT_T dl  = (ACT_T)vget_u(dlt,   (unsigned)l);
                ACT_T adl = a * dl;
                adl = clamp_fixed<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);
                EXP_T ddA_fx = exp2_poly_fx(adl);

                ACC_T H0  = (ACC_T)vget_u(H0v,  (unsigned)l);
                ACC_T Bx  = (ACC_T)vget_u(B_vec,(unsigned)l);
                ACC_T Xs  = (ACC_T)vget_u(xssm, (unsigned)l);

                ACC_T H1  = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * Xs;

                // MATCH DUT:
                //   DTYPE h1d = (DTYPE)H1;
                //   acc += (ACC_T)h1d * c;
                DTYPE h1d = (DTYPE)H1;
                ACC_T c   = (ACC_T)vget_u(C_vec, (unsigned)l);

                acc_s[j][l] += (ACC_T)h1d * c;
            }
        }
    }

    static DTYPE_VEC ssm_core_out[VEC_D];
    for (int j=0; j<VEC_D; ++j) {
        DTYPE_VEC gate = X_gate_stream[j]; // quantized value, no dequant (matches DUT)
        DTYPE_VEC outv;
        for (int l=0; l<VEC_FACTOR; ++l) {
            ACC_T g = (ACC_T)vget_u(gate, (unsigned)l);
            ACC_T y = acc_s[j][l];
            vset_u(outv, (unsigned)l, (DTYPE)(g * y));
        }
        ssm_core_out[j] = outv;
    }

    // ----------------------------------------------------------
    // 8) out_proj_stream_local
    // ----------------------------------------------------------
    static DTYPE_VEC out_proj[VEC_D];
    for (int i=0; i<VEC_D; ++i) {
        ACC_T accY[VEC_FACTOR];
        for (int l=0; l<VEC_FACTOR; ++l) accY[l]=0;

        for (int j=0; j<VEC_D; ++j) {
            DTYPE_VEC x = ssm_core_out[j];
            DTYPE_VEC w = W_out[i][j];
            for (int l=0; l<VEC_FACTOR; ++l) {
                accY[l] += (ACC_T)vget_u(x, (unsigned)l) * (ACC_T)vget_u(w, (unsigned)l);
            }
        }

        DTYPE_VEC y;
        for (int l=0; l<VEC_FACTOR; ++l) vset_u(y, (unsigned)l, (DTYPE)accY[l]);
        out_proj[i] = y;
    }

    // ----------------------------------------------------------
    // 9) add_D_skip_local : y + D_diag * X_in_D
    // ----------------------------------------------------------
    static DTYPE_VEC after_D[VEC_D];
    for (int j=0; j<VEC_D; ++j) {
        DTYPE_VEC y = out_proj[j];
        DTYPE_VEC x = X_in_D[j];
        DTYPE_VEC d = D_diag[j];
        DTYPE_VEC o;
        for (int l=0; l<VEC_FACTOR; ++l) {
            ACC_T yy = (ACC_T)vget_u(y, (unsigned)l);
            ACC_T xx = (ACC_T)vget_u(x, (unsigned)l);
            ACC_T dd = (ACC_T)vget_u(d, (unsigned)l);
            vset_u(o, (unsigned)l, (DTYPE)(yy + dd * xx));
        }
        after_D[j] = o;
    }

    // ----------------------------------------------------------
    // 10) add_residual_local : after_D + X_in_res
    // ----------------------------------------------------------
    for (int j=0; j<VEC_D; ++j) {
        DTYPE_VEC y = after_D[j];
        DTYPE_VEC x = X_in_res[j];
        DTYPE_VEC o;
        for (int l=0; l<VEC_FACTOR; ++l) {
            ACC_T yy = (ACC_T)vget_u(y, (unsigned)l);
            ACC_T xx = (ACC_T)vget_u(x, (unsigned)l);
            vset_u(o, (unsigned)l, (DTYPE)(yy + xx));
        }
        out_golden[j] = o;
    }
}

// ============================================================
// main()
// ============================================================
int main() {
    std::printf("[TB] tb_ssm.cpp (quant-matched golden)\n");

    const int   CASES = 1;
    const float tol   = 0.10f;

    // ✅ MUST match SSMU.h top: DTYPE_VEC* (NOT volatile)
    static DTYPE_VEC C_ddr[HUGE_LEN];
    static DTYPE_VEC H1_ddr[HUGE_LEN];

    for (int tc=0; tc<CASES; ++tc) {
        unsigned seed = 1u + (unsigned)tc;

        // ---- input vectors ----
        static DTYPE_VEC X_in_raw[VEC_D];
        for (unsigned j=0; j<(unsigned)VEC_D; ++j)
            X_in_raw[j] = rand_vec(seed, -1.0f, 1.0f);

        // ---- kernel array + stream ----
        DTYPE kernel[K];
        for (unsigned kk=0; kk<(unsigned)K; ++kk)
            kernel[kk] = (DTYPE)frand(seed, -0.5f, 0.5f);

        hls::stream<DTYPE> kernel_in;
        for (unsigned kk=0; kk<(unsigned)K; ++kk)
            kernel_in.write(kernel[kk]);

        // ---- weights ----
        static DTYPE_VEC A_fixed[N];
        static DTYPE_VEC RMS_weight[VEC_D];
        static DTYPE_VEC W_in_x[VEC_D][VEC_D];
        static DTYPE_VEC W_in_z[VEC_D][VEC_D];
        static DTYPE_VEC W_B[N][VEC_D];
        static DTYPE_VEC W_C[N][VEC_D];
        static DTYPE_VEC W_delta[VEC_D][VEC_D];
        static DTYPE_VEC W_out[VEC_D][VEC_D];
        static DTYPE_VEC D_diag[VEC_D];

        for (unsigned i=0; i<(unsigned)N; ++i) A_fixed[i] = rand_vec(seed, -0.5f, 0.5f);
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) RMS_weight[j] = rand_vec(seed, 0.5f, 1.5f);
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) D_diag[j] = rand_vec(seed, -0.2f, 0.2f);

        for (unsigned i=0; i<(unsigned)VEC_D; ++i)
            for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
                W_in_x[i][j]  = rand_vec(seed, -0.5f, 0.5f);
                W_in_z[i][j]  = rand_vec(seed, -0.5f, 0.5f);
                W_delta[i][j] = rand_vec(seed, -0.5f, 0.5f);
                W_out[i][j]   = rand_vec(seed, -0.5f, 0.5f);
            }

        for (unsigned i=0; i<(unsigned)N; ++i)
            for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
                W_B[i][j] = rand_vec(seed, -0.5f, 0.5f);
                W_C[i][j] = rand_vec(seed, -0.5f, 0.5f);
            }

        // ---- H0 stream + mirror array for golden ----
        static DTYPE_VEC H0_in_arr[HUGE_LEN];
        hls::stream<DTYPE_VEC> H0_in_stream;
        for (unsigned idx=0; idx<(unsigned)HUGE_LEN; ++idx) {
            DTYPE_VEC v = rand_vec(seed, -0.2f, 0.2f);
            H0_in_arr[idx] = v;
            H0_in_stream.write(v);
        }

        // ---- X stream ----
        hls::stream<DTYPE_VEC> X_in_stream;
        for (unsigned j=0; j<(unsigned)VEC_D; ++j)
            X_in_stream.write(X_in_raw[j]);

        // ---- output stream ----
        hls::stream<DTYPE_VEC> out_stream;

        // ---- golden (quant matched) ----
        static DTYPE_VEC golden[VEC_D];
        golden_model_quant_matched(kernel, X_in_raw, A_fixed, RMS_weight, H0_in_arr,
                                   W_in_x, W_in_z, W_B, W_C, W_delta, W_out, D_diag,
                                   golden);

        // ---- DUT call ----
        SSMU(kernel_in, A_fixed, RMS_weight,
             W_in_x, W_in_z, W_B, W_C, W_delta, W_out, D_diag,
             X_in_stream, H0_in_stream,
             C_ddr, H1_ddr,
             out_stream);

        // ---- compare ----
        int fail = 0;
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            if (out_stream.empty()) {
                std::printf("[TB] FAIL: DUT out_stream empty at j=%u\n", j);
                fail = 1;
                break;
            }
            DTYPE_VEC dut = out_stream.read();
            fail |= check_close(dut, golden[j], tol, j);
            if (fail) break;
        }

        if (fail) {
            std::printf("[TB] FAIL (tc=%d)\n", tc);
            return 1;
        } else {
            std::printf("[TB] PASS (tc=%d)\n", tc);
        }
    }

    return 0;
}
