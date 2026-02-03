// tb_ssm.cpp  (INT8-W_VEC aware testbench) - UPDATED for current SSMU interface
// Matches ssm.cpp (Version C):
// - conv shift behavior fixed (shift once per tile, not per lane)
// - exp uses exp_poly_fx (e^t poly), aligned with DUT
//
// IMPORTANT FIX (NEW):
// - Golden now emulates DUT quantization of dA:
//     stage3 stores (DTYPE)e into stream,
//     stage45 reads it back as ACC_T.
//   => golden now does: EXP_T -> (DTYPE) -> (ACC_T) before using ddA.
//
// IMPORTANT FIX (existing):
// - TB scale defaults now follow DUT rules:
//     * SSMU_USE_INT8==1 => default 1/128
//     * else             => default 1.0f
// - Added sanity prints + static_asserts to catch TB/DUT mismatch early.
// - Added output count checks (detect extra/missing outputs).
// - Added compile-time macro dumps (helps catch hidden -D overrides).

#include "SSMU.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

// ============================================================
// stringify helper for compile-time macro dump
// ============================================================
#define TB_STR_IMPL(x) #x
#define TB_STR(x) TB_STR_IMPL(x)

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
// scales MUST match DUT (conditional defaults)
// NOTE: if any -DSSMU_W_SCALE_* is passed in build flags,
//       these #ifndef blocks will NOT override them.
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
// Local vget/vset (TB-side; don't depend on DUT helpers)
// ============================================================
static inline DTYPE vget_u(const DTYPE_VEC &v, unsigned idx) { return v[idx]; }
static inline void  vset_u(DTYPE_VEC &v, unsigned idx, DTYPE val) { v[idx] = val; }

#if SSMU_USE_INT8
static inline ap_int<8> wraw_get_u(const W_VEC &w, unsigned idx) { return w[idx]; }
static inline ACC_T wget_scaled_u(const W_VEC &w, unsigned idx, ACC_T scale_fx) {
    ACC_T wi = (ACC_T)((int)wraw_get_u(w, idx));
    return wi * scale_fx;
}
#else
static inline DTYPE wraw_get_u(const W_VEC &w, unsigned idx) { return w[idx]; }
static inline ACC_T wget_scaled_u(const W_VEC &w, unsigned idx, ACC_T /*scale_fx*/) {
    return (ACC_T)wraw_get_u(w, idx);
}
#endif

template<typename T>
static inline T clamp_fixed(T x, T lo, T hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static inline ACT_T sigmoid_pwl_fx(ACT_T x) {
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fixed<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

static inline DTYPE silu_elem(DTYPE a) {
    ACT_T x = (ACT_T)a;
    ACT_T s = sigmoid_pwl_fx(x);
    return (DTYPE)(x * s);
}

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

// exp approx for e^t (NOT exp2): clamp then 2nd-order poly
static inline EXP_T exp_poly_fx(ACT_T t) {
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

static inline DTYPE_VEC rand_dvec(unsigned &seed, float lo=-1.0f, float hi=1.0f) {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        vset_u(v, l, (DTYPE)frand(seed, lo, hi));
    }
    return v;
}

static inline W_VEC rand_wvec(unsigned &seed, float lo=-1.0f, float hi=1.0f) {
    W_VEC w;
#if SSMU_USE_INT8
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        float x = frand(seed, lo, hi);
        int qi = (int)std::lrintf(x * 127.0f);
        if (qi > 127) qi = 127;
        if (qi < -127) qi = -127;
        w[l] = (ap_int<8>)qi;
    }
#else
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        w[l] = (DTYPE)frand(seed, lo, hi);
    }
#endif
    return w;
}

// ============================================================
// Golden model (MATCHES CURRENT DUT)
// ============================================================
static void golden_model(
    const DTYPE kernel[SSMU_K],
    const DTYPE_VEC X_in_raw[SSMU_D_T],
    const DTYPE_VEC A_fixed[SSMU_STATE],
    const DTYPE_VEC RMS_weight[SSMU_D_T],
    const DTYPE_VEC H0_in[HUGE_LEN],

    const W_VEC W_inproj[SSMU_D_T][SSMU_CIN_T],

    const W_VEC W_B[SSMU_STATE][SSMU_C2_T],
    const W_VEC W_C[SSMU_STATE][SSMU_C2_T],
    const W_VEC W_delta[SSMU_C2_T][SSMU_C2_T],
    const W_VEC W_out[SSMU_D_T][SSMU_C2_T],

    const DTYPE_VEC D_diag[SSMU_C2_T],

    DTYPE_VEC out_golden[SSMU_D_T]
) {
    const float eps = 1e-5f;

    // ----------------------------
    // RMSNorm stats on D-domain
    // ----------------------------
    ACC_T sumsq = 0;
    for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j) {
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T v = (ACC_T)vget_u(X_in_raw[j], l);
            sumsq += v * v;
        }
    }
    float ms = (float)sumsq / (float)(SSMU_D_T * VEC_FACTOR);
    if (!(ms >= 0.0f)) ms = 0.0f;
    if (ms < 0.0f) ms = 0.0f;
    float inv = 1.0f / std::sqrt(ms + eps);

    // RMSNorm apply
    DTYPE_VEC X_normed[SSMU_D_T];
    for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j) {
        DTYPE_VEC o;
        DTYPE_VEC w = RMS_weight[j];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float xv = (float)(ACC_T)vget_u(X_in_raw[j], l);
            float ww = (float)(ACC_T)vget_u(w, l);
            vset_u(o, l, (DTYPE)(xv * inv * ww));
        }
        X_normed[j] = o;
    }

    // ----------------------------
    // in_proj packed => Z, XBC, DT
    // weight layout: W_inproj[j][i]
    // ----------------------------
    const ACC_T wscale_in = (ACC_T)SSMU_W_SCALE_IN;

    DTYPE_VEC Z_for_gate[SSMU_C2_T];
    DTYPE_VEC XBC_all[SSMU_CCONV_T];

    for (unsigned i=0; i<(unsigned)SSMU_CIN_T; ++i) {
        ACC_T acc[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) acc[l] = 0;

        for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j) {
            DTYPE_VEC x = X_normed[j];
            W_VEC w = W_inproj[j][i];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACC_T xv = (ACC_T)vget_u(x, l);
                acc[l] += xv * wget_scaled_u(w, l, wscale_in);
            }
        }

        DTYPE_VEC outv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset_u(outv, l, (DTYPE)acc[l]);

        if (i < (unsigned)SSMU_C2_T) {
            Z_for_gate[i] = outv;
        } else if (i < (unsigned)(SSMU_C2_T + SSMU_CCONV_T)) {
            XBC_all[i - SSMU_C2_T] = outv;
        } else {
            // DT part ignored (DUT drains)
        }
    }

    // ----------------------------
    // gate = silu(Z)
    // ----------------------------
    DTYPE_VEC X_gate[SSMU_C2_T];
    for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
        DTYPE_VEC gv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            vset_u(gv, l, silu_elem(vget_u(Z_for_gate[j], l)));
        }
        X_gate[j] = gv;
    }

    // ----------------------------
    // conv1d + silu on XBC_all
    // shift line_buffer ONCE per tile (matches DUT)
    // ----------------------------
    DTYPE lb[SSMU_K-1][VEC_FACTOR];
    for (unsigned t=0; t<(unsigned)(SSMU_K-1); ++t)
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
            lb[t][l] = (DTYPE)0;

    DTYPE_VEC X_ssm[SSMU_C2_T];

    for (unsigned i=0; i<(unsigned)SSMU_CCONV_T; ++i) {
        bool do_c2 = (i < (unsigned)SSMU_C2_T);
        const DTYPE_VEC x_in = XBC_all[i];

        DTYPE w0[VEC_FACTOR], w1[VEC_FACTOR], w2[VEC_FACTOR], w3[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            w0[l] = lb[2][l];
            w1[l] = lb[1][l];
            w2[l] = lb[0][l];
            w3[l] = vget_u(x_in, l);
        }

        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            lb[2][l] = lb[1][l];
            lb[1][l] = lb[0][l];
            lb[0][l] = w3[l];
        }

        if (do_c2) {
            DTYPE_VEC ssmv;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACC_T sum = 0;
                sum += (ACC_T)kernel[0] * (ACC_T)w0[l];
                sum += (ACC_T)kernel[1] * (ACC_T)w1[l];
                sum += (ACC_T)kernel[2] * (ACC_T)w2[l];
                sum += (ACC_T)kernel[3] * (ACC_T)w3[l];
                vset_u(ssmv, l, silu_elem((DTYPE)sum));
            }
            X_ssm[i] = ssmv;
        }
    }

    // ----------------------------
    // delta proj
    // ----------------------------
    const ACC_T wscale_delta = (ACC_T)SSMU_W_SCALE_DELTA;
    DTYPE_VEC delta[SSMU_C2_T];
    for (unsigned d=0; d<(unsigned)SSMU_C2_T; ++d) {
        ACC_T acc[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) acc[l]=0;

        for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
            DTYPE_VEC x = X_ssm[j];
            W_VEC w = W_delta[d][j];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                acc[l] += (ACC_T)vget_u(x,l) * wget_scaled_u(w,l,wscale_delta);
            }
        }

        DTYPE_VEC dv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
            vset_u(dv, l, softplus_pwl_fx(acc[l]));
        delta[d] = dv;
    }

    // ----------------------------
    // B/C proj
    // ----------------------------
    const ACC_T wscale_bc = (ACC_T)SSMU_W_SCALE_BC;
    DTYPE_VEC Bv[SSMU_STATE];
    DTYPE_VEC Cv[SSMU_STATE];

    for (unsigned i=0; i<(unsigned)SSMU_STATE; ++i) {
        ACC_T accB[VEC_FACTOR], accC[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) { accB[l]=0; accC[l]=0; }

        for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
            DTYPE_VEC x = X_ssm[j];
            W_VEC wB = W_B[i][j];
            W_VEC wC = W_C[i][j];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACC_T xv = (ACC_T)vget_u(x,l);
                accB[l] += xv * wget_scaled_u(wB,l,wscale_bc);
                accC[l] += xv * wget_scaled_u(wC,l,wscale_bc);
            }
        }

        DTYPE_VEC ob, oc;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            vset_u(ob, l, (DTYPE)accB[l]);
            vset_u(oc, l, (DTYPE)accC[l]);
        }
        Bv[i]=ob; Cv[i]=oc;
    }

    // ----------------------------
    // fused core accumulation
    // ----------------------------
    DTYPE_VEC acc_out[SSMU_C2_T];
    for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
        DTYPE_VEC z; for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset_u(z,l,(DTYPE)0);
        acc_out[j]=z;
    }

    for (unsigned i=0; i<(unsigned)SSMU_STATE; ++i) {
        const DTYPE_VEC B_vec = Bv[i];
        const DTYPE_VEC C_vec = Cv[i];
        const DTYPE_VEC A_vec = A_fixed[i];

        for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
            unsigned idx = i*(unsigned)SSMU_C2_T + j;
            const DTYPE_VEC H0v = H0_in[idx];
            const DTYPE_VEC dlt = delta[j];
            const DTYPE_VEC dx  = X_ssm[j];

            DTYPE_VEC H1v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACT_T a  = (ACT_T)vget_u(A_vec, l);
                ACT_T dl = (ACT_T)vget_u(dlt,  l);

                ACT_T adl = a * dl;
                adl = clamp_fixed<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);

                // ✅ emulate DUT quantization boundary:
                // stage3: EXP_T -> DTYPE
                // stage45: DTYPE -> ACC_T
                EXP_T  ddA_fx = exp_poly_fx(adl);
                DTYPE  ddA_dt = (DTYPE)ddA_fx;
                ACC_T  ddA_q  = (ACC_T)ddA_dt;

                ACC_T H0  = (ACC_T)vget_u(H0v, l);
                ACC_T Bx  = (ACC_T)vget_u(B_vec, l);
                ACC_T Xs  = (ACC_T)vget_u(dx,   l);

                ACC_T H1 = H0 * ddA_q + (Bx * (ACC_T)dl) * Xs;
                vset_u(H1v, l, (DTYPE)H1);
            }

            DTYPE_VEC aold = acc_out[j];
            DTYPE_VEC anew;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACC_T base = (ACC_T)vget_u(aold, l);
                ACC_T addt = (ACC_T)vget_u(H1v,  l) * (ACC_T)vget_u(C_vec, l);
                vset_u(anew, l, (DTYPE)(base + addt));
            }
            acc_out[j] = anew;
        }
    }

    // ----------------------------
    // core output before out_proj
    // yz = (acc_out + D_diag * X_ssm) * X_gate
    // ----------------------------
    DTYPE_VEC ssm_core_out[SSMU_C2_T];
    for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
        DTYPE_VEC ov;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T htC = (ACC_T)vget_u(acc_out[j], l);
            ACC_T x   = (ACC_T)vget_u(X_ssm[j],  l);
            ACC_T d   = (ACC_T)vget_u(D_diag[j], l);
            ACC_T z   = (ACC_T)vget_u(X_gate[j], l);

            ACC_T y  = htC + d * x;
            ACC_T yz = y * z;
            vset_u(ov, l, (DTYPE)yz);
        }
        ssm_core_out[j] = ov;
    }

    // ----------------------------
    // out_proj: C2 -> D
    // ----------------------------
    const ACC_T wscale_out = (ACC_T)SSMU_W_SCALE_OUT;
    DTYPE_VEC y_outproj[SSMU_D_T];

    for (unsigned i=0; i<(unsigned)SSMU_D_T; ++i) {
        ACC_T accY[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) accY[l]=0;

        for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
            DTYPE_VEC x = ssm_core_out[j];
            W_VEC w = W_out[i][j];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                accY[l] += (ACC_T)vget_u(x,l) * wget_scaled_u(w,l,wscale_out);
            }
        }

        DTYPE_VEC yv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset_u(yv,l,(DTYPE)accY[l]);
        y_outproj[i] = yv;
    }

    // ----------------------------
    // residual add on D-domain
    // ----------------------------
    for (unsigned i=0; i<(unsigned)SSMU_D_T; ++i) {
        DTYPE_VEC yv   = y_outproj[i];
        DTYPE_VEC xraw = X_in_raw[i];
        DTYPE_VEC ov;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T yy = (ACC_T)vget_u(yv,   l);
            ACC_T xx = (ACC_T)vget_u(xraw, l);
            vset_u(ov, l, (DTYPE)(yy + xx));
        }
        out_golden[i] = ov;
    }
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

int main() {
    // ----------------------------
    // Sanity checks (catch TB/DUT mismatch early)
    // ----------------------------
    static_assert(sizeof(DTYPE_VEC) == (size_t)VEC_FACTOR * sizeof(DTYPE),
                  "DTYPE_VEC is not VEC_FACTOR * sizeof(DTYPE)");
#if SSMU_USE_INT8
    static_assert(sizeof(W_VEC) == (size_t)VEC_FACTOR * sizeof(ap_int<8>),
                  "W_VEC size mismatch for INT8 path");
#else
    static_assert(sizeof(W_VEC) == (size_t)VEC_FACTOR * sizeof(DTYPE),
                  "W_VEC size mismatch for non-INT8 path");
#endif

#ifndef __SYNTHESIS__
    // Compile-time macro dump (helps catch hidden -D overrides)
    std::printf("[TB] (compile-time macro dump)\n");
    std::printf("     SSMU_USE_INT8=%s\n", TB_STR(SSMU_USE_INT8));
    std::printf("     SSMU_W_SCALE_IN=%s\n", TB_STR(SSMU_W_SCALE_IN));
    std::printf("     SSMU_W_SCALE_BC=%s\n", TB_STR(SSMU_W_SCALE_BC));
    std::printf("     SSMU_W_SCALE_DELTA=%s\n", TB_STR(SSMU_W_SCALE_DELTA));
    std::printf("     SSMU_W_SCALE_OUT=%s\n", TB_STR(SSMU_W_SCALE_OUT));
#endif

    std::printf("[TB] tb_ssm.cpp (UPDATED for Version C) linked OK (main() present).\n");

    std::printf("[TB] SSMU_USE_INT8=%d\n", (int)SSMU_USE_INT8);
    std::printf("[TB] W_SCALE_IN=%f BC=%f DELTA=%f OUT=%f\n",
                (float)SSMU_W_SCALE_IN, (float)SSMU_W_SCALE_BC,
                (float)SSMU_W_SCALE_DELTA, (float)SSMU_W_SCALE_OUT);

    std::printf("[TB] D=%d N=%d CP=%d K=%d\n", (int)SSMU_D, (int)SSMU_N, (int)SSMU_CP, (int)SSMU_K);
    std::printf("[TB] D_T=%d C2_T=%d Cconv_T=%d CIN_T=%d CH_T=%d STATE=%d HUGE_LEN=%d\n",
                (int)SSMU_D_T, (int)SSMU_C2_T, (int)SSMU_CCONV_T, (int)SSMU_CIN_T, (int)SSMU_CH_T,
                (int)SSMU_STATE, (int)HUGE_LEN);

    std::printf("[TB] sizeof(DTYPE)=%zu sizeof(DTYPE_VEC)=%zu sizeof(W_VEC)=%zu\n",
                sizeof(DTYPE), sizeof(DTYPE_VEC), sizeof(W_VEC));

    const int   CASES = 1;
    const float tol   = 0.05f;

    static DTYPE_VEC C_ddr[HUGE_LEN];
    static DTYPE_VEC H1_ddr[HUGE_LEN];

    for (int tc=0; tc<CASES; ++tc) {
        unsigned seed = 1u + (unsigned)tc;

        // input vectors: D-domain tiles (D_T)
        DTYPE_VEC X_in_raw[SSMU_D_T];
        for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j)
            X_in_raw[j] = rand_dvec(seed, -1.0f, 1.0f);

        // kernel
        DTYPE kernel[SSMU_K];
        for (unsigned kk=0; kk<(unsigned)SSMU_K; ++kk)
            kernel[kk] = (DTYPE)frand(seed, -0.5f, 0.5f);

        hls::stream<DTYPE> kernel_in;
        for (unsigned kk=0; kk<(unsigned)SSMU_K; ++kk) kernel_in.write(kernel[kk]);

        // params
        static DTYPE_VEC A_fixed[SSMU_STATE];
        static DTYPE_VEC RMS_weight[SSMU_D_T];
        static DTYPE_VEC D_diag[SSMU_C2_T];

        for (unsigned i=0; i<(unsigned)SSMU_STATE; ++i) A_fixed[i] = rand_dvec(seed, -0.5f, 0.5f);
        for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j)  RMS_weight[j] = rand_dvec(seed, 0.5f, 1.5f);
        for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) D_diag[j]     = rand_dvec(seed, -0.2f, 0.2f);

        // weights
        static W_VEC W_inproj[SSMU_D_T][SSMU_CIN_T];
        static W_VEC W_delta[SSMU_C2_T][SSMU_C2_T];
        static W_VEC W_out[SSMU_D_T][SSMU_C2_T];
        static W_VEC W_B[SSMU_STATE][SSMU_C2_T];
        static W_VEC W_C[SSMU_STATE][SSMU_C2_T];

        for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j) {
            for (unsigned i=0; i<(unsigned)SSMU_CIN_T; ++i) {
                W_inproj[j][i] = rand_wvec(seed, -1.0f, 1.0f);
            }
        }

        for (unsigned i=0; i<(unsigned)SSMU_C2_T; ++i) {
            for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
                W_delta[i][j] = rand_wvec(seed, -1.0f, 1.0f);
            }
        }

        for (unsigned i=0; i<(unsigned)SSMU_D_T; ++i) {
            for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
                W_out[i][j] = rand_wvec(seed, -1.0f, 1.0f);
            }
        }

        for (unsigned i=0; i<(unsigned)SSMU_STATE; ++i) {
            for (unsigned j=0; j<(unsigned)SSMU_C2_T; ++j) {
                W_B[i][j] = rand_wvec(seed, -1.0f, 1.0f);
                W_C[i][j] = rand_wvec(seed, -1.0f, 1.0f);
            }
        }

        // H0 stream (HUGE_LEN = STATE*C2_T)
        static DTYPE_VEC H0_in_arr[HUGE_LEN];
        hls::stream<DTYPE_VEC> H0_in_stream;
        for (unsigned idx=0; idx<(unsigned)HUGE_LEN; ++idx) {
            DTYPE_VEC v = rand_dvec(seed, -0.2f, 0.2f);
            H0_in_arr[idx] = v;
            H0_in_stream.write(v);
        }

        // X stream: D_T tiles
        hls::stream<DTYPE_VEC> X_in_stream;
        for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j) X_in_stream.write(X_in_raw[j]);

        // out stream
        hls::stream<DTYPE_VEC> out_stream;

        // golden
        DTYPE_VEC golden[SSMU_D_T];
        golden_model(kernel, X_in_raw, A_fixed, RMS_weight, H0_in_arr,
                     W_inproj, W_B, W_C, W_delta, W_out,
                     D_diag, golden);

        // DUT (current interface)
        SSMU(kernel_in,
             A_fixed,
             RMS_weight,
             W_inproj,
             W_B, W_C, W_delta,
             W_out,
             D_diag,
             X_in_stream,
             H0_in_stream,
             C_ddr,
             H1_ddr,
             out_stream);

        // compare: D_T tiles
        int fail = 0;

        for (unsigned j=0; j<(unsigned)SSMU_D_T; ++j) {
            if (out_stream.empty()) {
                std::printf("[TB] FAIL: DUT out_stream empty at j=%u\n", j);
                fail = 1;
                break;
            }
            DTYPE_VEC dut = out_stream.read();
            fail |= check_close(dut, golden[j], tol, j);
        }

        // extra output guard: after consuming exactly D_T tiles, stream must be empty
        if (!fail && !out_stream.empty()) {
            int extra = 0;
            while (!out_stream.empty() && extra < 16) {
                (void)out_stream.read();
                extra++;
            }
            std::printf("[TB] FAIL: DUT produced EXTRA output tiles after D_T reads (peeked %d)\n", extra);
            fail = 1;
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
