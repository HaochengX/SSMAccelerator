#include "ssmu.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hls_stream.h>

// ============================================================
// TB-side typedefs & helpers (because DUT defines these in .cpp, not in ssmu.h)
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

static inline DTYPE vget_u(const DTYPE_VEC &v, int idx) {
    return v[(unsigned)idx];
}
static inline void vset_u(DTYPE_VEC &v, int idx, DTYPE val) {
    v[(unsigned)idx] = val;
}

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
    ap_fixed<24, 8> y  = (ap_fixed<24,8>)1.0 + (ap_fixed<24,8>)t + (ap_fixed<24,8>)0.5 * tt;

    if (y < 0) y = 0;
    return (EXP_T)y;
}

// ============================================================
// DUT align switches
// ============================================================
#ifndef TB_WOUT_TRANSPOSE
#define TB_WOUT_TRANSPOSE 0   // 0: W_out[i][j] , 1: W_out[j][i]
#endif

#ifndef TB_DBC_USE_X_SSM
#define TB_DBC_USE_X_SSM 1    // 1: use X_ssm , 0: use X_for_conv
#endif

#ifndef TB_COMBINE_GATE_MUL_ACC
#define TB_COMBINE_GATE_MUL_ACC 1 // 1: gate * acc_out, 0: gate*(acc_out + X_ssm)
#endif

#ifndef TB_DEBUG_PRINT
#define TB_DEBUG_PRINT 1
#endif

// ============================================================
// Golden model (your version, unchanged behavior-wise)
// ============================================================
static void golden_model(
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
    const float eps = 1e-5f;

    ACC_T sumsq = 0;
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T v = (ACC_T)vget_u(X_in_raw[j], l);
            sumsq += v * v;
        }
    }

    const float denom = (float)(VEC_D * VEC_FACTOR);
    float ms  = (float)sumsq / denom;
    if (!(ms >= 0.0f)) ms = 0.0f;
    if (ms < 0.0f) ms = 0.0f;

    float inv = 1.0f / std::sqrt(ms + eps);

    DTYPE_VEC X_normed[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC o;
        DTYPE_VEC w = RMS_weight[j];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float xv = (float)(ACC_T)vget_u(X_in_raw[j], l);
            float ww = (float)(ACC_T)vget_u(w, l);
            float yv = xv * inv * ww;
            vset_u(o, l, (DTYPE)yv);
        }
        X_normed[j] = o;
    }

    DTYPE_VEC X_for_conv[VEC_D];
    DTYPE_VEC Z_for_gate[VEC_D];

    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        ACC_T accX[VEC_FACTOR];
        ACC_T accZ[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) { accX[l]=0; accZ[l]=0; }

        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC x  = X_normed[j];
            DTYPE_VEC wx = W_in_x[i][j];
            DTYPE_VEC wz = W_in_z[i][j];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACC_T xv = (ACC_T)vget_u(x,  l);
                accX[l] += xv * (ACC_T)vget_u(wx, l);
                accZ[l] += xv * (ACC_T)vget_u(wz, l);
            }
        }

        DTYPE_VEC ox, oz;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            vset_u(ox, l, (DTYPE)accX[l]);
            vset_u(oz, l, (DTYPE)accZ[l]);
        }
        X_for_conv[i] = ox;
        Z_for_gate[i] = oz;
    }

    DTYPE_VEC X_gate[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC gv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            vset_u(gv, l, silu_elem(vget_u(Z_for_gate[j], l)));
        }
        X_gate[j] = gv;
    }

    DTYPE lb[K-1][VEC_FACTOR];
    for (unsigned t=0; t<(unsigned)(K-1); ++t)
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
            lb[t][l] = (DTYPE)0;

    DTYPE_VEC X_ssm[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE window[K][VEC_FACTOR];
        for (unsigned t=0; t<(unsigned)(K-1); ++t)
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
                window[t][l] = lb[t][l];

        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
            window[K-1][l] = vget_u(X_for_conv[j], l);

        for (int t=(int)K-2; t>0; --t)
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
                lb[t][l] = lb[t-1][l];

        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
            lb[0][l] = vget_u(X_for_conv[j], l);

        DTYPE_VEC ssmv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T sum = 0;
            for (unsigned kk=0; kk<(unsigned)K; ++kk) {
                sum += (ACC_T)kernel[kk] * (ACC_T)window[kk][l];
            }
            DTYPE conv = (DTYPE)sum;
            vset_u(ssmv, l, silu_elem(conv));
        }
        X_ssm[j] = ssmv;
    }

    const DTYPE_VEC *DBC_in = TB_DBC_USE_X_SSM ? X_ssm : X_for_conv;

#if TB_DEBUG_PRINT
    std::printf("[TB] dbg X_raw[0].lane0=%f\n", (float)vget_u(X_in_raw[0], 0));
    std::printf("[TB] dbg X_normed[0].lane0=%f\n", (float)vget_u(X_normed[0], 0));
    std::printf("[TB] dbg X_for_conv[0].lane0=%f\n", (float)vget_u(X_for_conv[0], 0));
    std::printf("[TB] dbg X_ssm[0].lane0=%f\n", (float)vget_u(X_ssm[0], 0));
#endif

    DTYPE_VEC delta[VEC_D];
    {
        const unsigned J_TILE = 8;
        DTYPE_VEC X_tile[J_TILE];

        for (unsigned d=0; d<(unsigned)VEC_D; ++d) {
            ACC_T acc[VEC_FACTOR];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) acc[l] = 0;

            for (unsigned jt=0; jt<(unsigned)VEC_D; jt+=J_TILE) {
                for (unsigned jj=0; jj<J_TILE; ++jj) {
                    X_tile[jj] = DBC_in[jt + jj];
                }
                for (unsigned jj=0; jj<J_TILE; ++jj) {
                    DTYPE_VEC w = W_delta[d][jt + jj];
                    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                        acc[l] += (ACC_T)vget_u(X_tile[jj], l) * (ACC_T)vget_u(w, l);
                    }
                }
            }

            DTYPE_VEC dv;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset_u(dv, l, softplus_pwl_fx(acc[l]));
            }
            delta[d] = dv;
        }
    }

    DTYPE_VEC Bv[N];
    DTYPE_VEC Cv[N];
    {
        const unsigned J_TILE = 8;
        DTYPE_VEC X_tile[J_TILE];

        for (unsigned i=0; i<(unsigned)N; ++i) {
            ACC_T accB[VEC_FACTOR], accC[VEC_FACTOR];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) { accB[l]=0; accC[l]=0; }

            for (unsigned jt=0; jt<(unsigned)VEC_D; jt+=J_TILE) {
                for (unsigned jj=0; jj<J_TILE; ++jj) {
                    X_tile[jj] = DBC_in[jt + jj];
                }
                for (unsigned jj=0; jj<J_TILE; ++jj) {
                    DTYPE_VEC wB = W_B[i][jt + jj];
                    DTYPE_VEC wC = W_C[i][jt + jj];
                    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                        ACC_T x = (ACC_T)vget_u(X_tile[jj], l);
                        accB[l] += x * (ACC_T)vget_u(wB, l);
                        accC[l] += x * (ACC_T)vget_u(wC, l);
                    }
                }
            }

            DTYPE_VEC ob, oc;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset_u(ob, l, (DTYPE)accB[l]);
                vset_u(oc, l, (DTYPE)accC[l]);
            }
            Bv[i]=ob; Cv[i]=oc;
        }
    }

    DTYPE_VEC acc_out[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC z;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset_u(z, l, (DTYPE)0);
        acc_out[j]=z;
    }

    for (unsigned i=0; i<(unsigned)N; ++i) {
        const DTYPE_VEC B_vec = Bv[i];
        const DTYPE_VEC C_vec = Cv[i];
        const DTYPE_VEC A_vec = A_fixed[i];

        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            unsigned idx = i*(unsigned)VEC_D + j;
            const DTYPE_VEC H0v = H0_in[idx];
            const DTYPE_VEC dlt = delta[j];
            const DTYPE_VEC dx  = X_ssm[j]; // MATCH DUT: dX term = X_ssm

            DTYPE_VEC H1v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACT_T a  = (ACT_T)vget_u(A_vec, l);
                ACT_T dl = (ACT_T)vget_u(dlt,  l);

                ACT_T adl = a * dl;
                adl = clamp_fixed<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);

                EXP_T ddA_fx = exp2_poly_fx(adl);

                ACC_T H0  = (ACC_T)vget_u(H0v, l);
                ACC_T Bx  = (ACC_T)vget_u(B_vec, l);
                ACC_T dX  = (ACC_T)vget_u(dx,   l);

                ACC_T H1 = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * dX;
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

    DTYPE_VEC ssm_core_out[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC ov;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T g = (ACC_T)vget_u(X_gate[j], l);
            ACC_T y = (ACC_T)vget_u(acc_out[j], l);
#if TB_COMBINE_GATE_MUL_ACC
            vset_u(ov, l, (DTYPE)(g * y));
#else
            ACC_T x = (ACC_T)vget_u(X_ssm[j], l);
            vset_u(ov, l, (DTYPE)(g * (y + x)));
#endif
        }
        ssm_core_out[j] = ov;
    }

    DTYPE_VEC y_outproj[VEC_D];
    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        ACC_T accY[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) accY[l] = 0;

        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC x = ssm_core_out[j];
#if TB_WOUT_TRANSPOSE
            DTYPE_VEC w = W_out[j][i];
#else
            DTYPE_VEC w = W_out[i][j];
#endif
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                accY[l] += (ACC_T)vget_u(x, l) * (ACC_T)vget_u(w, l);
            }
        }

        DTYPE_VEC yv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset_u(yv, l, (DTYPE)accY[l]);
        y_outproj[i] = yv;
    }

    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        DTYPE_VEC yv   = y_outproj[i];
        DTYPE_VEC xraw = X_in_raw[i];
        DTYPE_VEC ddi  = D_diag[i];

        DTYPE_VEC ov;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T yy = (ACC_T)vget_u(yv,   l);
            ACC_T xx = (ACC_T)vget_u(xraw, l);
            ACC_T dd = (ACC_T)vget_u(ddi,  l);
            vset_u(ov, l, (DTYPE)(yy + dd * xx + xx));
        }
        out_golden[i] = ov;
    }
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
static inline DTYPE_VEC rand_vec(unsigned &seed, float lo=-1.0f, float hi=1.0f) {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        vset_u(v, l, (DTYPE)frand(seed, lo, hi));
    }
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
// main(): drives DUT exactly matching ssmu.h prototype
// ============================================================
int main() {
    std::printf("[TB] tb_ssm.cpp linked OK (main() present).\n");

    // Keep it small for cosim debug:
    // you can bump CASES later.
    const int CASES = 1;
    const float tol = 0.05f;

    static DTYPE_VEC C_ddr[HUGE_LEN];
    static DTYPE_VEC H1_ddr[HUGE_LEN];

    for (int tc=0; tc<CASES; ++tc) {
        unsigned seed = 1u + (unsigned)tc;

        // ---- input vectors ----
        DTYPE_VEC X_in_raw[VEC_D];
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) X_in_raw[j] = rand_vec(seed, -1.0f, 1.0f);

        // ---- kernel array + stream ----
        DTYPE kernel[K];
        for (unsigned kk=0; kk<(unsigned)K; ++kk) kernel[kk] = (DTYPE)frand(seed, -0.5f, 0.5f);

        hls::stream<DTYPE> kernel_in;
        for (unsigned kk=0; kk<(unsigned)K; ++kk) kernel_in.write(kernel[kk]);

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

        for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
            for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
                W_in_x[i][j]  = rand_vec(seed, -0.5f, 0.5f);
                W_in_z[i][j]  = rand_vec(seed, -0.5f, 0.5f);
                W_delta[i][j] = rand_vec(seed, -0.5f, 0.5f);
                W_out[i][j]   = rand_vec(seed, -0.5f, 0.5f);
            }
        }
        for (unsigned i=0; i<(unsigned)N; ++i) {
            for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
                W_B[i][j] = rand_vec(seed, -0.5f, 0.5f);
                W_C[i][j] = rand_vec(seed, -0.5f, 0.5f);
            }
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
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) X_in_stream.write(X_in_raw[j]);

        // ---- output stream ----
        hls::stream<DTYPE_VEC> out_stream;

        // ---- golden ----
        DTYPE_VEC golden[VEC_D];
        golden_model(kernel, X_in_raw, A_fixed, RMS_weight, H0_in_arr,
                     W_in_x, W_in_z, W_B, W_C, W_delta, W_out, D_diag,
                     golden);

        // ---- DUT call (EXACTLY matches ssmu.h) ----
        SSMU(
            kernel_in,
            A_fixed,
            RMS_weight,
            W_in_x,
            W_in_z,
            W_B,
            W_C,
            W_delta,
            W_out,
            D_diag,
            X_in_stream,
            H0_in_stream,
            C_ddr,
            H1_ddr,
            out_stream
        );

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
