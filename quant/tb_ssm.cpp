// tb_ssmu.cpp (COSIM SAFE, FULL SIZE, matches CURRENT DUT with in_proj + out_proj + D skip)
// - Matches DUT: in_proj (W_in_x/W_in_z), gate from Z, conv on X_for_conv
// - Matches DUT: delta + (B,C) tiling order (J_TILE=8)
// - Matches DUT: dX term is X_ssm[j]
// - Matches DUT: output projection W_out applied to (X_gate + acc)
// - Matches DUT: D skip path (diagonal per-channel): y_final = y_outproj + D_diag * x_raw
// - ACC_T matches DUT (ap_fixed<32,10>)

#include "ssmu.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <hls_stream.h>
#include <ap_fixed.h>

#ifndef HUGE_LEN
#define HUGE_LEN (N * VEC_D)
#endif

// =========================
// Same fixed types as DUT
// =========================
#ifndef USE_FIXED_ACC
#define USE_FIXED_ACC 1
#endif

#if USE_FIXED_ACC
typedef ap_fixed<32, 10> ACC_T;   // MUST match DUT
#else
typedef float ACC_T;
#endif

typedef ap_fixed<18, 6> ACT_T;
typedef ap_fixed<20, 8> EXP_T;

static inline DTYPE vget_u(const DTYPE_VEC &v, unsigned idx) { return v[idx]; }
static inline void  vset_u(DTYPE_VEC &v, unsigned idx, DTYPE val) { v[idx] = val; }

template<typename T>
static inline T clamp_fixed(T x, T lo, T hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

// sigmoid(x) ≈ clamp(0.5 + x/4, 0, 1)
static inline ACT_T sigmoid_pwl_fx(ACT_T x) {
    const ACT_T half = (ACT_T)0.5;
    const ACT_T qtr  = (ACT_T)0.25;
    ACT_T y = half + qtr * x;
    return clamp_fixed<ACT_T>(y, (ACT_T)0.0, (ACT_T)1.0);
}

// SiLU(x)=x*sigmoid(x)
static inline DTYPE silu_elem(DTYPE a) {
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

// =========================
// Deterministic RNG (LCG)
// =========================
static inline unsigned lcg_next(unsigned &s) {
    s = 1664525u * s + 1013904223u;
    return s;
}
static inline float frand(unsigned &s, float lo=-1.0f, float hi=1.0f) {
    unsigned r = lcg_next(s);
    float u = (r & 0x00FFFFFFu) / float(0x01000000u);
    return lo + (hi - lo) * u;
}

static inline DTYPE f2dt(float x) { return (DTYPE)x; }
static inline float dt2f(DTYPE x) { return (float)x; }

// ============================================================
// Golden model (matches CURRENT DUT + D skip)
// ============================================================
static void golden_model(
    const DTYPE kernel[K],
    const DTYPE_VEC X_in_raw[VEC_D],         // raw X input to DUT (also D skip source)
    const DTYPE_VEC A_in_per_i[N],           // per-i A tokens
    const DTYPE_VEC H0_in[HUGE_LEN],         // per (i,j)
    const DTYPE_VEC W_in_x[VEC_D][VEC_D],
    const DTYPE_VEC W_in_z[VEC_D][VEC_D],
    const DTYPE_VEC W_B[N][VEC_D],
    const DTYPE_VEC W_C[N][VEC_D],
    const DTYPE_VEC W_delta[VEC_D][VEC_D],
    const DTYPE_VEC W_out[VEC_D][VEC_D],
    const DTYPE_VEC D_diag[VEC_D],           // <<< NEW (diagonal D skip)
    DTYPE_VEC out_golden[VEC_D]
) {
    // --------------------------------------------------------
    // 1) Input projection (linear-style): X_for_conv, Z_for_gate
    // --------------------------------------------------------
    DTYPE_VEC X_for_conv[VEC_D];
    DTYPE_VEC Z_for_gate[VEC_D];

    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        ACC_T accX[VEC_FACTOR];
        ACC_T accZ[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) { accX[l]=0; accZ[l]=0; }

        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC x  = X_in_raw[j];
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

    // --------------------------------------------------------
    // 2) Gate = SiLU(Z_for_gate)
    // --------------------------------------------------------
    DTYPE_VEC X_gate[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC gv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            vset_u(gv, l, silu_elem(vget_u(Z_for_gate[j], l)));
        }
        X_gate[j] = gv;
    }

    // --------------------------------------------------------
    // 3) conv1d on X_for_conv + SiLU => X_ssm (also dX term)
    // --------------------------------------------------------
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

        // shift lb
        for (int t=(int)K-2; t>0; --t)
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
                lb[t][l] = lb[t-1][l];

        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l)
            lb[0][l] = vget_u(X_for_conv[j], l);

        // conv sum + silu
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

    // --------------------------------------------------------
    // 4) delta (tiling order matches DUT: J_TILE=8)
    // --------------------------------------------------------
    DTYPE_VEC delta[VEC_D];
    {
        const unsigned J_TILE = 8;
        DTYPE_VEC X_tile[J_TILE];

        for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
            ACC_T acc[VEC_FACTOR];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) acc[l] = 0;

            for (unsigned jt=0; jt<(unsigned)VEC_D; jt+=J_TILE) {
                for (unsigned jj=0; jj<J_TILE; ++jj) {
                    X_tile[jj] = X_ssm[jt + jj];
                }
                for (unsigned jj=0; jj<J_TILE; ++jj) {
                    DTYPE_VEC w = W_delta[i][jt + jj];
                    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                        acc[l] += (ACC_T)vget_u(X_tile[jj], l) * (ACC_T)vget_u(w, l);
                    }
                }
            }

            DTYPE_VEC dv;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset_u(dv, l, softplus_pwl_fx(acc[l]));
            }
            delta[i] = dv;
        }
    }

    // --------------------------------------------------------
    // 5) B/C per i (tiling order matches DUT: J_TILE=8)
    // --------------------------------------------------------
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
                    X_tile[jj] = X_ssm[jt + jj];
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

    // --------------------------------------------------------
    // 6) fused accumulate (matches DUT)
    // --------------------------------------------------------
    DTYPE_VEC acc_out[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC z;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset_u(z, l, (DTYPE)0);
        acc_out[j]=z;
    }

    for (unsigned i=0; i<(unsigned)N; ++i) {
        const DTYPE_VEC B_vec = Bv[i];
        const DTYPE_VEC C_vec = Cv[i];
        const DTYPE_VEC A_vec = A_in_per_i[i];

        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            unsigned idx = i*(unsigned)VEC_D + j;
            const DTYPE_VEC H0v = H0_in[idx];
            const DTYPE_VEC dlt = delta[j];
            const DTYPE_VEC dx  = X_ssm[j];

            DTYPE_VEC H1v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                ACT_T a  = (ACT_T)vget_u(A_vec, l);
                ACT_T dl = (ACT_T)vget_u(dlt,  l);
                EXP_T ddA_fx = exp2_poly_fx(a * dl);

                ACC_T H0  = (ACC_T)vget_u(H0v, l);
                ACC_T Bx  = (ACC_T)vget_u(B_vec, l);
                ACC_T dX  = (ACC_T)vget_u(dx,   l);

                ACC_T H1 = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * dX;
                vset_u(H1v, l, (DTYPE)H1);
            }

            // acc[j] += H1 * C
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

    // ssm_core_out = X_gate + acc
    DTYPE_VEC ssm_core_out[VEC_D];
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC ov;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T y = (ACC_T)vget_u(X_gate[j], l) + (ACC_T)vget_u(acc_out[j], l);
            vset_u(ov, l, (DTYPE)y);
        }
        ssm_core_out[j] = ov;
    }

    // --------------------------------------------------------
    // 7) Output projection: y = W_out * ssm_core_out
    // --------------------------------------------------------
    DTYPE_VEC y_outproj[VEC_D];
    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        ACC_T accY[VEC_FACTOR];
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) accY[l] = 0;

        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC x = ssm_core_out[j];
            DTYPE_VEC w = W_out[i][j];
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                accY[l] += (ACC_T)vget_u(x, l) * (ACC_T)vget_u(w, l);
            }
        }

        DTYPE_VEC yv;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            vset_u(yv, l, (DTYPE)accY[l]);
        }
        y_outproj[i] = yv;
    }

    // --------------------------------------------------------
    // 8) D skip add (diagonal): out = y_outproj + D_diag[i] * X_in_raw[i]
    // --------------------------------------------------------
    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        DTYPE_VEC yv   = y_outproj[i];
        DTYPE_VEC xraw = X_in_raw[i];
        DTYPE_VEC ddi  = D_diag[i];

        DTYPE_VEC ov;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            ACC_T yy = (ACC_T)vget_u(yv,   l);
            ACC_T xx = (ACC_T)vget_u(xraw, l);
            ACC_T dd = (ACC_T)vget_u(ddi,  l);
            vset_u(ov, l, (DTYPE)(yy + dd * xx));
        }
        out_golden[i] = ov;
    }
}

int main() {
    unsigned seed = 12345u;

    // Streams
    hls::stream<DTYPE>     kernel_in("kernel_in");
    hls::stream<DTYPE_VEC> A_in_s("A_in");
    hls::stream<DTYPE_VEC> X_in_s("X_in");
    hls::stream<DTYPE_VEC> H0_in_s("H0_in");
    hls::stream<DTYPE_VEC> out_s("out");

    // DDR buffers
    static DTYPE_VEC C_ddr[HUGE_LEN];
    static DTYPE_VEC H1_ddr[HUGE_LEN];

    // Weights (const arrays to DUT)
    static DTYPE_VEC W_in_x[VEC_D][VEC_D];
    static DTYPE_VEC W_in_z[VEC_D][VEC_D];
    static DTYPE_VEC W_out [VEC_D][VEC_D];

    static DTYPE_VEC W_B[N][VEC_D];
    static DTYPE_VEC W_C[N][VEC_D];
    static DTYPE_VEC W_delta[VEC_D][VEC_D];

    // ---- NEW: D diagonal skip weights ----
    static DTYPE_VEC D_diag[VEC_D];

    // Host-side copies for golden
    static DTYPE kernel[K];
    static DTYPE_VEC X_in_raw[VEC_D];
    static DTYPE_VEC A_in_host[N];          // per-i
    static DTYPE_VEC H0_in[HUGE_LEN];
    static DTYPE_VEC out_golden[VEC_D];

    // init kernel
    for (unsigned i=0; i<(unsigned)K; ++i) {
        float r = frand(seed, -0.5f, 0.5f);
        kernel[i] = (DTYPE)r;
        kernel_in.write(kernel[i]);
    }

    // init X (VEC_D)
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC v;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float r = frand(seed, -1.0f, 1.0f);
            vset_u(v, l, (DTYPE)r);
        }
        X_in_raw[j] = v;
        X_in_s.write(v);
    }

    // init A (IMPORTANT: N tokens; per-i, matches DUT fused stage)
    for (unsigned i=0; i<(unsigned)N; ++i) {
        DTYPE_VEC v;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float r = frand(seed, -1.0f, 1.0f);
            vset_u(v, l, (DTYPE)r);
        }
        A_in_host[i] = v;
        A_in_s.write(v);
    }

    // init H0 (HUGE_LEN tokens)
    for (unsigned idx=0; idx<(unsigned)HUGE_LEN; ++idx) {
        DTYPE_VEC v;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float r = frand(seed, -0.5f, 0.5f);
            vset_u(v, l, (DTYPE)r);
        }
        H0_in[idx] = v;
        H0_in_s.write(v);
    }

    // init weights for W_in_x/W_in_z/W_out
    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC wix, wiz, wout;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                float rx  = frand(seed, -0.25f, 0.25f);
                float rz  = frand(seed, -0.25f, 0.25f);
                float ro  = frand(seed, -0.25f, 0.25f);
                vset_u(wix,  l, (DTYPE)rx);
                vset_u(wiz,  l, (DTYPE)rz);
                vset_u(wout, l, (DTYPE)ro);
            }
            W_in_x[i][j] = wix;
            W_in_z[i][j] = wiz;
            W_out [i][j] = wout;
        }
    }

    // init weights W_B/W_C/W_delta
    for (unsigned i=0; i<(unsigned)N; ++i) {
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC wb, wc;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                float rb = frand(seed, -0.25f, 0.25f);
                float rc = frand(seed, -0.25f, 0.25f);
                vset_u(wb, l, (DTYPE)rb);
                vset_u(wc, l, (DTYPE)rc);
            }
            W_B[i][j] = wb;
            W_C[i][j] = wc;
        }
    }
    for (unsigned i=0; i<(unsigned)VEC_D; ++i) {
        for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
            DTYPE_VEC wd;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                float r = frand(seed, -0.25f, 0.25f);
                vset_u(wd, l, (DTYPE)r);
            }
            W_delta[i][j] = wd;
        }
    }

    // init D_diag
    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC d;
        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float r = frand(seed, -0.25f, 0.25f);
            vset_u(d, l, (DTYPE)r);
        }
        D_diag[j] = d;
    }

    // Run DUT (UPDATED signature: includes D_diag)
    SSMU(kernel_in, A_in_s,
         W_in_x, W_in_z,
         W_B, W_C, W_delta,
         W_out,
         D_diag,                 // <<< NEW
         X_in_s, H0_in_s,
         C_ddr, H1_ddr, out_s);

    // Run golden (UPDATED signature: includes D_diag)
    golden_model(kernel, X_in_raw, A_in_host, H0_in,
                 W_in_x, W_in_z, W_B, W_C, W_delta, W_out,
                 D_diag,                 // <<< NEW
                 out_golden);

    // Compare OUT (DUT stream is VEC_D tokens after out_proj + D skip)
    const float tol = 1e-2f;
    int fail = 0;

    for (unsigned j=0; j<(unsigned)VEC_D; ++j) {
        DTYPE_VEC dutv = out_s.read();
        DTYPE_VEC refv = out_golden[j];

        for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
            float a = (float)vget_u(dutv, l);
            float b = (float)vget_u(refv, l);
            float e = std::fabs(a - b);

            if (e > tol) {
                if (fail < 10) {
                    std::printf("[TB] FAIL j=%u lane=%u dut=%f ref=%f err=%f (tol=%f)\n",
                                j, l, a, b, e, tol);
                }
                fail++;
            }
        }

        if ((j & 0x3FFu) == 0 && j != 0) {
            std::printf("[TB] compare progress j=%u/%u\n", j, (unsigned)VEC_D);
        }
    }

    if (fail) {
        std::printf("[TB] FAIL: tolerance exceeded, count=%d\n", fail);
        return 1;
    } else {
        std::printf("[TB] PASS\n");
        return 0;
    }
}
