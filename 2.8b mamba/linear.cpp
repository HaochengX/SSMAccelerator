#include "ssmu.h"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>

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

// ------------------------------------------------------------
// Vector accessors
// ------------------------------------------------------------
static inline DTYPE vget(const DTYPE_VEC &v, int idx) {
#pragma HLS INLINE
    return v[(unsigned)idx];
}
static inline void vset(DTYPE_VEC &v, int idx, DTYPE val) {
#pragma HLS INLINE
    v[(unsigned)idx] = val;
}

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
// - keep lane-parallelism a bit, but not full UNROLL to reduce LUT
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
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
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
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            vset(o, l, (DTYPE)((ACC_T)vget(y, l) + (ACC_T)vget(x, l)));
        }
        y_out.write(o);
    }
}

// ============================================================
// RMSNorm (pre-norm): y = x * rsqrt(mean(x^2) + eps) * weight
// - Adds clamp to prevent NaN due to negative/denorm
// ============================================================
static void rmsnorm_vecD_stream_local(hls::stream<DTYPE_VEC>& x_in,
                                      const DTYPE_VEC RMS_weight[VEC_D],
                                      hls::stream<DTYPE_VEC>& y_out) {
#pragma HLS INLINE off
    const float eps = 1e-5f;

    DTYPE_VEC xbuf[VEC_D];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_s2p impl=lutram

    // read full vector
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        xbuf[j] = x_in.read();
    }

    // compute mean square over all D elements
    ACC_T sumsq = 0;
    for (int j = 0; j < VEC_D; ++j) {
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS PIPELINE II=1
            ACC_T v = (ACC_T)vget(xbuf[j], l);
            sumsq += v * v;
        }
    }

    const float denom = (float)(VEC_D * VEC_FACTOR);
    float ms = (float)sumsq / denom;

    // NaN-guard / clamp (prevents sqrt(neg) -> NaN)
    if (!(ms >= 0.0f)) ms = 0.0f;                 // catches NaN/neg
    if (ms < 0.0f) ms = 0.0f;
    float inv = 1.0f / hls::sqrtf(ms + eps);

    // write normalized * weight
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC w = RMS_weight[j];
        DTYPE_VEC o;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
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
// - Keep PIPELINE on j-loop; reduce lane UNROLL from full to factor=4
// ============================================================
static void in_proj_stream_local(hls::stream<DTYPE_VEC>& X_in_raw,
                                 const DTYPE_VEC W_in_x[VEC_D][VEC_D],
                                 const DTYPE_VEC W_in_z[VEC_D][VEC_D],
                                 hls::stream<DTYPE_VEC>& X_for_conv,
                                 hls::stream<DTYPE_VEC>& Z_for_gate) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in_raw.read();
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] in_proj(linear): start (produce %d)\n", VEC_D);
#endif

    for (int i = 0; i < VEC_D; ++i) {
        ACC_T accX[VEC_FACTOR];
        ACC_T accZ[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accX complete dim=1
#pragma HLS ARRAY_PARTITION variable=accZ complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            accX[l] = (ACC_T)0;
            accZ[l] = (ACC_T)0;
        }

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC x  = X_buf[j];
            DTYPE_VEC wx = W_in_x[i][j];
            DTYPE_VEC wz = W_in_z[i][j];

            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                ACC_T xv = (ACC_T)vget(x,  l);
                accX[l] += xv * (ACC_T)vget(wx, l);
                accZ[l] += xv * (ACC_T)vget(wz, l);
            }
        }

        DTYPE_VEC outX, outZ;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            vset(outX, l, (DTYPE)accX[l]);
            vset(outZ, l, (DTYPE)accZ[l]);
        }
        X_for_conv.write(outX);
        Z_for_gate.write(outZ);

#ifndef __SYNTHESIS__
        if ((i & 15) == 0) DUT_PRINTF("[DUT] in_proj(linear): i=%d/%d\n", i, VEC_D);
#endif
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] in_proj(linear): done\n");
#endif
}

// ============================================================
// conv1d + SiLU (gate path uses SiLU(z), SSM path uses SiLU(conv))
// - Reduce lane UNROLL from full to factor=4
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

    // clear line buffer
    for (int i = 0; i < K-1; ++i)
        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[i][k] = 0;

    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=2
        DTYPE_VEC x_in = X_for_conv_in.read();
        DTYPE_VEC z_in = Z_for_gate_in.read();

        // gate = SiLU(z)
        DTYPE_VEC gate_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL factor=4
            vset(gate_out, k, silu_elem(vget(z_in, k)));
        }
        X_gate_out.write(gate_out);

        // build window
        DTYPE window[K][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window complete dim=2

        for (int j = 0; j < K-1; ++j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                window[j][k] = line_buffer[j][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            window[K-1][k] = vget(x_in, k);

        // shift line buffer
        for (int j = K-2; j > 0; --j)
            for (int k = 0; k < VEC_FACTOR; ++k)
                line_buffer[j][k] = line_buffer[j-1][k];

        for (int k = 0; k < VEC_FACTOR; ++k)
            line_buffer[0][k] = vget(x_in, k);

        // conv
        DTYPE_VEC conv_out;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL factor=4
            ACC_T sum = 0;
            for (int kk = 0; kk < K; ++kk) {
#pragma HLS UNROLL
                sum += (ACC_T)kernel_buffer[kk] * (ACC_T)window[kk][lane];
            }
            vset(conv_out, lane, (DTYPE)sum);
        }

        // ssm input = SiLU(conv)
        DTYPE_VEC ssm_out;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL factor=4
            vset(ssm_out, k, silu_elem(vget(conv_out, k)));
        }
        X_ssm_out.write(ssm_out);
    }
}

// ============================================================
// projections (delta) then (B,C) interleaved per i
// - Keep J_TILE unroll (8) but reduce lane unroll to factor=4
// ============================================================
static void projection_streams_local(hls::stream<DTYPE_VEC>& X_ssm_in,
                                     const DTYPE_VEC W_B[N][VEC_D],
                                     const DTYPE_VEC W_C[N][VEC_D],
                                     const DTYPE_VEC W_delta[VEC_D][VEC_D],
                                     hls::stream<DTYPE_VEC>& B_out_N,
                                     hls::stream<DTYPE_VEC>& C_out_N,
                                     hls::stream<DTYPE_VEC>& delta_out) {
#pragma HLS INLINE off

    const int J_TILE = 8;

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] delta: start (produce %d)\n", VEC_D);
#endif

    for (int i = 0; i < VEC_D; ++i) {
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            acc[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=8
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC w = W_delta[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                    acc[l] += (ACC_T)vget(X_tile[jj], l) * (ACC_T)vget(w, l);
                }
            }
        }

        DTYPE_VEC delta_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            vset(delta_vec, l, softplus_pwl_fx(acc[l]));
        }
        delta_out.write(delta_vec);

#ifndef __SYNTHESIS__
        if ((i & 15) == 0) DUT_PRINTF("[DUT] delta: i=%d/%d\n", i, VEC_D);
#endif
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] B+C: start (produce %d each)\n", N);
#endif

    for (int i = 0; i < N; ++i) {
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            accB[l] = (ACC_T)0;
            accC[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=8
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC wB = W_B[i][jt + jj];
                DTYPE_VEC wC = W_C[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                    ACC_T x = (ACC_T)vget(X_tile[jj], l);
                    accB[l] += x * (ACC_T)vget(wB, l);
                    accC[l] += x * (ACC_T)vget(wC, l);
                }
            }
        }

        DTYPE_VEC outB, outC;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            vset(outB, l, (DTYPE)accB[l]);
            vset(outC, l, (DTYPE)accC[l]);
        }
        B_out_N.write(outB);
        C_out_N.write(outC);

#ifndef __SYNTHESIS__
        if ((i & 511) == 0) DUT_PRINTF("[DUT] B+C: i=%d/%d\n", i, N);
#endif
    }
}

// ============================================================
// FUSED stage (Mamba-aligned):
// - Reduce lane UNROLL to factor=4 to cut LUT
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
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            vset(z, l, (DTYPE)0);
        }
        acc[j] = z;
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] FUSED(Mamba): start update/write/acc (N=%d, VEC_D=%d, HUGE_LEN=%d)\n", N, VEC_D, (int)HUGE_LEN);
#endif

    for (int i = 0; i < N; ++i) {
        DTYPE_VEC A_vec = A_fixed[i];
        DTYPE_VEC B_vec = B_in.read();
        DTYPE_VEC C_vec = C_in.read();

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            int idx = i * VEC_D + j;

            DTYPE_VEC H0v  = H0_in.read();
            DTYPE_VEC dlt  = delta_buf[j];
            DTYPE_VEC xssm = X_ssm_buf[j];

            DTYPE_VEC H1v;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                ACT_T a  = (ACT_T)vget(A_vec, l);
                ACT_T dl = (ACT_T)vget(dlt,  l);
                // (optional safety clamp for polynomial range)
                ACT_T adl = a * dl;
                adl = clamp_fx<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);
                EXP_T ddA_fx = exp2_poly_fx(adl);

                ACC_T H0  = (ACC_T)vget(H0v, l);
                ACC_T Bx  = (ACC_T)vget(B_vec, l);
                ACC_T Xs  = (ACC_T)vget(xssm, l);

                ACC_T H1  = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * Xs;
                vset(H1v, l, (DTYPE)H1);
            }

            C_ddr[idx]  = C_vec;
            H1_ddr[idx] = H1v;

            DTYPE_VEC aold = acc[j];
            DTYPE_VEC anew;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                ACC_T base = (ACC_T)vget(aold, l);
                ACC_T addt = (ACC_T)vget(H1v,  l) * (ACC_T)vget(C_vec, l);
                vset(anew, l, (DTYPE)(base + addt));
            }
            acc[j] = anew;
        }

#ifndef __SYNTHESIS__
        if ((i & 511) == 0) DUT_PRINTF("[DUT] FUSED(Mamba): i=%d/%d\n", i, N);
#endif
    }

    // Mamba: output = gate ⊙ y_ssm
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC yssm = acc[j];
        DTYPE_VEC gate = X_gate[j];
        DTYPE_VEC outv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            ACC_T g = (ACC_T)vget(gate, l);
            ACC_T y = (ACC_T)vget(yssm, l);
            vset(outv, l, (DTYPE)(g * y));
        }
        out.write(outv);
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] FUSED(Mamba): done\n");
#endif
}

// ============================================================
// Output projection (out_proj): y = W_out * x
// - Reduce lane UNROLL to factor=4
// ============================================================
static void out_proj_stream_local(hls::stream<DTYPE_VEC>& X_in,
                                  const DTYPE_VEC W_out[VEC_D][VEC_D],
                                  hls::stream<DTYPE_VEC>& Y_out) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        X_buf[j] = X_in.read();
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] out_proj(linear): start (produce %d)\n", VEC_D);
#endif

    for (int i = 0; i < VEC_D; ++i) {
        ACC_T accY[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accY complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            accY[l] = (ACC_T)0;
        }

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
            DTYPE_VEC x = X_buf[j];
            DTYPE_VEC w = W_out[i][j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
                accY[l] += (ACC_T)vget(x, l) * (ACC_T)vget(w, l);
            }
        }

        DTYPE_VEC y;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=4
            vset(y, l, (DTYPE)accY[l]);
        }
        Y_out.write(y);

#ifndef __SYNTHESIS__
        if ((i & 15) == 0) DUT_PRINTF("[DUT] out_proj(linear): i=%d/%d\n", i, VEC_D);
#endif
    }

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] out_proj(linear): done\n");
#endif
}

// ============================================================
// TOP: Full Mamba Block (Pre-Norm + MambaCore + OutProj + D-skip + Residual)
// ============================================================
void SSMU(hls::stream<DTYPE>& kernel_in,
          const DTYPE_VEC A_fixed[N],                // ✅ fixed A
          const DTYPE_VEC RMS_weight[VEC_D],         // ✅ RMSNorm weight (gamma), shape like x
          const DTYPE_VEC W_in_x[VEC_D][VEC_D],
          const DTYPE_VEC W_in_z[VEC_D][VEC_D],
          const DTYPE_VEC W_B[N][VEC_D],
          const DTYPE_VEC W_C[N][VEC_D],
          const DTYPE_VEC W_delta[VEC_D][VEC_D],
          const DTYPE_VEC W_out[VEC_D][VEC_D],
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

#pragma HLS STREAM variable=kernel_in depth=512
#pragma HLS STREAM variable=X_in      depth=512
#pragma HLS STREAM variable=H0_in     depth=512
#pragma HLS STREAM variable=out       depth=512

#pragma HLS DATAFLOW

    // 0) Tee input into: norm path / D-skip / residual
    hls::stream<DTYPE_VEC> X_in_norm("X_in_norm");
    hls::stream<DTYPE_VEC> X_in_D   ("X_in_D");
    hls::stream<DTYPE_VEC> X_in_res ("X_in_res");
#pragma HLS STREAM variable=X_in_norm depth=512
#pragma HLS STREAM variable=X_in_D    depth=512
#pragma HLS STREAM variable=X_in_res  depth=512

    // 0.5) RMSNorm output
    hls::stream<DTYPE_VEC> X_normed("X_normed");
#pragma HLS STREAM variable=X_normed depth=512

    // 1) In-proj outputs
    hls::stream<DTYPE_VEC> X_for_conv_stream("X_for_conv_stream");
    hls::stream<DTYPE_VEC> Z_for_gate_stream("Z_for_gate_stream");
#pragma HLS STREAM variable=X_for_conv_stream depth=512
#pragma HLS STREAM variable=Z_for_gate_stream depth=512

    // 2) Conv outputs
    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_upd_stream ("X_ssm_upd_stream");
#pragma HLS STREAM variable=X_gate_stream      depth=512
#pragma HLS STREAM variable=X_ssm_stream       depth=512
#pragma HLS STREAM variable=X_ssm_proj_stream  depth=512
#pragma HLS STREAM variable=X_ssm_upd_stream   depth=512

    // 3) Projections outputs
    hls::stream<DTYPE_VEC> B_stream_N("B_stream_N");
    hls::stream<DTYPE_VEC> C_stream_N("C_stream_N");
    hls::stream<DTYPE_VEC> delta_stream("delta_stream");
#pragma HLS STREAM variable=delta_stream       depth=512
#pragma HLS STREAM variable=B_stream_N         depth=512
#pragma HLS STREAM variable=C_stream_N         depth=512

    // 4) Core output (gate ⊙ y_ssm)
    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
#pragma HLS STREAM variable=ssm_core_out_stream depth=512

    // 5) out_proj
    hls::stream<DTYPE_VEC> out_proj_stream("out_proj_stream");
#pragma HLS STREAM variable=out_proj_stream depth=512

    // 6) after D-skip
    hls::stream<DTYPE_VEC> after_D_stream("after_D_stream");
#pragma HLS STREAM variable=after_D_stream depth=512

    // --- stage 0: split input ---
    tee_vecD_stream3_local(X_in, X_in_norm, X_in_D, X_in_res);

    // --- stage 0.5: pre-norm (RMSNorm) ---
    rmsnorm_vecD_stream_local(X_in_norm, RMS_weight, X_normed);

    // --- stage 1: in_proj ---
    in_proj_stream_local(X_normed, W_in_x, W_in_z, X_for_conv_stream, Z_for_gate_stream);

    // --- stage 2: conv + gate / ssm input ---
    conv1d_silu_stream_local(X_for_conv_stream, Z_for_gate_stream, kernel_in, X_gate_stream, X_ssm_stream);

    // --- stage 3: duplicate ssm input for proj + update ---
    dup_vecD_stream_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_upd_stream);

    // --- stage 4: delta/B/C projections ---
    projection_streams_local(X_ssm_proj_stream, W_B, W_C, W_delta,
                             B_stream_N, C_stream_N, delta_stream);

    // --- stage 5: fused SSM scan + gate⊙ output (Mamba aligned) ---
    fused_update_write_accum_output_mamba(
        X_gate_stream,
        X_ssm_upd_stream,
        delta_stream,
        A_fixed,
        B_stream_N,
        C_stream_N,
        H0_in,
        C_ddr,
        H1_ddr,
        ssm_core_out_stream
    );

    // --- stage 6: output projection ---
    out_proj_stream_local(ssm_core_out_stream, W_out, out_proj_stream);

    // --- stage 7: add D skip (diagonal) ---
    add_D_skip_local(out_proj_stream, X_in_D, D_diag, after_D_stream);

    // --- stage 8: residual add (x + ...) ---
    add_residual_local(after_D_stream, X_in_res, out);
}
