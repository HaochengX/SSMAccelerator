// ============================================================
// quant.cpp (SSMU with quant insertion + warning/cosim/RTGEN fixes)
// (UPDATED + latency optimizations inspired by Layer.h style)
//
// What we can / cannot truly “fix” from your warning list:
// - [HLS 207-1016] -Wmisleading-indentation : comes from Xilinx headers -> cannot fix in user code.
// - [HLS 200-1449]/[HLS 200-1450] : throughput advisories (dataflow + reading/writing caller ports).
//   We can *mitigate* by inserting explicit “copy” processes so stages only talk via streams.
// - [HLS 214-366] std::array/operator[] duplication : ensure every access uses ONE signature
//   -> always go through vget/vset with `unsigned` index.
// - [RTGEN 206-101] dangling AXI ports : commonly caused by reinterpreting pointers to different types.
//   -> FIXED here by writing to the original DTYPE_VEC* ports (no ap_uint* reinterpret-cast).
// - [HLS 214-464] array undecay on “variable length array” : often triggered by casts/undecay attempts.
//   -> Reduced by removing reinterpret-casts and keeping plain pointer writes.
// - [HLS 214-187] cannot unroll loop with variable tripcount : ensure we don’t request UNROLL there,
//   and add LOOP_TRIPCOUNT hints.
// - [SYNCHK 200-120]/[SYNCHK 200-23] are QoR advisories; we reduce risk by widening arithmetic,
//   and keeping pack loop fully unrolled.
//
// ============================================================

#include "SSMU.h"
#include <hls_stream.h>
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
// IMPORTANT: always use unsigned idx to avoid signature duplication warnings.
// ------------------------------------------------------------
static inline DTYPE vget(const DTYPE_VEC &v, unsigned idx) {
#pragma HLS INLINE
    return v[idx];
}
static inline void vset(DTYPE_VEC &v, unsigned idx, DTYPE val) {
#pragma HLS INLINE
    v[idx] = val;
}

// ============================================================
// Quant insertion points
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
// BFP Quant Toolkit
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
#pragma HLS INLINE
    // Make constants in a wider C type to avoid SYNCHK “overflow assumed” style warnings.
    const int QMAX_I = (int)((1u << (Q_BITS - 1)) - 1u);
    const int QMIN_I = (int)(-(1 << (Q_BITS - 1)));

    if (x32 > QMAX_I) return (QINT_T)QMAX_I;
    if (x32 < QMIN_I) return (QINT_T)QMIN_I;
    return (QINT_T)x32;
}

static inline ap_uint<32> abs_i32(ap_int<32> x) {
#pragma HLS INLINE
    ap_int<33> xe = (ap_int<33>)x;
    ap_int<33> me = (xe < 0) ? (ap_int<33>)(-xe) : xe;
    return (ap_uint<32>)me;
}

// avoid unsigned underflow on (s-1) by using int
static inline ap_int<32> rshift_round_i32(ap_int<32> x, QSCALE_T s) {
#pragma HLS INLINE
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
#pragma HLS INLINE
    const ap_uint<32> QMAX_U = (ap_uint<32>)((1u << (Q_BITS - 1)) - 1u);

    QSCALE_T s = 0;
    ap_uint<32> v = max_abs;

    while (v > QMAX_U && s < (QSCALE_T)Q_S_MAX) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=15
#pragma HLS UNROLL off
        v >>= 1;
        s++;
    }
    return s;
}

static inline void quant_vec(const DTYPE_VEC &in, DTYPE_VEC &q_out, QSCALE_T &s_out) {
#pragma HLS INLINE
    ap_uint<32> max_abs = 0;

    for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        ap_int<32> v = (ap_int<32>)(ACC_T)vget(in, (unsigned)l);
        ap_uint<32> a = abs_i32(v);
        if (a > max_abs) max_abs = a;
    }

    QSCALE_T s = calc_shared_scale_u32(max_abs);
    s_out = s;

    for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        ap_int<32> v   = (ap_int<32>)(ACC_T)vget(in, (unsigned)l);
        ap_int<32> q32 = rshift_round_i32(v, s);
        QINT_T q = clamp_qint(q32);
        vset(q_out, (unsigned)l, (DTYPE)(ACC_T)q);
    }
}

static inline ACC_T deq_lane(DTYPE q, QSCALE_T s) {
#pragma HLS INLINE
    ap_int<32> qi = (ap_int<32>)(ACC_T)q;
    ap_int<32> xi = (s == 0) ? qi : (ap_int<32>)(qi << (int)s);
    return (ACC_T)xi;
}

// "Fake quant": pass through values but set s=0
static void passthru_vecD_stream_as_q(hls::stream<DTYPE_VEC>& in,
                                      hls::stream<DTYPE_VEC>& q_out,
                                      hls::stream<QSCALE_T>&  s_out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        q_out.write(in.read());
        s_out.write((QSCALE_T)0);
    }
}

static void quant_vecD_stream(hls::stream<DTYPE_VEC>& in,
                             hls::stream<DTYPE_VEC>& q_out,
                             hls::stream<QSCALE_T>&  s_out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC v = in.read();
        DTYPE_VEC qv;
        QSCALE_T  s;
        quant_vec(v, qv, s);
        q_out.write(qv);
        s_out.write(s);
    }
}

static void drop_scale_vecD_stream(hls::stream<DTYPE_VEC>& q_in,
                                  hls::stream<QSCALE_T>&  s_in,
                                  hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        (void)s_in.read();
        out.write(q_in.read());
    }
}

} // namespace bfpq

// ============================================================
// NO exp/log approximations (fixed-point)
// ============================================================
typedef ap_fixed<18, 6>  ACT_T;
typedef ap_fixed<20, 8>  EXP_T;

template<typename T>
static inline T clamp_fx(T x, T lo, T hi) {
#pragma HLS INLINE
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

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

static inline EXP_T exp2_poly_fx(ACT_T t) {
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
// tee / dup
// ============================================================
static void tee_vecD_stream3_local(hls::stream<DTYPE_VEC>& in,
                                   hls::stream<DTYPE_VEC>& out1,
                                   hls::stream<DTYPE_VEC>& out2,
                                   hls::stream<DTYPE_VEC>& out3) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
        out3.write(v);
    }
}

static void dup_vecD_stream_local(hls::stream<DTYPE_VEC>& in,
                                  hls::stream<DTYPE_VEC>& out1,
                                  hls::stream<DTYPE_VEC>& out2) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

// ============================================================
// Copy helpers (mitigate HLS 200-1449/1450 “reads/writes from caller”)
// These make stages consume ONLY streams inside DATAFLOW.
// ============================================================
static void copy_vecD_stream(hls::stream<DTYPE_VEC>& in,
                             hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        out.write(in.read());
    }
}

static void copy_huge_stream(hls::stream<DTYPE_VEC>& in,
                             hls::stream<DTYPE_VEC>& out) {
#pragma HLS INLINE off
    for (int i = 0; i < (int)HUGE_LEN; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=1048576
        out.write(in.read());
    }
}

// ============================================================
// D skip / residual
// ============================================================
static void add_D_skip_local(hls::stream<DTYPE_VEC>& y_in,
                             hls::stream<DTYPE_VEC>& x_in,
                             const DTYPE_VEC D_diag[VEC_D],
                             hls::stream<DTYPE_VEC>& y_out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC y = y_in.read();
        DTYPE_VEC x = x_in.read();
        DTYPE_VEC d = D_diag[j];

        DTYPE_VEC o;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T yy = (ACC_T)vget(y, (unsigned)l);
            ACC_T xx = (ACC_T)vget(x, (unsigned)l);
            ACC_T dd = (ACC_T)vget(d, (unsigned)l);
            vset(o, (unsigned)l, (DTYPE)(yy + dd * xx));
        }
        y_out.write(o);
    }
}

static void add_residual_local(hls::stream<DTYPE_VEC>& y_in,
                               hls::stream<DTYPE_VEC>& x_res_in,
                               hls::stream<DTYPE_VEC>& y_out) {
#pragma HLS INLINE off
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC y = y_in.read();
        DTYPE_VEC x = x_res_in.read();
        DTYPE_VEC o;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(o, (unsigned)l, (DTYPE)((ACC_T)vget(y, (unsigned)l) + (ACC_T)vget(x, (unsigned)l)));
        }
        y_out.write(o);
    }
}

// ============================================================
// RMSNorm
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
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        xbuf[j] = x_in.read();
    }

    ACC_T sumsq_lane[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=sumsq_lane complete dim=1
    for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        sumsq_lane[l] = (ACC_T)0;
    }

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC xv = xbuf[j];
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T v = (ACC_T)vget(xv, (unsigned)l);
            sumsq_lane[l] += v * v;
        }
    }

    ACC_T sumsq = 0;
    for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        sumsq += sumsq_lane[l];
    }

    const float denom = (float)(VEC_D * VEC_FACTOR);
    float ms = (float)sumsq / denom;
    if (!(ms >= 0.0f)) ms = 0.0f;
    if (ms < 0.0f) ms = 0.0f;

    float inv = 1.0f / hls::sqrtf(ms + eps);

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC w = RMS_weight[j];
        DTYPE_VEC o;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            float xv = (float)(ACC_T)vget(xbuf[j], (unsigned)l);
            float ww = (float)(ACC_T)vget(w,     (unsigned)l);
            float yv = xv * inv * ww;
            vset(o, (unsigned)l, (DTYPE)yv);
        }
        y_out.write(o);
    }
}

// ============================================================
// Input projection
// ============================================================
static void in_proj_stream_local_q(hls::stream<DTYPE_VEC>& X_in_q,
                                   hls::stream<QSCALE_T>&  X_in_s,
                                   const DTYPE_VEC W_in_x[VEC_D][VEC_D],
                                   const DTYPE_VEC W_in_z[VEC_D][VEC_D],
                                   hls::stream<DTYPE_VEC>& X_for_conv_q,
                                   hls::stream<QSCALE_T>&  X_for_conv_s,
                                   hls::stream<DTYPE_VEC>& Z_for_gate_q,
                                   hls::stream<QSCALE_T>&  Z_for_gate_s) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[VEC_D];
    QSCALE_T  S_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
#pragma HLS BIND_STORAGE variable=S_buf type=ram_s2p impl=lutram

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        X_buf[j] = X_in_q.read();
        S_buf[j] = X_in_s.read();
    }

    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        ACC_T accX[VEC_FACTOR];
        ACC_T accZ[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accX complete dim=1
#pragma HLS ARRAY_PARTITION variable=accZ complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accX[l] = (ACC_T)0;
            accZ[l] = (ACC_T)0;
        }

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
            DTYPE_VEC xq = X_buf[j];
            QSCALE_T  xs = S_buf[j];
            DTYPE_VEC wx = W_in_x[i][j];
            DTYPE_VEC wz = W_in_z[i][j];

            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACC_T xv = bfpq::deq_lane(vget(xq, (unsigned)l), xs);
                accX[l] += xv * (ACC_T)vget(wx, (unsigned)l);
                accZ[l] += xv * (ACC_T)vget(wz, (unsigned)l);
            }
        }

        DTYPE_VEC outX_raw, outZ_raw;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outX_raw, (unsigned)l, (DTYPE)accX[l]);
            vset(outZ_raw, (unsigned)l, (DTYPE)accZ[l]);
        }

#if INS_Q_AFTER_INPROJ
        DTYPE_VEC outX_q, outZ_q;
        QSCALE_T sx, sz;
        bfpq::quant_vec(outX_raw, outX_q, sx);
        bfpq::quant_vec(outZ_raw, outZ_q, sz);
        X_for_conv_q.write(outX_q);  X_for_conv_s.write(sx);
        Z_for_gate_q.write(outZ_q);  Z_for_gate_s.write(sz);
#else
        X_for_conv_q.write(outX_raw);  X_for_conv_s.write((QSCALE_T)0);
        Z_for_gate_q.write(outZ_raw);  Z_for_gate_s.write((QSCALE_T)0);
#endif
    }
}

// ============================================================
// conv1d + SiLU (no window copy)
// ============================================================
static void conv1d_silu_stream_local_q(hls::stream<DTYPE_VEC>& X_for_conv_q,
                                       hls::stream<QSCALE_T>&  X_for_conv_s,
                                       hls::stream<DTYPE_VEC>& Z_for_gate_q,
                                       hls::stream<QSCALE_T>&  Z_for_gate_s,
                                       const DTYPE kernel_buffer[K],
                                       hls::stream<DTYPE_VEC>& X_gate_q,
                                       hls::stream<QSCALE_T>&  X_gate_s,
                                       hls::stream<DTYPE_VEC>& X_ssm_q,
                                       hls::stream<QSCALE_T>&  X_ssm_s) {
#pragma HLS INLINE off

    static DTYPE line_buffer[K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=lutram

    // init (determinism)
    for (int i = 0; i < K-1; ++i) {
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS PIPELINE II=1
            line_buffer[i][k] = 0;
        }
    }

    // DO NOT request UNROLL for this loop; give tripcount instead (fixes HLS 214-187 spam).
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
#pragma HLS PIPELINE II=2
        DTYPE_VEC xq = X_for_conv_q.read();
        QSCALE_T  xs = X_for_conv_s.read();
        DTYPE_VEC zq = Z_for_gate_q.read();
        QSCALE_T  zs = Z_for_gate_s.read();

        // gate = SiLU(dequant(z))
        DTYPE_VEC gate_raw;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            ACC_T z_deq = bfpq::deq_lane(vget(zq, (unsigned)k), zs);
            vset(gate_raw, (unsigned)k, silu_elem((DTYPE)z_deq));
        }

#if INS_Q_AFTER_CONV
        DTYPE_VEC gate_q;
        QSCALE_T  gs;
        bfpq::quant_vec(gate_raw, gate_q, gs);
        X_gate_q.write(gate_q);
        X_gate_s.write(gs);
#else
        X_gate_q.write(gate_raw);
        X_gate_s.write((QSCALE_T)0);
#endif

        // x_new (dequant)
        DTYPE x_new[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=x_new complete dim=1
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            ACC_T x_deq = bfpq::deq_lane(vget(xq, (unsigned)k), xs);
            x_new[k] = (DTYPE)x_deq;
        }

        // conv
        DTYPE_VEC conv_raw;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
            ACC_T sum = 0;
            for (int kk = 0; kk < K-1; ++kk) {
#pragma HLS UNROLL
                sum += (ACC_T)kernel_buffer[kk] * (ACC_T)line_buffer[kk][lane];
            }
            sum += (ACC_T)kernel_buffer[K-1] * (ACC_T)x_new[lane];
            vset(conv_raw, (unsigned)lane, (DTYPE)sum);
        }

        // shift
        for (int kk = K-2; kk > 0; --kk) {
            for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
                line_buffer[kk][lane] = line_buffer[kk-1][lane];
            }
        }
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
            line_buffer[0][lane] = x_new[lane];
        }

        // ssm input = SiLU(conv)
        DTYPE_VEC ssm_raw;
        for (int k = 0; k < VEC_FACTOR; ++k) {
#pragma HLS UNROLL
            vset(ssm_raw, (unsigned)k, silu_elem(vget(conv_raw, (unsigned)k)));
        }

#if INS_Q_AFTER_CONV
        DTYPE_VEC ssm_q;
        QSCALE_T  ss;
        bfpq::quant_vec(ssm_raw, ssm_q, ss);
        X_ssm_q.write(ssm_q);
        X_ssm_s.write(ss);
#else
        X_ssm_q.write(ssm_raw);
        X_ssm_s.write((QSCALE_T)0);
#endif
    }
}

// ============================================================
// projections
// ============================================================
static void projection_streams_local(hls::stream<DTYPE_VEC>& X_ssm_in,
                                     const DTYPE_VEC W_B[N][VEC_D],
                                     const DTYPE_VEC W_C[N][VEC_D],
                                     const DTYPE_VEC W_delta[VEC_D][VEC_D],
                                     hls::stream<DTYPE_VEC>& B_out_N,
                                     hls::stream<DTYPE_VEC>& C_out_N,
                                     hls::stream<DTYPE_VEC>& delta_out) {
#pragma HLS INLINE off
    const int J_TILE = 2;

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=bram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=2 dim=1

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        X_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    // delta
    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        ACC_T acc[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=2
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC w = W_delta[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    acc[l] += (ACC_T)vget(X_tile[jj], (unsigned)l) * (ACC_T)vget(w, (unsigned)l);
                }
            }
        }

        DTYPE_VEC delta_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(delta_vec, (unsigned)l, softplus_pwl_fx(acc[l]));
        }
        delta_out.write(delta_vec);
    }

    // B + C
    for (int i = 0; i < N; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        ACC_T accB[VEC_FACTOR];
        ACC_T accC[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accB complete dim=1
#pragma HLS ARRAY_PARTITION variable=accC complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accB[l] = (ACC_T)0;
            accC[l] = (ACC_T)0;
        }

        for (int jt = 0; jt < VEC_D; jt += J_TILE) {
#pragma HLS PIPELINE II=2
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                X_tile[jj] = X_buf[jt + jj];
            }

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                DTYPE_VEC wB = W_B[i][jt + jj];
                DTYPE_VEC wC = W_C[i][jt + jj];
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    ACC_T x = (ACC_T)vget(X_tile[jj], (unsigned)l);
                    accB[l] += x * (ACC_T)vget(wB, (unsigned)l);
                    accC[l] += x * (ACC_T)vget(wC, (unsigned)l);
                }
            }
        }

        DTYPE_VEC outB, outC;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(outB, (unsigned)l, (DTYPE)accB[l]);
            vset(outC, (unsigned)l, (DTYPE)accC[l]);
        }
        B_out_N.write(outB);
        C_out_N.write(outC);
    }
}

// ============================================================
// fused update
// KEY FIXES here:
// - NO reinterpret_cast to ap_uint*  -> avoids RTGEN dangling AXI ports.
// - Use scalar banked accumulators -> avoids vector RAM RMW hazard.
// - In synthesis: normal pointer writes allow burst inference.
// - In cosim: DO NOT use volatile alias (breaks operator= for hls::vector).
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

    // IMPORTANT: keep non-volatile pointers for both synth + cosim
    // (volatile hls::vector breaks assignment operator in Vitis HLS C-sim/cosim)
    DTYPE_VEC* Cw  = C_ddr;
    DTYPE_VEC* H1w = H1_ddr;

    DTYPE_VEC X_gate[VEC_D];
#pragma HLS BIND_STORAGE variable=X_gate type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        X_gate[j] = X_gate_in.read();
    }

    DTYPE_VEC X_ssm_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_ssm_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        X_ssm_buf[j] = X_ssm_in.read();
    }

    DTYPE_VEC delta_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=delta_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        delta_buf[j] = delta_in.read();
    }

    // scalar banked accumulator
    ACC_T acc_s[VEC_D][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=acc_s complete dim=2
#pragma HLS ARRAY_PARTITION variable=acc_s cyclic factor=4 dim=1

    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            acc_s[j][l] = (ACC_T)0;
        }
    }

    for (int i = 0; i < N; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=16384
        DTYPE_VEC A_vec = A_fixed[i];
        DTYPE_VEC B_vec = B_in.read();
        DTYPE_VEC C_vec = C_in.read();

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
            // use a wider type for idx math (avoid SYNCHK overflow assumptions)
            ap_uint<32> idx = (ap_uint<32>)i * (ap_uint<32>)VEC_D + (ap_uint<32>)j;

            DTYPE_VEC H0v  = H0_in.read();
            DTYPE_VEC dlt  = delta_buf[j];
            DTYPE_VEC xssm = X_ssm_buf[j];

            DTYPE_VEC H1v;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                ACT_T a   = (ACT_T)vget(A_vec, (unsigned)l);
                ACT_T dl  = (ACT_T)vget(dlt,   (unsigned)l);
                ACT_T adl = a * dl;
                adl = clamp_fx<ACT_T>(adl, (ACT_T)-3.0, (ACT_T)3.0);
                EXP_T ddA_fx = exp2_poly_fx(adl);

                ACC_T H0  = (ACC_T)vget(H0v,  (unsigned)l);
                ACC_T Bx  = (ACC_T)vget(B_vec,(unsigned)l);
                ACC_T Xs  = (ACC_T)vget(xssm, (unsigned)l);

                ACC_T H1  = H0 * (ACC_T)ddA_fx + (Bx * (ACC_T)dl) * Xs;
                DTYPE h1d = (DTYPE)H1;
                vset(H1v, (unsigned)l, h1d);

                // accumulate
                ACC_T c = (ACC_T)vget(C_vec, (unsigned)l);
                acc_s[j][l] += (ACC_T)h1d * c;
            }

            // AXI writes (now real, no dangling RTGEN ports)
            Cw[(unsigned)idx]  = C_vec;
            H1w[(unsigned)idx] = H1v;
        }
    }

    // output = gate * acc
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        DTYPE_VEC gate = X_gate[j];
        DTYPE_VEC outv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T g = (ACC_T)vget(gate, (unsigned)l);
            ACC_T y = acc_s[j][l];
            vset(outv, (unsigned)l, (DTYPE)(g * y));
        }
        out.write(outv);
    }
}

// ============================================================
// out proj
// ============================================================
static void out_proj_stream_local(hls::stream<DTYPE_VEC>& X_in,
                                  const DTYPE_VEC W_out[VEC_D][VEC_D],
                                  hls::stream<DTYPE_VEC>& Y_out) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[VEC_D];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=lutram
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        X_buf[j] = X_in.read();
    }

    for (int i = 0; i < VEC_D; ++i) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
        ACC_T accY[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=accY complete dim=1
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            accY[l] = (ACC_T)0;
        }

        for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=4096
            DTYPE_VEC x = X_buf[j];
            DTYPE_VEC w = W_out[i][j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                accY[l] += (ACC_T)vget(x, (unsigned)l) * (ACC_T)vget(w, (unsigned)l);
            }
        }

        DTYPE_VEC y;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(y, (unsigned)l, (DTYPE)accY[l]);
        }
        Y_out.write(y);
    }
}

// ============================================================
// TOP
// ============================================================
void SSMU(hls::stream<DTYPE>& kernel_in,
          const DTYPE_VEC A_fixed[N],
          const DTYPE_VEC RMS_weight[VEC_D],
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

    // ---- interfaces ----
#pragma HLS INTERFACE ap_fifo port=kernel_in depth=512
#pragma HLS INTERFACE ap_fifo port=X_in      depth=512
#pragma HLS INTERFACE ap_fifo port=H0_in     depth=512
#pragma HLS INTERFACE ap_fifo port=out       depth=512

    // Enable bursting/outstanding to help latency (tune as needed).
#pragma HLS INTERFACE m_axi port=C_ddr  offset=slave bundle=gmem0 depth=HUGE_LEN \
  max_write_burst_length=64 num_write_outstanding=16
#pragma HLS INTERFACE m_axi port=H1_ddr offset=slave bundle=gmem1 depth=HUGE_LEN \
  max_write_burst_length=64 num_write_outstanding=16

#pragma HLS INTERFACE s_axilite port=C_ddr  bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    // ---- preload kernel before DATAFLOW ----
    DTYPE kernel_local[K];
#pragma HLS ARRAY_PARTITION variable=kernel_local complete dim=1
    for (int i = 0; i < K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_local[i] = kernel_in.read();
    }

    // ========================================================
    // Internal streams (declare before DATAFLOW)
    // ========================================================
    hls::stream<DTYPE_VEC> X_in_norm("X_in_norm");
    hls::stream<DTYPE_VEC> X_in_D   ("X_in_D");
    hls::stream<DTYPE_VEC> X_in_res ("X_in_res");
#pragma HLS STREAM variable=X_in_norm depth=512
#pragma HLS STREAM variable=X_in_D    depth=512
#pragma HLS STREAM variable=X_in_res  depth=512

    // H0: copy from caller stream into internal stream (mitigate 200-1449/1450)
    hls::stream<DTYPE_VEC> H0_stream("H0_stream");
#pragma HLS STREAM variable=H0_stream depth=512

    hls::stream<DTYPE_VEC> X_normed("X_normed");
#pragma HLS STREAM variable=X_normed depth=512

    hls::stream<DTYPE_VEC> X_norm_q("X_norm_q");
    hls::stream<QSCALE_T>  X_norm_s("X_norm_s");
#pragma HLS STREAM variable=X_norm_q depth=512
#pragma HLS STREAM variable=X_norm_s depth=512

    hls::stream<DTYPE_VEC> X_for_conv_q("X_for_conv_q");
    hls::stream<QSCALE_T>  X_for_conv_s("X_for_conv_s");
    hls::stream<DTYPE_VEC> Z_for_gate_q("Z_for_gate_q");
    hls::stream<QSCALE_T>  Z_for_gate_s("Z_for_gate_s");
#pragma HLS STREAM variable=X_for_conv_q depth=512
#pragma HLS STREAM variable=X_for_conv_s depth=512
#pragma HLS STREAM variable=Z_for_gate_q depth=512
#pragma HLS STREAM variable=Z_for_gate_s depth=512

    hls::stream<DTYPE_VEC> X_gate_q("X_gate_q");
    hls::stream<QSCALE_T>  X_gate_s("X_gate_s");
    hls::stream<DTYPE_VEC> X_ssm_q("X_ssm_q");
    hls::stream<QSCALE_T>  X_ssm_s("X_ssm_s");
#pragma HLS STREAM variable=X_gate_q depth=512
#pragma HLS STREAM variable=X_gate_s depth=512
#pragma HLS STREAM variable=X_ssm_q  depth=512
#pragma HLS STREAM variable=X_ssm_s  depth=512

    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream ("X_ssm_stream");
#pragma HLS STREAM variable=X_gate_stream depth=512
#pragma HLS STREAM variable=X_ssm_stream  depth=512

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_upd_stream ("X_ssm_upd_stream");
#pragma HLS STREAM variable=X_ssm_proj_stream depth=512
#pragma HLS STREAM variable=X_ssm_upd_stream  depth=512

    hls::stream<DTYPE_VEC> B_stream_N("B_stream_N");
    hls::stream<DTYPE_VEC> C_stream_N("C_stream_N");
    hls::stream<DTYPE_VEC> delta_stream("delta_stream");
#pragma HLS STREAM variable=B_stream_N   depth=512
#pragma HLS STREAM variable=C_stream_N   depth=512
#pragma HLS STREAM variable=delta_stream depth=512

    // Optional: copy delta stream before fused_update (sometimes helps 200-1449/1450)
    hls::stream<DTYPE_VEC> delta_for_upd("delta_for_upd");
#pragma HLS STREAM variable=delta_for_upd depth=512

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
#pragma HLS STREAM variable=ssm_core_out_stream depth=512

    hls::stream<DTYPE_VEC> out_proj_stream("out_proj_stream");
#pragma HLS STREAM variable=out_proj_stream depth=512

    hls::stream<DTYPE_VEC> after_D_stream("after_D_stream");
#pragma HLS STREAM variable=after_D_stream depth=512

    // ========================================================
    // Canonical DATAFLOW: ONLY function calls below
    // ========================================================
#pragma HLS DATAFLOW

    tee_vecD_stream3_local(X_in, X_in_norm, X_in_D, X_in_res);

    // copy caller H0 into internal stream
    copy_huge_stream(H0_in, H0_stream);

    rmsnorm_vecD_stream_local(X_in_norm, RMS_weight, X_normed);

#if INS_Q_AFTER_RMSNORM
    bfpq::quant_vecD_stream(X_normed, X_norm_q, X_norm_s);
#else
    bfpq::passthru_vecD_stream_as_q(X_normed, X_norm_q, X_norm_s);
#endif

    in_proj_stream_local_q(X_norm_q, X_norm_s,
                           W_in_x, W_in_z,
                           X_for_conv_q, X_for_conv_s,
                           Z_for_gate_q, Z_for_gate_s);

    conv1d_silu_stream_local_q(X_for_conv_q, X_for_conv_s,
                               Z_for_gate_q, Z_for_gate_s,
                               kernel_local,
                               X_gate_q, X_gate_s,
                               X_ssm_q,  X_ssm_s);

    bfpq::drop_scale_vecD_stream(X_gate_q, X_gate_s, X_gate_stream);
    bfpq::drop_scale_vecD_stream(X_ssm_q,  X_ssm_s,  X_ssm_stream);

    dup_vecD_stream_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_upd_stream);

    projection_streams_local(X_ssm_proj_stream, W_B, W_C, W_delta,
                             B_stream_N, C_stream_N, delta_stream);

    // copy delta before update (mitigation)
    copy_vecD_stream(delta_stream, delta_for_upd);

    fused_update_write_accum_output_mamba(
        X_gate_stream,
        X_ssm_upd_stream,
        delta_for_upd,
        A_fixed,
        B_stream_N,
        C_stream_N,
        H0_stream,
        C_ddr,
        H1_ddr,
        ssm_core_out_stream
    );

    out_proj_stream_local(ssm_core_out_stream, W_out, out_proj_stream);

    add_D_skip_local(out_proj_stream, X_in_D, D_diag, after_D_stream);

    add_residual_local(after_D_stream, X_in_res, out);
}
