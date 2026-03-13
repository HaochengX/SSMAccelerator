// =============================================================================
// ssm_scan.hpp — SSM state-space scan: dA, stage45 update/reduce, DDR writer
// =============================================================================
#ifndef __SSM_SCAN_HPP__
#define __SSM_SCAN_HPP__

#include "../config/macro.hpp"

// =============================================================
// Stage 3: Compute dA = exp(A * delta) for each state
// =============================================================
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

// =============================================================
// Stage 4+5: SSM state update + reduction
// H1 = dA * H0 + (B * delta) * X
// htC = sum_n(H1 * C)
// =============================================================
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

// =============================================================
// DDR writer for trace streams
// =============================================================
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

#endif // __SSM_SCAN_HPP__
