// =============================================================================
// dt_adapt.hpp — DT adaptation, delta projection, and softplus transform
// =============================================================================
#ifndef __DT_ADAPT_HPP__
#define __DT_ADAPT_HPP__

#include "../config/macro.hpp"

// =============================================================
// DT adaptation: expand CH_T → C2_T by cyclic repeat
// =============================================================
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

// =============================================================
// Drain DT and pad zero for C2_T
// =============================================================
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

// =============================================================
// DT → delta via softplus
// =============================================================
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

// =============================================================
// W_delta tile streamers
// =============================================================
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

// =============================================================
// Delta projection (X_ssm × W_delta → delta via softplus)
// =============================================================
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

#endif // __DT_ADAPT_HPP__
