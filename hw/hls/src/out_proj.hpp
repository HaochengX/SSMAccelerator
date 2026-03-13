// =============================================================================
// out_proj.hpp — Stage-6 output (y*z*g) + low-rank output projection
// =============================================================================
#ifndef __OUT_PROJ_HPP__
#define __OUT_PROJ_HPP__

#include "../config/macro.hpp"

// =============================================================
// Stage 6: out = gate * (htC + D*X)
// =============================================================
static void stage6_out_yz_vec_local(
    hls::stream<DTYPE_VEC>& htC_in,
    const DTYPE_VEC         D_diag[C2_T],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& G_in,
    hls::stream<DTYPE_VEC>& out
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC htC  = htC_in.read();
        DTYPE_VEC xvec = X_in.read();
        DTYPE_VEC gvec = G_in.read();
        DTYPE_VEC dvec = D_diag[j];

        DTYPE_VEC outv;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            ACC_T ht = (ACC_T)vget(htC,  l);
            ACC_T x  = (ACC_T)vget(xvec, l);
            ACC_T d  = (ACC_T)vget(dvec, l);
            ACC_T g  = (ACC_T)vget(gvec, l);
            ACC_T s6_dx, s6_yg;
#pragma HLS BIND_OP variable=s6_dx op=mul impl=fabric
#pragma HLS BIND_OP variable=s6_yg op=mul impl=fabric
            s6_dx = d * x;
            ACC_T y  = ht + s6_dx;
            s6_yg = y * g;
            vset(outv, l, (DTYPE)s6_yg);
        }
#ifndef __SYNTHESIS__
        if (dbg_tok_sel(j)) {
            DUT_PRINTF("[DBG][s6  ] tok=%d lane=%d htC=% .6f x=% .6f g=% .6f d=% .6f out=% .6f\n",
                       j, DBG_LANE,
                       (float)vget(htC, DBG_LANE),
                       (float)vget(xvec, DBG_LANE),
                       (float)vget(gvec, DBG_LANE),
                       (float)vget(dvec, DBG_LANE),
                       (float)vget(outv, DBG_LANE));
        }
#endif
        out.write(outv);
    }
}

// ############################################################
// LOW-RANK OUTPUT PROJECTION
// ############################################################

// ---- Stage 1 tile streamer: W_out_B[RANK_T][C2_T] ----
static void stream_WoutB_tiles_local(
    const W_VEC W_out_B[RANK_T][C2_T],
    hls::stream<vec_tuple8> WoB_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_out_B cyclic factor=8 dim=2

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {
        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int jidx = jt + jj;
                    if ((i < RANK_T) && (jidx < C2_T)) wbuf[jj] = W_out_B[i][jidx];
                    else                                 wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                WoB_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 1 consumer: W_out_B @ X -> temp stream[RANK_T] ----
static void outproj_stage1_consume_local(
    const DTYPE_VEC X_buf[C2_T],
    hls::stream<vec_tuple8> WoB_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int jt = 0; jt < C2_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < C2_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = WoB_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(X_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_out_fx);
                    ACC_T p1 = (ACC_T)vget(X_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_out_fx);
                    ACC_T p2 = (ACC_T)vget(X_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_out_fx);
                    ACC_T p3 = (ACC_T)vget(X_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_out_fx);
                    ACC_T p4 = (ACC_T)vget(X_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_out_fx);
                    ACC_T p5 = (ACC_T)vget(X_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_out_fx);
                    ACC_T p6 = (ACC_T)vget(X_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_out_fx);
                    ACC_T p7 = (ACC_T)vget(X_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_out_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < RANK_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }
                temp_out.write(outv);
            }
        }
    }
}

// ---- Stage 2 tile streamer: W_out_A[D_T][RANK_T] ----
static void stream_WoutA_tiles_local(
    const W_VEC W_out_A[D_T][RANK_T],
    hls::stream<vec_tuple8> WoA_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_out_A cyclic factor=8 dim=2

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {
        for (int rt = 0; rt < RANK_T; rt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int ridx = rt + jj;
                    if ((i < D_T) && (ridx < RANK_T)) wbuf[jj] = W_out_A[i][ridx];
                    else                                wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                WoA_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 2 consumer: W_out_A @ temp -> Y[D_T] stream ----
static void outproj_stage2_consume_local(
    const DTYPE_VEC temp_buf[RANK_T],
    hls::stream<vec_tuple8> WoA_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

#ifndef __SYNTHESIS__
    FILE* f_opj_dbg = std::fopen("dut_out_proj_f32.bin", "wb");
#endif

    DTYPE_VEC T_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=T_tile complete dim=1

    for (int it = 0; it < D_T; it += SSMU_I_TILE) {

        hls::vector<ACC_T, VEC_FACTOR> accv[SSMU_I_TILE];
#pragma HLS ARRAY_PARTITION variable=accv complete dim=1
        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
            hls::vector<ACC_T, VEC_FACTOR> z;
            for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                z[l] = (ACC_T)0;
            }
            accv[ii] = z;
        }

        for (int rt = 0; rt < RANK_T; rt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int ridx = rt + jj;
                T_tile[jj] = (ridx < RANK_T) ? temp_buf[ridx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = WoA_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(T_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_out_fx);
                    ACC_T p1 = (ACC_T)vget(T_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_out_fx);
                    ACC_T p2 = (ACC_T)vget(T_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_out_fx);
                    ACC_T p3 = (ACC_T)vget(T_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_out_fx);
                    ACC_T p4 = (ACC_T)vget(T_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_out_fx);
                    ACC_T p5 = (ACC_T)vget(T_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_out_fx);
                    ACC_T p6 = (ACC_T)vget(T_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_out_fx);
                    ACC_T p7 = (ACC_T)vget(T_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_out_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < D_T) {
                DTYPE_VEC y;
                typedef ap_fixed<16,6,AP_RND_CONV,AP_SAT> DTYPE_Q_T;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    DTYPE_Q_T qv = (DTYPE_Q_T)accv[ii][l];
                    vset(y, (unsigned)l, (DTYPE)qv);
                }
#ifndef __SYNTHESIS__
                if (dbg_tok_sel(i)) {
                    DUT_PRINTF("[DBG][opj ] tok=%d lane=%d out=% .6f\n",
                               i, DBG_LANE, (float)vget(y, DBG_LANE));
                }
#endif
#ifndef __SYNTHESIS__
                if (f_opj_dbg) dump_vec_token(f_opj_dbg, y);
#endif
                Y_out.write(y);
            }
        }
    }
#ifndef __SYNTHESIS__
    if (f_opj_dbg) std::fclose(f_opj_dbg);
#endif
}

// ---- low-rank out_proj Stage 1: stage6_out x W_out_B -> temp_stream ----
static void out_proj_lr_stage1(
    hls::stream<DTYPE_VEC>& X_in,
    const W_VEC W_out_B[RANK_T][C2_T],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[C2_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> WoB_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=WoB_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_x_buf_C2_local(X_in, X_buf);
    stream_WoutB_tiles_local(W_out_B, WoB_tiles);
    outproj_stage1_consume_local(X_buf, WoB_tiles, temp_out, wscale_out_fx);
}

// ---- low-rank out_proj Stage 2: temp x W_out_A -> Y[D_T] stream ----
static void out_proj_lr_stage2(
    hls::stream<DTYPE_VEC>& temp_in,
    const W_VEC W_out_A[D_T][RANK_T],
    hls::stream<DTYPE_VEC>& Y_out,
    ACC_T wscale_out_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC temp_buf[RANK_T];
#pragma HLS BIND_STORAGE variable=temp_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> WoA_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=WoA_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_temp_buf_RANK_local(temp_in, temp_buf);
    stream_WoutA_tiles_local(W_out_A, WoA_tiles);
    outproj_stage2_consume_local(temp_buf, WoA_tiles, Y_out, wscale_out_fx);
}

#endif // __OUT_PROJ_HPP__
