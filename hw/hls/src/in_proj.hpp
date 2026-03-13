#ifndef __IN_PROJ_HPP__
#define __IN_PROJ_HPP__

#include "../config/macro.hpp"
#include "stream_utils.hpp"

static void stream_Win1_tiles_local(
    const W_VEC W_in_1[D_T][RANK_T],
    hls::stream<vec_tuple8> Win1_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_in_1 cyclic factor=8 dim=1

    for (int it = 0; it < RANK_T; it += SSMU_I_TILE) {
        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;

                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int jidx = jt + jj;
                    if ((i < RANK_T) && (jidx < D_T)) wbuf[jj] = W_in_1[jidx][i];
                    else                                wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                Win1_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 1 consumer: X @ W_in_1 → temp stream[RANK_T] ----
static void inproj_stage1_consume_local(
    const DTYPE_VEC X_buf[D_T],
    hls::stream<vec_tuple8> Win1_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_in_fx
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

        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false

            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = Win1_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(X_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_in_fx);
                    ACC_T p1 = (ACC_T)vget(X_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_in_fx);
                    ACC_T p2 = (ACC_T)vget(X_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_in_fx);
                    ACC_T p3 = (ACC_T)vget(X_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_in_fx);
                    ACC_T p4 = (ACC_T)vget(X_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_in_fx);
                    ACC_T p5 = (ACC_T)vget(X_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_in_fx);
                    ACC_T p6 = (ACC_T)vget(X_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_in_fx);
                    ACC_T p7 = (ACC_T)vget(X_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_in_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < RANK_T) {
                DTYPE_VEC outv;
                typedef ap_fixed<16,6,AP_RND_CONV,AP_SAT> DTYPE_Q_T;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    DTYPE_Q_T qv = (DTYPE_Q_T)accv[ii][l];
                    vset(outv, (unsigned)l, (DTYPE)qv);
                }
                temp_out.write(outv);
            }
        }
    }
}

// ---- Stage 2 tile streamer: W_in_2[RANK_T][INP_X_T] (middle 5120 only) ----
static void stream_Win2_tiles_local(
    const W_VEC W_in_2[RANK_T][INP_X_T],
    hls::stream<vec_tuple8> Win2_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_in_2 cyclic factor=8 dim=1

    for (int it = 0; it < INP_X_T; it += SSMU_I_TILE) {
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
                    if ((i < INP_X_T) && (ridx < RANK_T)) wbuf[jj] = W_in_2[ridx][i];
                    else                                   wbuf[jj] = wvec_zero();
                }

                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                Win2_tiles[ii].write(tup);
            }
        }
    }
}

// ---- Stage 2 consumer: temp @ W_in_2 → X(mid 5120) stream ----
static void inproj_stage2_consume_local(
    const DTYPE_VEC temp_buf[RANK_T],
    hls::stream<vec_tuple8> Win2_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& X_mid_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    DTYPE_VEC T_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=T_tile complete dim=1

    for (int it = 0; it < INP_X_T; it += SSMU_I_TILE) {

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
                vec_tuple8 wt = Win2_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1

                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(T_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_in_fx);
                    ACC_T p1 = (ACC_T)vget(T_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_in_fx);
                    ACC_T p2 = (ACC_T)vget(T_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_in_fx);
                    ACC_T p3 = (ACC_T)vget(T_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_in_fx);
                    ACC_T p4 = (ACC_T)vget(T_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_in_fx);
                    ACC_T p5 = (ACC_T)vget(T_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_in_fx);
                    ACC_T p6 = (ACC_T)vget(T_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_in_fx);
                    ACC_T p7 = (ACC_T)vget(T_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_in_fx);

                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < INP_X_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }
                X_mid_out.write(outv);
            }
        }
    }
}

// ---- Stage 1: X_normed × W_in_1 → temp_stream ----
static void in_proj_lr_stage1(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_in_1[D_T][RANK_T],
    hls::stream<DTYPE_VEC>& temp_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> Win1_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Win1_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_x_buf_D_local(X_in_d, X_buf);
    stream_Win1_tiles_local(W_in_1, Win1_tiles);
    inproj_stage1_consume_local(X_buf, Win1_tiles, temp_out, wscale_in_fx);
}

// ---- Stage 2: temp × W_in_2 → Z/XBC/DT streams ----
static void in_proj_lr_stage2(
    hls::stream<DTYPE_VEC>& temp_in,
    const W_VEC W_in_2[RANK_T][INP_X_T],
    hls::stream<DTYPE_VEC>& X_mid_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC temp_buf[RANK_T];
#pragma HLS BIND_STORAGE variable=temp_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=temp_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> Win2_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=Win2_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_temp_buf_RANK_local(temp_in, temp_buf);
    stream_Win2_tiles_local(W_in_2, Win2_tiles);
    inproj_stage2_consume_local(temp_buf, Win2_tiles, X_mid_out, wscale_in_fx);
}

template<int OUT_T>
static void stream_Wfull_tiles_local(
    const W_VEC W_full[D_T][OUT_T],
    hls::stream<vec_tuple8> W_tiles[SSMU_I_TILE]
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=W_full cyclic factor=8 dim=1

    for (int it = 0; it < OUT_T; it += SSMU_I_TILE) {
        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                int i = it + ii;
                W_VEC wbuf[J_TILE];
#pragma HLS ARRAY_PARTITION variable=wbuf complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    int jidx = jt + jj;
                    if ((i < OUT_T) && (jidx < D_T)) wbuf[jj] = W_full[jidx][i];
                    else                               wbuf[jj] = wvec_zero();
                }
                vec_tuple8 tup;
#pragma HLS ARRAY_PARTITION variable=tup.w complete dim=1
                for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL
                    tup.w[jj] = wbuf[jj];
                }
                W_tiles[ii].write(tup);
            }
        }
    }
}

template<int OUT_T>
static void inproj_full_consume_local(
    const DTYPE_VEC X_buf[D_T],
    hls::stream<vec_tuple8> W_tiles[SSMU_I_TILE],
    hls::stream<DTYPE_VEC>& out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    DTYPE_VEC X_tile[J_TILE];
#pragma HLS ARRAY_PARTITION variable=X_tile complete dim=1

    for (int it = 0; it < OUT_T; it += SSMU_I_TILE) {
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

        for (int jt = 0; jt < D_T; jt += J_TILE) {
#pragma HLS PIPELINE II=1
#pragma HLS DEPENDENCE variable=accv inter false
            for (int jj = 0; jj < J_TILE; ++jj) {
#pragma HLS UNROLL factor=SSMU_JJ_UNROLL
                int jidx = jt + jj;
                X_tile[jj] = (jidx < D_T) ? X_buf[jidx] : dvec_zero();
            }

            for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS UNROLL
                vec_tuple8 wt = W_tiles[ii].read();
#pragma HLS ARRAY_PARTITION variable=wt.w complete dim=1
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL factor=SSMU_LANE_UNROLL
                    ACC_T p0 = (ACC_T)vget(X_tile[0], (unsigned)l) * wget_scaled(wt.w[0], (unsigned)l, wscale_in_fx);
                    ACC_T p1 = (ACC_T)vget(X_tile[1], (unsigned)l) * wget_scaled(wt.w[1], (unsigned)l, wscale_in_fx);
                    ACC_T p2 = (ACC_T)vget(X_tile[2], (unsigned)l) * wget_scaled(wt.w[2], (unsigned)l, wscale_in_fx);
                    ACC_T p3 = (ACC_T)vget(X_tile[3], (unsigned)l) * wget_scaled(wt.w[3], (unsigned)l, wscale_in_fx);
                    ACC_T p4 = (ACC_T)vget(X_tile[4], (unsigned)l) * wget_scaled(wt.w[4], (unsigned)l, wscale_in_fx);
                    ACC_T p5 = (ACC_T)vget(X_tile[5], (unsigned)l) * wget_scaled(wt.w[5], (unsigned)l, wscale_in_fx);
                    ACC_T p6 = (ACC_T)vget(X_tile[6], (unsigned)l) * wget_scaled(wt.w[6], (unsigned)l, wscale_in_fx);
                    ACC_T p7 = (ACC_T)vget(X_tile[7], (unsigned)l) * wget_scaled(wt.w[7], (unsigned)l, wscale_in_fx);
                    accv[ii][l] = accv[ii][l] + tree_sum8(p0,p1,p2,p3,p4,p5,p6,p7);
                }
            }
        }

        for (int ii = 0; ii < SSMU_I_TILE; ++ii) {
#pragma HLS PIPELINE II=1
            int i = it + ii;
            if (i < OUT_T) {
                DTYPE_VEC outv;
                for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                    vset(outv, (unsigned)l, (DTYPE)accv[ii][l]);
                }
                out.write(outv);
            }
        }
    }
}

static void in_proj_nonlr_stage(
    hls::stream<DTYPE_VEC>& X_in_d,
    const W_VEC W_in_nonlr[D_T][INP_NONLR_T],
    hls::stream<DTYPE_VEC>& nlr_out,
    ACC_T wscale_in_fx
) {
#pragma HLS INLINE off

    DTYPE_VEC X_buf[D_T];
#pragma HLS BIND_STORAGE variable=X_buf type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=X_buf cyclic factor=8 dim=1

    hls::stream<vec_tuple8> W_tiles[SSMU_I_TILE];
#pragma HLS STREAM variable=W_tiles depth=SSMU_DEPTH_TILE

#pragma HLS DATAFLOW
    read_x_buf_D_local(X_in_d, X_buf);
    stream_Wfull_tiles_local<INP_NONLR_T>(W_in_nonlr, W_tiles);
    inproj_full_consume_local<INP_NONLR_T>(X_buf, W_tiles, nlr_out, wscale_in_fx);
}

static void demux_nonlr_local(
    hls::stream<DTYPE_VEC>& nlr_in,
    hls::stream<DTYPE_VEC>& Z_out,
    hls::stream<DTYPE_VEC>& DT_out,
    hls::stream<DTYPE_VEC>& B_out,
    hls::stream<DTYPE_VEC>& C_out
) {
#pragma HLS INLINE off
    for (int i = 0; i < INP_NONLR_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = nlr_in.read();
        if (i < INP_Z_T) {
#ifndef __SYNTHESIS__
            if (dbg_tok_sel(i)) {
                DUT_PRINTF("[DBG][Z   ] tok=%d lane=%d val=% .6f\n",
                           i, DBG_LANE, (float)vget(v, DBG_LANE));
            }
#endif
            Z_out.write(v);
        } else if (i < INP_Z_T + INP_DT_T) {
#if SSMU_ENABLE_DT
            DT_out.write(v);
#else
            DT_out.write(dvec_zero());
#endif
        } else if (i < INP_Z_T + INP_DT_T + INP_B_T) {
            B_out.write(v);
        } else {
            C_out.write(v);
        }
    }
}

static void assemble_xbc_local(
    hls::stream<DTYPE_VEC>& B_in,
    hls::stream<DTYPE_VEC>& C_in,
    hls::stream<DTYPE_VEC>& X_mid_in,
    hls::stream<DTYPE_VEC>& XBC_out
) {
#pragma HLS INLINE off
    for (int i = 0; i < INP_B_T; ++i) {
#pragma HLS PIPELINE II=1
        XBC_out.write(B_in.read());
    }
    for (int i = 0; i < INP_C_T; ++i) {
#pragma HLS PIPELINE II=1
        XBC_out.write(C_in.read());
    }
    for (int i = 0; i < INP_X_T; ++i) {
#pragma HLS PIPELINE II=1
        XBC_out.write(X_mid_in.read());
    }
}

#endif // __IN_PROJ_HPP__
