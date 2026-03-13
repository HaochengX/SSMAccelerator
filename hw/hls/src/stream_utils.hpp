// =============================================================================
// stream_utils.hpp — Stream copy, tee, drain, dup utilities
// =============================================================================
#ifndef __STREAM_UTILS_HPP__
#define __STREAM_UTILS_HPP__

#include "../config/macro.hpp"

// =============================================================
// Copy / Drain
// =============================================================
static void copy_kernel_k(hls::stream<DTYPE>& in, hls::stream<DTYPE>& out) {
#pragma HLS INLINE off
    for (int i = 0; i < CONV_K; ++i) {
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }
}

static void copy_vec_n(hls::stream<DTYPE_VEC>& in, hls::stream<DTYPE_VEC>& out, int count) {
#pragma HLS INLINE off
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        out.write(in.read());
    }
}

static void drain_vec_n(hls::stream<DTYPE_VEC>& in, int count) {
#pragma HLS INLINE off
    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        (void)in.read();
    }
}

// =============================================================
// Tee / Dup
// =============================================================
static void tee_vec_n_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out_main,
    hls::stream<DTYPE_VEC>& out_trace,
    int n
) {
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out_main.write(v);
#if SSMU_ENABLE_TRACE_STREAMS
        out_trace.write(v);
#else
        (void)out_trace;
#endif
    }
}

static void tee_vecDT_stream2_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2
) {
#pragma HLS INLINE off
    for (int i = 0; i < D_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

static void dup_vecC2_stream3_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2,
    hls::stream<DTYPE_VEC>& out3
) {
#pragma HLS INLINE off
    for (int i = 0; i < C2_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
        out3.write(v);
    }
}

static void dup_vecC2_stream2_local(
    hls::stream<DTYPE_VEC>& in,
    hls::stream<DTYPE_VEC>& out1,
    hls::stream<DTYPE_VEC>& out2
) {
#pragma HLS INLINE off
    for (int i = 0; i < C2_T; ++i) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

// =============================================================
// Preload (GMEM → local)
// =============================================================
template<int N>
static void preload_vec_table_local(const DTYPE_VEC in_gmem[N], DTYPE_VEC out_local[N]) {
#pragma HLS INLINE off
    for (int i = 0; i < N; ++i) {
#pragma HLS PIPELINE II=1
        out_local[i] = in_gmem[i];
    }
}

static void preload_vec_table_local_dyn(const DTYPE_VEC* in_gmem, DTYPE_VEC* out_local, int n) {
#pragma HLS INLINE off
    for (int i = 0; i < n; ++i) {
#pragma HLS PIPELINE II=1
        out_local[i] = in_gmem[i];
    }
}

// =============================================================
// Residual add
// =============================================================
static void add_residual_local_D(
    hls::stream<DTYPE_VEC>& y_in,
    hls::stream<DTYPE_VEC>& x_res_in,
    hls::stream<DTYPE_VEC>& y_out
) {
#pragma HLS INLINE off
#ifndef __SYNTHESIS__
    FILE* f_out_dbg = std::fopen("dut_out_f32.bin", "wb");
#endif
    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC y = y_in.read();
        DTYPE_VEC x = x_res_in.read();
        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(o, l, (DTYPE)((ACC_T)vget(y, l) + (ACC_T)vget(x, l)));
        }
#ifndef __SYNTHESIS__
        if (dbg_tok_sel(j)) {
            DUT_PRINTF("[DBG][out ] tok=%d lane=%d pre=% .6f res=% .6f out=% .6f\n",
                       j, DBG_LANE,
                       (float)vget(y, DBG_LANE),
                       (float)vget(x, DBG_LANE),
                       (float)vget(o, DBG_LANE));
        }
        if (f_out_dbg) dump_vec_token(f_out_dbg, o);
#endif
        y_out.write(o);
    }
#ifndef __SYNTHESIS__
    if (f_out_dbg) std::fclose(f_out_dbg);
#endif
}

// =============================================================
// Buffer readers
// =============================================================
static void read_x_buf_D_local(
    hls::stream<DTYPE_VEC>& x_in,
    DTYPE_VEC x_buf[D_T]
) {
#pragma HLS INLINE off
    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        x_buf[j] = x_in.read();
    }
}

static void read_temp_buf_RANK_local(
    hls::stream<DTYPE_VEC>& t_in,
    DTYPE_VEC t_buf[RANK_T]
) {
#pragma HLS INLINE off
    for (int j = 0; j < RANK_T; ++j) {
#pragma HLS PIPELINE II=1
        t_buf[j] = t_in.read();
    }
}

static void read_x_buf_C2_local(
    hls::stream<DTYPE_VEC>& x_in,
    DTYPE_VEC x_buf[C2_T]
) {
#pragma HLS INLINE off
    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        x_buf[j] = x_in.read();
    }
}

#endif // __STREAM_UTILS_HPP__
