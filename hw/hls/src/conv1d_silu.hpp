// =============================================================================
// conv1d_silu.hpp — Conv1D + SiLU with state management
// =============================================================================
#ifndef __CONV1D_SILU_HPP__
#define __CONV1D_SILU_HPP__

#include "../config/macro.hpp"

// =============================================================
// Conv1D + SiLU stream with state (v2: unchanged from v2)
// Processes XBC bundle: [B:STATE_V][C:STATE_V][x:C2_T]
// =============================================================
static void conv1d_silu_stream_local_with_state(
    hls::stream<DTYPE_VEC>& XBC_in,
    hls::stream<DTYPE_VEC>& Z_in,
    hls::stream<DTYPE>&     kernel_in,
    hls::stream<DTYPE_VEC>& conv_state_in,
    hls::stream<DTYPE_VEC>& conv_state_out,
    hls::stream<DTYPE_VEC>& G_out,
    hls::stream<DTYPE_VEC>& X_ssm_out,
    hls::stream<DTYPE_VEC>& B_conv_out,
    hls::stream<DTYPE_VEC>& C_conv_out
) {
#pragma HLS INLINE off

    DTYPE line_buffer[CONV_K-1][VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=line_buffer complete dim=2
#pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=lutram

    for (int k = 0; k < CONV_K-1; ++k) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC sv = conv_state_in.read();
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            line_buffer[k][l] = vget(sv, (unsigned)l);
        }
    }

    DTYPE kernel_buffer[CONV_K];
#pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
    for (int i = 0; i < CONV_K; ++i) {
#pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }

    for (int i = 0; i < CCONV_T; ++i) {
#pragma HLS PIPELINE II=1
        // LightMamba-aligned XBC ordering: [B:STATE_V][C:STATE_V][x:C2_T]
        const bool do_b  = (i < STATE_V);
        const bool do_c  = (i >= STATE_V && i < 2 * STATE_V);
        const bool do_c2 = (i >= 2 * STATE_V);

        if (do_c2) {
            DTYPE_VEC z_in = Z_in.read();
            DTYPE_VEC gate_out;
            for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
                vset(gate_out, l, silu_fx(vget(z_in, l)));
            }
            G_out.write(gate_out);
        }

        DTYPE_VEC x_in = XBC_in.read();

        DTYPE window0[VEC_FACTOR], window1[VEC_FACTOR], window2[VEC_FACTOR], window3[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=window0 complete
#pragma HLS ARRAY_PARTITION variable=window1 complete
#pragma HLS ARRAY_PARTITION variable=window2 complete
#pragma HLS ARRAY_PARTITION variable=window3 complete

        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            window0[l] = line_buffer[2][l];
            window1[l] = line_buffer[1][l];
            window2[l] = line_buffer[0][l];
            window3[l] = vget(x_in, (unsigned)l);
        }

        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            line_buffer[2][l] = line_buffer[1][l];
            line_buffer[1][l] = line_buffer[0][l];
            line_buffer[0][l] = vget(x_in, (unsigned)l);
        }

        {
            DTYPE_VEC ssm_out;
            for (unsigned lane = 0; lane < (unsigned)VEC_FACTOR; ++lane) {
#pragma HLS UNROLL
                // R16: fabric BIND_OP removed — conv muls use DSP (saves ~12,800 LUT, costs ~32 DSP)
                ACC_T cp0, cp1, cp2, cp3;
                cp0 = (ACC_T)kernel_buffer[0] * (ACC_T)window0[lane];
                cp1 = (ACC_T)kernel_buffer[1] * (ACC_T)window1[lane];
                cp2 = (ACC_T)kernel_buffer[2] * (ACC_T)window2[lane];
                cp3 = (ACC_T)kernel_buffer[3] * (ACC_T)window3[lane];
                vset(ssm_out, lane, silu_fx((DTYPE)(cp0 + cp1 + cp2 + cp3)));
            }
            if (do_b)       B_conv_out.write(ssm_out);
            else if (do_c)  C_conv_out.write(ssm_out);
            else if (do_c2) X_ssm_out.write(ssm_out);
        }
    }

    for (int k = 0; k < CONV_K-1; ++k) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC sv;
        for (int l = 0; l < VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            vset(sv, (unsigned)l, line_buffer[k][l]);
        }
        conv_state_out.write(sv);
    }
}

#endif // __CONV1D_SILU_HPP__
