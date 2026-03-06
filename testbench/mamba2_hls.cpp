// ============================================================
// Mamba2 Layer - Vitis HLS (Simplified for fast synthesis)
//
// Design choices to keep synthesis fast:
//   - Small fixed dimensions (not 2.7B scale - that needs
//     external DRAM + multi-layer tiling, out of scope here)
//   - All arrays are fixed-size with HLS pragmas
//   - Inner loops are pipelined; outer loops are NOT unrolled
//   - No recursion, no dynamic allocation
//   - float arithmetic for easier validation/debug
//
// Dimensions (easy to change via #defines):
//   D_MODEL    = 32
//   D_INNER    = 64   (expand=2)
//   D_STATE    = 8
//   NHEADS     = 4    (headdim = D_INNER/NHEADS = 16)
//   CONV_K     = 4
//   SEQ_LEN    = 8
// ============================================================

#include "mamba2_hls.h"
#include <hls_math.h>

// ---- Activation helpers ------------------------------------

static data_t silu(data_t x) {
#pragma HLS INLINE
    return x / (data_t(1) + hls::exp(-x));
}

static data_t softplus(data_t x) {
#pragma HLS INLINE
    // clamp for numerical safety
    if (x > data_t(20)) return x;
    return hls::log(data_t(1) + hls::exp(x));
}

// ---- RMSNorm (over d_inner, applied per token) -------------

static void rmsnorm_gated(
    data_t y[D_INNER],
    const data_t z[D_INNER],
    const data_t norm_w[D_INNER])
{
#pragma HLS INLINE off
    const data_t eps = data_t(1e-5);

    // gate first (norm_before_gate = false, matching reference)
    data_t sq = 0;
    GATE_LOOP: for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
        y[i] = y[i] * silu(z[i]);
        sq += y[i] * y[i];
    }

    data_t inv_rms = data_t(1) / hls::sqrt(sq / data_t(D_INNER) + eps);

    SCALE_LOOP: for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
        y[i] = y[i] * inv_rms * norm_w[i];
    }
}

// ---- Top-level kernel --------------------------------------
//
// Interfaces:
//   x        - input  [SEQ_LEN * D_MODEL]  (AXI stream or array)
//   y        - output [SEQ_LEN * D_MODEL]
//   weights  - read-only parameter bundle
//
// For simplicity all weights are passed as flat arrays.
// In a real design you'd DMA them in once and keep in BRAMs.

void mamba2_forward(
    const data_t x[SEQ_LEN * D_MODEL],
    data_t       y[SEQ_LEN * D_MODEL],
    // input projection
    const data_t w_in [D_MODEL * IN_PROJ_W],
    const data_t b_in [IN_PROJ_W],
    // conv
    const data_t conv_w[XBC_DIM * CONV_K],
    const data_t conv_b[XBC_DIM],
    // SSM params
    const data_t a_log  [NHEADS],
    const data_t d_skip [NHEADS],
    const data_t dt_bias[NHEADS],
    // norm
    const data_t norm_w [D_INNER],
    // output projection
    const data_t w_out[D_INNER * D_MODEL],
    const data_t b_out[D_MODEL])
{
// ---- Interface / memory pragmas ----------------------------
#pragma HLS INTERFACE mode=bram port=x
#pragma HLS INTERFACE mode=bram port=y
#pragma HLS INTERFACE mode=bram port=w_in
#pragma HLS INTERFACE mode=bram port=b_in
#pragma HLS INTERFACE mode=bram port=conv_w
#pragma HLS INTERFACE mode=bram port=conv_b
#pragma HLS INTERFACE mode=bram port=a_log
#pragma HLS INTERFACE mode=bram port=d_skip
#pragma HLS INTERFACE mode=bram port=dt_bias
#pragma HLS INTERFACE mode=bram port=norm_w
#pragma HLS INTERFACE mode=bram port=w_out
#pragma HLS INTERFACE mode=bram port=b_out
#pragma HLS INTERFACE mode=ap_ctrl_hs port=return

    // ---- State (persistent across tokens in one call) ------
    // ssm_state[i][n]  – per-channel per-state
    // conv_buf[pos][i] – circular buffer for depthwise conv
    static data_t ssm_state[D_INNER][D_STATE];
    static data_t conv_buf [CONV_K][XBC_DIM];
    static int    conv_pos;

#pragma HLS ARRAY_PARTITION variable=ssm_state complete dim=2  // partition state dim
#pragma HLS ARRAY_PARTITION variable=conv_buf  complete dim=1  // all conv taps live

    // Reset state at start of each call (recurrent reuse
    // across calls: remove these loops and make state
    // a module-level variable instead).
    RESET_SSM: for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
        for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL
            ssm_state[i][n] = 0;
        }
    }
    RESET_CONV: for (int p = 0; p < CONV_K; p++) {
#pragma HLS PIPELINE II=1
        for (int i = 0; i < XBC_DIM; i++) {
#pragma HLS UNROLL
            conv_buf[p][i] = 0;
        }
    }
    conv_pos = 0;

    // ---- Temporaries (on-chip, reused each token) ----------
    data_t proj    [IN_PROJ_W];
    data_t conv_xbc[XBC_DIM];
    data_t z_tok   [D_INNER];
    data_t inner_out[D_INNER];

#pragma HLS ARRAY_PARTITION variable=proj     cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=conv_xbc cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=z_tok    cyclic factor=4
#pragma HLS ARRAY_PARTITION variable=inner_out cyclic factor=4

    // ========================================================
    // Token loop – NOT pipelined at this level;
    // inner loops are pipelined.
    // ========================================================
    SEQ_LOOP: for (int t = 0; t < SEQ_LEN; t++) {

        // ---- 1. Input projection: proj = x_t @ w_in + b_in
        INIT_PROJ: for (int j = 0; j < IN_PROJ_W; j++) {
#pragma HLS PIPELINE II=1
            proj[j] = b_in[j];
        }

        IN_PROJ_I: for (int i = 0; i < D_MODEL; i++) {
            data_t xi = x[t * D_MODEL + i];
            IN_PROJ_J: for (int j = 0; j < IN_PROJ_W; j++) {
#pragma HLS PIPELINE II=1
                proj[j] += xi * w_in[i * IN_PROJ_W + j];
            }
        }

        // ---- 2. Update conv circular buffer ----------------
        CONV_BUF_WR: for (int i = 0; i < XBC_DIM; i++) {
#pragma HLS PIPELINE II=1
            conv_buf[conv_pos][i] = proj[OFF_XBC + i];
        }

        // ---- 3. Depthwise conv + silu ----------------------
        CONV_CH: for (int i = 0; i < XBC_DIM; i++) {
#pragma HLS PIPELINE II=1
            data_t acc = conv_b[i];
            CONV_TAP: for (int off = 0; off < CONV_K; off++) {
#pragma HLS UNROLL
                int idx = (conv_pos - off + CONV_K) % CONV_K;
                acc += conv_w[i * CONV_K + off] * conv_buf[idx][i];
            }
            conv_xbc[i] = silu(acc);
        }
        conv_pos = (conv_pos + 1) % CONV_K;

        // ---- 4. SSM step (per channel) ---------------------
        //
        // x_tok = conv_xbc[0 .. D_INNER)
        // b_tok = conv_xbc[D_INNER .. D_INNER+D_STATE)
        // c_tok = conv_xbc[D_INNER+D_STATE .. XBC_DIM)
        // dt    = proj[OFF_DT .. OFF_DT+NHEADS)

        SSM_LOOP: for (int i = 0; i < D_INNER; i++) {
#pragma HLS PIPELINE II=1
            int h  = i / HEADDIM;
            z_tok[i] = proj[OFF_Z + i];

            data_t u  = conv_xbc[i];
            data_t dt = softplus(proj[OFF_DT + h] + dt_bias[h]) + data_t(1e-4);
            data_t a  = -hls::exp(a_log[h]);
            data_t decay = hls::exp(dt * a);

            data_t ssm_acc = 0;
            SSM_STATE_LOOP: for (int n = 0; n < D_STATE; n++) {
#pragma HLS UNROLL
                data_t b_n   = conv_xbc[D_INNER + n];
                data_t c_n   = conv_xbc[D_INNER + D_STATE + n];
                data_t s_new = decay * ssm_state[i][n] + dt * b_n * u;
                ssm_state[i][n] = s_new;
                ssm_acc += c_n * s_new;
            }
            inner_out[i] = ssm_acc + d_skip[h] * u;
        }

        // ---- 5. RMSNorm + gate -----------------------------
        rmsnorm_gated(inner_out, z_tok, norm_w);

        // ---- 6. Output projection: y_t = inner_out @ w_out + b_out
        INIT_OUT: for (int j = 0; j < D_MODEL; j++) {
#pragma HLS PIPELINE II=1
            y[t * D_MODEL + j] = b_out[j];
        }

        OUT_PROJ_I: for (int i = 0; i < D_INNER; i++) {
            data_t yi = inner_out[i];
            OUT_PROJ_J: for (int j = 0; j < D_MODEL; j++) {
#pragma HLS PIPELINE II=1
                y[t * D_MODEL + j] += yi * w_out[i * D_MODEL + j];
            }
        }
    } // SEQ_LOOP
}
