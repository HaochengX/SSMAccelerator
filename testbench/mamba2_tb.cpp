// ============================================================
// mamba2_tb.cpp  –  Vitis HLS C-simulation testbench
// ============================================================

#include "mamba2_hls.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>

#define MAMBA_SOFTWARE_NO_MAIN
#include "softwareMamba.c"

static float randf(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static void fill(data_t *p, int n, float lo, float hi) {
    for (int i = 0; i < n; i++) p[i] = data_t(randf(lo, hi));
}

int main() {
    srand(42);

    // Allocate weights
    static data_t w_in  [D_MODEL * IN_PROJ_W];
    static data_t b_in  [IN_PROJ_W];
    static data_t conv_w[XBC_DIM * CONV_K];
    static data_t conv_b[XBC_DIM];
    static data_t a_log  [NHEADS];
    static data_t d_skip [NHEADS];
    static data_t dt_bias[NHEADS];
    static data_t norm_w [D_INNER];
    static data_t w_out  [D_INNER * D_MODEL];
    static data_t b_out  [D_MODEL];

    fill(w_in,   D_MODEL * IN_PROJ_W, -0.02f, 0.02f);
    fill(b_in,   IN_PROJ_W,           -0.01f, 0.01f);
    fill(conv_w, XBC_DIM * CONV_K,    -0.03f, 0.03f);
    fill(conv_b, XBC_DIM,             -0.01f, 0.01f);
    fill(a_log,  NHEADS,              -1.5f,  0.5f);
    fill(d_skip, NHEADS,              -0.05f, 0.05f);
    fill(dt_bias,NHEADS,              -2.0f, -0.5f);
    for (int i = 0; i < D_INNER; i++) norm_w[i] = data_t(1);
    fill(w_out, D_INNER * D_MODEL,    -0.02f, 0.02f);
    fill(b_out, D_MODEL,              -0.01f, 0.01f);

    // Input / output tensors
    static data_t x[SEQ_LEN * D_MODEL];
    static data_t y[SEQ_LEN * D_MODEL];
    static data_t y_ref[SEQ_LEN * D_MODEL];

    fill(x, SEQ_LEN * D_MODEL, -1.0f, 1.0f);

    // Run once
    mamba2_forward(x, y,
                   w_in, b_in,
                   conv_w, conv_b,
                   a_log, d_skip, dt_bias,
                   norm_w,
                   w_out, b_out);

    // Run software reference using explicit config + params.
    Mamba2Config ref_cfg;
    ref_cfg.d_model = D_MODEL;
    ref_cfg.expand = EXPAND;
    ref_cfg.d_inner = D_INNER;
    ref_cfg.d_state = D_STATE;
    ref_cfg.nheads = NHEADS;
    ref_cfg.headdim = HEADDIM;
    ref_cfg.conv_kernel = CONV_K;
    ref_cfg.rmsnorm = true;
    ref_cfg.norm_before_gate = false;

    bool ref_ok = mamba2_software_top(
        &ref_cfg,
        x, SEQ_LEN, y_ref,
        w_in, b_in,
        conv_w, conv_b,
        a_log, d_skip, dt_bias,
        norm_w,
        w_out, b_out);

    if (!ref_ok) {
        printf("Software reference call failed\n");
        return 1;
    }

    // Basic sanity checks
    bool all_finite = true;
    float mean_abs  = 0.0f;
    float mean_abs_ref = 0.0f;
    float max_abs_diff = 0.0f;
    float mean_abs_diff = 0.0f;
    for (int i = 0; i < SEQ_LEN * D_MODEL; i++) {
        float v_hls = (float)y[i];
        float v_ref = (float)y_ref[i];
        if (!std::isfinite(v_hls) || !std::isfinite(v_ref)) {
            all_finite = false;
            break;
        }
        float d = fabsf(v_hls - v_ref);
        mean_abs += fabsf(v_hls);
        mean_abs_ref += fabsf(v_ref);
        max_abs_diff = std::max(max_abs_diff, d);
        mean_abs_diff += d;
    }
    mean_abs /= (float)(SEQ_LEN * D_MODEL);
    mean_abs_ref /= (float)(SEQ_LEN * D_MODEL);
    mean_abs_diff /= (float)(SEQ_LEN * D_MODEL);

    const float atol = 1e-5f;
    bool match = all_finite && (max_abs_diff <= atol);

    printf("mean(|y_hls|) = %.6f\n", mean_abs);
    printf("mean(|y_ref|) = %.6f\n", mean_abs_ref);
    printf("max_abs_diff = %.8f\n", max_abs_diff);
    printf("mean_abs_diff = %.8f\n", mean_abs_diff);
    printf("all_finite = %s\n", all_finite ? "true" : "false");

    if (match) {
        printf("COMPARE PASSED (atol=%.1e)\n", atol);
        return 0;
    } else {
        printf("COMPARE FAILED (atol=%.1e)\n", atol);
        return 1;
    }
}
