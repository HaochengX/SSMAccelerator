#pragma once

// ============================================================
// mamba2_hls.h  –  Dimension constants & type definitions
// ============================================================

// ---- Numeric type -----------------------------------------
// Use float throughout for easier bring-up/debug.
typedef float data_t;

// ---- Model dimensions (edit here to scale up/down) --------
#define D_MODEL   768
#define EXPAND    2
#define D_INNER   (D_MODEL * EXPAND)
#define D_STATE   64
#define HEADDIM   64
#define NHEADS    (D_INNER / HEADDIM)
#define CONV_K    4
#define SEQ_LEN   2

// ---- Derived constants ------------------------------------
#define XBC_DIM   (D_INNER + 2 * D_STATE)          
#define IN_PROJ_W (2 * D_INNER + 2 * D_STATE + NHEADS) 

// Offsets within the packed projection vector
#define OFF_Z    0
#define OFF_XBC  D_INNER
#define OFF_DT   (OFF_XBC + XBC_DIM)

// ---- Top-level function prototype -------------------------
void mamba2_forward(
    const data_t x    [SEQ_LEN * D_MODEL],
    data_t       y    [SEQ_LEN * D_MODEL],
    const data_t w_in [D_MODEL * IN_PROJ_W],
    const data_t b_in [IN_PROJ_W],
    const data_t conv_w[XBC_DIM * CONV_K],
    const data_t conv_b[XBC_DIM],
    const data_t a_log  [NHEADS],
    const data_t d_skip [NHEADS],
    const data_t dt_bias[NHEADS],
    const data_t norm_w [D_INNER],
    const data_t w_out[D_INNER * D_MODEL],
    const data_t b_out[D_MODEL]);
