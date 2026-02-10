#ifndef __UCI_EECS_SSMU_HEADER_LAYERSH_ALIGNED__
#define __UCI_EECS_SSMU_HEADER_LAYERSH_ALIGNED__

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

// =============================================================
// Quantization build switch
// =============================================================
#ifndef SSMU_USE_INT8
#define SSMU_USE_INT8 1
#endif

// =============================================================
// Base constants (layers.h-aligned numbers)
//   layers.h: T=1, D=2560, N=128, K=4, P=64, CP=8
// =============================================================
#ifndef SSMU_T
#define SSMU_T 1
#endif

#ifndef SSMU_D
#define SSMU_D 2560
#endif

#ifndef SSMU_N
#define SSMU_N 128
#endif

#ifndef SSMU_K
#define SSMU_K 4
#endif

#ifndef SSMU_P
#define SSMU_P 64
#endif

#ifndef SSMU_CP
#define SSMU_CP 8
#endif

// =============================================================
// Derived channel sizes (match layers.h equations)
// =============================================================
#ifndef SSMU_C2
#define SSMU_C2 (2 * SSMU_D)                        // 5120
#endif

#ifndef SSMU_C_CONV
#define SSMU_C_CONV (2 * SSMU_D + 2 * SSMU_N)       // 5376
#endif

#ifndef SSMU_CH
#define SSMU_CH (2 * SSMU_D / SSMU_P)               // 80
#endif

#ifndef SSMU_C_IN
#define SSMU_C_IN (SSMU_C2 + SSMU_C_CONV + SSMU_CH) // 10576
#endif

// =============================================================
// Tile counts (CP=8 => tiles = channels/8)
// =============================================================
#ifndef SSMU_D_T
#define SSMU_D_T (SSMU_D / SSMU_CP)                 // 320
#endif

#ifndef SSMU_C2_T
#define SSMU_C2_T (SSMU_C2 / SSMU_CP)               // 640
#endif

#ifndef SSMU_CCONV_T
#define SSMU_CCONV_T (SSMU_C_CONV / SSMU_CP)        // 672
#endif

#ifndef SSMU_CH_T
#define SSMU_CH_T (SSMU_CH / SSMU_CP)               // 10
#endif

#ifndef SSMU_CIN_T
#define SSMU_CIN_T (SSMU_C_IN / SSMU_CP)            // 1322
#endif

// =============================================================
// Sanity checks
// =============================================================
static_assert(SSMU_T == 1, "This SSMU assumes T=1 per call.");
static_assert((SSMU_D % SSMU_CP) == 0, "D % CP must be 0");
static_assert((SSMU_C2 % SSMU_CP) == 0, "C2 % CP must be 0");
static_assert((SSMU_C_CONV % SSMU_CP) == 0, "C_conv % CP must be 0");
static_assert((SSMU_CH % SSMU_CP) == 0, "CH % CP must be 0");
static_assert((SSMU_C_IN % SSMU_CP) == 0, "C_in % CP must be 0");

// =============================================================
// Backward-compatible macros (single source of truth)
// =============================================================
#ifndef SSMU_STATE
#define SSMU_STATE SSMU_N
#endif

#ifndef VEC_FACTOR
#define VEC_FACTOR SSMU_CP
#endif

#ifndef VEC_D
#define VEC_D SSMU_C2_T
#endif

// Some older code uses K macro; keep it but prefer SSMU_K in new code.
#ifndef K
#define K SSMU_K
#endif

// Conv state tiles = K-1 vectors (each vector is VEC_FACTOR lanes)
#ifndef SSMU_CONV_STATE_TILES
#define SSMU_CONV_STATE_TILES (SSMU_K - 1)
#endif

#ifndef HUGE_LEN
#define HUGE_LEN (SSMU_STATE * VEC_D)   // N * C2_T
#endif

// =============================================================
// Types (single source of truth for ssm.cpp)
// =============================================================
typedef ap_fixed<16, 4> DTYPE;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

typedef ap_int<8>  Q8_T;
typedef ap_int<32> QACC_T;

typedef hls::vector<Q8_T,   VEC_FACTOR> Q8VEC;
typedef hls::vector<QACC_T, VEC_FACTOR> QACC_VEC;

#if SSMU_USE_INT8
  typedef Q8VEC W_VEC;      // INT8 packed weights
#else
  typedef DTYPE_VEC W_VEC;  // DTYPE weights
#endif

// Tell ssm.cpp not to redefine types / vget/vset
#ifndef SSMU_HAVE_TYPES
#define SSMU_HAVE_TYPES 1
#endif

// =============================================================
// vget/vset helpers (single source of truth)
// =============================================================
#ifndef SSMU_HAVE_VGET_VSET
#define SSMU_HAVE_VGET_VSET 1
#endif

static inline DTYPE vget(const DTYPE_VEC& v, unsigned idx) {
#pragma HLS INLINE
    return v[idx];
}
static inline void vset(DTYPE_VEC& v, unsigned idx, DTYPE val) {
#pragma HLS INLINE
    v[idx] = val;
}

#if SSMU_USE_INT8
static inline Q8_T vget(const Q8VEC& v, unsigned idx) {
#pragma HLS INLINE
    return v[idx];
}
static inline void vset(Q8VEC& v, unsigned idx, Q8_T val) {
#pragma HLS INLINE
    v[idx] = val;
}
#endif

// =============================================================
// TOP prototype (MATCHES your current ssm.cpp SSMU() signature)
// =============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,

    const DTYPE_VEC A_fixed[SSMU_STATE],
    const DTYPE_VEC RMS_weight[SSMU_D_T],

    const W_VEC W_inproj[SSMU_D_T][SSMU_CIN_T],

    const W_VEC W_B[SSMU_STATE][SSMU_C2_T],
    const W_VEC W_C[SSMU_STATE][SSMU_C2_T],
    const W_VEC W_delta[SSMU_C2_T][SSMU_C2_T],

    const W_VEC W_out[SSMU_D_T][SSMU_C2_T],

    const DTYPE_VEC D_diag[SSMU_C2_T],

    hls::stream<DTYPE_VEC>& X_in,    // D_T tiles
    hls::stream<DTYPE_VEC>& H0_in,   // HUGE_LEN tiles (N*C2_T)

    // conv_state in/out: exactly (K-1) tiles
    hls::stream<DTYPE_VEC>& conv_state_in,
    hls::stream<DTYPE_VEC>& conv_state_out,

    // DDR traces (C and H1) length HUGE_LEN
    DTYPE_VEC* C_ddr,
    DTYPE_VEC* H1_ddr,

    // Export H1 state stream (HUGE_LEN tiles)
    hls::stream<DTYPE_VEC>& H1_out,

    // Final output (D_T tiles)
    hls::stream<DTYPE_VEC>& out,

    // Runtime scales (pass 0.0f to use compile-time defaults)
    float w_scale_in,
    float w_scale_bc,
    float w_scale_delta,
    float w_scale_out
);

#endif // __UCI_EECS_SSMU_HEADER_LAYERSH_ALIGNED__
