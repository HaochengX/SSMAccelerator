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
#define SSMU_USE_INT8 0
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

// =============================================================
// In-projection output channel count
//   Layout expected by ssm.cpp (channel units):
//   [ Z: C2 ] [ XBC: C_CONV ] [ DT: CH ] [ B: N ] [ C: N ]
// =============================================================
#ifndef SSMU_C_IN
#define SSMU_C_IN (SSMU_C2 + SSMU_C_CONV + SSMU_CH + 2 * SSMU_N) // 10832
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
#define SSMU_CIN_T (SSMU_C_IN / SSMU_CP)            // 1354
#endif

// N tiles (vector-token count for STATE dimension)
#ifndef SSMU_STATE_T
#define SSMU_STATE_T (SSMU_N / SSMU_CP)             // 16
#endif

// =============================================================
// Sanity checks
// =============================================================
static_assert(SSMU_T == 1, "This SSMU assumes T=1 per call.");
static_assert((SSMU_D % SSMU_CP) == 0, "D % CP must be 0");
static_assert((SSMU_N % SSMU_CP) == 0, "N % CP must be 0");
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

#ifndef K
#define K SSMU_K
#endif

#ifndef SSMU_CONV_STATE_TILES
#define SSMU_CONV_STATE_TILES (SSMU_K - 1)
#endif

// =============================================================
// ✅ FIX: define HUGE_LEN / H1 lengths in *vector tokens*
// =============================================================
#ifndef HUGE_LEN
#define HUGE_LEN (SSMU_STATE_T * SSMU_C2_T)   // 16 * 640 = 10240 vectors
#endif

#ifndef SSMU_H1_OUT_LEN
#define SSMU_H1_OUT_LEN (SSMU_STATE_T * SSMU_C2_T)
#endif

// =============================================================
// Types
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
// TOP prototype (MUST match ssm.cpp SSMU() signature)
//   ✅ W_B/W_C removed
//   ✅ w_scale_bc removed
//   ✅ ✅ FIX: A_fixed must be STATE_T (vectorized state tokens), not STATE
// =============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,

    const DTYPE_VEC A_fixed[SSMU_STATE_T],
    const DTYPE_VEC RMS_weight[SSMU_D_T],

    const W_VEC W_inproj[SSMU_D_T][SSMU_CIN_T],
    const W_VEC W_delta[SSMU_C2_T][SSMU_C2_T],
    const W_VEC W_out[SSMU_D_T][SSMU_C2_T],

    const DTYPE_VEC D_diag[SSMU_C2_T],

    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,

    hls::stream<DTYPE_VEC>& conv_state_in,
    hls::stream<DTYPE_VEC>& conv_state_out,

    DTYPE_VEC* C_ddr,
    DTYPE_VEC* H1_ddr,

    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out,

    float w_scale_in,
    float w_scale_delta,
    float w_scale_out
);

#endif // __UCI_EECS_SSMU_HEADER_LAYERSH_ALIGNED__
