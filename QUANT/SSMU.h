#ifndef __UCI_EECS_SSMU_HEADER_20260106_BIG__
#define __UCI_EECS_SSMU_HEADER_20260106_BIG__

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_vector.h>

// =============================================================
// Optional debug banner
// =============================================================
#ifndef SSMU_HEADER_DEBUG
#define SSMU_HEADER_DEBUG 0
#endif

#if SSMU_HEADER_DEBUG
#warning ">>> ssmu.h INCLUDED (FULL SIZE MODE) <<<"
#endif

// =============================================================
// Model sizes (FULL checkpoint sizes ONLY)
// NOTE: DO NOT use generic macros like N/K/DIM (they break Vitis HLS templates).
// Use SSMU_* macros only.
// =============================================================
#ifndef SSMU_DIM
#define SSMU_DIM 2560
#endif

#ifndef SSMU_N
#define SSMU_N 5120
#endif

#ifndef SSMU_VEC_FACTOR
#define SSMU_VEC_FACTOR 16
#endif

#ifndef SSMU_K
#define SSMU_K 4
#endif

// =============================================================
// Derived sizes
// =============================================================
#ifndef SSMU_VEC_D
#define SSMU_VEC_D (SSMU_DIM / SSMU_VEC_FACTOR)
#endif

#ifndef SSMU_HUGE_LEN
#define SSMU_HUGE_LEN (SSMU_N * SSMU_VEC_D)
#endif

// -------------------------------------------------------------
// Backward-compat aliases (macros) for common non-conflicting names
// These are generally safe.
// -------------------------------------------------------------
#ifndef VEC_FACTOR
#define VEC_FACTOR SSMU_VEC_FACTOR
#endif

#ifndef VEC_D
#define VEC_D SSMU_VEC_D
#endif

#ifndef HUGE_LEN
#define HUGE_LEN SSMU_HUGE_LEN
#endif

// =============================================================
// Quantization config (optional, header-level single source of truth)
// - Keeps SSMU.cpp and other modules consistent.
// - DOES NOT change the top-level SSMU() prototype.
// =============================================================
#ifndef SSMU_ENABLE_QUANT
#define SSMU_ENABLE_QUANT 1
#endif

// default quant bits (match your linear quant setting if needed)
#ifndef SSMU_Q_BITS
#define SSMU_Q_BITS 8
#endif

// clamp on S (scale exponent) if you use log2-style scaling, typical 0..15
#ifndef SSMU_Q_S_MAX
#define SSMU_Q_S_MAX 15
#endif

// scale type (enough bits for 0..SSMU_Q_S_MAX)
typedef ap_uint<5> SSMU_QSCALE_T;

// -------------------------------------------------------------
// Optional helper constants (header-visible)
// -------------------------------------------------------------
static constexpr int SSMU_Q_BITS_C  = SSMU_Q_BITS;
static constexpr int SSMU_Q_S_MAX_C = SSMU_Q_S_MAX;

// =============================================================
// Types
// =============================================================
typedef ap_fixed<16,4> DTYPE;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

// If your quant flow carries a (value, scale) pair internally,
// you can use these typedefs in .cpp without exposing new ports.
typedef DTYPE      SSMU_QVAL_T;
typedef DTYPE_VEC  SSMU_QVEC_T;

// =============================================================
// constexpr mirrors
// =============================================================
static constexpr int SSMU_DIM_C        = SSMU_DIM;
static constexpr int SSMU_N_C          = SSMU_N;
static constexpr int SSMU_K_C          = SSMU_K;
static constexpr int SSMU_VEC_FACTOR_C = SSMU_VEC_FACTOR;
static constexpr int SSMU_VEC_D_C      = SSMU_VEC_D;
static constexpr int SSMU_HUGE_LEN_C   = SSMU_HUGE_LEN;

static constexpr int DIM_C        = SSMU_DIM_C;
static constexpr int N_C          = SSMU_N_C;
static constexpr int K_C          = SSMU_K_C;
static constexpr int VEC_FACTOR_C = SSMU_VEC_FACTOR_C;
static constexpr int VEC_D_C      = SSMU_VEC_D_C;
static constexpr int HUGE_LEN_C   = SSMU_HUGE_LEN_C;

// =============================================================
// Legacy constant aliases (SAFE, NOT macros)
// - lets old .cpp/.tb use N/DIM/K without template-macro pollution.
// =============================================================
static constexpr int DIM = DIM_C;
static constexpr int N   = N_C;
static constexpr int K   = K_C;

// =============================================================
// Compile-time safety checks
// =============================================================
static_assert(SSMU_VEC_FACTOR_C > 0, "SSMU_VEC_FACTOR must be > 0");
static_assert(SSMU_DIM_C > 0,        "SSMU_DIM must be > 0");
static_assert((SSMU_DIM_C % SSMU_VEC_FACTOR_C) == 0,
              "SSMU_DIM must be divisible by SSMU_VEC_FACTOR");

// keep these (your code assumes tiles of 8/16)
static_assert((SSMU_VEC_D_C % 8)  == 0, "SSMU_VEC_D must be multiple of 8");
static_assert((SSMU_VEC_D_C % 16) == 0, "SSMU_VEC_D must be multiple of 16");

static_assert(SSMU_HUGE_LEN_C > 0, "SSMU_HUGE_LEN must be > 0");

// sanity on quant settings
static_assert(SSMU_Q_BITS_C > 0, "SSMU_Q_BITS must be > 0");
static_assert(SSMU_Q_BITS_C <= 16, "SSMU_Q_BITS is expected <= 16 for this design");
static_assert(SSMU_Q_S_MAX_C >= 0, "SSMU_Q_S_MAX must be >= 0");
static_assert(SSMU_Q_S_MAX_C <= 31, "SSMU_Q_S_MAX must be <= 31 (QSCALE_T is 5-bit)");

// =============================================================
// TOP prototype (MUST match your current SSMU.cpp)
// Full Mamba Block: Pre-Norm + MambaCore + OutProj + D-skip + Residual
// =============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,

    // ---- fixed SSM parameter A (per i) ----
    const DTYPE_VEC A_fixed[SSMU_N],

    // ---- RMSNorm weight (gamma), shape like x ----
    const DTYPE_VEC RMS_weight[SSMU_VEC_D],

    // ---- input projections (linear) ----
    const DTYPE_VEC W_in_x[SSMU_VEC_D][SSMU_VEC_D],
    const DTYPE_VEC W_in_z[SSMU_VEC_D][SSMU_VEC_D],

    // ---- projections for selective scan ----
    const DTYPE_VEC W_B[SSMU_N][SSMU_VEC_D],
    const DTYPE_VEC W_C[SSMU_N][SSMU_VEC_D],
    const DTYPE_VEC W_delta[SSMU_VEC_D][SSMU_VEC_D],

    // ---- output projection (linear) ----
    const DTYPE_VEC W_out[SSMU_VEC_D][SSMU_VEC_D],

    // ---- D skip path (diagonal per-channel) ----
    const DTYPE_VEC D_diag[SSMU_VEC_D],

    // ---- streams / DDR ----
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    DTYPE_VEC* C_ddr,      // length = HUGE_LEN
    DTYPE_VEC* H1_ddr,     // length = HUGE_LEN
    hls::stream<DTYPE_VEC>& out
);

#endif // __UCI_EECS_SSMU_HEADER_20260106_BIG__
