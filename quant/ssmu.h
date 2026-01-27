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
#warning ">>> SSMU.h INCLUDED (FULL SIZE MODE) <<<"
#endif

// =============================================================
// Quantization build switch
// - When 1: weights are INT8 vectors (Q8VEC) for true quantized matmul
// - When 0: weights remain DTYPE_VEC (legacy fixed-point path)
// =============================================================
#ifndef SSMU_USE_INT8
#define SSMU_USE_INT8 1
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
#define SSMU_VEC_FACTOR 32
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
// Types (DTYPE path)
// =============================================================
typedef ap_fixed<16,4> DTYPE;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

// =============================================================
// Types (INT8 quantized weights path)
// =============================================================
typedef ap_int<8>  Q8_T;
typedef ap_int<32> QACC_T;
typedef hls::vector<Q8_T,  VEC_FACTOR> Q8VEC;
typedef hls::vector<QACC_T, VEC_FACTOR> QACC_VEC;

// =============================================================
// Weight type selection
// - Only weights are switched to int8 when SSMU_USE_INT8=1
// - Activations/streams remain DTYPE_VEC for now
// =============================================================
#if SSMU_USE_INT8
  typedef Q8VEC W_VEC;     // quantized weights
#else
  typedef DTYPE_VEC W_VEC; // legacy weights
#endif

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

// =============================================================
// TOP prototype (MUST match your current quant.cpp / SSMU.cpp)
// Full Mamba Block: Pre-Norm + MambaCore + OutProj + D-skip + Residual
//
// IMPORTANT:
// - Streams stay DTYPE_VEC
// - A_fixed / RMS_weight / D_diag stay DTYPE_VEC
// - Weights switch to W_VEC (DTYPE_VEC or Q8VEC) depending on SSMU_USE_INT8
// =============================================================
void SSMU(
    hls::stream<DTYPE>& kernel_in,

    // ---- fixed SSM parameter A (per i) ----
    const DTYPE_VEC A_fixed[SSMU_N],

    // ---- RMSNorm weight (gamma), shape like x ----
    const DTYPE_VEC RMS_weight[SSMU_VEC_D],

    // ---- input projections (linear) ----
    const W_VEC W_in_x[SSMU_VEC_D][SSMU_VEC_D],
    const W_VEC W_in_z[SSMU_VEC_D][SSMU_VEC_D],

    // ---- projections for selective scan ----
    const W_VEC W_B[SSMU_N][SSMU_VEC_D],
    const W_VEC W_C[SSMU_N][SSMU_VEC_D],
    const W_VEC W_delta[SSMU_VEC_D][SSMU_VEC_D],

    // ---- output projection (linear) ----
    const W_VEC W_out[SSMU_VEC_D][SSMU_VEC_D],

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
