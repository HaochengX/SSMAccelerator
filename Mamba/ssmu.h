#ifndef SSMU_H
#define SSMU_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
// DTYPE can be integer or fixed point
typedef ap_int<8> DTYPE;
//typedef float DTYPE;

//typedef ap_fixed<16,4> DTYPE;   // 16 total bits, 4 integer bits (12 fractional)

// Vector element type (lanes)
constexpr int VEC_FACTOR = 8;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

// Problem sizes (keep same as before)
#define BATCH 1
#define LENGTH 64
#define N 16
#define Dim 2560
#define K 4
#define VEC_D (Dim / VEC_FACTOR)
//Dim=2560, N=16

// local type for accurate math (change to ap_fixed if desired)
//typedef float FDTYPE;
typedef ap_fixed<32,8> FDTYPE;
//fixed_point 32
// Stream-based interfaces for different parts
// compute sl = x * sigmoid(x) using float math, return as DTYPE
static inline DTYPE silu_elem(DTYPE a) {
    FDTYPE x = (FDTYPE) a;
    FDTYPE expv = hls::exp(-x);
    FDTYPE sig = (FDTYPE)1.0 / ((FDTYPE)1.0 + expv);
    FDTYPE res = x * sig;
    // cast back to DTYPE (beware saturation if DTYPE is narrow)
    return (DTYPE) res;
}

static inline DTYPE exp_elem(DTYPE a) {
    FDTYPE x = (FDTYPE) a;
    FDTYPE y = hls::exp(x);
    return (DTYPE) y;
}

static inline DTYPE softplus_elem(DTYPE a) {
    FDTYPE x = (FDTYPE) a;
    FDTYPE y = hls::log((FDTYPE)1.0 + hls::exp(x));
    return (DTYPE) y;
}
void input_projection(
    hls::stream<DTYPE_VEC>& X_in,
    DTYPE_VEC W_in_ssm[VEC_D][VEC_D],
    DTYPE_VEC W_in_gate[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_ssm_out,
    hls::stream<DTYPE_VEC>& X_gate_out
);

void output_projection(
    hls::stream<DTYPE_VEC>& Y_in,
    DTYPE_VEC W_out[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& Y_out
);

void silu_gate(
    hls::stream<DTYPE_VEC> &X_gate_in,
    hls::stream<DTYPE_VEC> &X_gate_out
);
void conv1d_stream(hls::stream<DTYPE_VEC>& X_in, hls::stream<DTYPE>& kernel_in,
                 hls::stream<DTYPE_VEC>& X_ssm_out);
void projection_streams(hls::stream<DTYPE_VEC>& X_ssm_in,
                        DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
                        hls::stream<DTYPE_VEC>& B_out, hls::stream<DTYPE_VEC>& C_out, 
                        hls::stream<DTYPE_VEC>& delta_out_A, hls::stream<DTYPE_VEC>& delta_out_B);

// Part 2: A to ddA
void A_to_ddA_stream(hls::stream<DTYPE_VEC>& A_in, hls::stream<DTYPE_VEC>& delta_in, 
                     hls::stream<DTYPE_VEC>& ddA_out);

// Part 3: B to dB
void B_to_dB_stream(hls::stream<DTYPE_VEC>& B_in, hls::stream<DTYPE_VEC>& delta_in, 
                    hls::stream<DTYPE_VEC>& dB_out);

// Part 4: H update and final output
void update_H_stream(hls::stream<DTYPE_VEC>& ddA_in, hls::stream<DTYPE_VEC>& dX_in, 
                     hls::stream<DTYPE_VEC>& dB_in, hls::stream<DTYPE_VEC>& H0_in,
                     hls::stream<DTYPE_VEC>& H1_out);
void Xssm_output(
    hls::stream<DTYPE_VEC>& H1_in,
    hls::stream<DTYPE_VEC>& C_in,
    hls::stream<DTYPE_VEC>& out
);

// lightweight duplicators (kept minimal)
void duplicate_H1_stream(hls::stream<DTYPE_VEC>& in,
                        hls::stream<DTYPE_VEC>& out1,
                        hls::stream<DTYPE_VEC>& out2);
void gate_mul(
    hls::stream<DTYPE_VEC> &X_ssm,
    hls::stream<DTYPE_VEC> &X_gate,
    hls::stream<DTYPE_VEC> &Y_out
);
// Complete stream-based SSMU
void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
    DTYPE_VEC W_in_ssm[VEC_D][VEC_D], 
    DTYPE_VEC W_in_gate[VEC_D][VEC_D], 
    DTYPE_VEC W_out[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out);

#endif