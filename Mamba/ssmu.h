#ifndef SSMU_H
#define SSMU_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
// DTYPE can be integer or fixed point
//typedef ap_fixed<8, 3> DTYPE;
typedef ap_int<8> DTYPE;
//typedef float DTYPE;

constexpr int VEC_FACTOR = 8;
typedef hls::vector<DTYPE, VEC_FACTOR> DTYPE_VEC;

#define BATCH 1
#define LENGTH 64
#define N 128
#define Dim 2560
#define K 4
#define VEC_D (Dim / VEC_FACTOR)

// Stream-based interfaces for different parts

// Part 1: X to X_gate, B, C, delta
void conv1d_silu_stream(hls::stream<DTYPE_VEC>& X_in, DTYPE kernel[K], 
                        hls::stream<DTYPE_VEC>& X_gate_out, hls::stream<DTYPE_VEC>& X_ssm_out);
void projection_streams(hls::stream<DTYPE_VEC>& X_ssm_in, 
                        DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
                        hls::stream<DTYPE_VEC>& B_out, hls::stream<DTYPE_VEC>& C_out, hls::stream<DTYPE_VEC>& delta_out);

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
void final_output_stream(hls::stream<DTYPE_VEC>& X_gate_in, hls::stream<DTYPE_VEC>& H1_in, 
                         hls::stream<DTYPE_VEC>& C_in, hls::stream<DTYPE_VEC>& out);

void duplicate_X_ssm_stream(hls::stream<DTYPE_VEC>& in, 
                           hls::stream<DTYPE_VEC>& out1);
void duplicate_delta_stream(hls::stream<DTYPE_VEC>& in,
                           hls::stream<DTYPE_VEC>& out1,
                           hls::stream<DTYPE_VEC>& out2);
void duplicate_H1_stream(hls::stream<DTYPE_VEC>& in,
                        hls::stream<DTYPE_VEC>& out1,
                        hls::stream<DTYPE_VEC>& out2);
// Complete stream-based SSMU
void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out);

#endif
