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
constexpr int VEC_FACTOR = 16;
typedef hls::vector<DTYPE,VEC_FACTOR> DTYPE_VEC;

#define BATCH 1
#define LENGTH 64
#define N 16
#define Dim 64
#define K 4
#define VEC_D (Dim / VEC_FACTOR)

void silu(DTYPE_VEC in[VEC_D], DTYPE_VEC out[VEC_D]);
void exp1(DTYPE_VEC in[VEC_D], DTYPE_VEC out[VEC_D]);
void conv1d_vec(DTYPE_VEC input_X[VEC_D], DTYPE kernel[K], DTYPE_VEC Y[VEC_D]);
void projection(DTYPE_VEC in[VEC_D], DTYPE_VEC weight[N][VEC_D], DTYPE_VEC out[N]);
void projection_delta(DTYPE_VEC in[VEC_D], DTYPE_VEC weight[VEC_D][VEC_D], DTYPE_VEC out[VEC_D]);
void EMU_2D(DTYPE_VEC A[N][VEC_D], DTYPE_VEC B[VEC_D], DTYPE_VEC out[N][VEC_D]);
void UpdateH_producer(
    DTYPE_VEC ddA[N][VEC_D], DTYPE_VEC dX[VEC_D], DTYPE_VEC dB[N][VEC_D], DTYPE_VEC C[N],
    DTYPE_VEC H0[N][VEC_D],
    hls::stream<DTYPE_VEC> &stream_ddA,
    hls::stream<DTYPE_VEC> &stream_dX,
    hls::stream<DTYPE_VEC> &stream_dB,
    hls::stream<DTYPE_VEC> &stream_C,
    hls::stream<DTYPE_VEC> &stream_H0_in);
void UpdateH_consumer(
    hls::stream<DTYPE_VEC> &stream_ddA,
    hls::stream<DTYPE_VEC> &stream_X_ssm,
    hls::stream<DTYPE_VEC> &stream_dB,
    hls::stream<DTYPE_VEC> &stream_C,
    hls::stream<DTYPE_VEC> &stream_H0_in,
    DTYPE_VEC H1[N][VEC_D]);
void SSMU(
    DTYPE kernel[K],
    DTYPE_VEC A[N][VEC_D], DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D],
    DTYPE_VEC X[VEC_D],
    DTYPE_VEC H0[N][VEC_D], DTYPE_VEC H1[N][VEC_D],
    DTYPE_VEC W_delta[VEC_D][VEC_D],
    DTYPE_VEC out[VEC_D]);
#endif
