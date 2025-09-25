#ifndef LIGHTMAMBA_H
#define LIGHTMAMBA_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <hls_vector.h>
// DTYPE can be integer or fixed point
typedef ap_fixed<8, 3> DTYPE;
//typedef ap_int<8> DTYPE;
constexpr int VEC_FACTOR = 16;
typedef hls::vector<DTYPE,VEC_FACTOR> DTYPE_VEC;

//N: dimension; M: width; K: kernel
#define M 16
#define N 64
#define K 4
#define INPUT_DIM (N + K - 1)
#define VEC_N (N / VEC_FACTOR)

void silu(DTYPE_VEC in[VEC_N], DTYPE_VEC out[VEC_N]);
void exp1(DTYPE_VEC in[VEC_N], DTYPE_VEC out[VEC_N]);
void conv1d(DTYPE_VEC input_X[VEC_N], DTYPE kernel[K], DTYPE_VEC Y[VEC_N]);
void softplus(DTYPE_VEC in[VEC_N], DTYPE_VEC out[VEC_N]);
void EMU(DTYPE_VEC A[VEC_N], DTYPE_VEC B[VEC_N], DTYPE_VEC out[VEC_N]);
void EAU(DTYPE_VEC A[VEC_N], DTYPE_VEC B[VEC_N], DTYPE_VEC out[VEC_N]);
void SSMU(
    DTYPE kernel[K],
    DTYPE_VEC A[VEC_N], DTYPE_VEC B[VEC_N], DTYPE_VEC C[VEC_N], DTYPE_VEC D[VEC_N],
    DTYPE_VEC X[VEC_N], DTYPE_VEC Z[VEC_N],
    DTYPE_VEC H0[M][VEC_N], DTYPE_VEC H1[M][VEC_N],
    DTYPE_VEC delta[VEC_N], DTYPE_VEC bias[VEC_N],
    DTYPE_VEC out[VEC_N]);
#endif
