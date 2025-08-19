#ifndef SSMU_H
#define SSMU_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

//typedef ap_fixed<8, 3> DTYPE;
typedef ap_int<8> DTYPE;
//N: dimension; M: width; K: kernel
constexpr int M = 16;
constexpr int N = 64;
constexpr int K = 4;
constexpr int INPUT_DIM = N + K - 1;
constexpr int pp = 16;
constexpr int np = 4;

void silu(DTYPE in[N], DTYPE out[N]);
void exp1(DTYPE in[N], DTYPE out[N]);
void conv1d(DTYPE input_X[INPUT_DIM], DTYPE kernel[K], DTYPE Y[N]);
DTYPE softplus(DTYPE x);
void EMU(DTYPE A[N], DTYPE B[N], DTYPE out[N]);
void EAU(DTYPE A[N], DTYPE B[N], DTYPE out[N]);
void EMU_tiled(const DTYPE A_tile[pp], const DTYPE B_tile[pp], DTYPE out_tile[pp]);
void EAU_tiled(const DTYPE A_tile[pp], const DTYPE B_tile[pp], DTYPE out_tile[pp]);
void ACU(DTYPE input[np][pp], DTYPE output[pp]);
void SSMU(
DTYPE kernel[K], DTYPE A[N], DTYPE B[N], DTYPE C[N], DTYPE D[N],
DTYPE X[N], DTYPE Z[N],
DTYPE H0[M][N], DTYPE H1[M][N], DTYPE delta[N], DTYPE bias[N], DTYPE out[N]);
#endif

