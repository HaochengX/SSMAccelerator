#include "lightmamba.h"
DTYPE sigmoid(DTYPE x){
	return 1.0 / (1.0 + hls::expf(-x));
}

DTYPE softplus(DTYPE x){
	return (DTYPE)(hls::logf(1.0+hls::expf(x)));
}

void silu(DTYPE in[N], DTYPE out[N]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in  type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
#pragma HLS PIPELINE II=1
	for(int j=0; j<N; j++){
		out[j]=in[j]*sigmoid(in[j]);
	}
}



void exp1(DTYPE in[N], DTYPE out[N]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=in type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
#pragma HLS PIPELINE II=1
	for(int j=0; j<N; j++){
		out[j]=(DTYPE)(hls::expf(in[j]));
	}
}

void conv1d(DTYPE input_X[INPUT_DIM], DTYPE kernel[K], DTYPE Y[N]) {
#pragma HLS ARRAY_PARTITION variable=kernel type=block dim=1 factor=2
	DTYPE window[K];
#pragma HLS ARRAY_PARTITION variable=window type=block dim=1 factor=2

	for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
		for (int j = K-1; j > 0; j--) {
			window[j] = window[j-1];
		}
		window[0] = input_X[i + K - 1];

		DTYPE sum = 0;
		for (int j = 0; j < K; j++) {
			sum += kernel[j] * window[j];
		}
		Y[i] = sum;
	}
}
template <int size>
void EMU(DTYPE A[size], DTYPE B[size], DTYPE out[size]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=A   type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B   type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
#pragma HLS PIPELINE II=1
	for(int j=0; j<size; j++){
		out[j]=A[j]*B[j];
	}
}


template <int size>
void EAU(DTYPE A[size], DTYPE B[size], DTYPE out[size]){
#pragma HLS INLINE off
#pragma HLS ARRAY_PARTITION variable=A type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block factor=4 dim=1
#pragma HLS PIPELINE II=1
	for(int j=0; j<size; j++){
		out[j]=A[j]+B[j];
	}
}

void SSMU(
DTYPE kernel[K], DTYPE A[N], DTYPE B[N], DTYPE C[N], DTYPE D[N],
DTYPE X[N], DTYPE Z[N],
DTYPE H0[M][N], DTYPE H1[M][N],
DTYPE delta[N], DTYPE bias[N], DTYPE out[N]){

#pragma HLS ARRAY_PARTITION variable=kernel type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=A  type=block   factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=B  type=block   factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=C  type=block   factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=D  type=block   factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=X  type=block   factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=Z  type=block   factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=delta type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=bias type=block factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=out type=block  factor=4 dim=1
#pragma HLS INTERFACE m_axi port=H0 offset=slave bundle=gmem0 depth=1000
#pragma HLS INTERFACE m_axi port=H1 offset=slave bundle=gmem1 depth=1000
    DTYPE dd[N], dA[N], dB[N], dC[N], dX[N], dZ[N], ddA[N], ddX[N], ddB[N], yy1[N], accu_sum[N];
    EAU<N>(delta, bias, dd);
    for(int j=0; j<N; j++){
	    dd[j]=softplus(dd[j]);
    }
    conv1d(X, kernel, dX);
    silu(B, dB);
    silu(C, dC);
    silu(dX, ddX);
    silu(Z, dZ);
    EMU<N>(dB, dd, ddB);
    EMU<N>(A, dd, dA);
    EMU<N>(ddX, D, yy1);
    exp1(dA, ddA);
	for (int i = 0; i < N; i++) {
		accu_sum[i] = 0;
	}
	for (int i_tile = 0; i_tile < M/np; i_tile++) {
		for (int j_tile = 0; j_tile < N/pp; j_tile++) {
			DTYPE ddA_tile[pp];
			DTYPE ddB_tile[pp];
			DTYPE ddX_tile[pp];
			DTYPE H0_tile[np][pp];
			DTYPE dC_tile[pp];
			for (int k = 0; k < pp; k++) {
				ddA_tile[k] = ddA[j_tile * pp + k];
				ddB_tile[k] = ddB[j_tile * pp + k];
				ddX_tile[k] = ddX[j_tile * pp + k];
				dC_tile[k] = dC[j_tile * pp + k];
			}
			for (int i = 0; i < np; i++) {
				for (int k = 0; k < pp; k++) {
					H0_tile[i][k] = H0[i_tile * np + i][j_tile * pp + k];
				}
			}
			DTYPE hh0_local[np][pp];
			DTYPE hh1_local[np][pp];
			DTYPE dH_local[np][pp];
			DTYPE H1_write_back[np][pp];
			DTYPE accu_sum_tile[pp];
			for (int i = 0; i < np; i++) {
#pragma HLS PIPELINE II=1
			EMU<pp>(ddA_tile, H0_tile[i], hh0_local[i]);
			EMU<pp>(ddB_tile, ddX_tile, hh1_local[i]);
			EAU<pp>(hh0_local[i], hh1_local[i], H1_write_back[i]);
			EMU<pp>(dC_tile, H1_write_back[i], dH_local[i]);
			}
			for (int kk=0; kk<pp; kk++) accu_sum_tile[kk]=0;
			for (int ii=0; ii<np; ii++)
			  for (int kk=0; kk<pp; kk++)
			    accu_sum_tile[kk] += dH_local[ii][kk];

			for (int i=0; i<pp; i++) {
				accu_sum[j_tile*pp + i] += accu_sum_tile[i];
			}
			for (int i = 0; i < np; i++) {
				for (int k = 0; k < pp; k++) {
					H1[i_tile * np + i][j_tile * pp + k] = H1_write_back[i][k];
				}
			}
		}
	}
	DTYPE Y[N];
	EAU<N>(accu_sum, yy1, Y);
	EMU<N>(Y, dZ, out);
}
