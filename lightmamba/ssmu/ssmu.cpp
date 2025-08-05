#include "ssmu.h"
DTYPE sigmoid(DTYPE x){
	return 1.0 / (1.0 + hls::expf(-x));
}

DTYPE softplus(DTYPE x){
	return hls::logf(1.0+hls::expf(x));
}

void silu(DTYPE in[N], DTYPE out[N]){
#pragma HLS PIPELINE II=2
	for(int j=0; j<N; j++){
		out[j]=in[j]*sigmoid(in[j]);
	}
}



void exp1(DTYPE in[N], DTYPE out[N]){
#pragma HLS PIPELINE II=1
	for(int j=0; j<N; j++){
		out[j]=(DTYPE)(hls::expf(in[j]));
	}
}

void conv1d(DTYPE input_X[INPUT_DIM], DTYPE kernel[K], DTYPE Y[N]) {
#pragma HLS ARRAY_PARTITION variable=kernel type=complete dim=1
	DTYPE window[K];
#pragma HLS ARRAY_PARTITION variable=window type=complete dim=1

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

void EMU(DTYPE A[N], DTYPE B[N], DTYPE out[N]){
#pragma HLS PIPELINE II=1
	for(int j=0; j<N; j++){
		out[j]=A[j]*B[j];
	}
}



void EAU(DTYPE A[N], DTYPE B[N], DTYPE out[N]){
#pragma HLS PIPELINE II=2
	for(int j=0; j<N; j++){
		out[j]=A[j]+B[j];
	}
}



void EMU_tiled(const DTYPE A_tile[pp], const DTYPE B_tile[pp], DTYPE out_tile[pp]){
#pragma HLS PIPELINE II=1
	for (int j = 0; j < pp; j++) {
		out_tile[j] = A_tile[j] * B_tile[j];
	}
}



void EAU_tiled(const DTYPE A_tile[pp], const DTYPE B_tile[pp], DTYPE out_tile[pp]){
#pragma HLS PIPELINE II=1
	for (int j = 0; j < pp; j++) {
		out_tile[j] = A_tile[j] + B_tile[j];
	}
}

void ACU(DTYPE input[np][pp], DTYPE output[pp]){
#pragma HLS ARRAY_PARTITION variable=input type=complete dim=2
#pragma HLS PIPELINE II=1
	DTYPE sum[pp];
#pragma HLS ARRAY_PARTITION variable=sum type=complete dim=1
	for(int i=0; i<pp; i++){
		sum[i]=0;
	}
	for(int j=0; j<np; j++){
		for(int i=0; i<pp; i++){
			sum[i]+=input[j][i];
		}
	}
	for(int i=0; i<pp; i++){
		output[i]=sum[i];
	}
}


void SSMU(
DTYPE kernel[K], DTYPE A[N], DTYPE B[N], DTYPE C[N], DTYPE D[N],
DTYPE X[N], DTYPE Z[N],
DTYPE H0[M][N], DTYPE H1[M][N],
DTYPE delta[N], DTYPE bias[N], DTYPE out[N]){

#pragma HLS ARRAY_PARTITION variable=kernel type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=A type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=B type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=C type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=D type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=X type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=Z type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=bias type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=delta type=cyclic dim=1 factor=2
#pragma HLS ARRAY_PARTITION variable=out type=cyclic dim=1 factor=2
#pragma HLS INTERFACE m_axi port=H0 offset=slave bundle=gmem0 depth=1000
#pragma HLS INTERFACE m_axi port=H1 offset=slave bundle=gmem1 depth=1000

    DTYPE dd[N], dA[N], dB[N], dC[N], dX[N], dZ[N], ddA[N], ddX[N], ddB[N], yy1[N], accu_sum[N];
    EAU(delta, bias, dd);
    for(int j=0; j<N; j++){
	    dd[j]=softplus(dd[j]);
    }
    conv1d(X, kernel, dX);
    silu(B, dB);
    silu(C, dC);
    silu(dX, ddX);
    silu(Z, dZ);
    EMU(dB, dd, ddB);
    EMU(A, dd, dA);
    EMU(ddX, D, yy1);
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
			EMU_tiled(ddA_tile, H0_tile[i], hh0_local[i]);
			EMU_tiled(ddB_tile, ddX_tile, hh1_local[i]);
			EAU_tiled(hh0_local[i], hh1_local[i], H1_write_back[i]);
			EMU_tiled(dC_tile, H1_write_back[i], dH_local[i]);
			}
			ACU(dH_local, accu_sum_tile);
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
	EAU(accu_sum, yy1, Y);
	EMU(Y, dZ, out);
}

