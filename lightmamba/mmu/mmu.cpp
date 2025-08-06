#include "mmu.h"

void mmu(DTYPE in[D_IN], DTYPE weight[D_IN][D_OUT], DTYPE out[D_OUT]){
#pragma HLS INLINE
	for(int i=0; i<D_TEMP; i++){
#pragma HLS PIPELINE
		DTYPE sum1=0, sum2=0;
		for(int j=0; j<D_IN; j++){
#pragma HLS UNROLL
			sum1+=weight[j][i]*in[j];
			sum2+=weight[j][D_TEMP+i]*in[j];
		}
		out[i]=sum1;
		out[i+D_TEMP]=sum2;
	}
}
