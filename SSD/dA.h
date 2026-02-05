#ifndef DA_H
#define DA_H

#include "Mamba.h"

inline void dA_sequence(
    hls::stream<DTYPE_VEC>& dt_stream,
    const FDTYPE A[H],
    hls::stream<DTYPE_VEC>& da_stream
) {
    constexpr int VEC_PER_SAMPLE = H / VEC_FACTOR;
    
    BATCH_LOOP: for (int l = 0; l < LENGTH; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        TIMESTEP_LOOP: for (int t = 0; t < VEC_PER_SAMPLE; t++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            
            DTYPE_VEC dt_vec = dt_stream.read();
            
            DTYPE_VEC da_vec;
            
            HEAD_LOOP: for (int h = 0; h < VEC_FACTOR; h++) {
                #pragma HLS UNROLL
                
                int global_idx = t * VEC_FACTOR + h;
                
                if (global_idx < H) {
                    FDTYPE dt_val = (FDTYPE)dt_vec[h];
                    FDTYPE a_val = A[global_idx];
                    da_vec[h] = (DTYPE)(dt_val * a_val);
                } else {
                    da_vec[h] = DTYPE(0);
                }
            }
            da_stream.write(da_vec);
        }
    }
}

#endif