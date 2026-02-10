// dt.h
#ifndef DT_H
#define DT_H

#include "Mamba.h"

inline void softplus(
    hls::stream<DTYPE_VEC>& dt_vec_input_stream,
    hls::stream<DTYPE_VEC>& dt_vec_output_stream
) {
    BATCH_LOOP: for (int l = 0; l < LENGTH; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        TIMESTEP_LOOP: for (int t = 0; t < H/VEC_FACTOR; t++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=1024
            
            DTYPE_VEC dt_vec = dt_vec_input_stream.read();
            DTYPE_VEC processed_vec;
            
            PROCESS_VECTOR: for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS PIPELINE
                processed_vec[v] = softplus_elem(dt_vec[v]);
            }
            
            dt_vec_output_stream.write(processed_vec);
        }
    }
}

#endif // DT_H