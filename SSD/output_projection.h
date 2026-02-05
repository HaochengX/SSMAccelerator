// OutputProjection.h
#ifndef OUTPUT_PROJECTION_H
#define OUTPUT_PROJECTION_H

#include "Mamba.h"

inline void output_projection(
    hls::stream<DTYPE_VEC>& y_stream,
    const DTYPE weight_block[VEC_I][VEC_D][VEC_FACTOR][VEC_FACTOR],
    
    hls::stream<DTYPE_VEC>& out_stream
) {
    constexpr int INPUT_BLOCK_SIZE = 32;  
    constexpr int OUTPUT_BLOCK_SIZE = 32;  
    constexpr int NUM_INPUT_BLOCKS = (VEC_I + INPUT_BLOCK_SIZE - 1) / INPUT_BLOCK_SIZE;
    constexpr int NUM_OUTPUT_BLOCKS = (VEC_D + OUTPUT_BLOCK_SIZE - 1) / OUTPUT_BLOCK_SIZE;
    SEQUENCE_LOOP: for (int l = 0; l < LENGTH; l++) {
    #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        DTYPE y_buffer[I];
        #pragma HLS ARRAY_PARTITION variable=y_buffer cyclic factor=VEC_FACTOR/2
        
        FDTYPE out_buffer[DIM];
        #pragma HLS ARRAY_PARTITION variable=out_buffer cyclic factor=16
        
        READ_FEATURES: for (int i = 0; i < VEC_I; i++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC vec = y_stream.read();
            
            for (int j = 0; j < VEC_FACTOR; j++) {
                #pragma HLS UNROLL
                y_buffer[i * VEC_FACTOR + j] = vec[j];
            }
        }
        
        INIT_OUTPUT: for (int i = 0; i < DIM; i++) {
            #pragma HLS UNROLL factor=16
            out_buffer[i] = 0;
        }
        

        OUTPUT_BLOCK_LOOP: for (int ob = 0; ob < NUM_OUTPUT_BLOCKS; ob++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5  
            INPUT_BLOCK_LOOP: for (int ib = 0; ib < NUM_INPUT_BLOCKS; ib++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=10  
                
                int output_start = ob * OUTPUT_BLOCK_SIZE;
                int output_end = hls::min(output_start + OUTPUT_BLOCK_SIZE, VEC_D);
                
                int input_start = ib * INPUT_BLOCK_SIZE;
                int input_end = hls::min(input_start + INPUT_BLOCK_SIZE, VEC_I);
                
                COMPUTE_BLOCK: for (int out_vec_idx = output_start; out_vec_idx < output_end; out_vec_idx++) {
                    #pragma HLS PIPELINE II=4
                    
                    for (int out_elem_idx = 0; out_elem_idx < VEC_FACTOR; out_elem_idx++) {
                        #pragma HLS UNROLL factor=4
                        
                        FDTYPE acc = out_buffer[out_vec_idx * VEC_FACTOR + out_elem_idx];
                        
                        for (int in_vec_idx = input_start; in_vec_idx < input_end; in_vec_idx++) {
                            #pragma HLS UNROLL factor=2
                            
                            for (int in_elem_idx = 0; in_elem_idx < VEC_FACTOR; in_elem_idx++) {
                                #pragma HLS UNROLL
                                
                                FDTYPE weight_val = weight_block[out_vec_idx][in_vec_idx][out_elem_idx][in_elem_idx];
                                FDTYPE y_val = y_buffer[in_vec_idx * VEC_FACTOR + in_elem_idx];
                                acc += weight_val * y_val;
                            }
                        }
                        
                        out_buffer[out_vec_idx * VEC_FACTOR + out_elem_idx] = acc;
                    }
                }
            }
        }
        OUTPUT_LOOP: for (int i = 0; i < VEC_D; i++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC vec;
            int base_idx = i * VEC_FACTOR;
            for (int j = 0; j < VEC_FACTOR; j++) {
                #pragma HLS UNROLL
                int idx = base_idx + j;
                vec[j] = DTYPE(out_buffer[idx]);
            }
            
            out_stream.write(vec);
        }
    }
}
#endif // OUTPUT_PROJECTION_H