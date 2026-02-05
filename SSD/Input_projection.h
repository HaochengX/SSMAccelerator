// MatrixBlockCompute.h
#ifndef MATRIX_BLOCK_COMPUTE_H
#define MATRIX_BLOCK_COMPUTE_H

#include "Mamba.h"
inline void input_projection(
    hls::stream<DTYPE_VEC>& u_stream,
    
    const DTYPE weight_block[VEC_INPUT_LINEAR][VEC_D][VEC_FACTOR],
    
    hls::stream<DTYPE_VEC>& proj_stream
) {
    
    constexpr int INPUT_BLOCK_SIZE = 32;
    constexpr int OUTPUT_BLOCK_SIZE = 32;
    constexpr int NUM_INPUT_BLOCKS = (VEC_D + INPUT_BLOCK_SIZE - 1) / INPUT_BLOCK_SIZE;
    constexpr int NUM_OUTPUT_BLOCKS = (VEC_INPUT_LINEAR + OUTPUT_BLOCK_SIZE - 1) / OUTPUT_BLOCK_SIZE;
    SEQUENCE_LOOP: for (int l = 0; l < LENGTH; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        DTYPE u_buffer[DIM];
        #pragma HLS ARRAY_PARTITION variable=u_buffer cyclic factor=VEC_FACTOR
        
        FDTYPE proj[INPUT_LINEAR_SIZE];
        #pragma HLS ARRAY_PARTITION variable=proj block factor=16
        
        READ_LOOP: for (int i = 0; i < VEC_D; i++) {
            #pragma HLS PIPELINE II=1
            DTYPE_VEC vec = u_stream.read();
            
            for (int j = 0; j < VEC_FACTOR; j++) {
                #pragma HLS UNROLL
                u_buffer[i * VEC_FACTOR + j] = vec[j];
            }
        }
        
        INIT_OUTPUT: for (int i = 0; i < INPUT_LINEAR_SIZE; i++) {
            #pragma HLS UNROLL factor=16
            proj[i] = 0;
        }
        
        OUTPUT_BLOCK_LOOP: for (int ob = 0; ob < NUM_OUTPUT_BLOCKS; ob++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=21
            
            INPUT_BLOCK_LOOP: for (int ib = 0; ib < NUM_INPUT_BLOCKS; ib++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=5
                
                int output_start = ob * OUTPUT_BLOCK_SIZE;
                int output_end = hls::min(output_start + OUTPUT_BLOCK_SIZE, VEC_INPUT_LINEAR);
                
                int input_start = ib * INPUT_BLOCK_SIZE;
                int input_end = hls::min(input_start + INPUT_BLOCK_SIZE, VEC_D);
                
                COMPUTE_OUTPUT_BLOCK: for (int o_idx = output_start; o_idx < output_end; o_idx++) {
                    #pragma HLS PIPELINE II=4
                    
                    int output_base = o_idx * VEC_FACTOR;
                    
                    COMPUTE_INPUT_BLOCK: for (int i_idx = input_start; i_idx < input_end; i_idx++) {
                        #pragma HLS UNROLL factor=4
                        
                        int input_base = i_idx * VEC_FACTOR;
                        
                        for (int v = 0; v < VEC_FACTOR; v++) {
                            #pragma HLS UNROLL
                            
                            FDTYPE w_val = weight_block[o_idx][i_idx][v];
                            FDTYPE u_val = u_buffer[input_base + v];
                            
                            for (int ov = 0; ov < VEC_FACTOR; ov++) {
                                #pragma HLS UNROLL
                                proj[output_base + ov] = w_val * u_val + proj[output_base + ov];
                            }
                        }
                    }
                }
            }
        }
        OUTPUT_LOOP: for (int i = 0; i < VEC_INPUT_LINEAR; i++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC vec;
            int base_idx = i * VEC_FACTOR;
            
            // 填充向量
            for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                int idx = base_idx + v;
                    vec[v] = proj[idx];
                }
            proj_stream.write(vec);
        }
    }
}

#endif // MATRIX_BLOCK_COMPUTE_H