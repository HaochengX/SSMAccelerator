// Splitter.h
#ifndef SPLITTER_H
#define SPLITTER_H

#include "Mamba.h"

template<int SUM, int X, int Y, int Z>
void splitter(
    hls::stream<DTYPE_VEC>& in_stream,
    hls::stream<DTYPE_VEC>& a_stream,
    hls::stream<DTYPE_VEC>& b_stream,
    hls::stream<DTYPE>& c_stream
) {
    constexpr int VEC_PER_SAMPLE = (SUM + VEC_FACTOR - 1) / VEC_FACTOR;
    
    constexpr int A_START = 0;
    constexpr int A_END = X;
    
    constexpr int B_START = A_END;
    constexpr int B_END = B_START + Y;
    
    constexpr int C_START = B_END;
    constexpr int C_END = C_START + Z;
    
    DTYPE_VEC temp_vec;
    int vec_offset = 0;
    
    BATCH_LOOP: for (int l = 0; l < LENGTH; l++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32
        
        vec_offset = 0;
        SAMPLE_LOOP: for (int vec_idx = 0; vec_idx < VEC_PER_SAMPLE; vec_idx++) {
            #pragma HLS PIPELINE II=1
            
            DTYPE_VEC input_vec = in_stream.read();
            
            PROCESS_VEC_ELEMENTS: for (int v = 0; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                
                int element_idx = vec_idx * VEC_FACTOR + v;
                
                if (element_idx >= SUM) {
                    continue;
                }
                
                DTYPE element = input_vec[v];
                
                if (element_idx < A_END) {
                    temp_vec[vec_offset] = element;
                    vec_offset++;
                    
                    if (vec_offset == VEC_FACTOR) {
                        a_stream.write(temp_vec);
                        vec_offset = 0;
                    }
                } 
                else if (element_idx < B_END) {
                    temp_vec[vec_offset] = element;
                    vec_offset++;
                    
                    if (vec_offset == VEC_FACTOR) {
                        b_stream.write(temp_vec);
                        vec_offset = 0;
                    }
                }
                else if (element_idx < C_END) {
                    c_stream.write(element);
                }
            }
        }
        
        FLUSH_REMAINING: if (vec_offset > 0) {
            for (int v = vec_offset; v < VEC_FACTOR; v++) {
                #pragma HLS UNROLL
                temp_vec[v] = DTYPE(0);
            }
            
            int last_element_idx = (VEC_PER_SAMPLE - 1) * VEC_FACTOR + (vec_offset - 1);
            
            if (last_element_idx < A_END) {
                a_stream.write(temp_vec);
            } 
            else if (last_element_idx < B_END) {
                b_stream.write(temp_vec);
            }
        }
    }
}

#endif // SPLITTER_H