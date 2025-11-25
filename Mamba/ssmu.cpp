#include "ssmu.h"
#include "lut_optimized.h"

// ==================== Part 1: X to X_gate, B, C, delta ====================

// Optimized conv1d and silu with streams
void conv1d_silu_stream(hls::stream<DTYPE_VEC>& X_in, hls::stream<DTYPE>& kernel_in, 
                        hls::stream<DTYPE_VEC>& X_gate_out, hls::stream<DTYPE_VEC>& X_ssm_out) {
    #pragma HLS INLINE OFF
    
    static DTYPE shift_reg[K + VEC_FACTOR - 1];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete

    DTYPE kernel_buffer[K];
    #pragma HLS ARRAY_PARTITION variable=kernel_buffer complete
    read_kernel: for(int i = 0; i < K; i++) {
        #pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel_in.read();
    }
    DTYPE_VEC X_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=X_buffer cyclic factor=2 dim=1
    // Read and buffer X for X_gate
    read_X: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=2
        DTYPE_VEC x_val = X_in.read();
        X_buffer[i] = x_val;
        
        // Apply SiLU for X_gate
        // DTYPE_VEC x_gate;
        // for(int k = 0; k < VEC_FACTOR; k++) {
        //     #pragma HLS UNROLL
        //     DTYPE val = x_val[k];
        //     DTYPE exp_val = hls::exp(-val);
        //     DTYPE sigmoid = (DTYPE)1.0 / ((DTYPE)1.0 + exp_val);
        //     x_gate[k] = val * sigmoid;
        // }
        // X_gate_out.write(x_gate);
        X_gate_out.write(lut_silu_vec(x_val));
    }
    
    // Initialize shift register
    init_shift_reg: for(int i = 0; i < K + VEC_FACTOR - 1; i++) {
        #pragma HLS UNROLL
        shift_reg[i] = 0;
    }
    
    // Process convolution for X_ssm
    process_vectors: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=2
        
        DTYPE_VEC input_vec = X_buffer[i];
        DTYPE_VEC output_vec;
        
        // Update shift register
        update_shift_reg: for(int j = K + VEC_FACTOR - 2; j >= VEC_FACTOR; j--) {
            #pragma HLS UNROLL
            shift_reg[j] = shift_reg[j - VEC_FACTOR];
        }
        
        load_data: for(int j = 0; j < VEC_FACTOR; j++) {
            #pragma HLS UNROLL
            shift_reg[j] = input_vec[j];
        }
        
        // Compute convolution output
        compute_output: for(int j = 0; j < VEC_FACTOR; j++) {
            #pragma HLS UNROLL
            DTYPE sum = 0;
            conv_calc: for(int k = 0; k < K; k++) {
                #pragma HLS UNROLL
                sum += kernel_buffer[k] * shift_reg[j + k];
            }
            output_vec[j] = sum;
        }
        
        // Apply SiLU for X_ssm
        // DTYPE_VEC x_ssm;
        // for(int k = 0; k < VEC_FACTOR; k++) {
        //     #pragma HLS UNROLL
        //     DTYPE val = output_vec[k];
        //     DTYPE exp_val = hls::exp(-val);
        //     DTYPE sigmoid = (DTYPE)1.0 / ((DTYPE)1.0 + exp_val);
        //     x_ssm[k] = val * sigmoid;
        // }
        // X_ssm_out.write(x_ssm);
        X_ssm_out.write(lut_silu_vec(output_vec));
    }
}

// Optimized projections with streams for B, C, and delta
void projection_streams(hls::stream<DTYPE_VEC>& X_ssm_in, 
                        DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
                        hls::stream<DTYPE_VEC>& B_out, hls::stream<DTYPE_VEC>& C_out, hls::stream<DTYPE_VEC>& delta_out) {
    
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=W_B cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=W_B complete dim=2
    #pragma HLS ARRAY_PARTITION variable=W_C cyclic factor=2 dim=1  
    #pragma HLS ARRAY_PARTITION variable=W_C complete dim=2
    #pragma HLS ARRAY_PARTITION variable=W_delta cyclic factor=2 dim=1
    #pragma HLS ARRAY_PARTITION variable=W_delta cyclic factor=2 dim=2
    
    DTYPE_VEC X_ssm_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=X_ssm_buffer complete
    
    read_X_ssm: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        X_ssm_buffer[j] = X_ssm_in.read();
    }
    
    projection_B: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=4

        DTYPE_VEC out_vec_B;
      compute_B: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=2
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                if (j == 0) out_vec_B[k] = 0; 
                out_vec_B[k] += X_ssm_buffer[j][k] * W_B[i][j][k];
            }
        }
        B_out.write(out_vec_B);
    }
    
    projection_C: for(int i = 0; i < N; i++) {
                #pragma HLS PIPELINE II=4
        
        DTYPE_VEC out_vec_C;
        
        compute_C: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=2
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                if (j == 0) out_vec_C[k] = 0;
                out_vec_C[k] += X_ssm_buffer[j][k] * W_C[i][j][k];
            }
        }

        C_out.write(out_vec_C);
    }
    
    projection_delta_stream: for(int i = 0; i < VEC_D; i++) {
#pragma HLS PIPELINE II=4
        
        DTYPE_VEC out_vec_delta;
        
        compute_delta: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=2
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                if (j == 0) out_vec_delta[k] = 0;
                out_vec_delta[k] += X_ssm_buffer[j][k] * W_delta[i][j][k];
            }
        }
        
        // apply_softplus: for(int k = 0; k < VEC_FACTOR; k++) {
        //     #pragma HLS UNROLL
        //     DTYPE val =out_vec_delta[k];

        //    out_vec_delta[k] = (DTYPE)hls::log((DTYPE)1.0 + hls::exp(val));
        // }
        // delta_out.write(out_vec_delta);
        delta_out.write(lut_softplus_vec(out_vec_delta));
    }
}

// ==================== Part 2: A to ddA ====================

void A_to_ddA_stream(hls::stream<DTYPE_VEC>& A_in, hls::stream<DTYPE_VEC>& delta_in, 
                     hls::stream<DTYPE_VEC>& ddA_out) {
     #pragma HLS INLINE OFF
    DTYPE_VEC delta_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=delta_buffer complete
    
    cache_delta: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=2
        delta_buffer[j] = delta_in.read();
    }
    // Process A and delta to produce ddA
 process_A_delta: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=2
        
        DTYPE_VEC A_val = A_in.read();
        
        for(int j = 0; j < VEC_D; j++) {
            DTYPE_VEC delta_val = delta_buffer[j];
            //  dA = A * delta
            DTYPE_VEC dA_vec = A_val * delta_val;
            
            //  ddA = exp(dA)
            // DTYPE_VEC ddA_vec;
            // for(int k = 0; k < VEC_FACTOR; k++) {
            //     DTYPE dA_val = dA_vec[k];

            //      ddA_vec[k] = (DTYPE)hls::exp(dA_vec[k]);
            // }
            // ddA_out.write(ddA_vec);
            ddA_out.write(lut_exp_vec(dA_vec));
        }
    }
}

// ==================== Part 3: B to dB ====================

void B_to_dB_stream(hls::stream<DTYPE_VEC>& B_in, hls::stream<DTYPE_VEC>& delta_in, 
                    hls::stream<DTYPE_VEC>& dB_out) {
    #pragma HLS INLINE OFF
    
    DTYPE_VEC delta_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=delta_buffer complete
    
    read_delta_dB: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=2
        delta_buffer[j] = delta_in.read();
    }
    
    process_B_delta: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=2
        
        DTYPE_VEC B_val = B_in.read();
        
        for(int j = 0; j < VEC_D; j++) {
            DTYPE_VEC delta_val = delta_buffer[j];
            DTYPE_VEC dB_vec = B_val * delta_val;
            dB_out.write(dB_vec);
        }
    }
}

// ==================== Part 4: H update and final output ====================

void update_H_stream(hls::stream<DTYPE_VEC>& ddA_in, hls::stream<DTYPE_VEC>& dX_in, 
                     hls::stream<DTYPE_VEC>& dB_in, hls::stream<DTYPE_VEC>& H0_in,
                     hls::stream<DTYPE_VEC>& H1_out) {
    #pragma HLS INLINE OFF
    
    DTYPE_VEC dX_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=dX_buffer complete
    
    read_dX: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        dX_buffer[j] = dX_in.read();
    }
    
     update_H_loop: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        
        for(int j = 0; j < VEC_D; j++) {
            DTYPE_VEC H0_val = H0_in.read();
            DTYPE_VEC ddA_val = ddA_in.read();
            DTYPE_VEC dB_val = dB_in.read();
            DTYPE_VEC dX_val = dX_buffer[j];
            
            // H1 = H0 * ddA + dB * dX
            DTYPE_VEC H1_val = H0_val * ddA_val + dB_val * dX_val;
            H1_out.write(H1_val);
        }
    }
}

void final_output_stream(hls::stream<DTYPE_VEC>& X_gate_in, hls::stream<DTYPE_VEC>& H1_in, 
                         hls::stream<DTYPE_VEC>& C_in, hls::stream<DTYPE_VEC>& out) {
    
    DTYPE_VEC X_gate_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=X_gate_buffer complete
    
    DTYPE_VEC C_buffer[N];
    #pragma HLS ARRAY_PARTITION variable=C_buffer cyclic factor=8 dim=1
    
    read_X_gate: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        X_gate_buffer[j] = X_gate_in.read();
    }
    
    read_C: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        C_buffer[i] = C_in.read();
    }
    
    DTYPE_VEC temp_acc[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=temp_acc complete
    
    init_acc: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS UNROLL
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            temp_acc[j][k] = 0;
        }
    }
    
    accumulate_H1_C: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=2
        DTYPE_VEC C_val = C_buffer[i];
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=2
            DTYPE_VEC H1_val = H1_in.read();
            temp_acc[j] += H1_val * C_val;
        }
    }
    
    final_output: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC out_val = X_gate_buffer[j] + temp_acc[j];
        out.write(out_val);
    }
}

void duplicate_X_ssm_stream(hls::stream<DTYPE_VEC>& in, 
                           hls::stream<DTYPE_VEC>& out1) {
    #pragma HLS INLINE OFF
    for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC val = in.read();
        out1.write(val);
    }
}

void duplicate_delta_stream(hls::stream<DTYPE_VEC>& in,
                           hls::stream<DTYPE_VEC>& out1,
                           hls::stream<DTYPE_VEC>& out2) {
    #pragma HLS INLINE OFF
    for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC val = in.read();
        out1.write(val);
        out2.write(val);
    }
}

void duplicate_H1_stream(hls::stream<DTYPE_VEC>& in,
                        hls::stream<DTYPE_VEC>& out1,
                        hls::stream<DTYPE_VEC>& out2) {
    #pragma HLS INLINE OFF
    for(int i = 0; i < N * VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC val = in.read();
        out1.write(val);
        out2.write(val);
    }
}

// ==================== Complete Stream-based SSMU ====================

void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out) {
    
    #pragma HLS DATAFLOW
    
    // Internal streams
        hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream("X_ssm_stream");
    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> B_stream("B_stream");
    hls::stream<DTYPE_VEC> C_stream("C_stream");
    hls::stream<DTYPE_VEC> delta_stream("delta_stream");
    hls::stream<DTYPE_VEC> delta_stream_A("delta_stream_A");
    hls::stream<DTYPE_VEC> delta_stream_B("delta_stream_B");
    hls::stream<DTYPE_VEC> ddA_stream("ddA_stream");
    hls::stream<DTYPE_VEC> dB_stream("dB_stream");
    hls::stream<DTYPE_VEC> H1_temp_stream("H1_temp_stream");
    hls::stream<DTYPE_VEC> H1_final_stream("H1_final_stream");

    // Set appropriate depths for streams
    #pragma HLS STREAM variable=X_gate_stream depth=2
    #pragma HLS STREAM variable=X_ssm_stream depth=2
    #pragma HLS STREAM variable=X_ssm_proj_stream depth=2
    #pragma HLS STREAM variable=B_stream depth=2
    #pragma HLS STREAM variable=C_stream depth=2
    #pragma HLS STREAM variable=delta_stream depth=2
    #pragma HLS STREAM variable=delta_stream_A depth=2
    #pragma HLS STREAM variable=delta_stream_B depth=2
    #pragma HLS STREAM variable=ddA_stream depth=2
    #pragma HLS STREAM variable=dB_stream depth=2
    #pragma HLS STREAM variable=H1_temp_stream depth=2
    #pragma HLS STREAM variable=H1_final_stream depth=2
    
    // Part 1: X to X_gate, B, C, delta
    conv1d_silu_stream(X_in, kernel_in, X_gate_stream, X_ssm_stream);
    
    // Split X_ssm for multiple consumers - now as a function call
    duplicate_X_ssm_stream(X_ssm_stream, X_ssm_proj_stream);
    
    projection_streams(X_ssm_proj_stream, W_B, W_C, W_delta, B_stream, C_stream, delta_stream);

    // Split delta for multiple consumers - now as a function call
    duplicate_delta_stream(delta_stream, delta_stream_A, delta_stream_B);
    
    // Part 2: A to ddA
    A_to_ddA_stream(A_in, delta_stream_A, ddA_stream);
    
    // Part 3: B to dB  
    B_to_dB_stream(B_stream, delta_stream_B, dB_stream);
    
    // Part 4: H update and final output
    update_H_stream(ddA_stream, X_ssm_proj_stream, dB_stream, H0_in, H1_temp_stream);
    
    // Split H1 for output and final computation - now as a function call
    duplicate_H1_stream(H1_temp_stream, H1_final_stream, H1_out);
    
    // Final output computation
    final_output_stream(X_gate_stream, H1_final_stream, C_stream, out);
}
