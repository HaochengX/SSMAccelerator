#include "ssmu.h"

// ==================== Part 1: X to X_gate, B, C, delta ====================

// Optimized conv1d and silu with streams
void conv1d_silu_stream(hls::stream<DTYPE_VEC>& X_in, DTYPE kernel[K], 
                        hls::stream<DTYPE_VEC>& X_gate_out, hls::stream<DTYPE_VEC>& X_ssm_out) {
    #pragma HLS INLINE OFF
    
    static DTYPE shift_reg[K + VEC_FACTOR - 1];
    #pragma HLS ARRAY_PARTITION variable=shift_reg complete
    
    DTYPE_VEC X_buffer[VEC_D];
    
    // Read and buffer X for X_gate
    read_X: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC x_val = X_in.read();
        X_buffer[i] = x_val;
        
        // Apply SiLU for X_gate
        DTYPE_VEC x_gate;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            DTYPE val = x_val[k];
            DTYPE exp_val = hls::exp(-val);
            DTYPE sigmoid = (DTYPE)1.0 / ((DTYPE)1.0 + exp_val);
            x_gate[k] = val * sigmoid;
        }
        X_gate_out.write(x_gate);
    }
    
    // Initialize shift register
    init_shift_reg: for(int i = 0; i < K + VEC_FACTOR - 1; i++) {
        #pragma HLS UNROLL
        shift_reg[i] = 0;
    }
    
    // Process convolution for X_ssm
    process_vectors: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        
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
                sum += kernel[k] * shift_reg[j + k];
            }
            output_vec[j] = sum;
        }
        
        // Apply SiLU for X_ssm
        DTYPE_VEC x_ssm;
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            DTYPE val = output_vec[k];
            DTYPE exp_val = hls::exp(-val);
            DTYPE sigmoid = (DTYPE)1.0 / ((DTYPE)1.0 + exp_val);
            x_ssm[k] = val * sigmoid;
        }
        X_ssm_out.write(x_ssm);
    }
}

// Optimized projections with streams for B, C, and delta
void projection_streams(hls::stream<DTYPE_VEC>& X_ssm_in, 
                        DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
                        hls::stream<DTYPE_VEC>& B_out, hls::stream<DTYPE_VEC>& C_out, hls::stream<DTYPE_VEC>& delta_out) {
    
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=W_B cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=W_B complete dim=2
    #pragma HLS ARRAY_PARTITION variable=W_C cyclic factor=8 dim=1  
    #pragma HLS ARRAY_PARTITION variable=W_C complete dim=2
    #pragma HLS ARRAY_PARTITION variable=W_delta cyclic factor=8 dim=1
    #pragma HLS ARRAY_PARTITION variable=W_delta cyclic factor=8 dim=2
    
    DTYPE_VEC X_ssm_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=X_ssm_buffer complete
    
    read_X_ssm: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        X_ssm_buffer[j] = X_ssm_in.read();
    }
    
    projection_B: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=2

        DTYPE_VEC out_vec_B;
      compute_B: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=4
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                if (j == 0) out_vec_B[k] = 0; 
                out_vec_B[k] += X_ssm_buffer[j][k] * W_B[i][j][k];
            }
        }
        B_out.write(out_vec_B);
    }
    
    projection_C: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        
                #pragma HLS PIPELINE II=2
        
        DTYPE_VEC out_vec_C;
        
        compute_C: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=4
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                if (j == 0) out_vec_C[k] = 0;
                out_vec_C[k] += X_ssm_buffer[j][k] * W_C[i][j][k];
            }
        }

        C_out.write(out_vec_C);
    }
    
    projection_delta_stream: for(int i = 0; i < VEC_D; i++) {
#pragma HLS PIPELINE II=2
        
        DTYPE_VEC out_vec_delta;
        
        compute_delta: for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL factor=4
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                if (j == 0) out_vec_delta[k] = 0;
                out_vec_delta[k] += X_ssm_buffer[j][k] * W_delta[i][j][k];
            }
        }
        
        apply_softplus: for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            DTYPE val =out_vec_delta[k];
            out_vec_delta[k] = (DTYPE)hls::log((DTYPE)1.0 + hls::exp(val));
        }
        delta_out.write(out_vec_delta);
    }
}

// ==================== Part 2: A to ddA ====================

void A_to_ddA_stream(hls::stream<DTYPE_VEC>& A_in, hls::stream<DTYPE_VEC>& delta_in, 
                     hls::stream<DTYPE_VEC>& ddA_out) {
     #pragma HLS INLINE OFF
    DTYPE_VEC delta_buffer[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=delta_buffer complete
    
    cache_delta: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        delta_buffer[j] = delta_in.read();
    }
    // Process A and delta to produce ddA
 process_A_delta: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        
        DTYPE_VEC A_val = A_in.read();
        
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
            DTYPE_VEC delta_val = delta_buffer[j];
            //  dA = A * delta
            DTYPE_VEC dA_vec = A_val * delta_val;
            
            //  ddA = exp(dA)
            DTYPE_VEC ddA_vec;
            for(int k = 0; k < VEC_FACTOR; k++) {
                #pragma HLS UNROLL
                DTYPE dA_val = dA_vec[k];
                ddA_vec[k] = (DTYPE)hls::exp(dA_vec[k]);
            }
            ddA_out.write(ddA_vec);
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
        #pragma HLS PIPELINE II=1
        delta_buffer[j] = delta_in.read();
    }
    
    process_B_delta: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        
        DTYPE_VEC B_val = B_in.read();
        
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS UNROLL
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
    DTYPE_VEC C_buffer[N];
    
    // Read X_gate into buffer
    read_X_gate: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        X_gate_buffer[j] = X_gate_in.read();
    }
    
    // Read C into buffer
    read_C: for(int i = 0; i < N; i++) {
        #pragma HLS PIPELINE II=1
        C_buffer[i] = C_in.read();
    }
    
    // Initialize accumulation buffer
    DTYPE_VEC temp_acc[VEC_D];
    #pragma HLS ARRAY_PARTITION variable=temp_acc complete
    
    init_acc: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS UNROLL
        for(int k = 0; k < VEC_FACTOR; k++) {
            #pragma HLS UNROLL
            temp_acc[j][k] = 0;
        }
    }
    
    // Accumulate H1 * C
    accumulate_H1_C: for(int i = 0; i < N; i++) {
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            DTYPE_VEC H1_val = H1_in.read();
            temp_acc[j] += H1_val * C_buffer[i];
        }
    }
    
    // Final output: out = X_gate + temp_acc
    final_output: for(int j = 0; j < VEC_D; j++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC out_val = X_gate_buffer[j] + temp_acc[j];
        out.write(out_val);
    }
}

// ==================== Complete Stream-based SSMU ====================

void SSMU_stream_complete(
    DTYPE kernel[K],
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
    hls::stream<DTYPE_VEC> B_stream("B_stream");
    hls::stream<DTYPE_VEC> C_stream("C_stream");
    hls::stream<DTYPE_VEC> delta_stream("delta_stream");
    hls::stream<DTYPE_VEC> delta_stream2("delta_stream2");
    hls::stream<DTYPE_VEC> delta_stream3("delta_stream3");
    hls::stream<DTYPE_VEC> ddA_stream("ddA_stream");
    hls::stream<DTYPE_VEC> dB_stream("dB_stream");
    hls::stream<DTYPE_VEC> H1_temp_stream("H1_temp_stream");
    
    #pragma HLS STREAM variable=X_gate_stream depth=128
    #pragma HLS STREAM variable=X_ssm_stream depth=128
    #pragma HLS STREAM variable=B_stream depth=128
    #pragma HLS STREAM variable=C_stream depth=128
    #pragma HLS STREAM variable=delta_stream depth=128
    #pragma HLS STREAM variable=delta_stream2 depth=128
    #pragma HLS STREAM variable=delta_stream3 depth=128
    #pragma HLS STREAM variable=ddA_stream depth=128
    #pragma HLS STREAM variable=dB_stream depth=128
    #pragma HLS STREAM variable=H1_temp_stream depth=128

DTYPE kernel_buffer[K];
    #pragma HLS ARRAY_PARTITION variable=kernel_buffer complete

    // 加载kernel
    load_kernel: for (int i = 0; i < K; i++) {
        #pragma HLS PIPELINE II=1
        kernel_buffer[i] = kernel[i];
    }
    
    // Part 1: X to X_gate, B, C, delta
    conv1d_silu_stream(X_in, kernel_buffer, X_gate_stream, X_ssm_stream);
    
    // Split X_ssm for multiple consumers
    hls::stream<DTYPE_VEC> X_ssm_delta("X_ssm_delta");
    #pragma HLS STREAM variable=X_ssm_delta depth=128
    
    duplicate_X_ssm: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC x_ssm_val = X_ssm_stream.read();
        X_ssm_delta.write(x_ssm_val);
    }
    projection_streams(X_ssm_delta, W_B, W_C, W_delta, B_stream, C_stream, delta_stream);

    // Split delta for multiple consumers
    duplicate_delta: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC delta_val = delta_stream.read();
        delta_stream2.write(delta_val);
        delta_stream3.write(delta_val);
    }
    
    // Part 2: A to ddA
    A_to_ddA_stream(A_in, delta_stream2, ddA_stream);
    
    // Part 3: B to dB  
    B_to_dB_stream(B_stream, delta_stream3, dB_stream);
    
    // Part 4: H update and final output
    update_H_stream(ddA_stream, X_ssm_delta, dB_stream, H0_in, H1_temp_stream);
    
    // Split H1 for output and final computation
        // For final output, we need to process H1 and C with X_gate
    // We'll create a separate path for this
    
    hls::stream<DTYPE_VEC> H1_for_final("H1_for_final");
    #pragma HLS STREAM variable=H1_for_final depth=128
    copy_H1_for_final: for(int i = 0; i < N * VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC H1_val = H1_temp_stream.read();
        H1_for_final.write(H1_val);
        H1_out.write(H1_val);
    }
    
    // Re-read H1 for final computation (in a real implementation, we'd avoid this duplication)
    // For now, we'll assume H1_out is a copy and we can use the original
    final_output_stream(X_gate_stream, H1_for_final, C_stream, out);
}

// Original SSMU function for compatibility
void SSMU(
    DTYPE kernel[K],
    DTYPE_VEC A[N][VEC_D], DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D],
    DTYPE_VEC X[VEC_D],
    DTYPE_VEC H0[N][VEC_D], DTYPE_VEC H1[N][VEC_D],
    DTYPE_VEC W_delta[VEC_D][VEC_D],
    DTYPE_VEC out[VEC_D]) {
    
    #pragma HLS DATAFLOW
    
    // Convert arrays to streams
    hls::stream<DTYPE_VEC> A_stream, X_stream, H0_stream, H1_stream, out_stream;
    #pragma HLS STREAM variable=A_stream depth=64
    #pragma HLS STREAM variable=X_stream depth=64
    #pragma HLS STREAM variable=H0_stream depth=64
    #pragma HLS STREAM variable=H1_stream depth=64
    #pragma HLS STREAM variable=out_stream depth=64
    
    // Write input arrays to streams
    write_A: for(int i = 0; i < N; i++) {
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            A_stream.write(A[i][j]);
        }
    }
    
    write_X: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        X_stream.write(X[i]);
    }
    
    write_H0: for(int i = 0; i < N; i++) {
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            H0_stream.write(H0[i][j]);
        }
    }
    
    // Call stream-optimized version
    SSMU_stream_complete(kernel, A_stream, W_B, W_C, W_delta, X_stream, H0_stream, H1_stream, out_stream);
    
    // Read outputs from streams
    read_H1: for(int i = 0; i < N; i++) {
        for(int j = 0; j < VEC_D; j++) {
            #pragma HLS PIPELINE II=1
            H1[i][j] = H1_stream.read();
        }
    }
    
    read_out: for(int i = 0; i < VEC_D; i++) {
        #pragma HLS PIPELINE II=1
        out[i] = out_stream.read();
    }
}
