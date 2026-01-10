#include "ssmu.h"
// #include "lut_optimized.h"



// ==================== Part 1: X to X_ssm, X_gate, B, C, delta ====================
void input_projection(
    hls::stream<DTYPE_VEC>& X_in,
    DTYPE_VEC W_in_ssm[VEC_D][VEC_D],
    DTYPE_VEC W_in_gate[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_ssm_out,
    hls::stream<DTYPE_VEC>& X_gate_out
) {
    #pragma HLS INLINE off
    
    DTYPE_VEC X_buf[VEC_D];
    #pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=bram
    
    read_x_loop:
    for (int j = 0; j < VEC_D; ++j) {
        #pragma HLS PIPELINE II=1
        X_buf[j] = X_in.read();
    }
    
    compute_xssm_loop:
    for (int i = 0; i < VEC_D; ++i) {
        DTYPE_VEC Wssm_row[VEC_D];
        DTYPE_VEC Wgate_row[VEC_D];
        
        preload_weights_loop:
        for (int j = 0; j < VEC_D; ++j) {
            #pragma HLS PIPELINE II=1
            Wssm_row[j] = W_in_ssm[i][j];
            Wgate_row[j] = W_in_gate[i][j];
        }
        DTYPE_VEC acc_ssm;
        DTYPE_VEC acc_gate;
        
        mac_ssm_loop:
        for (int j = 0; j < VEC_D; ++j) {
            #pragma HLS PIPELINE II=1
            DTYPE_VEC x = X_buf[j];
            DTYPE_VEC w_ssm = Wssm_row[j];
            DTYPE_VEC w_gate = Wgate_row[j];
            
            for (int l = 0; l < VEC_FACTOR; ++l) {
                #pragma HLS PIPELINE II=1
                FDTYPE prod_ssm = (FDTYPE)x[l] * (FDTYPE)w_ssm[l];
                FDTYPE prod_gate = (FDTYPE)x[l] * (FDTYPE)w_gate[l];
                
                if (j == 0) {
                    acc_ssm[l] = (DTYPE)prod_ssm;
                    acc_gate[l] = (DTYPE)prod_gate;
                } else {
                    acc_ssm[l] = (DTYPE)((FDTYPE)acc_ssm[l] + prod_ssm);
                    acc_gate[l] = (DTYPE)((FDTYPE)acc_gate[l] + prod_gate);
                }
            }
        }
        
        X_ssm_out.write(acc_ssm);
        X_gate_out.write(acc_gate);
    }
}

void silu_gate(
    hls::stream<DTYPE_VEC> &X_gate_in,
    hls::stream<DTYPE_VEC> &X_gate_out
) {

    DTYPE_VEC x;
    DTYPE_VEC y;
    x = X_gate_in.read();
        for (int k = 0; k < VEC_FACTOR; ++k) {
            #pragma HLS UNROLL
            y[k] = silu_elem(x[k]);
        }

    X_gate_out.write(y);
}


// Optimized conv1d and silu with streams
void conv1d_stream(hls::stream<DTYPE_VEC>& X_in, hls::stream<DTYPE>& kernel_in,
                 hls::stream<DTYPE_VEC>& X_ssm_out) {
    #pragma HLS INLINE off

    static DTYPE line_buffer[K-1][VEC_FACTOR];
    #pragma HLS BIND_STORAGE variable=line_buffer type=ram_s2p impl=bram

    DTYPE kernel_buffer[K];
    for (int i = 0; i < K; ++i) {
        #pragma HLS PIPELINE
        kernel_buffer[i] = kernel_in.read();
    }

    DTYPE_VEC X_buffer[VEC_D];
    #pragma HLS BIND_STORAGE variable=X_buffer type=ram_s2p impl=bram

    read_input: for (int i = 0; i < VEC_D; ++i) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC xv = X_in.read();
        X_buffer[i] = xv;
    }

    init_line_buffer: for (int i = 0; i < K-1; ++i) {
        for (int k = 0; k < VEC_FACTOR; ++k) {
                    #pragma HLS PIPELINE II=1
            line_buffer[i][k] = 0;
        }
    }
    conv_proc: for (int i = 0; i < VEC_D; ++i) {
        DTYPE_VEC in_vec = X_buffer[i];
        DTYPE window[K][VEC_FACTOR];
        
        for (int j = 0; j < K-1; ++j) {
            for (int k = 0; k < VEC_FACTOR; ++k) {
                        #pragma HLS PIPELINE II=1
                window[j][k] = line_buffer[j][k];
            }
        }
        for (int k = 0; k < VEC_FACTOR; ++k) {
                    #pragma HLS PIPELINE II=1
            window[K-1][k] = in_vec[k];
        }
        for (int j = K-2; j > 0; --j) {
            for (int k = 0; k < VEC_FACTOR; ++k) {
                        #pragma HLS PIPELINE II=1
                line_buffer[j][k] = line_buffer[j-1][k];
            }
        }
        for (int k = 0; k < VEC_FACTOR; ++k) {
                    #pragma HLS PIPELINE II=1
            line_buffer[0][k] = in_vec[k];
        }

        DTYPE_VEC conv_out;
        for (int lane = 0; lane < VEC_FACTOR; ++lane) {
            FDTYPE sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                        #pragma HLS PIPELINE II=1
                sum += (FDTYPE)kernel_buffer[k] * (FDTYPE)window[k][lane];
            }
            conv_out[lane] = (DTYPE) sum;
        }

        X_ssm_out.write(conv_out);
    }
}

// Optimized projections with streams for B, C, and delta
void projection_streams(hls::stream<DTYPE_VEC>& X_ssm_in,
                        DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
                        hls::stream<DTYPE_VEC>& B_out, hls::stream<DTYPE_VEC>& C_out, 
                        hls::stream<DTYPE_VEC>& delta_out_A, hls::stream<DTYPE_VEC>& delta_out_B) {
    #pragma HLS INLINE off

    // Avoid partitioning large weight matrices in outer dimensions to allow BRAM implementation.
    // lightly partition local buffer
    DTYPE_VEC X_buf[VEC_D];
    #pragma HLS BIND_STORAGE variable=X_buf type=ram_s2p impl=bram
    // Read X_ssm (VEC_D vectors)
    load_xssm_loop:for (int j = 0; j < VEC_D; ++j) {
        #pragma HLS PIPELINE II=1
        X_buf[j] = X_ssm_in.read();
    }
    
    compute_delta:
    for (int i = 0; i < VEC_D; ++i) {

    // ===============================
    // Stage 1: preload (NO pipeline)
    // ===============================
    DTYPE_VEC Wd_row[VEC_D];

preload_wdelta_row_loop:
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        Wd_row[j] = W_delta[i][j];
    }

    // ===============================
    // Stage 2: compute (PIPELINED)
    // ===============================

    DTYPE_VEC acc;
delta_mac_j_loop:
    for (int j = 0; j < VEC_D; ++j) {
                #pragma HLS PIPELINE II=1
        DTYPE_VEC x = X_buf[j];
        DTYPE_VEC w = Wd_row[j];

delta_mac_lane_loop:
        for (int l = 0; l < VEC_FACTOR; ++l) {
                    #pragma HLS PIPELINE II=1
            FDTYPE prod = (FDTYPE)x[l] * (FDTYPE)w[l];
            if (j == 0)
                acc[l] = (DTYPE)prod;
            else
                acc[l] = (DTYPE)((FDTYPE)acc[l] + prod);
        }
    }

    DTYPE_VEC delta_vec;
delta_softplus_lane_loop:
    for (int l = 0; l < VEC_FACTOR; ++l) {
                #pragma HLS PIPELINE II=1
        delta_vec[l] = softplus_elem(acc[l]);
    }

    delta_out_A.write(delta_vec);
    delta_out_B.write(delta_vec);
}

     for (int i = 0; i < N; ++i) {
        // preload W_B / W_C row
        DTYPE_VEC WB_row[VEC_D];
        DTYPE_VEC WC_row[VEC_D];

        for (int j = 0; j < VEC_D; ++j) {
                    #pragma HLS PIPELINE II=1
            WB_row[j] = W_B[i][j];
            WC_row[j] = W_C[i][j];
        }

        // ---------- B projection ----------
        DTYPE_VEC accB;
        b_mac_j_loop: for (int j = 0; j < VEC_D; ++j) {

            DTYPE_VEC x = X_buf[j];
            DTYPE_VEC w = WB_row[j];
            for (int l = 0; l < VEC_FACTOR; ++l) {
                        #pragma HLS PIPELINE II=1
        FDTYPE prod = (FDTYPE)x[l] * (FDTYPE)w[l];
                if (j == 0)
                    accB[l] = (DTYPE)prod;
                else
                    accB[l] = (DTYPE)((FDTYPE)accB[l] + prod);
                    }
        }
        B_out.write(accB);

        // ---------- C projection (emit per j) ----------
        c_j_loop: for (int j = 0; j < VEC_D; ++j) {
            DTYPE_VEC outC;
            DTYPE_VEC x = X_buf[j];
            DTYPE_VEC w = WC_row[j];

            for (int l = 0; l < VEC_FACTOR; ++l) {
                        #pragma HLS PIPELINE II=1
                outC[l] = (DTYPE)((FDTYPE)x[l] * (FDTYPE)w[l]);
            }
            C_out.write(outC);
        }
    }
}

void A_to_ddA_stream(
    hls::stream<DTYPE_VEC>& A_in,
    hls::stream<DTYPE_VEC>& delta_in,
    hls::stream<DTYPE_VEC>& ddA_out
) {
#pragma HLS INLINE off

    DTYPE_VEC ddA_buf[VEC_D];

compute_ddA_loop:
    for (int j = 0; j < VEC_D; ++j) {
        DTYPE_VEC Aij = A_in.read();
        DTYPE_VEC dij = delta_in.read();

compute_ddA_lane_loop:
        for (int l = 0; l < VEC_FACTOR; ++l) {
                    #pragma HLS PIPELINE II=1
            ddA_buf[j][l] =
                (DTYPE)((FDTYPE)Aij[l] * (FDTYPE)dij[l]);
        }
    }

write_ddA_loop:
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        ddA_out.write(ddA_buf[j]);
    }
}

// ==================== Part 3: B to dB ====================
void B_to_dB_stream(
    hls::stream<DTYPE_VEC>& B_in,
    hls::stream<DTYPE_VEC>& delta_in,
    hls::stream<DTYPE_VEC>& dB_out
) {
#pragma HLS INLINE off

    DTYPE_VEC dB_buf[VEC_D];

compute_dB_loop:
    for (int j = 0; j < VEC_D; ++j) {
        DTYPE_VEC Bij = B_in.read();
        DTYPE_VEC dij = delta_in.read();

compute_dB_lane_loop:
        for (int l = 0; l < VEC_FACTOR; ++l) {
                    #pragma HLS PIPELINE II=1
            dB_buf[j][l] =
                (DTYPE)((FDTYPE)Bij[l] * (FDTYPE)dij[l]);
        }
    }

write_dB_loop:
    for (int j = 0; j < VEC_D; ++j) {
#pragma HLS PIPELINE II=1
        dB_out.write(dB_buf[j]);
    }
}

// ==================== Part 4: H update and final output ====================
void update_H_stream(hls::stream<DTYPE_VEC>& ddA_in, hls::stream<DTYPE_VEC>& dX_in,
                     hls::stream<DTYPE_VEC>& dB_in, hls::stream<DTYPE_VEC>& H0_in,
                     hls::stream<DTYPE_VEC>& H1_out) {
 #pragma HLS INLINE off

update_i_loop:
    for (int i = 0; i < N; ++i) {
    update_j_loop:
        for (int j = 0; j < VEC_D; ++j) {

            // one (i,j) per iteration
            DTYPE_VEC H0v = H0_in.read();
            DTYPE_VEC ddA = ddA_in.read();
            DTYPE_VEC dBv = dB_in.read();
            DTYPE_VEC dXv = dX_in.read();

            DTYPE_VEC H1v;
        update_lane_loop:
            for (int l = 0; l < VEC_FACTOR; ++l) {
                        #pragma HLS PIPELINE II=1
                FDTYPE v =
                    (FDTYPE)H0v[l] * (FDTYPE)ddA[l] +
                    (FDTYPE)dBv[l] * (FDTYPE)dXv[l];
                H1v[l] = (DTYPE)v;
            }

            // exactly ONE write per iteration
            H1_out.write(H1v);
        }
    }
}

void Xssm_output(
    hls::stream<DTYPE_VEC>& H1_in,
    hls::stream<DTYPE_VEC>& C_in,
    hls::stream<DTYPE_VEC>& out
) {
#pragma HLS INLINE off

H1_C_loop:
    for (int j = 0; j < VEC_D; ++j) {
        DTYPE acc_local[VEC_FACTOR];
        
        for (int l = 0; l < VEC_FACTOR; ++l) {
                    #pragma HLS PIPELINE II=1
            acc_local[l] = (DTYPE)0;
        }
        
        for (int i = 0; i < N; ++i) {
            DTYPE_VEC H1v = H1_in.read();
            DTYPE_VEC Cv = C_in.read();
            
            for (int l = 0; l < VEC_FACTOR; ++l) {
                        #pragma HLS PIPELINE II=1
                acc_local[l] = (DTYPE)((FDTYPE)acc_local[l] + 
                                       (FDTYPE)H1v[l] * (FDTYPE)Cv[l]);
            }
        }

        DTYPE_VEC out_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
                    #pragma HLS PIPELINE II=1
            out_vec[l] = acc_local[l];
        }
        
        out.write(out_vec);
    }
}

void duplicate_H1_stream(hls::stream<DTYPE_VEC>& in,
                        hls::stream<DTYPE_VEC>& out1,
                        hls::stream<DTYPE_VEC>& out2) {
    #pragma HLS INLINE off
    for (int i = 0; i < N * VEC_D; ++i) {
        #pragma HLS PIPELINE II=1
        DTYPE_VEC v = in.read();
        out1.write(v);
        out2.write(v);
    }
}

void gate_mul(
    hls::stream<DTYPE_VEC> &X_ssm,
    hls::stream<DTYPE_VEC> &X_gate,
    hls::stream<DTYPE_VEC> &Y_out
) {
#pragma HLS INLINE off
  for (int j = 0; j < VEC_D; ++j) {
        DTYPE_VEC ssm = X_ssm.read();
        DTYPE_VEC gate = X_gate.read();
        
        DTYPE_VEC out_vec;
        for (int l = 0; l < VEC_FACTOR; ++l) {
                    #pragma HLS PIPELINE II=1
            out_vec[l] = (DTYPE)((FDTYPE)ssm[l] * (FDTYPE)gate[l]);
        }
        
        Y_out.write(out_vec);
    }
}

void output_projection(
    hls::stream<DTYPE_VEC>& Y_in,
    DTYPE_VEC W_out[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& Y_out
) {
    #pragma HLS INLINE off
    
    DTYPE_VEC Y_buf[VEC_D];
    #pragma HLS BIND_STORAGE variable=Y_buf type=ram_s2p impl=bram
    
    read_y_loop:
    for (int j = 0; j < VEC_D; ++j) {
        #pragma HLS PIPELINE II=1
        Y_buf[j] = Y_in.read();
    }
    
    compute_output_loop:
    for (int i = 0; i < VEC_D; ++i) {
        DTYPE_VEC Wout_row[VEC_D];
        
        preload_wout_row_loop:
        for (int j = 0; j < VEC_D; ++j) {
            #pragma HLS PIPELINE II=1
            Wout_row[j] = W_out[i][j];
        }
        
        DTYPE_VEC acc_out;
        
        mac_output_loop:
        for (int j = 0; j < VEC_D; ++j) {
            #pragma HLS PIPELINE II=1
            DTYPE_VEC y = Y_buf[j];
            DTYPE_VEC w = Wout_row[j];
            
            for (int l = 0; l < VEC_FACTOR; ++l) {
                #pragma HLS PIPELINE II=1
                FDTYPE prod = (FDTYPE)y[l] * (FDTYPE)w[l];
                
                if (j == 0) {
                    acc_out[l] = (DTYPE)prod;
                } else {
                    acc_out[l] = (DTYPE)((FDTYPE)acc_out[l] + prod);
                }
            }
        }
        
        Y_out.write(acc_out);
    }
}

// ==================== Complete Stream-based SSMU ====================

void SSMU(
    hls::stream<DTYPE>& kernel_in,
    hls::stream<DTYPE_VEC>& A_in,
    DTYPE_VEC W_B[N][VEC_D], DTYPE_VEC W_C[N][VEC_D], DTYPE_VEC W_delta[VEC_D][VEC_D],
    DTYPE_VEC W_in_ssm[VEC_D][VEC_D], 
    DTYPE_VEC W_in_gate[VEC_D][VEC_D], 
    DTYPE_VEC W_out[VEC_D][VEC_D],
    hls::stream<DTYPE_VEC>& X_in,
    hls::stream<DTYPE_VEC>& H0_in,
    hls::stream<DTYPE_VEC>& H1_out,
    hls::stream<DTYPE_VEC>& out) {

    #pragma HLS DATAFLOW

    // Internal streams
    hls::stream<DTYPE_VEC> X_proj_ssm("X_proj_ssm");
    hls::stream<DTYPE_VEC> X_proj_gate("X_proj_gate");
    hls::stream<DTYPE_VEC> X_gate_stream("X_gate_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream("X_ssm_stream");
    hls::stream<DTYPE_VEC> B_stream("B_stream");
    hls::stream<DTYPE_VEC> C_stream("C_stream");
    hls::stream<DTYPE_VEC> delta_stream_A("delta_stream_A");
    hls::stream<DTYPE_VEC> delta_stream_B("delta_stream_B");
    hls::stream<DTYPE_VEC> ddA_stream("ddA_stream");
    hls::stream<DTYPE_VEC> dB_stream("dB_stream");
    hls::stream<DTYPE_VEC> H1_temp_stream("H1_temp_stream");
    hls::stream<DTYPE_VEC> H1_final_stream("H1_final_stream");
    hls::stream<DTYPE_VEC> X_ssm_output("X_ssm_output");
    hls::stream<DTYPE_VEC> Y("Y");
    // Minimal FIFO depths
    #pragma HLS STREAM variable=X_proj_ssm depth=2
    #pragma HLS STREAM variable=X_proj_gate depth=2
    #pragma HLS STREAM variable=X_gate_stream depth=2
    #pragma HLS STREAM variable=X_ssm_stream depth=2
    #pragma HLS STREAM variable=B_stream depth=2
    #pragma HLS STREAM variable=C_stream depth=2
    #pragma HLS STREAM variable=delta_stream_A depth=2
    #pragma HLS STREAM variable=delta_stream_B depth=2
    #pragma HLS STREAM variable=ddA_stream depth=2
    #pragma HLS STREAM variable=dB_stream depth=2
    #pragma HLS STREAM variable=H1_temp_stream depth=2
    #pragma HLS STREAM variable=H1_final_stream depth=2
    #pragma HLS STREAM variable=X_ssm_output depth=2
    #pragma HLS STREAM variable=Y depth=2

    // Part 1: Compute X_gate and X_ssm
    input_projection(X_in, W_in_ssm, W_in_gate, X_proj_ssm, X_proj_gate);
    conv1d_stream(X_proj_ssm, kernel_in, X_ssm_stream);
    silu_gate(X_proj_gate, X_gate_stream);
    // Part 2: Projections (produces B, C, and delta for both A and B)
    projection_streams(X_ssm_stream, W_B, W_C, W_delta, 
                      B_stream, C_stream, delta_stream_A, delta_stream_B);

    // Part 3: A -> ddA and B -> dB (parallel)
    A_to_ddA_stream(A_in, delta_stream_A, ddA_stream);
    B_to_dB_stream(B_stream, delta_stream_B, dB_stream);

    // Part 4: Update H state
    update_H_stream(ddA_stream, X_ssm_stream, dB_stream, H0_in, H1_temp_stream);

    // Duplicate H1 for output and next iteration
    duplicate_H1_stream(H1_temp_stream, H1_final_stream, H1_out);

    // Part 5: Final output
    Xssm_output(H1_final_stream, C_stream, X_ssm_output);
    
    gate_mul(X_ssm_output, X_gate_stream, Y);

    output_projection(Y, W_out, out);
}