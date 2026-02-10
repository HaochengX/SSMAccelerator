#include "Mamba.h"
#include "Input_projection.h"
#include "Splitter.h"
#include "dt.h"
#include "dA.h"
#include "conv1d.h"
#include "ssd.h"
#include "output_projection.h"
#include "rmsnorm.h"

void PREPARE_KERNEL(const DTYPE *&conv_kernel, hls::stream<DTYPE> &kernel_stream) {
  for (int i = 0; i < K; i++) {
#pragma HLS PIPELINE II = 1
    kernel_stream.write(conv_kernel[i]);
  };
}

void mamba2_top(hls::stream<DTYPE_VEC> &input_stream,

           const DTYPE input_proj_weight[VEC_INPUT_LINEAR][VEC_D][VEC_FACTOR],
           const FDTYPE A_param[H], const DTYPE conv_kernel[K],
           const DTYPE output_proj_weight[VEC_I][VEC_D][VEC_FACTOR][VEC_FACTOR],

           hls::stream<DTYPE_VEC> &output_stream) {
#pragma HLS DATAFLOW

  hls::stream<DTYPE_VEC> proj_stream;
  hls::stream<DTYPE_VEC> dt_stream;
  hls::stream<DTYPE_VEC> XBC_stream;
  hls::stream<DTYPE_VEC> Z_stream;

  hls::stream<DTYPE_VEC> dt_softplus_stream;
  hls::stream<DTYPE_VEC> dA_stream;
  hls::stream<DTYPE> kernel_stream;
  hls::stream<DTYPE_VEC> X_conv_stream;
  hls::stream<DTYPE_VEC> B_conv_stream;
  hls::stream<DTYPE_VEC> C_conv_stream;

  hls::stream<DTYPE_VEC> X_ssd_stream;
  hls::stream<DTYPE_VEC> dA_ssd_stream;
  hls::stream<DTYPE_VEC> B_ssd_stream;
  hls::stream<DTYPE_VEC> C_ssd_stream;

  hls::stream<DTYPE_VEC> Y_stream;
  hls::stream<DTYPE_VEC> final_state_stream;

#pragma HLS STREAM variable = proj_stream depth = 64
#pragma HLS STREAM variable = dt_stream depth = 32
#pragma HLS STREAM variable = XBC_stream depth = 64
#pragma HLS STREAM variable = Z_stream depth = 64
#pragma HLS STREAM variable = dt_softplus_stream depth = 32
#pragma HLS STREAM variable = dA_stream depth = 32
#pragma HLS STREAM variable = kernel_stream depth = 4
#pragma HLS STREAM variable = X_conv_stream depth = 64
#pragma HLS STREAM variable = B_conv_stream depth = 32
#pragma HLS STREAM variable = C_conv_stream depth = 32
#pragma HLS STREAM variable = X_ssd_stream depth = 64
#pragma HLS STREAM variable = dA_ssd_stream depth = 32
#pragma HLS STREAM variable = B_ssd_stream depth = 32
#pragma HLS STREAM variable = C_ssd_stream depth = 32
#pragma HLS STREAM variable = Y_stream depth = 64
#pragma HLS STREAM variable = final_state_stream depth = 8

  input_projection(input_stream, input_proj_weight, proj_stream);

  splitter(proj_stream, XBC_stream, Z_stream, dt_stream);

  softplus(dt_stream, dt_softplus_stream);

  dA_sequence(dt_softplus_stream, A_param, dA_stream);

  PREPARE_KERNEL(conv_kernel, kernel_stream);

  conv1d_sequence(XBC_stream, kernel_stream, X_conv_stream, B_conv_stream,
                  C_conv_stream);

  rearrange_for_ssd(X_conv_stream, dA_stream, B_conv_stream, C_conv_stream,
                    X_ssd_stream, dA_ssd_stream, B_ssd_stream, C_ssd_stream);

  hls::stream<DTYPE_VEC> empty_state_stream;
  hls::stream<bool> has_init_stream;

#pragma HLS STREAM variable = empty_state_stream depth = 1
#pragma HLS STREAM variable = has_init_stream depth = 1

  has_init_stream.write(false);

  ssd_top(X_ssd_stream, dA_ssd_stream, B_ssd_stream, C_ssd_stream,
          empty_state_stream, has_init_stream, Y_stream, final_state_stream);

  output_projection(Y_stream, output_proj_weight, output_stream);
}