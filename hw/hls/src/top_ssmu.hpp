// =============================================================================
// top_ssmu.hpp — Top-level SSMU kernel + SSMU_STACK64 extern "C" wrapper
// =============================================================================
#ifndef __TOP_SSMU_HPP__
#define __TOP_SSMU_HPP__

// Include all sub-module headers
#include "stream_utils.hpp"
#include "rmsnorm.hpp"
#include "in_proj.hpp"
#include "conv1d_silu.hpp"
#include "dt_adapt.hpp"
#include "ssm_scan.hpp"
#include "out_proj.hpp"

// ============================================================
// AXI tuning macros
// ============================================================
#ifndef SSMU_AXI_RO_TUNE
#define SSMU_AXI_RO_TUNE max_read_burst_length=128 num_read_outstanding=32
#endif

// Stream depth: reduced from 650 -> 64 -> 16.
// At depth=16: 1 SRL16 stage = 128 LUTs per stream -> ~4k LUTs (~3.7%).
// Deadlock-critical streams (Z/G/X_ssm_out/X_residual) set explicitly.
#ifndef SSMU_STREAM_DEPTH
#define SSMU_STREAM_DEPTH 16
#endif

#ifndef SSMU_TRACE_DEPTH
#define SSMU_TRACE_DEPTH 800
#endif

// ============================================================
// SSMU: main kernel (low-rank interface)
// ============================================================
void SSMU(
    hls::stream<DTYPE>&      kernel_in,

    const DTYPE_VEC          A_fixed[STATE_V],
    const DTYPE_VEC          RMS_weight[D_T],
    const DTYPE_VEC          RMS_weight_2[C2_T],

    // LOW-RANK: W_inproj factored into W_in_1 x W_in_2
    const W_VEC              W_in_1[D_T][RANK_T],
    const W_VEC              W_in_2[RANK_T][INP_X_T],
    const W_VEC              W_in_nonlr[D_T][INP_NONLR_T],

    const W_VEC              W_delta[C2_T][C2_T],

    // LOW-RANK: W_out factored into W_out_A x W_out_B
    const W_VEC              W_out_A[D_T][RANK_T],
    const W_VEC              W_out_B[RANK_T][C2_T],

    const DTYPE_VEC          D_diag[C2_T],

    hls::stream<DTYPE_VEC>&  X_in,
    hls::stream<DTYPE_VEC>&  H0_in,

    hls::stream<DTYPE_VEC>&  conv_state_in,
    hls::stream<DTYPE_VEC>&  conv_state_out,

    DTYPE_VEC*               C_ddr,
    DTYPE_VEC*               H1_ddr,

    hls::stream<DTYPE_VEC>&  H1_out,
    hls::stream<DTYPE_VEC>&  out,

    float                    w_scale_in,
    float                    w_scale_delta,
    float                    w_scale_out
) {
    // =========================================================================
    // 1) Interfaces
    // =========================================================================
#pragma HLS INTERFACE ap_fifo   port=kernel_in
#pragma HLS INTERFACE ap_fifo   port=X_in
#pragma HLS INTERFACE ap_fifo   port=H0_in
#pragma HLS INTERFACE ap_fifo   port=conv_state_in
#pragma HLS INTERFACE ap_fifo   port=conv_state_out
#pragma HLS INTERFACE ap_fifo   port=H1_out
#pragma HLS INTERFACE ap_fifo   port=out

#pragma HLS INTERFACE m_axi port=A_fixed      offset=slave bundle=gmemA      depth=STATE_V             SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=RMS_weight   offset=slave bundle=gmemRMS    depth=D_T                SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=RMS_weight_2 offset=slave bundle=gmemRMS2   depth=C2_T               SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=D_diag       offset=slave bundle=gmemD      depth=C2_T               SSMU_AXI_RO_TUNE

// LOW-RANK weight interfaces (4 sub-matrices instead of 2 full matrices)
#pragma HLS INTERFACE m_axi port=W_in_1       offset=slave bundle=gmemIn1    depth=SSMU_DEPTH_IN1      SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_in_2       offset=slave bundle=gmemIn2    depth=SSMU_DEPTH_IN2_X    SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_in_nonlr   offset=slave bundle=gmemInNLR  depth=SSMU_DEPTH_IN_NONLR SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_delta      offset=slave bundle=gmemDelta  depth=SSMU_DEPTH_DELTA    SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_out_A      offset=slave bundle=gmemOutA   depth=SSMU_DEPTH_OUTA     SSMU_AXI_RO_TUNE
#pragma HLS INTERFACE m_axi port=W_out_B      offset=slave bundle=gmemOutB   depth=SSMU_DEPTH_OUTB     SSMU_AXI_RO_TUNE

#pragma HLS INTERFACE m_axi     port=C_ddr     offset=slave bundle=gmem0 depth=HUGE_LEN
#pragma HLS INTERFACE m_axi     port=H1_ddr    offset=slave bundle=gmem1 depth=HUGE_LEN
#pragma HLS INTERFACE s_axilite port=C_ddr     bundle=control
#pragma HLS INTERFACE s_axilite port=H1_ddr    bundle=control

#pragma HLS INTERFACE s_axilite port=w_scale_in     bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_delta  bundle=control
#pragma HLS INTERFACE s_axilite port=w_scale_out    bundle=control
#pragma HLS INTERFACE s_axilite port=return         bundle=control

    // =========================================================================
    // 2) Runtime scales
    // =========================================================================
    const ACC_T wscale_in_fx    = pick_scale_fx(w_scale_in,    (float)SSMU_W_SCALE_IN);
    const ACC_T wscale_delta_fx = pick_scale_fx(w_scale_delta, (float)SSMU_W_SCALE_DELTA);
    const ACC_T wscale_out_fx   = pick_scale_fx(w_scale_out,   (float)SSMU_W_SCALE_OUT);

#ifndef __SYNTHESIS__
    DUT_PRINTF("[DUT] SSMU LOW-RANK v3 (resource-optimized): RANK=%d RANK_T=%d\n", (int)SSMU_RANK, (int)RANK_T);
    DUT_PRINTF("[DUT] ACC_T=ap_fixed<24,8>  ACT_T=ap_fixed<16,6>  EXP_T=ap_fixed<16,8>  RMS_INV_T=ap_fixed<24,12>\n");
    DUT_PRINTF("[DUT] SSMU_STREAM_DEPTH=%d\n", (int)SSMU_STREAM_DEPTH);
    DUT_PRINTF("[DUT] SSMU_USE_INT8=%d\n", (int)SSMU_USE_INT8);
    DUT_PRINTF("[DUT] STATE_SCALAR=%d STATE_V(STATE_T)=%d\n", (int)STATE_SCALAR, (int)STATE_V);
    DUT_PRINTF("[DUT] W_in_1: [%d][%d], W_in_2(mid): [%d][%d], W_in_nonlr: [%d][%d]\n",
               D_T, RANK_T, RANK_T, INP_X_T, D_T, INP_NONLR_T);
    DUT_PRINTF("[DUT] W_out_A: [%d][%d], W_out_B: [%d][%d]\n", D_T, RANK_T, RANK_T, C2_T);
#endif

    // =========================================================================
    // 3) Preload constants
    // =========================================================================
    DTYPE_VEC A_local[STATE_V];
    DTYPE_VEC RMS_local[D_T];
    DTYPE_VEC RMS2_local[C2_T];
    DTYPE_VEC D_local[C2_T];

#pragma HLS BIND_STORAGE    variable=A_local    type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE    variable=RMS_local  type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE    variable=RMS2_local type=ram_2p impl=lutram
#pragma HLS BIND_STORAGE    variable=D_local    type=ram_2p impl=lutram
#pragma HLS ARRAY_PARTITION variable=A_local    cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=RMS_local  cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=RMS2_local cyclic factor=8 dim=1
#pragma HLS ARRAY_PARTITION variable=D_local    cyclic factor=8 dim=1

    preload_vec_table_local_dyn(A_fixed,         A_local,    STATE_V);
    preload_vec_table_local<D_T>(RMS_weight,     RMS_local);
    preload_vec_table_local<C2_T>(RMS_weight_2,  RMS2_local);
    preload_vec_table_local<C2_T>(D_diag,        D_local);

    // =========================================================================
    // 4) Streams
    // =========================================================================
    hls::stream<DTYPE_VEC> X_in_pre("X_in_pre");
    hls::stream<DTYPE_VEC> H0_in_pre("H0_in_pre");
    hls::stream<DTYPE_VEC> conv_state_pre("conv_state_pre");

    hls::stream<DTYPE>     kernel_local("kernel_local");
    hls::stream<DTYPE_VEC> X_local("X_local");
    hls::stream<DTYPE_VEC> X_for_norm("X_for_norm");
    hls::stream<DTYPE_VEC> X_residual("X_residual");
    hls::stream<DTYPE_VEC> X_normed("X_normed");
    hls::stream<DTYPE_VEC> X_normed_lr("X_normed_lr");
    hls::stream<DTYPE_VEC> X_normed_nonlr("X_normed_nonlr");

#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> trace_rms("trace_rms");
    hls::stream<DTYPE_VEC> trace_delta("trace_delta");
    hls::stream<DTYPE_VEC> trace_htC("trace_htC");
    hls::stream<DTYPE_VEC> trace_core("trace_core");
#endif

    hls::stream<DTYPE_VEC> Z_stream("Z_stream");
    hls::stream<DTYPE_VEC> XBC_stream("XBC_stream");
    hls::stream<DTYPE_VEC> DT_stream("DT_stream");
    hls::stream<DTYPE_VEC> X_mid_stream("X_mid_stream");
    hls::stream<DTYPE_VEC> nlr_stream("nlr_stream");
    hls::stream<DTYPE_VEC> B_raw_stream("B_raw_stream");
    hls::stream<DTYPE_VEC> C_raw_stream("C_raw_stream");

    hls::stream<DTYPE_VEC> DT_C2_stream("DT_C2_stream");
    hls::stream<DTYPE_VEC> G_stream("G_stream");
    hls::stream<DTYPE_VEC> X_ssm_stream("X_ssm_stream");
    hls::stream<DTYPE_VEC> conv_state_local_in("conv_state_local_in");
    hls::stream<DTYPE_VEC> conv_state_local_out("conv_state_local_out");

    hls::stream<vec_tuple8>  Wd_tiles("Wd_tiles");
    hls::stream<ap_uint<1> > start_wd("start_wd");

    hls::stream<DTYPE_VEC> X_ssm_proj_stream("X_ssm_proj_stream");
    hls::stream<DTYPE_VEC> X_ssm_scan_stream("X_ssm_scan_stream");
    hls::stream<DTYPE_VEC> X_ssm_out_stream("X_ssm_out_stream");

    hls::stream<DTYPE_VEC> B_stream_S("B_stream_S");
    hls::stream<DTYPE_VEC> C_stream_S("C_stream_S");

    hls::stream<DTYPE_VEC> delta_selected("delta_selected");
    hls::stream<DTYPE_VEC> delta_for_dA("delta_for_dA");
    hls::stream<DTYPE_VEC> delta_for_scan("delta_for_scan");
    hls::stream<DTYPE_VEC> dA_stream("dA_stream");

    hls::stream<DTYPE_VEC> htC_stream("htC_stream");
    hls::stream<DTYPE_VEC> C_trace_stream("C_trace_stream");
    hls::stream<DTYPE_VEC> H1_trace_stream("H1_trace_stream");
    hls::stream<DTYPE_VEC> H1_state_stream("H1_state_stream");

    hls::stream<DTYPE_VEC> ssm_core_out_stream("ssm_core_out_stream");
    hls::stream<DTYPE_VEC> ssm_normed_stream("ssm_normed_stream");
    hls::stream<DTYPE_VEC> out_proj_stream_s("out_proj_stream_s");
    hls::stream<DTYPE_VEC> out_local("out_local");

    hls::stream<DTYPE_VEC> B_conv_stream("B_conv_stream");
    hls::stream<DTYPE_VEC> C_conv_stream("C_conv_stream");

    // LOW-RANK intermediate temp streams
    hls::stream<DTYPE_VEC> inproj_temp_stream("inproj_temp_stream");
    hls::stream<DTYPE_VEC> outproj_temp_stream("outproj_temp_stream");

    // =========================================================================
    // 5) Stream depths
    // =========================================================================
#pragma HLS STREAM variable=X_in_pre             depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=H0_in_pre            depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=conv_state_pre       depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=kernel_local         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_local              depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_for_norm           depth=SSMU_STREAM_DEPTH
// X_residual must be >= D_T=320: tee writes X_for_norm and X_residual in SAME loop.
// add_residual consumes X_residual only at END of chain (after out_proj) -> deadlock if too small.
#pragma HLS STREAM variable=X_residual           depth=320
#pragma HLS STREAM variable=X_normed             depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_normed_lr          depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_normed_nonlr       depth=SSMU_STREAM_DEPTH

#if SSMU_ENABLE_TRACE_STREAMS
#pragma HLS STREAM variable=trace_rms            depth=SSMU_TRACE_DEPTH
#pragma HLS STREAM variable=trace_delta          depth=SSMU_TRACE_DEPTH
#pragma HLS STREAM variable=trace_htC            depth=SSMU_TRACE_DEPTH
#pragma HLS STREAM variable=trace_core           depth=SSMU_TRACE_DEPTH
#endif

#pragma HLS STREAM variable=Z_stream             depth=640  // >= C2_T (written before XBC by stage2)
#pragma HLS STREAM variable=XBC_stream           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=DT_stream            depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_mid_stream         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=nlr_stream           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=B_raw_stream         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_raw_stream         depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=DT_C2_stream         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=G_stream             depth=640  // >= C2_T (produced before htC ready)
#pragma HLS STREAM variable=X_ssm_stream         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=conv_state_local_in  depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=conv_state_local_out depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=Wd_tiles             depth=SSMU_DEPTH_TILE
#pragma HLS STREAM variable=start_wd             depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=X_ssm_proj_stream    depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=X_ssm_scan_stream    depth=SSMU_STREAM_DEPTH
// X_ssm_out_stream must be >= C2_T=640: dup writes proj/scan/out in same iteration.
// stage6 consumes out only after htC ready (end of stage45) -> deadlock if too small.
#pragma HLS STREAM variable=X_ssm_out_stream     depth=640

#pragma HLS STREAM variable=B_stream_S           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_stream_S           depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=delta_selected       depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=delta_for_dA         depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=delta_for_scan       depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=dA_stream            depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=htC_stream           depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_trace_stream       depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=H1_trace_stream      depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=H1_state_stream      depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=ssm_core_out_stream  depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=ssm_normed_stream    depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=out_proj_stream_s    depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=out_local            depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=B_conv_stream        depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=C_conv_stream        depth=SSMU_STREAM_DEPTH

#pragma HLS STREAM variable=inproj_temp_stream   depth=SSMU_STREAM_DEPTH
#pragma HLS STREAM variable=outproj_temp_stream  depth=SSMU_STREAM_DEPTH

    // =========================================================================
    // 6) Dataflow graph
    // =========================================================================
#pragma HLS DATAFLOW

    // predecessor copies
    copy_vec_n(X_in,          X_in_pre,       D_T);
    copy_vec_n(H0_in,         H0_in_pre,      SSMU_H1_OUT_LEN);
    copy_vec_n(conv_state_in, conv_state_pre, CONV_K-1);

    // main flow
    copy_kernel_k(kernel_in, kernel_local);
    copy_vec_n(X_in_pre, X_local, D_T);
    copy_vec_n(conv_state_pre, conv_state_local_in, CONV_K-1);

    // residual split
    tee_vecDT_stream2_local(X_local, X_for_norm, X_residual);

    // RMSNorm
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> X_normed_raw("X_normed_raw");
#pragma HLS STREAM variable=X_normed_raw depth=SSMU_STREAM_DEPTH
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_local, X_normed_raw);
    tee_vec_n_local(X_normed_raw, X_normed, trace_rms, D_T);
#else
    rmsnorm_vecDT_stream_local(X_for_norm, RMS_local, X_normed);
#endif

    // input projection split:
    // [5120] Z(full-rank) + [5120] X(low-rank) + [80+128+128] DT/B/C(full-rank)
    tee_vecDT_stream2_local(X_normed, X_normed_lr, X_normed_nonlr);

    in_proj_lr_stage1(X_normed_lr, W_in_1, inproj_temp_stream, wscale_in_fx);
    in_proj_lr_stage2(inproj_temp_stream, W_in_2, X_mid_stream, wscale_in_fx);

    in_proj_nonlr_stage(X_normed_nonlr, W_in_nonlr, nlr_stream, wscale_in_fx);
    demux_nonlr_local(nlr_stream, Z_stream, DT_stream, B_raw_stream, C_raw_stream);
    assemble_xbc_local(B_raw_stream, C_raw_stream, X_mid_stream, XBC_stream);

    // conv + state
    conv1d_silu_stream_local_with_state(
        XBC_stream, Z_stream, kernel_local,
        conv_state_local_in, conv_state_local_out,
        G_stream, X_ssm_stream,
        B_conv_stream, C_conv_stream
    );
    copy_vec_n(conv_state_local_out, conv_state_out, CONV_K-1);

    // route conv'd B/C into SSM scan
    copy_vec_n(B_conv_stream, B_stream_S, SSMU_STATE_T);
    copy_vec_n(C_conv_stream, C_stream_S, SSMU_STATE_T);

    // duplicate X_ssm
    dup_vecC2_stream3_local(X_ssm_stream, X_ssm_proj_stream, X_ssm_scan_stream, X_ssm_out_stream);

    // delta selection
#if (SSMU_ENABLE_DT && SSMU_DELTA_FROM_DT)
    dtadapt_stream_local(DT_stream, DT_C2_stream);
    dt_to_delta_stream_local(DT_C2_stream, delta_selected);
    drain_vec_n(X_ssm_proj_stream, C2_T);
#else
    drain_vec_n(DT_stream, CH_T);
    write_token1_local(start_wd);
    stream_Wdelta_tiles_gated_local(W_delta, start_wd, Wd_tiles);
    projection_delta_only_local(X_ssm_proj_stream, Wd_tiles, delta_selected, wscale_delta_fx);
#endif

    // delta tee
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> delta_for_scan_raw("delta_for_scan_raw");
#pragma HLS STREAM variable=delta_for_scan_raw depth=SSMU_STREAM_DEPTH
    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan_raw);
    tee_vec_n_local(delta_for_scan_raw, delta_for_scan, trace_delta, C2_T);
#else
    dup_vecC2_stream2_local(delta_selected, delta_for_dA, delta_for_scan);
#endif

    // stage3: dA
    stage3_dA_stream_local(delta_for_dA, A_local, dA_stream);

    // stage45: SSM state update + reduction
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> htC_raw("htC_raw");
#pragma HLS STREAM variable=htC_raw depth=SSMU_STREAM_DEPTH
    stage45_update_reduce_local(
        X_ssm_scan_stream, delta_for_scan, dA_stream, B_stream_S, C_stream_S, H0_in_pre,
        htC_raw, C_trace_stream, H1_trace_stream, H1_state_stream
    );
    tee_vec_n_local(htC_raw, htC_stream, trace_htC, C2_T);
#else
    stage45_update_reduce_local(
        X_ssm_scan_stream, delta_for_scan, dA_stream, B_stream_S, C_stream_S, H0_in_pre,
        htC_stream, C_trace_stream, H1_trace_stream, H1_state_stream
    );
#endif

    // optional DDR trace
#if SSMU_ENABLE_TRACE_DDR
    ddr_writer_local(C_trace_stream, H1_trace_stream, C_ddr, H1_ddr);
#endif

    // H1 stream out
#if SSMU_ENABLE_H1_STREAM_OUT
    copy_vec_n(H1_state_stream, H1_out, SSMU_H1_OUT_LEN);
#else
    drain_vec_n(H1_state_stream, SSMU_H1_OUT_LEN);
#endif

    // stage6: output = gate * (htC + D*X)
#if SSMU_ENABLE_TRACE_STREAMS
    hls::stream<DTYPE_VEC> core_raw("core_raw");
#pragma HLS STREAM variable=core_raw depth=SSMU_STREAM_DEPTH
    stage6_out_yz_vec_local(htC_stream, D_local, X_ssm_out_stream, G_stream, core_raw);
    tee_vec_n_local(core_raw, ssm_core_out_stream, trace_core, C2_T);
#else
    stage6_out_yz_vec_local(htC_stream, D_local, X_ssm_out_stream, G_stream, ssm_core_out_stream);
#endif

    // second RMSNorm between stage6 and out_proj
    rmsnorm_vecC2T_stream_local(ssm_core_out_stream, RMS2_local, ssm_normed_stream);

    // LOW-RANK out-proj: two DATAFLOW stages
    out_proj_lr_stage1(ssm_normed_stream, W_out_B, outproj_temp_stream, wscale_out_fx);
    out_proj_lr_stage2(outproj_temp_stream, W_out_A, out_proj_stream_s, wscale_out_fx);

    add_residual_local_D(out_proj_stream_s, X_residual, out_local);
    copy_vec_n(out_local, out, D_T);
}

// ============================================================
// SSMU_STACK64: extern "C" wrapper (low-rank interface)
// ============================================================
extern "C" void SSMU_STACK64(
    hls::stream<DTYPE>&      kernel_in,

    const DTYPE_VEC          A_fixed[STATE_V],
    const DTYPE_VEC          RMS_weight[D_T],
    const DTYPE_VEC          RMS_weight_2[C2_T],

    const W_VEC              W_in_1[D_T][RANK_T],
    const W_VEC              W_in_2[RANK_T][INP_X_T],
    const W_VEC              W_in_nonlr[D_T][INP_NONLR_T],

    const W_VEC              W_delta[C2_T][C2_T],

    const W_VEC              W_out_A[D_T][RANK_T],
    const W_VEC              W_out_B[RANK_T][C2_T],

    const DTYPE_VEC          D_diag[C2_T],

    hls::stream<DTYPE_VEC>&  X_in,
    hls::stream<DTYPE_VEC>&  H0_in,

    hls::stream<DTYPE_VEC>&  conv_state_in,
    hls::stream<DTYPE_VEC>&  conv_state_out,

    DTYPE_VEC*               C_ddr,
    DTYPE_VEC*               H1_ddr,

    hls::stream<DTYPE_VEC>&  H1_out,
    hls::stream<DTYPE_VEC>&  out,

    float                    w_scale_in,
    float                    w_scale_delta,
    float                    w_scale_out
) {
#pragma HLS INLINE off

    SSMU(kernel_in,
         A_fixed, RMS_weight, RMS_weight_2,
         W_in_1, W_in_2, W_in_nonlr,
         W_delta,
         W_out_A, W_out_B,
         D_diag,
         X_in, H0_in,
         conv_state_in, conv_state_out,
         C_ddr, H1_ddr,
         H1_out, out,
         w_scale_in, w_scale_delta, w_scale_out);
}

#endif // __TOP_SSMU_HPP__
