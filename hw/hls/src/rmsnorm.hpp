// =============================================================================
// rmsnorm.hpp — RMSNorm for D_T and C2_T token streams
// =============================================================================
#ifndef __RMSNORM_HPP__
#define __RMSNORM_HPP__

#include "../config/macro.hpp"

// =============================================================
// RMSNorm for D_T tokens (input normalization)
// R4: float inv replaced with ap_fixed<24,12>
// =============================================================
static void rmsnorm_vecDT_stream_local(
    hls::stream<DTYPE_VEC>& x_in,
    const DTYPE_VEC RMS_weight[D_T],
    hls::stream<DTYPE_VEC>& y_out
) {
#pragma HLS INLINE off
    const float eps = 1e-5f;

    DTYPE_VEC xbuf[D_T];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_s2p impl=lutram

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        xbuf[j] = x_in.read();
    }

    ACC_T lane_sumsq[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=lane_sumsq complete dim=1
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        lane_sumsq[l] = 0;
    }

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = xbuf[j];
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            DTYPE vraw = vget(xv, (unsigned)l);
            typedef ap_fixed<16,8> RMS_NRW_T;
            RMS_NRW_T vsq_n;
#pragma HLS BIND_OP variable=vsq_n op=mul impl=fabric
            vsq_n = (RMS_NRW_T)vraw * (RMS_NRW_T)vraw;
            lane_sumsq[l] += (ACC_T)vsq_n;
        }
    }

    ACC_T sumsq = 0;
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        sumsq += lane_sumsq[l];
    }

    typedef ap_ufixed<40,12> RMS_MS_T_D;
    static const ap_ufixed<32,4> k_inv_dt_n = (ap_ufixed<32,4>)(1.0 / (double)(D_T * VEC_FACTOR));
    RMS_MS_T_D sumsq_u = (sumsq > (ACC_T)0) ? (RMS_MS_T_D)sumsq : (RMS_MS_T_D)0;
    RMS_MS_T_D ms      = sumsq_u * (RMS_MS_T_D)k_inv_dt_n;
    RMS_MS_T_D ms_eps  = ms + (RMS_MS_T_D)1e-5f;
    RMS_MS_T_D sq      = hls::sqrt(ms_eps);
    RMS_INV_T sq_safe_d = (sq > (RMS_MS_T_D)0) ? (RMS_INV_T)sq : (RMS_INV_T)1.0;
    RMS_INV_T inv = (RMS_INV_T)1.0 / sq_safe_d;
    (void)eps;

    for (int j = 0; j < D_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC w = RMS_weight[j];
        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            typedef ap_fixed<16,8> RMS_NRW_T;
            RMS_NRW_T xvn  = (RMS_NRW_T)vget(xbuf[j], l);
            RMS_NRW_T invn = (RMS_NRW_T)inv;
            RMS_NRW_T wwn  = (RMS_NRW_T)vget(w, l);
            RMS_NRW_T rms_xi, yv;
#pragma HLS BIND_OP variable=rms_xi op=mul impl=fabric
#pragma HLS BIND_OP variable=yv     op=mul impl=fabric
            rms_xi = xvn * invn;
            yv     = rms_xi * wwn;
            vset(o, l, (DTYPE)yv);
        }
        y_out.write(o);
    }
}

// =============================================================
// RMSNorm for C2_T tokens (between stage6 and out_proj)
// =============================================================
static void rmsnorm_vecC2T_stream_local(
    hls::stream<DTYPE_VEC>& x_in,
    const DTYPE_VEC RMS_weight_2[C2_T],
    hls::stream<DTYPE_VEC>& y_out
) {
#pragma HLS INLINE off
    const float eps = 1e-5f;

    DTYPE_VEC xbuf[C2_T];
#pragma HLS BIND_STORAGE variable=xbuf type=ram_s2p impl=lutram

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        xbuf[j] = x_in.read();
    }

#ifndef __SYNTHESIS__
    FILE* f_core_dbg = std::fopen("dut_ssm_core_f32.bin", "wb");
    if (f_core_dbg) {
        for (int j = 0; j < C2_T; ++j) dump_vec_token(f_core_dbg, xbuf[j]);
        std::fclose(f_core_dbg);
    }
    FILE* f_norm_dbg = std::fopen("dut_ssm_normed_f32.bin", "wb");
#endif

    typedef ap_fixed<16,8,AP_RND_CONV,AP_SAT> RMS_NRW_T;
    ACC_T lane_sumsq[VEC_FACTOR];
#pragma HLS ARRAY_PARTITION variable=lane_sumsq complete dim=1
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        lane_sumsq[l] = 0;
    }

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC xv = xbuf[j];
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            DTYPE vraw = vget(xv, (unsigned)l);
            RMS_NRW_T x_nrw = (RMS_NRW_T)vraw;
            RMS_NRW_T vsq_n;
#pragma HLS BIND_OP variable=vsq_n op=mul impl=fabric
            vsq_n = (RMS_NRW_T)(x_nrw * x_nrw);
            lane_sumsq[l] += (ACC_T)vsq_n;
        }
    }

    ACC_T sumsq = 0;
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
        sumsq += lane_sumsq[l];
    }

    typedef ap_ufixed<40,12> RMS_MS_T_C;
    static const ap_ufixed<32,4> k_inv_c2_n = (ap_ufixed<32,4>)(1.0 / (double)(C2_T * VEC_FACTOR));
    RMS_MS_T_C sumsq_u = (sumsq > (ACC_T)0) ? (RMS_MS_T_C)sumsq : (RMS_MS_T_C)0;
    RMS_MS_T_C ms      = sumsq_u * (RMS_MS_T_C)k_inv_c2_n;
    RMS_MS_T_C ms_eps  = ms + (RMS_MS_T_C)1e-5f;
    RMS_MS_T_C sq      = hls::sqrt(ms_eps);
    RMS_INV_T sq_safe_c = (sq > (RMS_MS_T_C)0) ? (RMS_INV_T)sq : (RMS_INV_T)1.0;
    RMS_INV_T inv = (RMS_INV_T)1.0 / sq_safe_c;
    (void)eps;

    for (int j = 0; j < C2_T; ++j) {
#pragma HLS PIPELINE II=1
        DTYPE_VEC w = RMS_weight_2[j];
        DTYPE_VEC o;
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
#pragma HLS UNROLL
            RMS_NRW_T xvn  = (RMS_NRW_T)vget(xbuf[j], l);
            RMS_NRW_T invn = (RMS_NRW_T)inv;
            RMS_NRW_T wwn  = (RMS_NRW_T)vget(w, l);
            RMS_NRW_T rms_xi, yv;
#pragma HLS BIND_OP variable=rms_xi op=mul impl=fabric
#pragma HLS BIND_OP variable=yv     op=mul impl=fabric
            rms_xi = xvn * invn;
            yv     = rms_xi * wwn;
            vset(o, l, (DTYPE)yv);
        }
#ifndef __SYNTHESIS__
        if (dbg_tok_sel(j)) {
            DUT_PRINTF("[DBG][rms2] tok=%d lane=%d in=% .6f w=% .6f out=% .6f\n",
                       j, DBG_LANE,
                       (float)vget(xbuf[j], DBG_LANE),
                       (float)vget(w, DBG_LANE),
                       (float)vget(o, DBG_LANE));
        }
        if (f_norm_dbg) dump_vec_token(f_norm_dbg, o);
#endif
        y_out.write(o);
    }
#ifndef __SYNTHESIS__
    if (f_norm_dbg) std::fclose(f_norm_dbg);
#endif
}

#endif // __RMSNORM_HPP__
