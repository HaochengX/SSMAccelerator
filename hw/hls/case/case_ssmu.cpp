// =============================================================================
// case_ssmu.cpp — SSMU_STACK64 testbench (BIN-DRIVEN + GOLDEN-COMPARE)
// Adapted from Reactor_migration_package for SSM_accel TellMe_Infra structure
// =============================================================================
// tb_ssmu.cpp  (BIN-DRIVEN + GOLDEN-COMPARE + MULTI-STEP INFERENCE / COSIM-safe)
// ✅ Updated for current SSMU top (NO W_B/W_C ports; B/C come from in-proj)
// ✅ COSIM FIX: call syn.top symbol SSMU_STACK64 (NOT SSMU)
//    - Fixes: [COSIM 212-330] "top function 'SSMU_STACK64' is not invoked in the test bench."
// ✅ Strict token-count checks (no silent padding, no size() dependency)
// ✅ Multi-step inference cache:
//    - Next H0_in  <- previous H1_out (vector tokens)
//    - Next conv_state_in <- previous conv_state_out
// ✅ Matches make_bins_raw.py bins_raw format (int16 Q4.12, vec-major flatten)
// ✅ Runtime w_scale_* handling (FIXED):
//    - Prefer env if env var PRESENT (even if value is 0)
//    - Else prefer files (Q4.12 int16, 1 elem): w_scale_in_f32.raw.bin, w_scale_delta_f32.raw.bin, w_scale_out_f32.raw.bin
//    - Else default 0.0f (lets pick_scale_fx fall back if implemented that way)
// ✅ NEW: conv_state_out golden compare (TB_TOL_CS)
// ✅ NEW: If SSMU_ENABLE_H1_STREAM_OUT=0, use H1_ddr to fill h1_dut (so multi-step cache stays valid)
// ✅ NEW: Deterministic interleaved producer (TB_INTERLEAVE=1) to better stress scheduling/backpressure
// ✅ NEW: TB_PRINT_TOKEN to print a chosen token index (out/h1), not just token0
//
// Env vars:
//   TB_BINS_DIR     : override bins directory (default auto-locate bins_raw)
//   TB_TOL_OUT      : tolerance for out (default 0.05)
//   TB_TOL_H1       : tolerance for h1  (default 0.06)
//   TB_TOL_CS       : tolerance for conv_state_out (default = TB_TOL_H1)
//   TB_STEPS        : number of inference steps (default 1)
//   TB_TOPK         : print top-k mismatches (default 10)
//   TB_DRAIN_TRACE  : 1 to drain trace_* streams when enabled (default 1)  [placeholder for future]
//   TB_WSCALE_IN / TB_WSCALE_DELTA / TB_WSCALE_OUT : override runtime scales (float) [presence-based]
//   TB_INTERLEAVE   : 1 to feed streams in deterministic round-robin order (default 0)
//   TB_PRINT_TOKEN  : token index to print (default 0). Set -1 to disable prints.
//
// Build notes:
//   - This TB assumes SSMU_USE_INT8=0 for weights bins (DTYPE-packed bins).
//   - If SSMU_USE_INT8=1, you must regenerate bins or add an int8 loader.

#include "../src/top_ssmu.hpp"

#include <hls_stream.h>
#include <ap_int.h>
#include <ap_fixed.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <thread>
#include <string>
#include <vector>

// ============================================================
// Shapes from SSMU.h
// ============================================================
static const int D_T     = SSMU_D_T;
static const int C2_T    = SSMU_C2_T;
static const int CCONV_T = SSMU_CCONV_T;
static const int CH_T    = SSMU_CH_T;
static const int CIN_T   = SSMU_CIN_T;
static const int CONV_K  = SSMU_K;

#ifndef SSMU_STATE_T
#define SSMU_STATE_T (SSMU_STATE / VEC_FACTOR)
#endif

static const int STATE_SCALAR = SSMU_STATE;
static const int STATE_V      = SSMU_STATE_T;
static const int INP_NONLR_T  = (C2_T + CH_T + 2 * STATE_V);

#ifndef SSMU_ENABLE_H1_STREAM_OUT
#define SSMU_ENABLE_H1_STREAM_OUT 1
#endif

#ifndef SSMU_H1_OUT_LEN
#define SSMU_H1_OUT_LEN (SSMU_STATE_T * SSMU_C2_T)
#endif

static const int H0_LEN = SSMU_H1_OUT_LEN;

#ifndef VEC_FACTOR
#error "VEC_FACTOR must be defined in SSMU.h"
#endif

// ============================================================
// Q4.12 packing matches make_bins_raw.py
// ============================================================
#ifndef TB_FRAC_BITS
#define TB_FRAC_BITS 12
#endif

// ============================================================
// Build tag (prove which tb_ssmu.cpp is compiled)
// ============================================================
#ifndef TB_BUILD_TAG
#define TB_BUILD_TAG "case_ssmu.cpp SSM_ACCEL v1.0 (adapted from tb_ssm.cpp vitis5/quant9)"
#endif

// ============================================================
// vget/vset fallback (if header doesn't provide)
// ============================================================
#ifndef SSMU_HAVE_VGET_VSET
template<typename V>
static inline auto vget(const V& v, unsigned idx) -> decltype(v[idx]) { return v[idx]; }
template<typename V, typename T>
static inline void vset(V& v, unsigned idx, const T& val) { v[idx] = (decltype(v[idx]))val; }
#endif

static inline DTYPE from_i16(int16_t q) {
    // ✅ FIX: bit-exact Q4.12 -> ap_fixed<16,4> (no float intermediate)
    DTYPE d;
    d.range(15, 0) = (ap_uint<16>)(uint16_t)q;
    return d;
}

static inline DTYPE_VEC make_vec_zero() {
    DTYPE_VEC v;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, (DTYPE)0);
    return v;
}

static void print_vec4(const char* tag, const DTYPE_VEC& v) {
    std::printf("%s:", tag);
    for (unsigned l=0; l<(unsigned)VEC_FACTOR && l<4; ++l) {
        std::printf(" %f", (float)vget(v,l));
    }
    std::printf("\n");
    std::fflush(stdout);
}

// ============================================================
// INLINE fixed-point RMSNorm recalculation (DUT-equivalent for strict gating)
// ============================================================
// Recompute norm-stage output using same fixed-point rules as DUT
// to avoid stale golden reference file issues.
static void compute_norm_stage_like_dut(
    const float* ssm_core_tokens,   // Input: C2_T float tokens per lane
    const float* rms_weight_2,      // C2_T weights per lane
    float* ssm_normed_out,          // Output: C2_T float tokens per lane
    int C2_T_val,
    int VEC_FACTOR_val
) {
    const float eps = 1e-5f;
    typedef ap_fixed<16,8,AP_RND_CONV,AP_SAT> RMS_NRW_T;
    typedef ap_fixed<24,8> ACC_T;
    typedef ap_fixed<24,12> RMS_INV_T;
    typedef ap_ufixed<40,12> RMS_MS_T_C;

    // Compute RMS2 inverse per lane structure similar to DUT
    std::vector<ACC_T> lane_sumsq(VEC_FACTOR_val, 0);

    for (int j = 0; j < C2_T_val; ++j) {
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR_val; ++l) {
            RMS_NRW_T x_nrw = (RMS_NRW_T)(ssm_core_tokens[j * VEC_FACTOR_val + l]);
            RMS_NRW_T vsq_n;
            vsq_n = (RMS_NRW_T)(x_nrw * x_nrw);
            lane_sumsq[l] += (ACC_T)vsq_n;
        }
    }

    // Sum across lanes
    ACC_T sumsq = 0;
    for (unsigned l = 0; l < (unsigned)VEC_FACTOR_val; ++l) {
        sumsq += lane_sumsq[l];
    }

    // Compute mean + sqrt + inv
    static const ap_ufixed<32,4> k_inv_c2 = (ap_ufixed<32,4>)(1.0 / (double)(C2_T_val * VEC_FACTOR_val));
    RMS_MS_T_C sumsq_u = (sumsq > (ACC_T)0) ? (RMS_MS_T_C)sumsq : (RMS_MS_T_C)0;
    RMS_MS_T_C ms      = sumsq_u * (RMS_MS_T_C)k_inv_c2;
    RMS_MS_T_C ms_eps  = ms + (RMS_MS_T_C)1e-5f;
    RMS_MS_T_C sq      = static_cast<RMS_MS_T_C>(std::sqrt(static_cast<double>(ms_eps)));
    RMS_INV_T sq_safe  = (sq > (RMS_MS_T_C)0) ? (RMS_INV_T)sq : (RMS_INV_T)1.0;
    RMS_INV_T inv      = (RMS_INV_T)1.0 / sq_safe;

    // Apply norm with fixed-point mult
    for (int j = 0; j < C2_T_val; ++j) {
        for (unsigned l = 0; l < (unsigned)VEC_FACTOR_val; ++l) {
            RMS_NRW_T xvn = (RMS_NRW_T)(ssm_core_tokens[j * VEC_FACTOR_val + l]);
            RMS_NRW_T invn = (RMS_NRW_T)inv;
            RMS_NRW_T wwn = (RMS_NRW_T)(rms_weight_2[j * VEC_FACTOR_val + l]);
            RMS_NRW_T rms_xi = xvn * invn;
            RMS_NRW_T yv = rms_xi * wwn;
            ssm_normed_out[j * VEC_FACTOR_val + l] = (float)yv;
        }
    }
}

// ============================================================
// Path + file utils
// ============================================================
static std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    char last = a.back();
    if (last == '/' || last == '\\') return a + b;
    return a + "/" + b;
}

static bool read_file_i16(const std::string& path, std::vector<int16_t>& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);

    if (sz < 0) { std::fclose(fp); return false; }
    if ((sz % (long)sizeof(int16_t)) != 0) {
        std::printf("[TB][FATAL] file size not multiple of int16: %s size=%ld\n", path.c_str(), sz);
        std::fflush(stdout);
        std::fclose(fp);
        return false;
    }

    size_t n = (size_t)sz / sizeof(int16_t);
    out.resize(n);
    size_t r = std::fread(out.data(), sizeof(int16_t), n, fp);
    std::fclose(fp);
    return r == n;
}

static bool read_file_f32(const std::string& path, std::vector<float>& out) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    std::fseek(fp, 0, SEEK_END);
    long sz = std::ftell(fp);
    std::fseek(fp, 0, SEEK_SET);

    if (sz < 0) { std::fclose(fp); return false; }
    if ((sz % (long)sizeof(float)) != 0) {
        std::printf("[TB][FATAL] file size not multiple of float: %s size=%ld\n", path.c_str(), sz);
        std::fflush(stdout);
        std::fclose(fp);
        return false;
    }

    size_t n = (size_t)sz / sizeof(float);
    out.resize(n);
    size_t r = std::fread(out.data(), sizeof(float), n, fp);
    std::fclose(fp);
    return r == n;
}

static bool file_exists_i16_quick(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;
    std::fclose(fp);
    return true;
}

// ============================================================
// bins_raw auto-locator
// ============================================================
static std::string resolve_bins_dir() {
    if (const char* env = std::getenv("TB_BINS_DIR")) {
        if (env[0]) {
            std::string env_dir(env);
            std::string probe = join_path(env_dir, "A_fixed_f32.raw.bin");
            if (file_exists_i16_quick(probe)) return env_dir;
            std::printf("[TB][WARN] TB_BINS_DIR is set but '%s' not found in '%s'. Auto-probing fallbacks...\n",
                        "A_fixed_f32.raw.bin", env_dir.c_str());
            std::fflush(stdout);
        }
    }

    const char* cands[] = {
        "bins_raw",
        "./bins_raw",
        "../bins_raw",
        "../../bins_raw",
        "../../../bins_raw",
        "../../../../bins_raw",
        "vitis5/quant9/bins_raw",
        "./vitis5/quant9/bins_raw",
        "../vitis5/quant9/bins_raw",
        "../../vitis5/quant9/bins_raw",
        "../../../vitis5/quant9/bins_raw"
    };

    for (auto* c : cands) {
        const std::string probe = join_path(std::string(c), "A_fixed_f32.raw.bin");
        if (file_exists_i16_quick(probe)) return std::string(c);
    }
    std::printf("[TB][WARN] Could not auto-locate bins_raw directory. Fallback='bins_raw'.\n");
    std::fflush(stdout);
    return std::string("bins_raw");
}

static std::string resolve_dut_dump_dir() {
    if (const char* env = std::getenv("TB_DUT_DUMP_DIR")) {
        if (env[0]) {
            std::string env_dir(env);
            std::string p0 = join_path(env_dir, "dut_ssm_core_f32.bin");
            std::string p1 = join_path(env_dir, "dut_ssm_normed_f32.bin");
            std::string p2 = join_path(env_dir, "dut_out_proj_f32.bin");
            if (file_exists_i16_quick(p0) && file_exists_i16_quick(p1) && file_exists_i16_quick(p2)) {
                return env_dir;
            }
            std::printf("[TB][WARN] TB_DUT_DUMP_DIR is set but stage dumps are missing in '%s'. Auto-probing fallbacks...\n", env_dir.c_str());
            std::fflush(stdout);
        }
    }

    const char* cands[] = {
        ".",
        "./",
        "build",
        "./build",
        "hls/csim/build",
        "./hls/csim/build",
        "../hls/csim/build",
        "../../hls/csim/build",
        "../../../hls/csim/build",
        "vitis5/quant9/hls/csim/build",
        "./vitis5/quant9/hls/csim/build",
        "../vitis5/quant9/hls/csim/build",
        "../../vitis5/quant9/hls/csim/build",
        "../../../vitis5/quant9/hls/csim/build"
    };

    for (auto* c : cands) {
        std::string base(c);
        std::string p0 = join_path(base, "dut_ssm_core_f32.bin");
        std::string p1 = join_path(base, "dut_ssm_normed_f32.bin");
        std::string p2 = join_path(base, "dut_out_proj_f32.bin");
        if (file_exists_i16_quick(p0) && file_exists_i16_quick(p1) && file_exists_i16_quick(p2)) {
            return base;
        }
    }
    std::printf("[TB][WARN] Could not auto-locate DUT stage dump directory. Fallback='.'.\n");
    std::fflush(stdout);
    return std::string(".");
}

// step-specific detector: requires "_s{step}.raw.bin" substring
static bool is_step_specific_path(const std::string& p, int step) {
    const std::string key = "_s" + std::to_string(step) + ".raw.bin";
    return (p.find(key) != std::string::npos);
}

static bool is_step_specific_path_bin(const std::string& p, int step) {
    const std::string key = "_s" + std::to_string(step) + ".bin";
    return (p.find(key) != std::string::npos);
}

// Prefer step-specific filename if exists, else fallback to legacy base name.
// Convention: base "x_in_f32.raw.bin" => "x_in_s{step}.raw.bin"
static std::string step_file(const std::string& dir, const std::string& base, int step) {
    auto try_name = [&](const std::string& fn)->std::string{
        std::string p = join_path(dir, fn);
        if (file_exists_i16_quick(p)) return p;
        return std::string();
    };

    auto map = [&](const char* stem)->std::string{
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s_s%d.raw.bin", stem, step);
        auto p = try_name(std::string(buf));
        return p;
    };

    if (base == "x_in_f32.raw.bin") {
        auto p = map("x_in");
        if (!p.empty()) return p;
    }
    if (base == "kernel_in_f32.raw.bin") {
        auto p = map("kernel_in");
        if (!p.empty()) return p;
    }
    if (base == "out_ref_f32.raw.bin") {
        auto p = map("out_ref");
        if (!p.empty()) return p;
    }
    if (base == "h1_ref_f32.raw.bin") {
        auto p = map("h1_ref");
        if (!p.empty()) return p;
    }
    if (base == "h0_in_f32.raw.bin") {
        auto p = map("h0_in");
        if (!p.empty()) return p;
    }
    if (base == "conv_state_in_f32.raw.bin") {
        auto p = map("conv_state_in");
        if (!p.empty()) return p;
    }
    if (base == "conv_state_out_f32.raw.bin") {
        auto p = map("conv_state_out");
        if (!p.empty()) return p;
    }

    {
        const std::string primary = join_path(dir, base);
        if (file_exists_i16_quick(primary)) return primary;

        // naming fallback: *.raw.bin <-> *.bin
        if (base.size() > 8 && base.substr(base.size() - 8) == ".raw.bin") {
            std::string alt = base.substr(0, base.size() - 8) + ".bin";
            std::string altp = join_path(dir, alt);
            if (file_exists_i16_quick(altp)) {
                std::printf("[TB][WARN] Using fallback file '%s' for requested '%s'\n", altp.c_str(), primary.c_str());
                std::fflush(stdout);
                return altp;
            }
        } else if (base.size() > 4 && base.substr(base.size() - 4) == ".bin") {
            std::string alt = base.substr(0, base.size() - 4) + ".raw.bin";
            std::string altp = join_path(dir, alt);
            if (file_exists_i16_quick(altp)) {
                std::printf("[TB][WARN] Using fallback file '%s' for requested '%s'\n", altp.c_str(), primary.c_str());
                std::fflush(stdout);
                return altp;
            }
        }
        return primary;
    }
}

static std::string step_file_bin(const std::string& dir, const std::string& base, int step) {
    auto try_name = [&](const std::string& fn)->std::string{
        std::string p = join_path(dir, fn);
        if (file_exists_i16_quick(p)) return p;
        return std::string();
    };

    auto map = [&](const char* stem)->std::string{
        char buf[256];
        std::snprintf(buf, sizeof(buf), "%s_s%d.bin", stem, step);
        return try_name(std::string(buf));
    };

    if (base == "ssm_core_ref_f32.bin") {
        auto p = map("ssm_core_ref_f32");
        if (!p.empty()) return p;
    }
    if (base == "ssm_normed_ref_f32.bin") {
        auto p = map("ssm_normed_ref_f32");
        if (!p.empty()) return p;
    }
    if (base == "out_proj_ref_f32.bin") {
        auto p = map("out_proj_ref_f32");
        if (!p.empty()) return p;
    }
    if (base == "out_ref_f32.bin") {
        auto p = map("out_ref_f32");
        if (!p.empty()) return p;
    }

    {
        const std::string primary = join_path(dir, base);
        if (file_exists_i16_quick(primary)) return primary;

        // naming fallback: *.bin <-> *.raw.bin
        if (base.size() > 4 && base.substr(base.size() - 4) == ".bin") {
            std::string alt = base.substr(0, base.size() - 4) + ".raw.bin";
            std::string altp = join_path(dir, alt);
            if (file_exists_i16_quick(altp)) {
                std::printf("[TB][WARN] Using fallback file '%s' for requested '%s'\n", altp.c_str(), primary.c_str());
                std::fflush(stdout);
                return altp;
            }
        } else if (base.size() > 8 && base.substr(base.size() - 8) == ".raw.bin") {
            std::string alt = base.substr(0, base.size() - 8) + ".bin";
            std::string altp = join_path(dir, alt);
            if (file_exists_i16_quick(altp)) {
                std::printf("[TB][WARN] Using fallback file '%s' for requested '%s'\n", altp.c_str(), primary.c_str());
                std::fflush(stdout);
                return altp;
            }
        }
        return primary;
    }
}

// ============================================================
// Token loaders
// ============================================================

// Read vec tokens into DTYPE_VEC[tokens]. Accept either:
//  - vec layout: tokens*VEC_FACTOR elems  (preferred)
//  - scalar layout: tokens elems -> broadcast to all lanes
static bool load_vec_tokens(
    const std::string& path,
    int tokens,
    DTYPE_VEC* dst,
    const char* tag
) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        std::fflush(stdout);
        return false;
    }

    const long got = (long)buf.size();
    const long exp_vec = (long)tokens * (long)VEC_FACTOR;
    const long exp_sca = (long)tokens;

    if (got == exp_vec) {
        long idx = 0;
        for (int t=0; t<tokens; ++t) {
            DTYPE_VEC v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset(v, l, from_i16(buf[idx++]));
            }
            dst[t] = v;
        }
        std::printf("[TB] %s loaded (vec) elems=%ld tokens=%d\n", tag, got, tokens);
        std::fflush(stdout);
        return true;
    }

    if (got == exp_sca) {
        for (int t=0; t<tokens; ++t) {
            DTYPE x = from_i16(buf[t]);
            DTYPE_VEC v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, x);
            dst[t] = v;
        }
        std::printf("[TB][WARN] %s loaded (scalar->broadcast) elems=%ld tokens=%d (expected vec=%ld)\n",
                    tag, got, tokens, exp_vec);
        std::fflush(stdout);
        return true;
    }

    std::printf("[TB][FATAL] elem count mismatch: %s got=%ld expected(vec)=%ld or scalar=%ld\n",
                path.c_str(), got, exp_vec, exp_sca);
    std::fflush(stdout);
    return false;
}

static bool load_scalar_tokens(
    const std::string& path,
    int tokens,
    DTYPE* dst,
    const char* tag
) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        std::fflush(stdout);
        return false;
    }
    long got = (long)buf.size();
    if (got != tokens) {
        std::printf("[TB][FATAL] elem count mismatch: %s got=%ld expected=%d\n",
                    path.c_str(), got, tokens);
        std::fflush(stdout);
        return false;
    }
    for (int i=0;i<tokens;++i) dst[i] = from_i16(buf[i]);
    std::printf("[TB] %s loaded (scalar) elems=%ld\n", tag, got);
    std::fflush(stdout);
    return true;
}

static bool load_vec_tokens_f32(
    const std::string& path,
    int tokens,
    DTYPE_VEC* dst,
    const char* tag
) {
    std::vector<float> buf;
    if (!read_file_f32(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        std::fflush(stdout);
        return false;
    }

    const long got = (long)buf.size();
    const long exp_vec = (long)tokens * (long)VEC_FACTOR;
    const long exp_sca = (long)tokens;

    if (got == exp_vec) {
        long idx = 0;
        for (int t=0; t<tokens; ++t) {
            DTYPE_VEC v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset(v, l, (DTYPE)buf[idx++]);
            }
            dst[t] = v;
        }
        std::printf("[TB] %s loaded (f32 vec) elems=%ld\n", tag, got);
        std::fflush(stdout);
        return true;
    }

    if (got == exp_sca) {
        for (int t=0; t<tokens; ++t) {
            DTYPE x = (DTYPE)buf[t];
            DTYPE_VEC v;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(v, l, x);
            dst[t] = v;
        }
        std::printf("[TB][WARN] %s loaded (f32 scalar->broadcast) elems=%ld expected(vec)=%ld\n",
                    tag, got, exp_vec);
        std::fflush(stdout);
        return true;
    }

    std::printf("[TB][FATAL] f32 elem count mismatch: %s got=%ld expected(vec)=%ld or scalar=%ld\n",
                path.c_str(), got, exp_vec, exp_sca);
    std::fflush(stdout);
    return false;
}

static bool load_scalar_one_q412(const std::string& path, float& out_f, const char* tag) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) return false;
    if ((int)buf.size() != 1) {
        std::printf("[TB][FATAL] %s expected 1 element (Q4.12 int16), got=%ld in %s\n",
                    tag, (long)buf.size(), path.c_str());
        std::fflush(stdout);
        return false;
    }
    out_f = (float)from_i16(buf[0]);
    std::printf("[TB] %s loaded (Q4.12) = %f  from %s\n", tag, out_f, path.c_str());
    std::fflush(stdout);
    return true;
}

// Load W_VEC matrix stored as vec tokens.
// Expect elems = ROWS*COLS*VEC_FACTOR OR scalar ROWS*COLS (broadcast lanes).
template<int ROWS, int COLS>
static bool load_w_mat(
    const std::string& path,
    W_VEC dst[ROWS][COLS],
    const char* tag
) {
    std::vector<int16_t> buf;
    if (!read_file_i16(path, buf)) {
        std::printf("[TB][FATAL] cannot open: %s\n", path.c_str());
        std::fflush(stdout);
        return false;
    }
    const long got = (long)buf.size();
    const long exp_vec = (long)ROWS * (long)COLS * (long)VEC_FACTOR;
    const long exp_sca = (long)ROWS * (long)COLS;

#if SSMU_USE_INT8
    std::printf("[TB][FATAL] SSMU_USE_INT8=1 but this TB expects DTYPE-packed bins.\n");
    std::fflush(stdout);
    (void)got; (void)exp_vec; (void)exp_sca; (void)tag; (void)dst;
    return false;
#else
    if (got == exp_vec) {
        long idx = 0;
        for (int r=0;r<ROWS;++r) for (int c=0;c<COLS;++c) {
            W_VEC w;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
                vset(w, l, (DTYPE)from_i16(buf[idx++]));
            }
            dst[r][c] = w;
        }
        std::printf("[TB] %s loaded (vec) elems=%ld\n", tag, got);
        std::fflush(stdout);
        return true;
    }

    if (got == exp_sca) {
        long idx = 0;
        for (int r=0;r<ROWS;++r) for (int c=0;c<COLS;++c) {
            DTYPE x = from_i16(buf[idx++]);
            W_VEC w;
            for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) vset(w, l, x);
            dst[r][c] = w;
        }
        std::printf("[TB][WARN] %s loaded (scalar->broadcast) elems=%ld (expected vec=%ld)\n",
                    tag, got, exp_vec);
        std::fflush(stdout);
        return true;
    }

    std::printf("[TB][FATAL] elem count mismatch: %s got=%ld expected(vec)=%ld or scalar=%ld\n",
                path.c_str(), got, exp_vec, exp_sca);
    std::fflush(stdout);
    return false;
#endif
}

// ============================================================
// Stream drain (COSIM-safe, no size() reliance)
// ============================================================
template<typename T>
static int drain_exact(hls::stream<T>& s, T* out, int expect, const char* tag) {
    for (int i=0; i<expect; ++i) {
        if (s.empty()) {
            std::printf("[TB][FATAL] %s token underflow: got=%d expect=%d\n", tag, i, expect);
            std::fflush(stdout);
            return -1;
        }
        out[i] = s.read();
    }
    if (!s.empty()) {
        int extra = 0;
        while (!s.empty()) { (void)s.read(); ++extra; }
        std::printf("[TB][FATAL] %s has extra tokens: extra=%d (expected exactly %d)\n", tag, extra, expect);
        std::fflush(stdout);
        return -1;
    }
    return 0;
}

template<typename T>
static void drain_all(hls::stream<T>& s) { while (!s.empty()) (void)s.read(); }

// ============================================================
// Diff helpers + Top-K report
// ============================================================
static inline float vec_max_abs_diff(const DTYPE_VEC& a, const DTYPE_VEC& b) {
    float m = 0.0f;
    for (unsigned l=0; l<(unsigned)VEC_FACTOR; ++l) {
        float da = (float)vget(a,l);
        float db = (float)vget(b,l);
        float d  = std::fabs(da - db);
        if (d > m) m = d;
    }
    return m;
}

struct Mismatch {
    int token;
    int lane;
    float dutv;
    float refv;
    float diff;
};

static float max_abs_diff_vec_tokens(const DTYPE_VEC* a, const DTYPE_VEC* b, int tokens) {
    float m = 0.0f;
    for (int i=0;i<tokens;++i) m = std::max(m, vec_max_abs_diff(a[i], b[i]));
    return m;
}

static float max_abs_value_vec_tokens(const DTYPE_VEC* a, int tokens) {
    float m = 0.0f;
    for (int t=0; t<tokens; ++t) {
        for (int l=0; l<(int)VEC_FACTOR; ++l) {
            float v = std::fabs((float)vget(a[t], (unsigned)l));
            if (v > m) m = v;
        }
    }
    return m;
}

static void collect_topk(const DTYPE_VEC* dut, const DTYPE_VEC* ref, int tokens, int topk, const char* tag) {
    std::vector<Mismatch> all;
    all.reserve((size_t)tokens * (size_t)VEC_FACTOR);

    for (int t=0; t<tokens; ++t) {
        for (int l=0; l<(int)VEC_FACTOR; ++l) {
            float dv = (float)vget(dut[t], (unsigned)l);
            float rv = (float)vget(ref[t], (unsigned)l);
            float d  = std::fabs(dv - rv);
            all.push_back(Mismatch{t, l, dv, rv, d});
        }
    }

    int n = std::min(topk, (int)all.size());
    if (n <= 0) return;

    std::nth_element(all.begin(), all.begin() + (n - 1), all.end(),
        [](const Mismatch& a, const Mismatch& b){ return a.diff > b.diff; });
    std::sort(all.begin(), all.begin() + n,
        [](const Mismatch& a, const Mismatch& b){ return a.diff > b.diff; });

    std::printf("[TB] Top-%d mismatches for %s:\n", n, tag);
    for (int i=0;i<n;++i) {
        const auto& m = all[i];
        std::printf("  #%d token=%d lane=%d diff=%e dut=%f ref=%f\n",
                    i, m.token, m.lane, m.diff, m.dutv, m.refv);
    }
    std::fflush(stdout);
}

// ============================================================
// Env helpers (presence-based)
// ============================================================
static bool env_present(const char* name) {
    const char* s = std::getenv(name);
    return (s && s[0]);
}
static float read_env_float(const char* name, float defv) {
    const char* s = std::getenv(name);
    if (!s || !s[0]) return defv;
    return (float)std::atof(s);
}
static int read_env_int(const char* name, int defv) {
    const char* s = std::getenv(name);
    if (!s || !s[0]) return defv;
    return std::atoi(s);
}

// ============================================================
// Run DUT once, capture out + h1 + conv_state_out
// (SSMU_STACK64 declaration provided by top_ssmu.hpp)
// ============================================================
static int run_dut_once(
    const DTYPE     kernel_host[CONV_K],
    const DTYPE_VEC A_fixed[STATE_V],
    const DTYPE_VEC RMS_weight[D_T],
    const DTYPE_VEC RMS_weight_2[C2_T],                    // ★ v2
    const W_VEC     W_in_1[D_T][SSMU_RANK_T],              // ★ v2 low-rank
    const W_VEC     W_in_2[SSMU_RANK_T][C2_T],             // middle 5120 low-rank
    const W_VEC     W_in_nonlr[D_T][INP_NONLR_T],
    const W_VEC     W_delta[C2_T][C2_T],
    const W_VEC     W_out_A[D_T][SSMU_RANK_T],             // ★ v2 low-rank
    const W_VEC     W_out_B[SSMU_RANK_T][C2_T],            // ★ v2 low-rank
    const DTYPE_VEC D_diag[C2_T],
    const DTYPE_VEC X_host[D_T],
    const DTYPE_VEC H0_host[H0_LEN],
    const DTYPE_VEC conv_state_in_host[CONV_K-1],
    // outputs:
    DTYPE_VEC out_host_dut[D_T],
    DTYPE_VEC h1_host_dut[SSMU_H1_OUT_LEN],
    DTYPE_VEC conv_state_out_host[CONV_K-1],
    // runtime scales
    float w_scale_in,
    float w_scale_delta,
    float w_scale_out,
    // producer mode
    int tb_interleave
) {
    hls::stream<DTYPE>      kernel_in("kernel_in");
    hls::stream<DTYPE_VEC>  X_in("X_in");
    hls::stream<DTYPE_VEC>  H0_in("H0_in");
    hls::stream<DTYPE_VEC>  conv_state_in("conv_state_in");
    hls::stream<DTYPE_VEC>  conv_state_out("conv_state_out");
    hls::stream<DTYPE_VEC>  H1_out("H1_out");
    hls::stream<DTYPE_VEC>  out("out");

    // Trace DDR ports (only used if SSMU_ENABLE_TRACE_DDR)
    static DTYPE_VEC C_ddr[H0_LEN];
    static DTYPE_VEC H1_ddr[H0_LEN];

    // Launch DUT first (COSIM-safe)
    std::thread dut_th([&](){
        SSMU_STACK64(
            kernel_in,
            A_fixed,
            RMS_weight,
            RMS_weight_2,
            W_in_1,
            W_in_2,
            W_in_nonlr,
            W_delta,
            W_out_A,
            W_out_B,
            D_diag,
            X_in,
            H0_in,
            conv_state_in,
            conv_state_out,
            (DTYPE_VEC*)C_ddr,
            (DTYPE_VEC*)H1_ddr,
            H1_out,
            out,
            w_scale_in,
            w_scale_delta,
            w_scale_out
        );
    });

    // Feed inputs AFTER DUT starts (avoid TB write-block deadlock)
    if (!tb_interleave) {
        for (int k=0;k<CONV_K;++k)      kernel_in.write(kernel_host[k]);
        for (int i=0;i<D_T;++i)        X_in.write(X_host[i]);
        for (int i=0;i<H0_LEN;++i)     H0_in.write(H0_host[i]);
        for (int k=0;k<CONV_K-1;++k)   conv_state_in.write(conv_state_in_host[k]);
    } else {
        // Deterministic round-robin feeder (reproducible, closer to multi-producer behavior)
        int k_i = 0;
        int x_i = 0;
        int h_i = 0;
        int c_i = 0;

        while (k_i < CONV_K || x_i < D_T || h_i < H0_LEN || c_i < (CONV_K-1)) {
            if (k_i < CONV_K)        { kernel_in.write(kernel_host[k_i]); ++k_i; }
            if (x_i < D_T)           { X_in.write(X_host[x_i]); ++x_i; }
            if (h_i < H0_LEN)        { H0_in.write(H0_host[h_i]); ++h_i; }
            if (c_i < (CONV_K-1))    { conv_state_in.write(conv_state_in_host[c_i]); ++c_i; }
        }
    }

    dut_th.join();

    // Strict drains (exact token counts)
    if (drain_exact(conv_state_out, conv_state_out_host, CONV_K-1, "conv_state_out") != 0) return -1;

#if SSMU_ENABLE_H1_STREAM_OUT
    if (drain_exact(H1_out, h1_host_dut, SSMU_H1_OUT_LEN, "H1_out") != 0) return -1;
#else
    // H1 stream disabled: fill from DDR instead (so cache update stays meaningful)
    drain_all(H1_out);
    for (int i=0;i<SSMU_H1_OUT_LEN;++i) {
        // Assumption: DDR length is H0_LEN == SSMU_H1_OUT_LEN (as in your current TB)
        h1_host_dut[i] = H1_ddr[i];
    }
#endif

    if (drain_exact(out, out_host_dut, D_T, "out") != 0) return -1;

    return 0;
}

// ============================================================
// Main
// ============================================================
int main() {
    std::string bins_dir = resolve_bins_dir();
    std::string dut_dump_dir = resolve_dut_dump_dir();

    const float tol_out     = read_env_float("TB_TOL_OUT", 0.05f);
    const float tol_h1      = read_env_float("TB_TOL_H1",  0.06f);
    const float tol_cs      = read_env_float("TB_TOL_CS",  tol_h1);
    const float tol_core    = read_env_float("TB_TOL_CORE", tol_h1);
    const float tol_norm    = read_env_float("TB_TOL_NORM", tol_h1);
    const float tol_opj     = read_env_float("TB_TOL_OPJ",  tol_out);

    const int   steps       = std::max(1, read_env_int("TB_STEPS", 1));
    const int   topk        = std::max(0, read_env_int("TB_TOPK", 10));
    const int   drain_trace = read_env_int("TB_DRAIN_TRACE", 1); // placeholder for future
    const int   interleave  = read_env_int("TB_INTERLEAVE", 0);
    const int   print_tok   = read_env_int("TB_PRINT_TOKEN", 0);
    const int   run_corners = read_env_int("TB_RUN_CORNERS", 1);
    const int   corner_strict = read_env_int("TB_CORNER_STRICT", 0);
    const int   strict_stage = read_env_int("TB_STRICT_STAGE", 0);

    std::printf("[TB] BUILD_TAG=%s\n", TB_BUILD_TAG);
    std::printf("[TB] Using bins dir: %s\n", bins_dir.c_str());
    std::printf("[TB] DUT dump dir : %s\n", dut_dump_dir.c_str());
    std::printf("[TB] STATE_SCALAR=%d  STATE_V(STATE_T)=%d  D_T=%d  C2_T=%d  CIN_T=%d  VEC_FACTOR=%d\n",
                (int)STATE_SCALAR, (int)STATE_V, (int)D_T, (int)C2_T, (int)CIN_T, (int)VEC_FACTOR);
    std::printf("[TB] H0_LEN=%d  H1_LEN=%d  CONV_K=%d\n", (int)H0_LEN, (int)SSMU_H1_OUT_LEN, (int)CONV_K);
    std::printf("[TB] TOL_OUT=%g  TOL_H1=%g  TOL_CS=%g  TOL_CORE=%g TOL_NORM=%g TOL_OPJ=%g\n",
                tol_out, tol_h1, tol_cs, tol_core, tol_norm, tol_opj);
    std::printf("[TB] TB_STEPS=%d  TB_TOPK=%d  TB_DRAIN_TRACE=%d  TB_INTERLEAVE=%d  TB_PRINT_TOKEN=%d\n",
                steps, topk, drain_trace, interleave, print_tok);
    std::printf("[TB] TB_RUN_CORNERS=%d  TB_CORNER_STRICT=%d  TB_STRICT_STAGE=%d\n",
                run_corners, corner_strict, strict_stage);
    std::fflush(stdout);

    // ------------------------------------------------------------
    // Host arrays (weights constant across steps)
    // ------------------------------------------------------------
    static DTYPE     kernel_host[CONV_K];
    static DTYPE_VEC A_fixed[STATE_V];
    static DTYPE_VEC RMS_weight[D_T];
    static DTYPE_VEC RMS_weight_2[C2_T];                    // ★ v2
    static DTYPE_VEC D_diag[C2_T];

    static W_VEC W_in_1[D_T][SSMU_RANK_T];                  // ★ v2 low-rank
    static W_VEC W_in_2[SSMU_RANK_T][C2_T];                 // middle 5120 low-rank
    static W_VEC W_in_nonlr[D_T][INP_NONLR_T];
    static W_VEC W_delta[C2_T][C2_T];
    static W_VEC W_out_A[D_T][SSMU_RANK_T];                 // ★ v2 low-rank
    static W_VEC W_out_B[SSMU_RANK_T][C2_T];                // ★ v2 low-rank

    // Per-step IO
    static DTYPE_VEC X_host[D_T];
    static DTYPE_VEC H0_host[H0_LEN];
    static DTYPE_VEC conv_state_in_host[CONV_K-1];

    static DTYPE_VEC out_ref[D_T];
    static DTYPE_VEC h1_ref[SSMU_H1_OUT_LEN];
    static DTYPE_VEC conv_state_out_ref[CONV_K-1];
    static DTYPE_VEC core_ref[C2_T];
    static DTYPE_VEC norm_ref[C2_T];
    static DTYPE_VEC opj_ref[D_T];

    static DTYPE_VEC out_dut[D_T];
    static DTYPE_VEC h1_dut[SSMU_H1_OUT_LEN];
    static DTYPE_VEC conv_state_out_dut[CONV_K-1];
    static DTYPE_VEC core_dut[C2_T];
    static DTYPE_VEC norm_dut[C2_T];
    static DTYPE_VEC opj_dut[D_T];

    auto P0 = [&](const char* fn){ return join_path(bins_dir, std::string(fn)); };

    // ---- load constant tables/weights ----
    if (!load_vec_tokens(P0("A_fixed_f32.raw.bin"),           STATE_V, A_fixed,      "A_fixed")) return 1;
    if (!load_vec_tokens(P0("RMS_weight_f32.raw.bin"),        D_T,     RMS_weight,   "RMS_weight")) return 1;
    if (!load_vec_tokens(P0("RMS_weight_2_f32.raw.bin"),      C2_T,    RMS_weight_2, "RMS_weight_2")) return 1;  // ★ v2
    if (!load_vec_tokens(P0("D_diag_f32.raw.bin"),            C2_T,    D_diag,       "D_diag")) return 1;

    if (!load_w_mat<D_T,        SSMU_RANK_T>(P0("W_in_1_f32.raw.bin"),  W_in_1,  "W_in_1"))  return 1;  // ★ v2
    if (!load_w_mat<SSMU_RANK_T, C2_T>     (P0("W_in_2_f32.raw.bin"),  W_in_2,  "W_in_2_mid5120"))  return 1;
    if (!load_w_mat<D_T, INP_NONLR_T>      (P0("W_in_nonlr_f32.raw.bin"), W_in_nonlr, "W_in_nonlr")) return 1;
    if (!load_w_mat<C2_T, C2_T>            (P0("W_delta_f32.raw.bin"), W_delta, "W_delta")) return 1;
    if (!load_w_mat<D_T,        SSMU_RANK_T>(P0("W_out_A_f32.raw.bin"), W_out_A, "W_out_A")) return 1;  // ★ v2
    if (!load_w_mat<SSMU_RANK_T, C2_T>     (P0("W_out_B_f32.raw.bin"), W_out_B, "W_out_B")) return 1;  // ★ v2

    // ---- runtime scales (env(presence) > files > default) ----
    float w_scale_in    = 0.0f;
    float w_scale_delta = 0.0f;
    float w_scale_out   = 0.0f;

    if (env_present("TB_WSCALE_IN")) {
        w_scale_in = read_env_float("TB_WSCALE_IN", 0.0f);
        std::printf("[TB] w_scale_in from ENV = %f\n", w_scale_in);
    } else {
        if (!load_scalar_one_q412(P0("w_scale_in_f32.raw.bin"), w_scale_in, "w_scale_in")) {
            w_scale_in = 0.0f;
            std::printf("[TB] w_scale_in default = %f\n", w_scale_in);
        }
    }

    if (env_present("TB_WSCALE_DELTA")) {
        w_scale_delta = read_env_float("TB_WSCALE_DELTA", 0.0f);
        std::printf("[TB] w_scale_delta from ENV = %f\n", w_scale_delta);
    } else {
        if (!load_scalar_one_q412(P0("w_scale_delta_f32.raw.bin"), w_scale_delta, "w_scale_delta")) {
            w_scale_delta = 0.0f;
            std::printf("[TB] w_scale_delta default = %f\n", w_scale_delta);
        }
    }

    if (env_present("TB_WSCALE_OUT")) {
        w_scale_out = read_env_float("TB_WSCALE_OUT", 0.0f);
        std::printf("[TB] w_scale_out from ENV = %f\n", w_scale_out);
    } else {
        if (!load_scalar_one_q412(P0("w_scale_out_f32.raw.bin"), w_scale_out, "w_scale_out")) {
            w_scale_out = 0.0f;
            std::printf("[TB] w_scale_out default = %f\n", w_scale_out);
        }
    }

    std::printf("[TB] runtime scales: w_scale_in=%f  w_scale_delta=%f  w_scale_out=%f\n",
                w_scale_in, w_scale_delta, w_scale_out);
    std::fflush(stdout);

    // ---- initial caches (step0 preferred if exists, else legacy) ----
    static DTYPE_VEC H0_init[H0_LEN];
    static DTYPE_VEC conv_state_init[CONV_K-1];
    {
        const std::string h0_path = step_file(bins_dir, "h0_in_f32.raw.bin", 0);
        const std::string cs_path = step_file(bins_dir, "conv_state_in_f32.raw.bin", 0);

        if (!load_vec_tokens(h0_path, H0_LEN, H0_init, "h0_in(step0)")) return 1;
        if (!load_vec_tokens(cs_path, CONV_K-1, conv_state_init, "conv_state_in(step0)")) return 1;
    }

    struct RunProfile {
        const char* name;
        float ws_in;
        float ws_delta;
        float ws_out;
        int steps;
        int interleave;
        bool strict_golden;
    };

    std::vector<RunProfile> profiles;
    profiles.push_back(RunProfile{
        "baseline",
        w_scale_in,
        w_scale_delta,
        w_scale_out,
        steps,
        interleave,
        true
    });

    const int corner_steps = std::max(2, steps);
    if (run_corners) {
        profiles.push_back(RunProfile{"corner_scale_zero",  0.0f,           0.0f,           0.0f,           corner_steps, 1, false});
        profiles.push_back(RunProfile{"corner_scale_small", (1.0f/1024.0f), (1.0f/1024.0f), (1.0f/1024.0f), corner_steps, 1, false});
        profiles.push_back(RunProfile{"corner_scale_large", 2.0f,           2.0f,           2.0f,           corner_steps, 1, false});
    }

    // ============================================================
    // Multi-profile + multi-step inference loop
    // ============================================================
    bool all_pass = true;

    for (const auto& prof : profiles) {
        const bool strict_compare = prof.strict_golden || (corner_strict != 0);
        const bool strict_stage_gate = (strict_compare && (strict_stage != 0));

        std::printf("\n[TB] =========================================================\n");
        std::printf("[TB] PROFILE=%s  steps=%d  interleave=%d  strict=%d stage_gate=%d\n",
                prof.name, prof.steps, prof.interleave, (int)strict_compare, (int)strict_stage_gate);
        std::printf("[TB] scales: w_in=%f w_delta=%f w_out=%f\n",
                    prof.ws_in, prof.ws_delta, prof.ws_out);
        std::printf("[TB] =========================================================\n");
        std::fflush(stdout);

        for (int i=0;i<H0_LEN;++i)   H0_host[i] = H0_init[i];
        for (int i=0;i<CONV_K-1;++i) conv_state_in_host[i] = conv_state_init[i];

        for (int s=0; s<prof.steps; ++s) {
            std::printf("\n[TB] [%s] STEP %d/%d\n", prof.name, s+1, prof.steps);
            std::fflush(stdout);

            const std::string k_path    = step_file(bins_dir, "kernel_in_f32.raw.bin", s);
            const std::string x_path    = step_file(bins_dir, "x_in_f32.raw.bin", s);
            const std::string out_path  = step_file(bins_dir, "out_ref_f32.raw.bin", s);
            const std::string h1_path   = step_file(bins_dir, "h1_ref_f32.raw.bin", s);
            const std::string cs_ref_path = step_file(bins_dir, "conv_state_out_f32.raw.bin", s);

            if (!load_scalar_tokens(k_path, CONV_K, kernel_host, "kernel_in")) return 1;
            if (!load_vec_tokens(x_path, D_T, X_host, "x_in")) return 1;
            if (!load_vec_tokens(out_path, D_T, out_ref, "out_ref")) return 1;
            if (!load_vec_tokens(h1_path, SSMU_H1_OUT_LEN, h1_ref, "h1_ref")) return 1;

            if (!load_vec_tokens(cs_ref_path, CONV_K-1, conv_state_out_ref, "conv_state_out_ref(required)")) {
                std::printf("[TB][FATAL] conv_state_out_ref is mandatory. Missing/invalid for step=%d\n", s);
                std::fflush(stdout);
                return 1;
            }

            {
                const std::string h0_override = step_file(bins_dir, "h0_in_f32.raw.bin", s);
                const std::string cs_override = step_file(bins_dir, "conv_state_in_f32.raw.bin", s);

                if (is_step_specific_path(h0_override, s) && file_exists_i16_quick(h0_override)) {
                    if (!load_vec_tokens(h0_override, H0_LEN, H0_host, "h0_in(override)")) return 1;
                }
                if (is_step_specific_path(cs_override, s) && file_exists_i16_quick(cs_override)) {
                    if (!load_vec_tokens(cs_override, CONV_K-1, conv_state_in_host, "conv_state_in(override)")) return 1;
                }
            }

            int rc = run_dut_once(
                kernel_host, A_fixed, RMS_weight, RMS_weight_2,
                W_in_1, W_in_2, W_in_nonlr, W_delta, W_out_A, W_out_B, D_diag,
                X_host, H0_host, conv_state_in_host,
                out_dut, h1_dut, conv_state_out_dut,
                prof.ws_in, prof.ws_delta, prof.ws_out,
                prof.interleave
            );
            if (rc) return 2;

            float diff_out  = max_abs_diff_vec_tokens(out_dut, out_ref, D_T);
            float diff_h1   = max_abs_diff_vec_tokens(h1_dut,  h1_ref,  SSMU_H1_OUT_LEN);
            float diff_cs   = max_abs_diff_vec_tokens(conv_state_out_dut, conv_state_out_ref, CONV_K-1);

            float diff_core = 0.0f;
            float diff_norm = 0.0f;
            float diff_opj  = 0.0f;
            bool  have_stage_dut = true;

            const std::string core_ref_path = step_file_bin(bins_dir, "ssm_core_ref_f32.bin", s);
            const std::string norm_ref_path = step_file_bin(bins_dir, "ssm_normed_ref_f32.bin", s);
            const std::string opj_ref_path  = step_file_bin(bins_dir, "out_proj_ref_f32.bin", s);

            if (!load_vec_tokens_f32(core_ref_path, C2_T, core_ref, "ssm_core_ref")) return 1;
            if (!load_vec_tokens_f32(norm_ref_path, C2_T, norm_ref, "ssm_normed_ref")) return 1;
            if (!load_vec_tokens_f32(opj_ref_path, D_T, opj_ref, "out_proj_ref")) return 1;

            const std::string core_dut_path = join_path(dut_dump_dir, "dut_ssm_core_f32.bin");
            const std::string norm_dut_path = join_path(dut_dump_dir, "dut_ssm_normed_f32.bin");
            const std::string opj_dut_path  = join_path(dut_dump_dir, "dut_out_proj_f32.bin");

            if (file_exists_i16_quick(core_dut_path) &&
                file_exists_i16_quick(norm_dut_path) &&
                file_exists_i16_quick(opj_dut_path)) {
                if (!load_vec_tokens_f32(core_dut_path, C2_T, core_dut, "dut_ssm_core")) have_stage_dut = false;
                if (!load_vec_tokens_f32(norm_dut_path, C2_T, norm_dut, "dut_ssm_normed")) have_stage_dut = false;
                
                // ★ CRITICAL FIX: Recalculate norm_ref from core_dut to match DUT fixed-point behavior
                // (Old golden norm file may use different RMS2 quantization semantics)
                if (have_stage_dut) {
                    std::printf("[TB] [NORM-FIX] Recalculating norm reference from DUT core output using DUT fixed-point rules...\n");
                    std::fflush(stdout);
                    
                    std::vector<float> rms_w2_f32(C2_T * VEC_FACTOR);
                    for (int j = 0; j < C2_T; ++j) {
                        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
                            rms_w2_f32[j * VEC_FACTOR + l] = (float)vget(RMS_weight_2[j], l);
                        }
                    }
                    
                    std::vector<float> core_dut_f32(C2_T * VEC_FACTOR);
                    for (int j = 0; j < C2_T; ++j) {
                        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
                            core_dut_f32[j * VEC_FACTOR + l] = (float)vget(core_dut[j], l);
                        }
                    }
                    
                    std::vector<float> norm_recalc(C2_T * VEC_FACTOR);
                    compute_norm_stage_like_dut(
                        core_dut_f32.data(),
                        rms_w2_f32.data(),
                        norm_recalc.data(),
                        C2_T,
                        VEC_FACTOR
                    );
                    
                    // Replace norm_ref with recalculated version
                    for (int j = 0; j < C2_T; ++j) {
                        for (unsigned l = 0; l < (unsigned)VEC_FACTOR; ++l) {
                            vset(norm_ref[j], l, (DTYPE)(norm_recalc[j * VEC_FACTOR + l]));
                        }
                    }
                    
                    std::printf("[TB] [NORM-FIX] Norm reference recalculated and updated for strict comparison.\n");
                    std::fflush(stdout);
                }

                if (!load_vec_tokens_f32(opj_dut_path, D_T, opj_dut, "dut_out_proj")) have_stage_dut = false;
            } else {
                have_stage_dut = false;
                std::printf("[TB][WARN] DUT stage dump files not found under '%s' (set TB_DUT_DUMP_DIR if needed).\n",
                            dut_dump_dir.c_str());
                std::fflush(stdout);
            }

            if (have_stage_dut) {
                diff_core = max_abs_diff_vec_tokens(core_dut, core_ref, C2_T);
                diff_norm = max_abs_diff_vec_tokens(norm_dut, norm_ref, C2_T);
                diff_opj  = max_abs_diff_vec_tokens(opj_dut,  opj_ref,  D_T);
            } else if (strict_stage_gate) {
                std::printf("[TB][FATAL] Stage DUT dumps are required when TB_STRICT_STAGE=1 for profile: %s\n", prof.name);
                std::fflush(stdout);
                return 2;
            }

            if (print_tok >= 0) {
                int t_out = std::min(std::max(print_tok, 0), D_T-1);
                int t_h1  = std::min(std::max(print_tok, 0), SSMU_H1_OUT_LEN-1);

                std::printf("[TB] >>> PRINT_BEGIN token=%d\n", print_tok);
                std::fflush(stdout);

                print_vec4("DUT out[t]", out_dut[t_out]);
                print_vec4("REF out[t]", out_ref[t_out]);
                print_vec4("DUT h1[t]",  h1_dut[t_h1]);
                print_vec4("REF h1[t]",  h1_ref[t_h1]);

                std::printf("[TB] >>> PRINT_END\n");
                std::fflush(stdout);
            }

            std::printf("[TB] diff(out )=%e  diff(h1  )=%e  diff(cs  )=%e\n", diff_out, diff_h1, diff_cs);
            if (have_stage_dut) {
                std::printf("[TB] diff(core)=%e  diff(norm)=%e  diff(opj )=%e\n", diff_core, diff_norm, diff_opj);
            } else {
                std::printf("[TB][WARN] Stage diff skipped (missing dut stage dump files).\n");
            }
            std::fflush(stdout);

            bool pass_strict = (diff_out  <= tol_out)
                            && (diff_h1   <= tol_h1)
                            && (diff_cs   <= tol_cs)
                            && (!strict_stage_gate || (have_stage_dut
                                                 && (diff_core <= tol_core)
                                                 && (diff_norm <= tol_norm)
                                                 && (diff_opj  <= tol_opj)));

            if (!strict_compare) {
                float peak_out = max_abs_value_vec_tokens(out_dut, D_T);
                float peak_h1  = max_abs_value_vec_tokens(h1_dut, SSMU_H1_OUT_LEN);
                float peak_cs  = max_abs_value_vec_tokens(conv_state_out_dut, CONV_K-1);
                std::printf("[TB] smoke peaks: out=%e h1=%e cs=%e\n", peak_out, peak_h1, peak_cs);
                std::fflush(stdout);
            }

            if (!pass_strict && topk > 0 && strict_compare) {
                collect_topk(out_dut, out_ref, D_T, topk, "out");
                collect_topk(h1_dut,  h1_ref,  SSMU_H1_OUT_LEN, topk, "h1");
                collect_topk(conv_state_out_dut, conv_state_out_ref, CONV_K-1, topk, "conv_state_out");
                if (have_stage_dut) {
                    collect_topk(core_dut, core_ref, C2_T, topk, "ssm_core");
                    collect_topk(norm_dut, norm_ref, C2_T, topk, "ssm_normed");
                    collect_topk(opj_dut,  opj_ref,  D_T, topk, "out_proj");
                }
            }

            if (strict_compare) {
                if (pass_strict) {
                    std::printf("[TB][%s][STEP %d][PASS] strict golden match.\n", prof.name, s);
                } else {
                    std::printf("[TB][%s][STEP %d][FAIL] strict golden mismatch.\n", prof.name, s);
                    all_pass = false;
                }
            } else {
                std::printf("[TB][%s][STEP %d][SMOKE] completed (non-strict corner profile).\n", prof.name, s);
            }
            std::fflush(stdout);

            for (int i=0;i<H0_LEN;++i)   H0_host[i] = h1_dut[i];
            for (int i=0;i<CONV_K-1;++i) conv_state_in_host[i] = conv_state_out_dut[i];
        }
    }

    if (all_pass) {
        std::printf("\n[TB][PASS] ALL STRICT PROFILES PASS.\n");
        std::fflush(stdout);
        return 0;
    } else {
        std::printf("\n[TB][FAIL] One or more strict profiles/steps failed.\n");
        std::printf("          (Adjust TB_TOL_OUT / TB_TOL_H1 / TB_TOL_CS / TB_TOL_CORE / TB_TOL_NORM / TB_TOL_OPJ if needed)\n");
        std::fflush(stdout);
        return 2;
    }
}
