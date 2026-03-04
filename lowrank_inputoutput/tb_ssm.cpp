// tb_ssmu.cpp  (LOW-RANK variant — BIN-DRIVEN + GOLDEN-COMPARE + MULTI-STEP INFERENCE / COSIM-safe)
// ✅ Updated for LOW-RANK SSMU top:
//      W_inproj[D_T][CIN_T]  → W_in_1[D_T][RANK_T] + W_in_2[RANK_T][CIN_T]
//      W_out[D_T][C2_T]      → W_out_A[D_T][RANK_T] + W_out_B[RANK_T][C2_T]
// ✅ COSIM FIX: call syn.top symbol SSMU_STACK64 (NOT SSMU)
// ✅ Strict token-count checks (no silent padding, no size() dependency)
// ✅ Multi-step inference cache
// ✅ Matches make_bins_raw_lowrank.py bins_raw format (int16 Q4.12, vec-major flatten)
// ✅ Runtime w_scale_* handling (presence-based)
// ✅ conv_state_out golden compare (TB_TOL_CS)
// ✅ H1_DDR fallback, interleave, print_token

#include "SSMU.h"

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
// ★ LOW-RANK Constants (must match ssm.cpp)
// ============================================================
#ifndef SSMU_RANK
#define SSMU_RANK 1024
#endif

#define SSMU_RANK_T (SSMU_RANK / VEC_FACTOR)
static const int RANK_T = SSMU_RANK_T;   // 128

// ============================================================
// Q4.12 packing matches make_bins_raw_lowrank.py
// ============================================================
#ifndef TB_FRAC_BITS
#define TB_FRAC_BITS 12
#endif

// ============================================================
// Build tag
// ============================================================
#ifndef TB_BUILD_TAG
#define TB_BUILD_TAG "tb_ssmu.cpp LOW-RANK + COSIM_TOP_CALL_SSMU_STACK64 + CS_COMPARE + H1_DDR_FALLBACK TAG_2026-03-04_v1"
#endif

// ============================================================
// vget/vset fallback
// ============================================================
#ifndef SSMU_HAVE_VGET_VSET
template<typename V>
static inline auto vget(const V& v, unsigned idx) -> decltype(v[idx]) { return v[idx]; }
template<typename V, typename T>
static inline void vset(V& v, unsigned idx, const T& val) { v[idx] = (decltype(v[idx]))val; }
#endif

static inline DTYPE from_i16(int16_t q) {
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
        if (env[0]) return std::string(env);
    }

    const char* cands[] = {
        "bins_raw",
        "../bins_raw",
        "../../bins_raw",
        "../../../bins_raw",
        "../../../../bins_raw"
    };

    for (auto* c : cands) {
        const std::string probe = join_path(std::string(c), "A_fixed_f32.raw.bin");
        if (file_exists_i16_quick(probe)) return std::string(c);
    }
    return std::string("bins_raw");
}

static bool is_step_specific_path(const std::string& p, int step) {
    const std::string key = "_s" + std::to_string(step) + ".raw.bin";
    return (p.find(key) != std::string::npos);
}

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

    if (base == "x_in_f32.raw.bin")             { auto p = map("x_in");             if (!p.empty()) return p; }
    if (base == "kernel_in_f32.raw.bin")         { auto p = map("kernel_in");         if (!p.empty()) return p; }
    if (base == "out_ref_f32.raw.bin")           { auto p = map("out_ref");           if (!p.empty()) return p; }
    if (base == "h1_ref_f32.raw.bin")            { auto p = map("h1_ref");            if (!p.empty()) return p; }
    if (base == "h0_in_f32.raw.bin")             { auto p = map("h0_in");             if (!p.empty()) return p; }
    if (base == "conv_state_in_f32.raw.bin")     { auto p = map("conv_state_in");     if (!p.empty()) return p; }
    if (base == "conv_state_out_f32.raw.bin")    { auto p = map("conv_state_out");    if (!p.empty()) return p; }

    return join_path(dir, base);
}

// ============================================================
// Token loaders
// ============================================================
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
// Stream drain (COSIM-safe)
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
// Env helpers
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
// ★ COSIM symbol: LOW-RANK SSMU_STACK64 declaration
// ============================================================
extern "C" void SSMU_STACK64(
    hls::stream<DTYPE>&      kernel_in,
    const DTYPE_VEC          A_fixed[STATE_V],
    const DTYPE_VEC          RMS_weight[D_T],

    // ★ LOW-RANK inproj
    const W_VEC              W_in_1[D_T][RANK_T],
    const W_VEC              W_in_2[RANK_T][CIN_T],

    const W_VEC              W_delta[C2_T][C2_T],

    // ★ LOW-RANK outproj
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
);

// ============================================================
// ★ Run DUT once (LOW-RANK interface)
// ============================================================
static int run_dut_once(
    const DTYPE     kernel_host[CONV_K],
    const DTYPE_VEC A_fixed[STATE_V],
    const DTYPE_VEC RMS_weight[D_T],
    // ★ LOW-RANK weights
    const W_VEC     W_in_1[D_T][RANK_T],
    const W_VEC     W_in_2[RANK_T][CIN_T],
    const W_VEC     W_delta[C2_T][C2_T],
    const W_VEC     W_out_A[D_T][RANK_T],
    const W_VEC     W_out_B[RANK_T][C2_T],
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

    static DTYPE_VEC C_ddr[H0_LEN];
    static DTYPE_VEC H1_ddr[H0_LEN];

    // Launch DUT first (COSIM-safe)
    std::thread dut_th([&](){
        SSMU_STACK64(
            kernel_in,
            A_fixed,
            RMS_weight,
            W_in_1, W_in_2,         // ★ LOW-RANK inproj
            W_delta,
            W_out_A, W_out_B,       // ★ LOW-RANK outproj
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

    // Feed inputs AFTER DUT starts
    if (!tb_interleave) {
        for (int k=0;k<CONV_K;++k)      kernel_in.write(kernel_host[k]);
        for (int i=0;i<D_T;++i)        X_in.write(X_host[i]);
        for (int i=0;i<H0_LEN;++i)     H0_in.write(H0_host[i]);
        for (int k=0;k<CONV_K-1;++k)   conv_state_in.write(conv_state_in_host[k]);
    } else {
        int k_i = 0, x_i = 0, h_i = 0, c_i = 0;
        while (k_i < CONV_K || x_i < D_T || h_i < H0_LEN || c_i < (CONV_K-1)) {
            if (k_i < CONV_K)        { kernel_in.write(kernel_host[k_i]); ++k_i; }
            if (x_i < D_T)           { X_in.write(X_host[x_i]); ++x_i; }
            if (h_i < H0_LEN)        { H0_in.write(H0_host[h_i]); ++h_i; }
            if (c_i < (CONV_K-1))    { conv_state_in.write(conv_state_in_host[c_i]); ++c_i; }
        }
    }

    dut_th.join();

    // Strict drains
    if (drain_exact(conv_state_out, conv_state_out_host, CONV_K-1, "conv_state_out") != 0) return -1;

#if SSMU_ENABLE_H1_STREAM_OUT
    if (drain_exact(H1_out, h1_host_dut, SSMU_H1_OUT_LEN, "H1_out") != 0) return -1;
#else
    drain_all(H1_out);
    for (int i=0;i<SSMU_H1_OUT_LEN;++i) h1_host_dut[i] = H1_ddr[i];
#endif

    if (drain_exact(out, out_host_dut, D_T, "out") != 0) return -1;

    return 0;
}

// ============================================================
// Main
// ============================================================
int main() {
    std::string bins_dir = resolve_bins_dir();

    const float tol_out     = read_env_float("TB_TOL_OUT", 0.05f);
    const float tol_h1      = read_env_float("TB_TOL_H1",  0.06f);
    const float tol_cs      = read_env_float("TB_TOL_CS",  tol_h1);
    const int   steps       = std::max(1, read_env_int("TB_STEPS", 1));
    const int   topk        = std::max(0, read_env_int("TB_TOPK", 10));
    const int   drain_trace = read_env_int("TB_DRAIN_TRACE", 1);
    const int   interleave  = read_env_int("TB_INTERLEAVE", 0);
    const int   print_tok   = read_env_int("TB_PRINT_TOKEN", 0);

    std::printf("[TB] BUILD_TAG=%s\n", TB_BUILD_TAG);
    std::printf("[TB] Using bins dir: %s\n", bins_dir.c_str());
    std::printf("[TB] RANK=%d RANK_T=%d\n", (int)SSMU_RANK, (int)RANK_T);
    std::printf("[TB] STATE_SCALAR=%d  STATE_V(STATE_T)=%d  D_T=%d  C2_T=%d  CIN_T=%d  VEC_FACTOR=%d\n",
                (int)STATE_SCALAR, (int)STATE_V, (int)D_T, (int)C2_T, (int)CIN_T, (int)VEC_FACTOR);
    std::printf("[TB] H0_LEN=%d  H1_LEN=%d  CONV_K=%d\n", (int)H0_LEN, (int)SSMU_H1_OUT_LEN, (int)CONV_K);
    std::printf("[TB] TOL_OUT=%g  TOL_H1=%g  TOL_CS=%g  TB_STEPS=%d  TB_TOPK=%d  TB_DRAIN_TRACE=%d\n",
                tol_out, tol_h1, tol_cs, steps, topk, drain_trace);
    std::printf("[TB] TB_INTERLEAVE=%d  TB_PRINT_TOKEN=%d\n", interleave, print_tok);
    std::fflush(stdout);

    (void)drain_trace;  // placeholder for future

    // ------------------------------------------------------------
    // Host arrays (weights constant across steps)
    // ------------------------------------------------------------
    static DTYPE     kernel_host[CONV_K];
    static DTYPE_VEC A_fixed[STATE_V];
    static DTYPE_VEC RMS_weight[D_T];
    static DTYPE_VEC D_diag[C2_T];

    // ★ LOW-RANK weight host arrays (4 sub-matrices + unchanged W_delta)
    static W_VEC W_in_1[D_T][RANK_T];
    static W_VEC W_in_2[RANK_T][CIN_T];
    static W_VEC W_delta[C2_T][C2_T];
    static W_VEC W_out_A[D_T][RANK_T];
    static W_VEC W_out_B[RANK_T][C2_T];

    // Per-step IO
    static DTYPE_VEC X_host[D_T];
    static DTYPE_VEC H0_host[H0_LEN];
    static DTYPE_VEC conv_state_in_host[CONV_K-1];

    static DTYPE_VEC out_ref[D_T];
    static DTYPE_VEC h1_ref[SSMU_H1_OUT_LEN];
    static DTYPE_VEC conv_state_out_ref[CONV_K-1];

    static DTYPE_VEC out_dut[D_T];
    static DTYPE_VEC h1_dut[SSMU_H1_OUT_LEN];
    static DTYPE_VEC conv_state_out_dut[CONV_K-1];

    auto P0 = [&](const char* fn){ return join_path(bins_dir, std::string(fn)); };

    // ---- load constant tables/weights ----
    if (!load_vec_tokens(P0("A_fixed_f32.raw.bin"),          STATE_V, A_fixed,    "A_fixed")) return 1;
    if (!load_vec_tokens(P0("RMS_weight_f32.raw.bin"),       D_T,     RMS_weight, "RMS_weight")) return 1;
    if (!load_vec_tokens(P0("D_diag_f32.raw.bin"),           C2_T,    D_diag,     "D_diag")) return 1;

    // ★ LOW-RANK weight loading (4 sub-matrices)
    if (!load_w_mat<D_T, RANK_T>(P0("W_in_1_f32.raw.bin"),    W_in_1,  "W_in_1"))  return 1;
    if (!load_w_mat<RANK_T, CIN_T>(P0("W_in_2_f32.raw.bin"),  W_in_2,  "W_in_2"))  return 1;
    if (!load_w_mat<C2_T, C2_T>(P0("W_delta_f32.raw.bin"),    W_delta, "W_delta"))  return 1;
    if (!load_w_mat<D_T, RANK_T>(P0("W_out_A_f32.raw.bin"),   W_out_A, "W_out_A")) return 1;
    if (!load_w_mat<RANK_T, C2_T>(P0("W_out_B_f32.raw.bin"),  W_out_B, "W_out_B")) return 1;

    // ---- runtime scales ----
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

    // ---- initial caches ----
    {
        const std::string h0_path = step_file(bins_dir, "h0_in_f32.raw.bin", 0);
        const std::string cs_path = step_file(bins_dir, "conv_state_in_f32.raw.bin", 0);

        if (!load_vec_tokens(h0_path, H0_LEN, H0_host, "h0_in(step0)")) return 1;
        if (!load_vec_tokens(cs_path, CONV_K-1, conv_state_in_host, "conv_state_in(step0)")) return 1;
    }

    // ============================================================
    // Multi-step inference loop
    // ============================================================
    bool all_pass = true;

    for (int s=0; s<steps; ++s) {
        std::printf("\n[TB] ====================== STEP %d/%d ======================\n", s, steps);
        std::fflush(stdout);

        const std::string k_path    = step_file(bins_dir, "kernel_in_f32.raw.bin", s);
        const std::string x_path    = step_file(bins_dir, "x_in_f32.raw.bin", s);
        const std::string out_path  = step_file(bins_dir, "out_ref_f32.raw.bin", s);
        const std::string h1_path   = step_file(bins_dir, "h1_ref_f32.raw.bin", s);

        if (!load_scalar_tokens(k_path, CONV_K, kernel_host, "kernel_in")) return 1;
        if (!load_vec_tokens(x_path, D_T, X_host, "x_in")) return 1;
        if (!load_vec_tokens(out_path, D_T, out_ref, "out_ref")) return 1;
        if (!load_vec_tokens(h1_path, SSMU_H1_OUT_LEN, h1_ref, "h1_ref")) return 1;

        bool have_cs_ref = false;
        {
            const std::string cs_ref_path = step_file(bins_dir, "conv_state_out_f32.raw.bin", s);
            if (file_exists_i16_quick(cs_ref_path)) {
                if (!load_vec_tokens(cs_ref_path, CONV_K-1, conv_state_out_ref, "conv_state_out_ref")) return 1;
                have_cs_ref = true;
            } else {
                for (int i=0;i<CONV_K-1;++i) conv_state_out_ref[i] = make_vec_zero();
                have_cs_ref = false;
            }
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

        // ---- run dut (LOW-RANK) ----
        int rc = run_dut_once(
            kernel_host, A_fixed, RMS_weight,
            W_in_1, W_in_2, W_delta, W_out_A, W_out_B,   // ★ 5 weight matrices
            D_diag,
            X_host, H0_host, conv_state_in_host,
            out_dut, h1_dut, conv_state_out_dut,
            w_scale_in, w_scale_delta, w_scale_out,
            interleave
        );
        if (rc) return 2;

        // ---- compare ----
        float diff_out = max_abs_diff_vec_tokens(out_dut, out_ref, D_T);
        float diff_h1  = max_abs_diff_vec_tokens(h1_dut,  h1_ref,  SSMU_H1_OUT_LEN);

        float diff_cs = 0.0f;
        if (have_cs_ref) {
            diff_cs = max_abs_diff_vec_tokens(conv_state_out_dut, conv_state_out_ref, CONV_K-1);
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

        std::printf("[TB] max_abs_diff(out) = %e\n", diff_out);
        std::printf("[TB] max_abs_diff(h1 ) = %e\n", diff_h1);
        if (have_cs_ref) {
            std::printf("[TB] max_abs_diff(cs ) = %e\n", diff_cs);
        } else {
            std::printf("[TB] conv_state_out_ref: (missing file) -> skip compare\n");
        }
        std::fflush(stdout);

        bool pass = (diff_out <= tol_out)
                 && (diff_h1  <= tol_h1)
                 && (!have_cs_ref || (diff_cs <= tol_cs));

        if (!pass && topk > 0) {
            collect_topk(out_dut, out_ref, D_T, topk, "out");
            collect_topk(h1_dut,  h1_ref,  SSMU_H1_OUT_LEN, topk, "h1");
            if (have_cs_ref) {
                collect_topk(conv_state_out_dut, conv_state_out_ref, CONV_K-1, topk, "conv_state_out");
            }
        }

        if (pass) {
            std::printf("[TB][STEP %d][PASS] Golden match within tolerance.\n", s);
        } else {
            std::printf("[TB][STEP %d][FAIL] Golden mismatch.\n", s);
            all_pass = false;
        }
        std::fflush(stdout);

        // ---- update cache for next step ----
        for (int i=0;i<H0_LEN;++i)        H0_host[i] = h1_dut[i];
        for (int i=0;i<CONV_K-1;++i)      conv_state_in_host[i] = conv_state_out_dut[i];
    }

    if (all_pass) {
        std::printf("\n[TB][PASS] ALL STEPS PASS.\n");
        std::fflush(stdout);
        return 0;
    } else {
        std::printf("\n[TB][FAIL] One or more steps failed.\n");
        std::printf("          (Adjust TB_TOL_OUT / TB_TOL_H1 / TB_TOL_CS if needed)\n");
        std::fflush(stdout);
        return 2;
    }
}
