"""
make_bins_raw_lowrank.py — Generate bins_raw for LOW-RANK SSMU variant.

Changes from make_bins_raw.py:
  W_inproj[D_T][CIN_T]  → W_in_1[D_T][RANK_T] + W_in_2[RANK_T][CIN_T]
  W_out[D_T][C2_T]      → W_out_A[D_T][RANK_T] + W_out_B[RANK_T][C2_T]

Usage:
  python make_bins_raw_lowrank.py --ssmu_h SSMU.h --out_dir bins_raw --rank 1024

The golden reference computes using the factored form, so the COSIM
comparison is exact (no rank-truncation error — we generate W_in_1/W_in_2
as independent random matrices, not via SVD of W_inproj).
"""

import os
import re
import argparse
import numpy as np

# =========================
# Quantization: int16 Q4.12
# =========================
TB_FRAC_BITS = 12
Q_SCALE = 1 << TB_FRAC_BITS
Q_MIN = -32768
Q_MAX = 32767

def f32_to_i16_q412(arr_f32: np.ndarray) -> np.ndarray:
    x = np.asarray(arr_f32, dtype=np.float32)
    q = np.round(x * Q_SCALE).astype(np.int32)
    q = np.clip(q, Q_MIN, Q_MAX).astype(np.int16)
    return q

def write_i16_raw(path: str, arr_i16: np.ndarray):
    arr_i16 = np.asarray(arr_i16, dtype=np.int16).ravel(order="C")
    with open(path, "wb") as f:
        f.write(arr_i16.tobytes(order="C"))

def write_raw_q412(path: str, arr_f32: np.ndarray):
    write_i16_raw(path, f32_to_i16_q412(arr_f32))

def write_f32(path: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32).ravel(order="C")
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))

# =========================
# Parse SSMU.h macros + knobs
# =========================
def parse_macros_from_header(path: str):
    txt = open(path, "r", encoding="utf-8", errors="ignore").read()

    macros = {}

    define_pat = re.compile(r"^\s*#\s*define\s+([A-Za-z_]\w*)\s+(.+?)\s*$", re.M)
    const_pat  = re.compile(r"^\s*static\s+const\s+int\s+([A-Za-z_]\w*)\s*=\s*(.+?)\s*;\s*$", re.M)

    def strip_comments(expr: str) -> str:
        expr = expr.strip()
        expr = expr.split("//")[0].strip()
        expr = expr.split("/*")[0].strip()
        return expr

    def maybe_expr(expr: str):
        expr = strip_comments(expr)
        if not re.match(r"^[0-9A-Za-z_+\-*/()%<>\s\.]+$", expr):
            return expr
        return expr

    for m in define_pat.finditer(txt):
        name, expr = m.group(1), m.group(2)
        macros[name] = maybe_expr(expr)

    for m in const_pat.finditer(txt):
        name, expr = m.group(1), m.group(2)
        macros[name] = maybe_expr(expr)

    def try_eval(expr: str):
        try:
            return eval(expr, {"__builtins__": {}}, {})
        except Exception:
            return None

    for _ in range(80):
        progress = False
        for k, v in list(macros.items()):
            if isinstance(v, (int, float)):
                continue
            if not isinstance(v, str):
                continue
            expr = v
            for kk, vv in macros.items():
                if isinstance(vv, (int, float)):
                    expr = re.sub(rf"\b{re.escape(kk)}\b", str(vv), expr)
            val = try_eval(expr)
            if isinstance(val, (int, float, np.integer, np.floating)):
                if abs(float(val) - int(float(val))) < 1e-9:
                    macros[k] = int(float(val))
                else:
                    macros[k] = float(val)
                progress = True
        if not progress:
            break

    def resolve_int(name: str, default=None):
        if name not in macros:
            if default is None:
                raise KeyError(f"Macro {name} not found in {path}")
            return int(default)
        v = macros[name]
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return int(v)
        expr = str(v)
        for kk, vv in macros.items():
            if isinstance(vv, (int, float)):
                expr = re.sub(rf"\b{re.escape(kk)}\b", str(vv), expr)
        val = try_eval(expr)
        if val is None:
            if default is None:
                raise ValueError(f"Could not resolve int macro {name}: expr={v}")
            return int(default)
        return int(val)

    def resolve_float(name: str, default=None):
        if name not in macros:
            if default is None:
                raise KeyError(f"Macro {name} not found in {path}")
            return float(default)
        v = macros[name]
        if isinstance(v, (int, float)):
            return float(v)
        expr = str(v)
        for kk, vv in macros.items():
            if isinstance(vv, (int, float)):
                expr = re.sub(rf"\b{re.escape(kk)}\b", str(vv), expr)
        val = try_eval(expr)
        if val is None:
            if default is None:
                raise ValueError(f"Could not resolve float macro {name}: expr={v}")
            return float(default)
        return float(val)

    D_T     = resolve_int("SSMU_D_T")
    C2_T    = resolve_int("SSMU_C2_T")
    CCONV_T = resolve_int("SSMU_CCONV_T")
    CH_T    = resolve_int("SSMU_CH_T")
    CIN_T   = resolve_int("SSMU_CIN_T")
    K       = resolve_int("SSMU_K", default=4)
    CP      = resolve_int("VEC_FACTOR", default=resolve_int("SSMU_CP", default=8))

    STATE_SCALAR = resolve_int("SSMU_STATE", default=resolve_int("SSMU_N"))
    STATE_V      = resolve_int("SSMU_STATE_T", default=STATE_SCALAR // CP)

    H1_LEN = STATE_V * C2_T

    knobs = {
        "SSMU_USE_INT8": resolve_int("SSMU_USE_INT8", default=0),
        "SSMU_ENABLE_DT": resolve_int("SSMU_ENABLE_DT", default=1),
        "SSMU_DELTA_FROM_DT": resolve_int("SSMU_DELTA_FROM_DT", default=1),
        "SSMU_ACCURATE_MATH_CSIM": resolve_int("SSMU_ACCURATE_MATH_CSIM", default=0),
        "SSMU_SOFTPLUS_POS_TH": resolve_float("SSMU_SOFTPLUS_POS_TH", default=8.0),
        "SSMU_SOFTPLUS_NEG_TH": resolve_float("SSMU_SOFTPLUS_NEG_TH", default=-8.0),
        "SSMU_EXP_CLAMP": resolve_float("SSMU_EXP_CLAMP", default=3.0),
    }

    shapes = {
        "CP": CP,
        "D_T": D_T,
        "C2_T": C2_T,
        "CCONV_T": CCONV_T,
        "CH_T": CH_T,
        "CIN_T": CIN_T,
        "K": K,
        "STATE_SCALAR": STATE_SCALAR,
        "STATE_V": STATE_V,
        "H1_LEN": H1_LEN,
    }
    return shapes, knobs

# =========================
# Math helpers (match ssm.cpp approximations)
# =========================
def sigmoid_fx(x, accurate):
    x = np.asarray(x, dtype=np.float32)
    if accurate:
        return (1.0 / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)
    y = (0.5 + 0.25 * x).astype(np.float32)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def silu_fx(x, accurate):
    x = np.asarray(x, dtype=np.float32)
    return (x * sigmoid_fx(x, accurate)).astype(np.float32)

def softplus_fx(x, accurate, pos_th, neg_th):
    x = np.asarray(x, dtype=np.float32)
    if accurate:
        y = np.empty_like(x, dtype=np.float32)
        y[x > pos_th] = x[x > pos_th]
        y[x < neg_th] = 0.0
        mid = (x <= pos_th) & (x >= neg_th)
        y[mid] = np.log1p(np.exp(x[mid], dtype=np.float32)).astype(np.float32)
        return y.astype(np.float32)
    y = np.empty_like(x, dtype=np.float32)
    y[x > pos_th] = x[x > pos_th]
    y[x < neg_th] = 0.0
    mid = (x <= pos_th) & (x >= neg_th)
    y[mid] = (0.5 * x[mid] + 1.0).astype(np.float32)
    return y.astype(np.float32)

def exp_fx(x, accurate, exp_clamp):
    x = np.asarray(x, dtype=np.float32)
    t = np.clip(x, -exp_clamp, exp_clamp).astype(np.float32)
    if accurate:
        return np.exp(t, dtype=np.float32).astype(np.float32)
    y = (1.0 + t + 0.5 * (t * t)).astype(np.float32)
    y = np.maximum(y, 0.0).astype(np.float32)
    return y

# =========================
# Golden compute blocks
# =========================
def rmsnorm_vec_tokens(x_tokens, rms_weight_tokens, eps=1e-5):
    x = x_tokens.astype(np.float32)
    w = rms_weight_tokens.astype(np.float32)
    ms = np.mean(x * x, dtype=np.float32)
    inv = 1.0 / np.sqrt(ms + eps, dtype=np.float32)
    y = x * inv * w
    return y.astype(np.float32)


# ★ LOW-RANK in-projection (two-stage)
def inproj_lowrank(x_normed, W_in_1, W_in_2, D_T, RANK_T, CIN_T, CP):
    """
    Stage 1: temp[r,l] = Σ_j x_normed[j,l] · W_in_1[j,r,l]
    Stage 2: out[i,l]  = Σ_r temp[r,l]      · W_in_2[r,i,l]
    """
    temp = np.zeros((RANK_T, CP), dtype=np.float32)
    for lane in range(CP):
        temp[:, lane] = x_normed[:, lane].astype(np.float32) @ W_in_1[:, :, lane].astype(np.float32)

    out = np.zeros((CIN_T, CP), dtype=np.float32)
    for lane in range(CP):
        out[:, lane] = temp[:, lane].astype(np.float32) @ W_in_2[:, :, lane].astype(np.float32)

    return out


def split_inproj(packed, C2_T, CCONV_T, CH_T):
    """New v2 layout: [ Z: C2_T ] [ XBC: CCONV_T ] [ DT: CH_T ]
    B and C come from conv1d output on the XBC bundle, not from in-proj.
    """
    baseZ  = 0
    baseX  = C2_T
    baseDT = C2_T + CCONV_T
    Z   = packed[baseZ:baseZ + C2_T]
    XBC = packed[baseX:baseX + CCONV_T]
    DT  = packed[baseDT:baseDT + CH_T]
    return Z, XBC, DT


def conv1d_silu_with_state(XBC, Z, kernel, conv_state_in, C2_T, CCONV_T, CP, accurate_math, K=4, STATE_V=16):
    """v2: conv+silu applied to ALL CCONV_T tokens.
    x tokens (i < C2_T):              → X_gate (silu(z)), X_ssm (conv+silu(x))
    B tokens (i in [C2_T, C2_T+STATE_V)): → B_conv (conv+silu(B))
    C tokens (i in [C2_T+STATE_V, CCONV_T)): → C_conv (conv+silu(C))
    """
    assert K == 4
    line = np.zeros((K-1, CP), dtype=np.float32)
    for k in range(K-1):
        line[k, :] = conv_state_in[k].astype(np.float32)

    X_gate = np.zeros((C2_T, CP), dtype=np.float32)
    X_ssm  = np.zeros((C2_T, CP), dtype=np.float32)
    B_conv = np.zeros((STATE_V, CP), dtype=np.float32)  # ★ v2
    C_conv = np.zeros((STATE_V, CP), dtype=np.float32)  # ★ v2

    k0, k1, k2, k3 = [np.float32(kernel[i]) for i in range(K)]

    for i in range(CCONV_T):
        # ★ LightMamba-aligned XBC ordering: [B:STATE_V][C:STATE_V][x:C2_T]
        do_b  = (i < STATE_V)                              # B first
        do_c  = (i >= STATE_V) and (i < 2 * STATE_V)      # C second
        do_c2 = (i >= 2 * STATE_V)                         # x last

        if do_c2:
            # Gate aligned with x token index (i - 2*STATE_V)
            X_gate[i - 2 * STATE_V, :] = silu_fx(Z[i - 2 * STATE_V, :], accurate_math)

        x_in = XBC[i, :].astype(np.float32)

        window0 = line[2, :].copy()
        window1 = line[1, :].copy()
        window2 = line[0, :].copy()
        window3 = x_in

        line[2, :] = line[1, :]
        line[1, :] = line[0, :]
        line[0, :] = x_in

        s = k0 * window0 + k1 * window1 + k2 * window2 + k3 * window3
        ssm_out = silu_fx(s, accurate_math)

        if do_b:                                            # B first
            B_conv[i, :] = ssm_out
        elif do_c:                                          # C second
            C_conv[i - STATE_V, :] = ssm_out
        elif do_c2:                                         # x last
            X_ssm[i - 2 * STATE_V, :] = ssm_out

    conv_state_out = np.zeros((K-1, CP), dtype=np.float32)
    for k in range(K-1):
        conv_state_out[k, :] = line[k, :]
    return X_gate, X_ssm, B_conv, C_conv, conv_state_out


def dtadapt(DT, CH_T, C2_T, CP):
    out = np.zeros((C2_T, CP), dtype=np.float32)
    for j in range(C2_T):
        out[j, :] = DT[j % CH_T, :].astype(np.float32)
    return out


def delta_from_dt_path(DT, CH_T, C2_T, CP, accurate_math, pos_th, neg_th):
    DT_C2 = dtadapt(DT, CH_T, C2_T, CP)
    delta = softplus_fx(DT_C2, accurate_math, pos_th, neg_th)
    return delta.astype(np.float32)


def delta_from_wdelta_path(X_ssm, W_delta, C2_T, CP, accurate_math, pos_th, neg_th):
    delta = np.zeros((C2_T, CP), dtype=np.float32)
    for lane in range(CP):
        pre = (W_delta[:, :, lane].astype(np.float32) @ X_ssm[:, lane].astype(np.float32)).astype(np.float32)
        delta[:, lane] = softplus_fx(pre, accurate_math, pos_th, neg_th)
    return delta.astype(np.float32)


def stage3_dA(delta, A_fixed, STATE_V, C2_T, CP, accurate_math, exp_clamp):
    dA = np.zeros((STATE_V * C2_T, CP), dtype=np.float32)
    idx = 0
    for i in range(STATE_V):
        Avec = A_fixed[i, :].astype(np.float32)
        for j in range(C2_T):
            dlt = delta[j, :].astype(np.float32)
            dA[idx, :] = exp_fx(Avec * dlt, accurate_math, exp_clamp)
            idx += 1
    return dA


def stage45_update_reduce(X_ssm, delta, dA_stream, B, C, H0, STATE_V, C2_T, CP):
    H1  = np.zeros((STATE_V * C2_T, CP), dtype=np.float32)
    acc = np.zeros((C2_T, CP), dtype=np.float32)

    idx = 0
    for i in range(STATE_V):
        Bv = B[i, :].astype(np.float32)
        Cv = C[i, :].astype(np.float32)
        for j in range(C2_T):
            H0v  = H0[idx, :].astype(np.float32)
            dlt  = delta[j, :].astype(np.float32)
            xssm = X_ssm[j, :].astype(np.float32)
            dA   = dA_stream[idx, :].astype(np.float32)

            H1v = H0v * dA + (Bv * dlt) * xssm
            H1[idx, :] = H1v.astype(np.float32)
            acc[j, :] = (acc[j, :] + H1v * Cv).astype(np.float32)
            idx += 1

    htC = acc
    return htC.astype(np.float32), H1.astype(np.float32)


def stage6(htC, D_diag, X_ssm, X_gate, accurate_math=True):
    """v2: y = htC + D*x, then yz = y * gate (no silu on y — matches Scala YZ)."""
    y  = htC.astype(np.float32) + D_diag.astype(np.float32) * X_ssm.astype(np.float32)
    yz = y * X_gate.astype(np.float32)
    return yz.astype(np.float32)


def rmsnorm_vec_tokens_c2t(x_tokens, rms_weight_tokens, eps=1e-5):
    """v2: second RMSNorm on C2_T token stream (shape [C2_T, CP])."""
    x = x_tokens.astype(np.float32)
    w = rms_weight_tokens.astype(np.float32)
    ms = np.mean(x * x, dtype=np.float32)
    inv = 1.0 / np.sqrt(ms + eps, dtype=np.float32)
    y = x * inv * w
    return y.astype(np.float32)


# ★ LOW-RANK out-projection (two-stage)
def out_proj_lowrank(ssm_core_out, W_out_A, W_out_B, D_T, RANK_T, C2_T, CP):
    """
    Stage 1: temp[r,l] = Σ_j W_out_B[r,j,l] · ssm_core_out[j,l]
    Stage 2: Y[i,l]    = Σ_r W_out_A[i,r,l] · temp[r,l]
    """
    temp = np.zeros((RANK_T, CP), dtype=np.float32)
    for lane in range(CP):
        temp[:, lane] = W_out_B[:, :, lane].astype(np.float32) @ ssm_core_out[:, lane].astype(np.float32)

    Y = np.zeros((D_T, CP), dtype=np.float32)
    for lane in range(CP):
        Y[:, lane] = W_out_A[:, :, lane].astype(np.float32) @ temp[:, lane].astype(np.float32)

    return Y.astype(np.float32)


# =========================
# Main: generate bins_raw (LOW-RANK)
# =========================
def main(out_dir="bins_raw", ssmu_h="SSMU.h", seed=0, rank=1024, also_write_f32=False,
         accurate_math_override=None, enable_dt_override=None, delta_from_dt_override=None):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    shp, knobs = parse_macros_from_header(ssmu_h)

    if accurate_math_override is not None:
        knobs["SSMU_ACCURATE_MATH_CSIM"] = int(accurate_math_override)
    if enable_dt_override is not None:
        knobs["SSMU_ENABLE_DT"] = int(enable_dt_override)
    if delta_from_dt_override is not None:
        knobs["SSMU_DELTA_FROM_DT"] = int(delta_from_dt_override)

    CP      = shp["CP"]
    D_T     = shp["D_T"]
    C2_T    = shp["C2_T"]
    CCONV_T = shp["CCONV_T"]
    CH_T    = shp["CH_T"]
    CIN_T   = shp["CIN_T"]
    K       = shp["K"]
    STATE_V = shp["STATE_V"]
    H1_LEN  = shp["H1_LEN"]

    RANK = rank
    assert RANK % CP == 0, f"RANK={RANK} must be divisible by VEC_FACTOR={CP}"
    RANK_T = RANK // CP

    accurate_math = (knobs["SSMU_ACCURATE_MATH_CSIM"] != 0)
    enable_dt     = (knobs["SSMU_ENABLE_DT"] != 0)
    delta_from_dt = (knobs["SSMU_DELTA_FROM_DT"] != 0)
    pos_th        = float(knobs["SSMU_SOFTPLUS_POS_TH"])
    neg_th        = float(knobs["SSMU_SOFTPLUS_NEG_TH"])
    exp_clamp     = float(knobs["SSMU_EXP_CLAMP"])

    print("[PY] Shapes:", shp)
    print("[PY] Knobs :", knobs)
    print(f"[PY] LOW-RANK: RANK={RANK} RANK_T={RANK_T}")
    print(f"[PY] Mode  : accurate_math={int(accurate_math)} enable_dt={int(enable_dt)} delta_from_dt={int(delta_from_dt)}")
    print("[PY] out_dir:", out_dir)

    def rand_vec(n_tokens, scale=0.1):
        return (rng.standard_normal((n_tokens, CP)).astype(np.float32) * scale)

    # Inputs
    x_in           = rand_vec(D_T,    scale=0.2)
    h0_in          = rand_vec(H1_LEN, scale=0.2)
    conv_state_in  = rand_vec(K-1,    scale=0.2)
    kernel_in      = (rng.standard_normal((K,)).astype(np.float32) * 0.05)

    # Tables
    A_fixed     = rand_vec(STATE_V, scale=0.02)
    RMS_weight  = (np.abs(rand_vec(D_T, scale=0.5)) + 0.5).astype(np.float32)
    RMS_weight_2 = (np.abs(rand_vec(C2_T, scale=0.5)) + 0.5).astype(np.float32)  # ★ v2
    D_diag      = rand_vec(C2_T, scale=0.02)

    # ★ LOW-RANK Weights (float32) — generate as independent random matrices
    W_in_1  = (rng.standard_normal((D_T,    RANK_T, CP)).astype(np.float32) * 0.02)
    W_in_2  = (rng.standard_normal((RANK_T, CIN_T,  CP)).astype(np.float32) * 0.02)
    W_delta = (rng.standard_normal((C2_T,   C2_T,   CP)).astype(np.float32) * 0.02)
    W_out_A = (rng.standard_normal((D_T,    RANK_T, CP)).astype(np.float32) * 0.02)
    W_out_B = (rng.standard_normal((RANK_T, C2_T,   CP)).astype(np.float32) * 0.02)

    # ----------------------------
    # Write TB-consumed RAW bins (*.raw.bin) as int16 Q4.12
    # ----------------------------
    def W(path, arr_f32):
        write_raw_q412(os.path.join(out_dir, path), arr_f32)

    W("x_in_f32.raw.bin",            x_in)
    W("h0_in_f32.raw.bin",           h0_in)
    W("conv_state_in_f32.raw.bin",   conv_state_in)
    W("kernel_in_f32.raw.bin",       kernel_in)

    W("A_fixed_f32.raw.bin",         A_fixed)
    W("RMS_weight_f32.raw.bin",      RMS_weight)
    W("RMS_weight_2_f32.raw.bin",    RMS_weight_2)  # ★ v2: second RMSNorm
    W("D_diag_f32.raw.bin",          D_diag)

    # ★ LOW-RANK weight bins (4 sub-matrices + W_delta)
    W("W_in_1_f32.raw.bin",          W_in_1.reshape(-1))
    W("W_in_2_f32.raw.bin",          W_in_2.reshape(-1))
    W("W_delta_f32.raw.bin",         W_delta.reshape(-1))
    W("W_out_A_f32.raw.bin",         W_out_A.reshape(-1))
    W("W_out_B_f32.raw.bin",         W_out_B.reshape(-1))

    print(f"[PY] W_in_1:  {W_in_1.shape}  elems={W_in_1.size}")
    print(f"[PY] W_in_2:  {W_in_2.shape}  elems={W_in_2.size}")
    print(f"[PY] W_delta: {W_delta.shape}  elems={W_delta.size}")
    print(f"[PY] W_out_A: {W_out_A.shape}  elems={W_out_A.size}")
    print(f"[PY] W_out_B: {W_out_B.shape}  elems={W_out_B.size}")

    # ----------------------------
    # Golden reference compute (LOW-RANK)
    # ----------------------------
    X_residual = x_in.copy()
    X_normed   = rmsnorm_vec_tokens(x_in, RMS_weight, eps=1e-5)

    # ★ LOW-RANK inproj: X_normed @ W_in_1 @ W_in_2
    packed = inproj_lowrank(X_normed, W_in_1, W_in_2, D_T, RANK_T, CIN_T, CP)

    Z, XBC, DT = split_inproj(packed, C2_T, CCONV_T, CH_T)

    X_gate, X_ssm, B_conv, C_conv, conv_state_out = conv1d_silu_with_state(
        XBC, Z, kernel_in, conv_state_in,
        C2_T=C2_T, CCONV_T=CCONV_T, CP=CP, accurate_math=accurate_math, K=K, STATE_V=STATE_V
    )

    if enable_dt and delta_from_dt:
        delta = delta_from_dt_path(DT, CH_T, C2_T, CP, accurate_math, pos_th, neg_th)
    else:
        delta = delta_from_wdelta_path(X_ssm, W_delta, C2_T, CP, accurate_math, pos_th, neg_th)

    dA = stage3_dA(delta, A_fixed, STATE_V, C2_T, CP, accurate_math, exp_clamp)

    htC, H1_ref = stage45_update_reduce(
        X_ssm=X_ssm, delta=delta, dA_stream=dA,
        B=B_conv, C=C_conv, H0=h0_in,       # ★ v2: use conv'd B/C
        STATE_V=STATE_V, C2_T=C2_T, CP=CP
    )

    ssm_core_out = stage6(htC, D_diag, X_ssm, X_gate, accurate_math=accurate_math)

    # ★ v2: second RMSNorm between stage6 and out_proj
    ssm_normed = rmsnorm_vec_tokens_c2t(ssm_core_out, RMS_weight_2, eps=1e-5)

    # ★ LOW-RANK outproj: W_out_A @ (W_out_B @ ssm_normed)  [★ v2: uses normed output]
    out_proj_y = out_proj_lowrank(ssm_normed, W_out_A, W_out_B, D_T, RANK_T, C2_T, CP)
    out_ref    = (out_proj_y + X_residual).astype(np.float32)

    # Write references + auxiliary bins
    W("conv_state_out_f32.raw.bin", conv_state_out)

    W("w_scale_in_f32.raw.bin",    np.array([1.0], dtype=np.float32))
    W("w_scale_delta_f32.raw.bin", np.array([1.0], dtype=np.float32))
    W("w_scale_out_f32.raw.bin",   np.array([1.0], dtype=np.float32))

    W("out_ref_f32.raw.bin", out_ref)
    W("h1_ref_f32.raw.bin",  H1_ref)

    if also_write_f32:
        write_f32(os.path.join(out_dir, "out_ref_f32.bin"), out_ref)
        write_f32(os.path.join(out_dir, "h1_ref_f32.bin"),  H1_ref)
        write_f32(os.path.join(out_dir, "delta_f32.bin"), delta)
        write_f32(os.path.join(out_dir, "x_normed_f32.bin"), X_normed)

    print("[PY] wrote LOW-RANK TB bins (*.raw.bin int16 Q4.12) to:", out_dir)
    print("[PY] out_ref:", out_ref.shape, "h1_ref:", H1_ref.shape)
    print("[PY] delta  :", delta.shape, "conv_state_out:", conv_state_out.shape)

    # Summary: list all bin files
    bins = sorted(os.listdir(out_dir))
    print(f"[PY] Total {len(bins)} files in {out_dir}:")
    for b in bins:
        sz = os.path.getsize(os.path.join(out_dir, b))
        print(f"  {b:40s}  {sz:>10,d} bytes")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="bins_raw")
    ap.add_argument("--ssmu_h",  default="SSMU.h")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--rank",    type=int, default=1024,
                    help="Low-rank dimension (SSMU_RANK). Default 1024.")
    ap.add_argument("--also_write_f32", action="store_true")
    ap.add_argument("--accurate_math", type=int, choices=[0,1], default=None)
    ap.add_argument("--enable_dt", type=int, choices=[0,1], default=None)
    ap.add_argument("--delta_from_dt", type=int, choices=[0,1], default=None)

    args = ap.parse_args()
    main(out_dir=args.out_dir,
         ssmu_h=args.ssmu_h,
         seed=args.seed,
         rank=args.rank,
         also_write_f32=args.also_write_f32,
         accurate_math_override=args.accurate_math,
         enable_dt_override=args.enable_dt,
         delta_from_dt_override=args.delta_from_dt)
