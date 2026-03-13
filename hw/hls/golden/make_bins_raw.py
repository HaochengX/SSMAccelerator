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
# Q4.12 representable float range is about [-8.0, 7.999755...]
# We'll clip to int16 range after scaling.

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
        # allow only safe-ish chars (names/operators/parens)
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

    # Iteratively resolve numeric macros
    for _ in range(80):
        progress = False
        # substitute known ints/floats into unresolved expressions
        for k, v in list(macros.items()):
            if isinstance(v, (int, float)):
                continue
            if not isinstance(v, str):
                continue
            expr = v
            # replace already-resolved numeric macros
            for kk, vv in macros.items():
                if isinstance(vv, (int, float)):
                    expr = re.sub(rf"\b{re.escape(kk)}\b", str(vv), expr)
            val = try_eval(expr)
            if isinstance(val, (int, float, np.integer, np.floating)):
                # prefer int when it is integral
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
        # last attempt: substitute + eval
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

    # ---- required shapes
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

    # ---- knobs (match ssm.cpp defaults)
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
# layers.h-aligned math (match ssm.cpp)
# =========================
def sigmoid_fx(x: np.ndarray, accurate: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if accurate:
        return (1.0 / (1.0 + np.exp(-x, dtype=np.float32))).astype(np.float32)
    # approx: half + qtr*x, clamp [0,1]
    y = (0.5 + 0.25 * x).astype(np.float32)
    return np.clip(y, 0.0, 1.0).astype(np.float32)

def silu_fx(x: np.ndarray, accurate: bool) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x * sigmoid_fx(x, accurate)).astype(np.float32)

def softplus_fx(x: np.ndarray, accurate: bool, pos_th: float, neg_th: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if accurate:
        # ssm.cpp accurate:
        # if x>pos_th => x; if x<neg_th => 0; else log1p(exp(x))
        y = np.empty_like(x, dtype=np.float32)
        y[x > pos_th] = x[x > pos_th]
        y[x < neg_th] = 0.0
        mid = (x <= pos_th) & (x >= neg_th)
        y[mid] = np.log1p(np.exp(x[mid], dtype=np.float32)).astype(np.float32)
        return y.astype(np.float32)
    # approx: if x>TH => x; if x<NTH => 0; else half*x + 1
    y = np.empty_like(x, dtype=np.float32)
    y[x > pos_th] = x[x > pos_th]
    y[x < neg_th] = 0.0
    mid = (x <= pos_th) & (x >= neg_th)
    y[mid] = (0.5 * x[mid] + 1.0).astype(np.float32)
    return y.astype(np.float32)

def exp_fx(x: np.ndarray, accurate: bool, exp_clamp: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    t = np.clip(x, -exp_clamp, exp_clamp).astype(np.float32)
    if accurate:
        return np.exp(t, dtype=np.float32).astype(np.float32)
    # approx: 1 + t + 0.5*t^2; clamp >=0
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

def inproj_pack(x_normed, W_inproj, D_T, CIN_T, CP):
    out = np.zeros((CIN_T, CP), dtype=np.float32)
    # W_inproj is (D_T, CIN_T, CP)
    for lane in range(CP):
        out[:, lane] = x_normed[:, lane].astype(np.float32) @ W_inproj[:, :, lane].astype(np.float32)
    return out

def split_inproj(packed, C2_T, CCONV_T, CH_T, STATE_V):
    baseZ  = 0
    baseX  = C2_T
    baseDT = C2_T + CCONV_T
    baseB  = C2_T + CCONV_T + CH_T
    baseC  = C2_T + CCONV_T + CH_T + STATE_V
    Z   = packed[baseZ:baseZ + C2_T]
    XBC = packed[baseX:baseX + CCONV_T]
    DT  = packed[baseDT:baseDT + CH_T]
    B   = packed[baseB:baseB + STATE_V]
    C   = packed[baseC:baseC + STATE_V]
    return Z, XBC, DT, B, C

def conv1d_silu_with_state(XBC, Z, kernel, conv_state_in, C2_T, CCONV_T, CP, accurate_math, K=4):
    assert K == 4
    line = np.zeros((K-1, CP), dtype=np.float32)
    for k in range(K-1):
        line[k, :] = conv_state_in[k].astype(np.float32)

    X_gate = np.zeros((C2_T, CP), dtype=np.float32)
    X_ssm  = np.zeros((C2_T, CP), dtype=np.float32)

    k0, k1, k2, k3 = [np.float32(kernel[i]) for i in range(K)]

    for i in range(CCONV_T):
        do_c2 = (i < C2_T)
        if do_c2:
            X_gate[i, :] = silu_fx(Z[i, :], accurate_math)

        x_in = XBC[i, :].astype(np.float32)

        window0 = line[2, :].copy()
        window1 = line[1, :].copy()
        window2 = line[0, :].copy()
        window3 = x_in

        line[2, :] = line[1, :]
        line[1, :] = line[0, :]
        line[0, :] = x_in

        if do_c2:
            s = k0 * window0 + k1 * window1 + k2 * window2 + k3 * window3
            X_ssm[i, :] = silu_fx(s, accurate_math)

    conv_state_out = np.zeros((K-1, CP), dtype=np.float32)
    for k in range(K-1):
        conv_state_out[k, :] = line[k, :]
    return X_gate, X_ssm, conv_state_out

def dtadapt(DT, CH_T, C2_T, CP):
    # DT shape: (CH_T, CP) -> map to (C2_T, CP) by modulo, matching ssm.cpp dtadapt_stream_local()
    out = np.zeros((C2_T, CP), dtype=np.float32)
    for j in range(C2_T):
        out[j, :] = DT[j % CH_T, :].astype(np.float32)
    return out

def delta_from_dt_path(DT, CH_T, C2_T, CP, accurate_math, pos_th, neg_th):
    DT_C2 = dtadapt(DT, CH_T, C2_T, CP)
    delta = softplus_fx(DT_C2, accurate_math, pos_th, neg_th)
    return delta.astype(np.float32)

def delta_from_wdelta_path(X_ssm, W_delta, C2_T, CP, accurate_math, pos_th, neg_th):
    # X_ssm: (C2_T, CP)
    # W_delta: (C2_T, C2_T, CP)  (per-lane matmul)
    delta = np.zeros((C2_T, CP), dtype=np.float32)
    for lane in range(CP):
        # W_delta[:,:,lane] @ X_ssm[:,lane]
        pre = (W_delta[:, :, lane].astype(np.float32) @ X_ssm[:, lane].astype(np.float32)).astype(np.float32)
        delta[:, lane] = softplus_fx(pre, accurate_math, pos_th, neg_th)
    return delta.astype(np.float32)

def stage3_dA(delta, A_fixed, STATE_V, C2_T, CP, accurate_math, exp_clamp):
    # dA[idx] where idx = i*C2_T + j
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
    y = htC.astype(np.float32) + D_diag.astype(np.float32) * X_ssm.astype(np.float32)
    # ✅ FIX: ssm.cpp applies uD_node = silu(y) before gating
    u = silu_fx(y, accurate_math)
    yz = u * X_gate.astype(np.float32)
    return yz.astype(np.float32)

def out_proj(ssm_core_out, W_out, D_T, C2_T, CP):
    Y = np.zeros((D_T, CP), dtype=np.float32)
    for lane in range(CP):
        Y[:, lane] = W_out[:, :, lane].astype(np.float32) @ ssm_core_out[:, lane].astype(np.float32)
    return Y.astype(np.float32)

# =========================
# Main: generate bins_raw
# =========================
def main(out_dir="bins_raw", ssmu_h="SSMU.h", seed=0, also_write_f32=False,
         accurate_math_override=None, enable_dt_override=None, delta_from_dt_override=None):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    shp, knobs = parse_macros_from_header(ssmu_h)

    # apply CLI overrides (if provided)
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

    accurate_math = (knobs["SSMU_ACCURATE_MATH_CSIM"] != 0)
    enable_dt     = (knobs["SSMU_ENABLE_DT"] != 0)
    delta_from_dt = (knobs["SSMU_DELTA_FROM_DT"] != 0)
    pos_th        = float(knobs["SSMU_SOFTPLUS_POS_TH"])
    neg_th        = float(knobs["SSMU_SOFTPLUS_NEG_TH"])
    exp_clamp     = float(knobs["SSMU_EXP_CLAMP"])

    print("[PY] Shapes:", shp)
    print("[PY] Knobs :", knobs)
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
    A_fixed     = rand_vec(STATE_V, scale=0.02)  # STATE_V tokens
    RMS_weight  = (np.abs(rand_vec(D_T, scale=0.5)) + 0.5).astype(np.float32)
    D_diag      = rand_vec(C2_T, scale=0.02)

    # Weights (float32)
    W_inproj = (rng.standard_normal((D_T, CIN_T, CP)).astype(np.float32) * 0.02)
    W_delta  = (rng.standard_normal((C2_T, C2_T, CP)).astype(np.float32) * 0.02)
    W_out    = (rng.standard_normal((D_T, C2_T, CP)).astype(np.float32) * 0.02)

    # ----------------------------
    # Write TB-consumed RAW bins (*.raw.bin) as int16 Q4.12
    # ----------------------------
    def W(path, arr_f32): write_raw_q412(os.path.join(out_dir, path), arr_f32)

    W("x_in_f32.raw.bin",            x_in)                 # (D_T, CP)
    W("h0_in_f32.raw.bin",           h0_in)                # (H1_LEN, CP)
    W("conv_state_in_f32.raw.bin",   conv_state_in)        # (K-1, CP)
    W("kernel_in_f32.raw.bin",       kernel_in)            # (K,)

    W("A_fixed_f32.raw.bin",         A_fixed)              # (STATE_V, CP)
    W("RMS_weight_f32.raw.bin",      RMS_weight)           # (D_T, CP)
    W("D_diag_f32.raw.bin",          D_diag)               # (C2_T, CP)

    W("W_inproj_f32.raw.bin",        W_inproj.reshape(-1)) # flatten
    W("W_delta_f32.raw.bin",         W_delta.reshape(-1))
    W("W_out_f32.raw.bin",           W_out.reshape(-1))

    # ----------------------------
    # Golden reference compute (float32)
    # ----------------------------
    X_residual = x_in.copy()
    X_normed   = rmsnorm_vec_tokens(x_in, RMS_weight, eps=1e-5)

    packed = inproj_pack(X_normed, W_inproj, D_T, CIN_T, CP)
    Z, XBC, DT, B, C = split_inproj(packed, C2_T, CCONV_T, CH_T, STATE_V)

    X_gate, X_ssm, conv_state_out = conv1d_silu_with_state(
        XBC, Z, kernel_in, conv_state_in,
        C2_T=C2_T, CCONV_T=CCONV_T, CP=CP, accurate_math=accurate_math, K=K
    )

    # delta selection must match ssm.cpp:
    if enable_dt and delta_from_dt:
        delta = delta_from_dt_path(DT, CH_T, C2_T, CP, accurate_math, pos_th, neg_th)
    else:
        delta = delta_from_wdelta_path(X_ssm, W_delta, C2_T, CP, accurate_math, pos_th, neg_th)

    dA    = stage3_dA(delta, A_fixed, STATE_V, C2_T, CP, accurate_math, exp_clamp)

    htC, H1_ref = stage45_update_reduce(
        X_ssm=X_ssm, delta=delta, dA_stream=dA,
        B=B, C=C, H0=h0_in,
        STATE_V=STATE_V, C2_T=C2_T, CP=CP
    )

    ssm_core_out = stage6(htC, D_diag, X_ssm, X_gate, accurate_math=accurate_math)
    out_proj_y   = out_proj(ssm_core_out, W_out, D_T, C2_T, CP)
    out_ref      = (out_proj_y + X_residual).astype(np.float32)

    # ✅ Write conv_state_out reference
    W("conv_state_out_f32.raw.bin", conv_state_out)

    # ✅ Write w_scale bins (Q4.12 single element) for TB loader
    W("w_scale_in_f32.raw.bin",    np.array([1.0], dtype=np.float32))
    W("w_scale_delta_f32.raw.bin", np.array([1.0], dtype=np.float32))
    W("w_scale_out_f32.raw.bin",   np.array([1.0], dtype=np.float32))

    # Write refs as RAW too (so TB can compare reading int16->DTYPE)
    W("out_ref_f32.raw.bin", out_ref)
    W("h1_ref_f32.raw.bin",  H1_ref)

    if also_write_f32:
        write_f32(os.path.join(out_dir, "out_ref_f32.bin"), out_ref)
        write_f32(os.path.join(out_dir, "h1_ref_f32.bin"),  H1_ref)
        # optional debug dumps
        write_f32(os.path.join(out_dir, "delta_f32.bin"), delta)
        write_f32(os.path.join(out_dir, "x_normed_f32.bin"), X_normed)
        write_f32(os.path.join(out_dir, "x_gate_f32.bin"), X_gate)
        write_f32(os.path.join(out_dir, "x_ssm_f32.bin"), X_ssm)

    print("[PY] wrote TB bins (*.raw.bin int16 Q4.12) to:", out_dir)
    print("[PY] out_ref:", out_ref.shape, "h1_ref:", H1_ref.shape)
    print("[PY] delta  :", delta.shape, "conv_state_out:", conv_state_out.shape)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="bins_raw")
    ap.add_argument("--ssmu_h",  default="SSMU.h")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--also_write_f32", action="store_true")

    # Optional overrides (leave unset to follow SSMU.h / defaults)
    ap.add_argument("--accurate_math", type=int, choices=[0,1], default=None,
                    help="Override SSMU_ACCURATE_MATH_CSIM (0=approx, 1=accurate).")
    ap.add_argument("--enable_dt", type=int, choices=[0,1], default=None,
                    help="Override SSMU_ENABLE_DT (0/1).")
    ap.add_argument("--delta_from_dt", type=int, choices=[0,1], default=None,
                    help="Override SSMU_DELTA_FROM_DT (0/1).")

    args = ap.parse_args()
    main(out_dir=args.out_dir,
         ssmu_h=args.ssmu_h,
         seed=args.seed,
         also_write_f32=args.also_write_f32,
         accurate_math_override=args.accurate_math,
         enable_dt_override=args.enable_dt,
         delta_from_dt_override=args.delta_from_dt)
