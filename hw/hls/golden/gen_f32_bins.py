import os
import numpy as np

def write_f32(path, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float32).ravel()
    with open(path, "wb") as f:
        f.write(arr.tobytes(order="C"))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def silu(x):
    return x * sigmoid(x)

def softplus(x):
    # numpy stable-ish softplus
    # softplus(x) = log1p(exp(x))
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def rmsnorm_vec_tokens(x_tokens, rms_weight_tokens, eps=1e-5):
    """
    x_tokens:        (D_T, CP)
    rms_weight:      (D_T, CP)
    matches your ssm.cpp RMSNorm: global mean square across all lanes+tokens.
    """
    x = x_tokens.astype(np.float32)
    w = rms_weight_tokens.astype(np.float32)

    ms = np.mean(x * x, dtype=np.float32)
    inv = 1.0 / np.sqrt(ms + eps, dtype=np.float32)
    y = x * inv * w
    return y.astype(np.float32)

def inproj_pack(x_normed, W_inproj, D_T, CIN_T, CP):
    """
    lane-wise dot over D_T:
      out[i, lane] = sum_j x_normed[j,lane] * W_inproj[j,i,lane]
    returns packed tokens: (CIN_T, CP)
    """
    # x_normed: (D_T, CP)
    # W_inproj: (D_T, CIN_T, CP)
    out = np.zeros((CIN_T, CP), dtype=np.float32)

    # do per-lane GEMV: (D_T) dot (D_T,CIN_T) -> (CIN_T)
    for lane in range(CP):
        # (D_T,) @ (D_T, CIN_T) = (CIN_T,)
        out[:, lane] = x_normed[:, lane].astype(np.float32) @ W_inproj[:, :, lane].astype(np.float32)
    return out

def split_inproj(packed, C2_T, CCONV_T, CH_T, STATE_T):
    """
    layout (token units):
      [ Z: C2_T ] [ XBC: CCONV_T ] [ DT: CH_T ] [ B: STATE_T ] [ C: STATE_T ]
    """
    baseZ  = 0
    baseX  = C2_T
    baseDT = C2_T + CCONV_T
    baseB  = C2_T + CCONV_T + CH_T
    baseC  = C2_T + CCONV_T + CH_T + STATE_T

    Z   = packed[baseZ:baseZ + C2_T]
    XBC = packed[baseX:baseX + CCONV_T]
    DT  = packed[baseDT:baseDT + CH_T]
    B   = packed[baseB:baseB + STATE_T]
    C   = packed[baseC:baseC + STATE_T]
    return Z, XBC, DT, B, C

def conv1d_silu_with_state(XBC, Z, kernel, conv_state_in, C2_T, CCONV_T, CP, K=4):
    """
    Matches your ssm.cpp conv1d_silu_stream_local_with_state (K=4)
    - line_buffer[0]=most recent, line_buffer[2]=oldest
    - init line_buffer[k] from conv_state_in[k] for k=0..2
    - for i in 0..CCONV_T-1:
        if i < C2_T: gate_out = silu(Z[i])
        window0=line_buffer[2], window1=line_buffer[1], window2=line_buffer[0], window3=XBC[i]
        shift buffers
        if i < C2_T: X_ssm[i] = silu(sum_k kernel[k]*windowk)
    - output conv_state_out tokens are line_buffer[k] for k=0..2 (same order as code writes)
    """
    assert K == 4
    # line_buffer shape (3, CP)
    line = np.zeros((K-1, CP), dtype=np.float32)

    # load conv_state_in: k=0..2
    for k in range(K-1):
        line[k, :] = conv_state_in[k].astype(np.float32)

    X_gate = np.zeros((C2_T, CP), dtype=np.float32)
    X_ssm  = np.zeros((C2_T, CP), dtype=np.float32)

    k0, k1, k2, k3 = [np.float32(kernel[i]) for i in range(K)]

    for i in range(CCONV_T):
        do_c2 = (i < C2_T)

        if do_c2:
            X_gate[i, :] = silu(Z[i, :].astype(np.float32)).astype(np.float32)

        x_in = XBC[i, :].astype(np.float32)

        window0 = line[2, :].copy()
        window1 = line[1, :].copy()
        window2 = line[0, :].copy()
        window3 = x_in

        # shift
        line[2, :] = line[1, :]
        line[1, :] = line[0, :]
        line[0, :] = x_in

        if do_c2:
            s = k0 * window0 + k1 * window1 + k2 * window2 + k3 * window3
            X_ssm[i, :] = silu(s).astype(np.float32)

    # write conv_state_out in order k=0..2
    conv_state_out = np.zeros((K-1, CP), dtype=np.float32)
    for k in range(K-1):
        conv_state_out[k, :] = line[k, :]

    return X_gate, X_ssm, conv_state_out

def dt_to_delta_DT_only(DT, CH_T, C2_T, CP):
    """
    Matches your current DT-only branch when CH_T != C2_T:
      drain CH_T tokens, then pad C2_T zeros, then softplus.
    => delta becomes all softplus(0)=~0.693?  BUT in your ssm.cpp, they pad *DTYPE_VEC zeros*
       then softplus_fx(x) where x=0 => log1p(exp(0)) = 0.693...
       HOWEVER in your ssm.cpp softplus_fx has special approx; in CSIM accurate math returns 0.693...
       In your posted ssm.cpp: if xf < neg_th -> 0; else log1p(exp(xf)).
       At x=0 => log1p(1)=0.693.
    But your code path: they drain DT then pad zero vectors into DT_C2_stream, then softplus => 0.693.
    """
    # drain DT (not used)
    _ = DT  # consumed conceptually

    zeros = np.zeros((C2_T, CP), dtype=np.float32)
    delta = softplus(zeros).astype(np.float32)
    return delta

def stage3_dA(delta, A_fixed, STATE_T, C2_T, CP):
    """
    dA_out token stream is STATE_T * C2_T tokens in order:
      for i in 0..STATE_T-1:
        for j in 0..C2_T-1:
          dA = exp(A_fixed[i] * delta[j]) lane-wise
    """
    dA = np.zeros((STATE_T * C2_T, CP), dtype=np.float32)
    idx = 0
    for i in range(STATE_T):
        Avec = A_fixed[i, :].astype(np.float32)
        for j in range(C2_T):
            dlt = delta[j, :].astype(np.float32)
            dA[idx, :] = np.exp(Avec * dlt).astype(np.float32)
            idx += 1
    return dA

def stage45_update_reduce(X_ssm, delta, dA_stream, B, C, H0, STATE_T, C2_T, CP):
    """
    Matches your stage45_update_reduce_local
    Inputs:
      X_ssm:   (C2_T, CP)
      delta:   (C2_T, CP)
      dA:      (STATE_T*C2_T, CP) in same order
      B:       (STATE_T, CP)
      C:       (STATE_T, CP)
      H0:      (STATE_T*C2_T, CP)  (H0_in tokens)
    Outputs:
      htC:     (C2_T, CP)
      H1_out:  (STATE_T*C2_T, CP) in produced order
    """
    htC = np.zeros((C2_T, CP), dtype=np.float32)
    H1  = np.zeros((STATE_T * C2_T, CP), dtype=np.float32)

    # accumulate per j
    acc = np.zeros((C2_T, CP), dtype=np.float32)

    idx = 0
    for i in range(STATE_T):
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

    htC[:, :] = acc
    return htC, H1

def stage6(htC, D_diag, X_ssm, X_gate, C2_T, CP):
    """
    out = (htC + D_diag * X_ssm) * X_gate
    token-wise lane-wise
    """
    y = htC.astype(np.float32) + D_diag.astype(np.float32) * X_ssm.astype(np.float32)
    yz = y * X_gate.astype(np.float32)
    return yz.astype(np.float32)

def out_proj(ssm_core_out, W_out, D_T, C2_T, CP):
    """
    lane-wise dot over C2_T:
      Y[i,lane] = sum_j ssm_core_out[j,lane] * W_out[i,j,lane]
    """
    Y = np.zeros((D_T, CP), dtype=np.float32)
    for lane in range(CP):
        # (D_T, C2_T) @ (C2_T,) via dot on last dim:
        # equivalent: Y[:,lane] = W_out[:,:,lane] @ x[:,lane]
        Y[:, lane] = W_out[:, :, lane].astype(np.float32) @ ssm_core_out[:, lane].astype(np.float32)
    return Y

def main(out_dir="bins_f32", seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- Shapes from your SSMU.h ----
    CP = 8
    D_T = 320
    C2_T = 640
    CCONV_T = 672
    CH_T = 10
    STATE_T = 16
    CIN_T = 1354
    K = 4
    H1_LEN = STATE_T * C2_T  # 10240

    # Helper: random vec tokens
    def rand_vec(n_tokens, scale=0.1):
        return (rng.standard_normal((n_tokens, CP)).astype(np.float32) * scale)

    # Inputs
    x_in = rand_vec(D_T, scale=0.2)
    h0_in = rand_vec(H1_LEN, scale=0.2)
    conv_state_in = rand_vec(K-1, scale=0.2)
    kernel_in = (rng.standard_normal((K,)).astype(np.float32) * 0.05)

    # Tables
    A_fixed = rand_vec(STATE_T, scale=0.02)
    RMS_weight = (np.abs(rand_vec(D_T, scale=0.5)) + 0.5).astype(np.float32)
    D_diag = rand_vec(C2_T, scale=0.02)

    # Weights (float32)
    W_inproj = (rng.standard_normal((D_T, CIN_T, CP)).astype(np.float32) * 0.02)
    W_delta  = (rng.standard_normal((C2_T, C2_T, CP)).astype(np.float32) * 0.02)  # unused in DT-only CH!=C2 path
    W_out    = (rng.standard_normal((D_T, C2_T, CP)).astype(np.float32) * 0.02)

    # ----------------------------
    # Write input/weights bins
    # ----------------------------
    write_f32(os.path.join(out_dir, "x_in_f32.bin"), x_in)
    write_f32(os.path.join(out_dir, "h0_in_f32.bin"), h0_in)
    write_f32(os.path.join(out_dir, "conv_state_in_f32.bin"), conv_state_in)
    write_f32(os.path.join(out_dir, "kernel_in_f32.bin"), kernel_in)

    write_f32(os.path.join(out_dir, "A_fixed_f32.bin"), A_fixed)
    write_f32(os.path.join(out_dir, "RMS_weight_f32.bin"), RMS_weight)
    write_f32(os.path.join(out_dir, "D_diag_f32.bin"), D_diag)

    write_f32(os.path.join(out_dir, "W_inproj_f32.bin"), W_inproj.reshape(-1))
    write_f32(os.path.join(out_dir, "W_delta_f32.bin"),  W_delta.reshape(-1))
    write_f32(os.path.join(out_dir, "W_out_f32.bin"),    W_out.reshape(-1))

    # ----------------------------
    # ✅ GOLDEN reference compute
    # ----------------------------
    # 1) residual copy
    X_residual = x_in.copy()

    # 2) RMSNorm
    X_normed = rmsnorm_vec_tokens(x_in, RMS_weight, eps=1e-5)

    # 3) in_proj pack and split
    packed = inproj_pack(X_normed, W_inproj, D_T, CIN_T, CP)
    Z, XBC, DT, B, C = split_inproj(packed, C2_T, CCONV_T, CH_T, STATE_T)

    # 4) conv+silu with conv_state
    X_gate, X_ssm, conv_state_out = conv1d_silu_with_state(
        XBC, Z, kernel_in, conv_state_in,
        C2_T=C2_T, CCONV_T=CCONV_T, CP=CP, K=K
    )

    # 5) DT-only (your current CH_T != C2_T behavior)
    delta = dt_to_delta_DT_only(DT, CH_T, C2_T, CP)

    # 6) stage3 dA
    dA = stage3_dA(delta, A_fixed, STATE_T, C2_T, CP)

    # 7) stage45 scan/update + reduce
    htC, H1_ref = stage45_update_reduce(
        X_ssm=X_ssm,
        delta=delta,
        dA_stream=dA,
        B=B,
        C=C,
        H0=h0_in,
        STATE_T=STATE_T,
        C2_T=C2_T,
        CP=CP
    )

    # 8) stage6
    ssm_core_out = stage6(htC, D_diag, X_ssm, X_gate, C2_T, CP)

    # 9) out_proj + residual
    out_proj_y = out_proj(ssm_core_out, W_out, D_T, C2_T, CP)
    out_ref = (out_proj_y + X_residual).astype(np.float32)

    # ----------------------------
    # ✅ Write golden outputs
    # ----------------------------
    write_f32(os.path.join(out_dir, "out_ref_f32.bin"), out_ref)
    write_f32(os.path.join(out_dir, "h1_ref_f32.bin"),  H1_ref)

    print("[PY] wrote float32 bins to:", out_dir)
    print("[PY] wrote golden outputs:")
    print("     - out_ref_f32.bin   (D_T x CP) =", out_ref.shape)
    print("     - h1_ref_f32.bin    (STATE_T*C2_T x CP) =", H1_ref.shape)
    print("     Next: pack_dtypes to create bins_raw, then TB dump out_dut, then compare vs out_ref/h1_ref.")

if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "bins_f32"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(out_dir=out_dir, seed=seed)
