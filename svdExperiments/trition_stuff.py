import argparse
import time

import torch
import triton
import triton.language as tl


# -----------------------------
# Helpers
# -----------------------------
def bench(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def torch_two_gemm(x, A, B):
    # (M,K) @ (K,R) -> (M,R) then (M,R) @ (R,N) -> (M,N)
    return (x @ A.t()) @ B.t()


def torch_dense_what(x, A, B):
    # W_hat = (N,R) @ (R,K) = (N,K), then x @ W_hat^T
    W_hat = B @ A
    return x @ W_hat.t()


def torch_dense_full(x, W):
    return x @ W.t()


# -----------------------------
# Triton fused kernel (group N)
# C = X @ A.T @ B.T
# where:
#   X:   (M,K)
#   A_T: (K,R)  (pretransposed, contiguous)
#   B_T: (R,N)  (pretransposed, contiguous)
#   C:   (M,N)
# -----------------------------
@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_R": 32, "GROUP_N": 8, "BLOCK_NG": 256},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "BLOCK_R": 32, "GROUP_N": 4, "BLOCK_NG": 256},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 32, "BLOCK_K": 32, "BLOCK_R": 32, "GROUP_N": 8, "BLOCK_NG": 256},
            num_warps=8,
            num_stages=5,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_K": 64, "BLOCK_R": 32, "GROUP_N": 8, "BLOCK_NG": 256},
            num_warps=8,
            num_stages=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_R": 32, "GROUP_N": 4, "BLOCK_NG": 256},
            num_warps=8,
            num_stages=4,
        ),
    ],
    key=["M", "N", "K", "R", "FAST_MATH"],
)
@triton.jit
def lowrank_fused_kernel_groupn(
    X, A_T, B_T, C,
    M: tl.constexpr, N: tl.constexpr, K, R,
    stride_xm, stride_xk,
    stride_atk, stride_atr,      # A_T is (K,R)
    stride_br, stride_bn,        # B_T is (R,N)
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    FAST_MATH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_NG: tl.constexpr,      # = BLOCK_N * GROUP_N
):
    pid_m = tl.program_id(0)
    pid_ng = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_ng * BLOCK_NG + tl.arange(0, BLOCK_NG)

    acc = tl.zeros((BLOCK_M, BLOCK_NG), dtype=tl.float32)

    r_tiles = tl.cdiv(R, BLOCK_R)
    k_tiles = tl.cdiv(K, BLOCK_K)

    # Loop over R tiles (runtime loop)
    for r_iter in range(0, r_tiles):
        r0 = r_iter * BLOCK_R
        offs_r = r0 + tl.arange(0, BLOCK_R)

        tmp = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

        # Loop over K tiles (runtime loop)
        for k_iter in range(0, k_tiles):
            k0 = k_iter * BLOCK_K
            offs_k = k0 + tl.arange(0, BLOCK_K)

            x = tl.load(
                X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
                other=0.0,
            )

            a = tl.load(
                A_T + offs_k[:, None] * stride_atk + offs_r[None, :] * stride_atr,
                mask=(offs_k[:, None] < K) & (offs_r[None, :] < R),
                other=0.0,
            )

            if FAST_MATH:
                tmp += tl.dot(x, a, out_dtype=tl.float32)
            else:
                tmp += tl.dot(x.to(tl.float32), a.to(tl.float32))

        b = tl.load(
            B_T + offs_r[:, None] * stride_br + offs_n[None, :] * stride_bn,
            mask=(offs_r[:, None] < R) & (offs_n[None, :] < N),
            other=0.0,
        )

        if FAST_MATH:
            acc += tl.dot(tmp.to(b.dtype), b, out_dtype=tl.float32)
        else:
            acc += tl.dot(tmp, b.to(tl.float32))

    tl.store(
        C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        acc.to(OUT_DTYPE),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def fused_lowrank_triton_preT(x: torch.Tensor, A_T: torch.Tensor, B_T: torch.Tensor, fast_math: bool):
    """
    Compute C = x @ A_T @ B_T
    x:   (M, K) contiguous
    A_T: (K, R) contiguous
    B_T: (R, N) contiguous
    """
    assert x.ndim == 2 and A_T.ndim == 2 and B_T.ndim == 2
    M, K = x.shape
    K2, R = A_T.shape
    R2, N = B_T.shape
    assert K == K2 and R == R2
    assert x.is_contiguous()
    assert A_T.is_contiguous()
    assert B_T.is_contiguous()

    C = torch.empty((M, N), device=x.device, dtype=x.dtype)

    if x.dtype == torch.float16:
        out_dtype = tl.float16
    elif x.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16
    else:
        out_dtype = tl.float32

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_NG"]),
    )

    lowrank_fused_kernel_groupn[grid](
        x, A_T, B_T, C,
        M, N, K, R,
        x.stride(0), x.stride(1),
        A_T.stride(0), A_T.stride(1),
        B_T.stride(0), B_T.stride(1),
        C.stride(0), C.stride(1),
        OUT_DTYPE=out_dtype,
        FAST_MATH=fast_math,
    )
    return C


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=2048)
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--n", type=int, default=2560)
    p.add_argument("--r", type=int, default=1536)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    torch.manual_seed(0)
    x = torch.randn(args.m, args.k, device="cuda", dtype=dtype).contiguous()
    W = torch.randn(args.n, args.k, device="cuda", dtype=dtype)

    # Build low-rank factors from SVD of W
    with torch.no_grad():
        Wf = W.float()
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        r = min(args.r, S.numel())
        U = U[:, :r]
        S = S[:r]
        Vh = Vh[:r, :]

        A = Vh.to(dtype=dtype).contiguous()                      # (R, K)
        B = (U * S.unsqueeze(0)).to(dtype=dtype).contiguous()    # (N, R)

        # IMPORTANT: pretranspose ONCE (this was your timing bug)
        A_T = A.t().contiguous()  # (K, R)
        B_T = B.t().contiguous()  # (R, N)

    # Correctness (FP32 reference)
    ref = torch_two_gemm(x.float(), A.float(), B.float())
    fused = fused_lowrank_triton_preT(x, A_T, B_T, fast_math=args.fast).float()
    max_err = (fused - ref).abs().max().item()
    rel_err = max_err / (ref.abs().max().item() + 1e-8)
    print(f"max error vs FP32 ref: {max_err:.6e} (rel {rel_err:.6e})")

    # Benchmarks (now fair: no transpose inside fused)
    t_two = bench(lambda: torch_two_gemm(x, A, B), iters=args.iters)
    t_dense = bench(lambda: torch_dense_what(x, A, B), iters=args.iters)
    t_full = bench(lambda: torch_dense_full(x, W), iters=args.iters)
    t_fused = bench(lambda: fused_lowrank_triton_preT(x, A_T, B_T, fast_math=args.fast), iters=args.iters)

    print(f"2xGEMM:      {t_two*1e3:.3f} ms")
    print(f"Dense W_hat: {t_dense*1e3:.3f} ms")
    print(f"Dense Full W:{t_full*1e3:.3f} ms")
    print(f"Fused Triton:{t_fused*1e3:.3f} ms")


if __name__ == "__main__":
    main()
