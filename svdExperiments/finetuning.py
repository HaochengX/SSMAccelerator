import os
import math
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def layer_indices(mode: str, n_layers: int) -> Set[int]:
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(0, n_layers // 3))
    if mode == "mid":
        return set(range(n_layers // 3, 2 * n_layers // 3))
    if mode == "late":
        return set(range(2 * n_layers // 3, n_layers))
    if mode == "early_mid":
        return set(range(0, 2 * n_layers // 3))
    if mode == "mid_late":
        return set(range(n_layers // 3, n_layers))
    raise ValueError(mode)


def _quantize_symmetric_lastdim(x: torch.Tensor, qmax: int, group_size: int) -> torch.Tensor:
    if x.numel() == 0:
        return x

    out_dtype = x.dtype
    x_fp = x.float()
    orig_shape = x_fp.shape
    x2 = x_fp.reshape(-1, orig_shape[-1])
    n = x2.shape[1]
    g = n if group_size <= 0 else group_size

    n_groups = (n + g - 1) // g
    pad = n_groups * g - n
    if pad > 0:
        x2 = F.pad(x2, (0, pad))

    xg = x2.view(-1, n_groups, g)
    max_val = xg.abs().amax(dim=2, keepdim=True)
    scale = max_val / float(qmax)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = torch.round(xg / scale).clamp(-qmax, qmax)
    xg_deq = q * scale

    x2_deq = xg_deq.view(-1, n_groups * g)
    x2_deq = x2_deq[:, :n]
    return x2_deq.view(orig_shape).to(dtype=out_dtype)


def quantize_tensor(x: torch.Tensor, qtype: str, group_size: int) -> torch.Tensor:
    if qtype == "none":
        return x
    if qtype == "int8":
        return _quantize_symmetric_lastdim(x, qmax=127, group_size=group_size)
    if qtype == "int4":
        return _quantize_symmetric_lastdim(x, qmax=7, group_size=group_size)
    raise ValueError(f"Unknown quant type: {qtype}")


def fake_quant_ste(x: torch.Tensor, qtype: str, group_size: int) -> torch.Tensor:
    xq = quantize_tensor(x, qtype=qtype, group_size=group_size)
    xq = xq.to(dtype=x.dtype)
    return x + (xq - x).detach()


class FactorizedLinearQAT(nn.Linear):
    """
    W(out,in) ≈ B(out,r) @ A(r,in)
    QAT path: fake-quant A/B with STE in forward.
    """

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.rank = 0
        self.A: Optional[nn.Parameter] = None
        self.B: Optional[nn.Parameter] = None
        self.quant_type = "none"
        self.quant_group_size = 0

    @torch.no_grad()
    def disable_low_rank(self):
        self.rank = 0
        self.A = None
        self.B = None

    @torch.no_grad()
    def set_low_rank_from_weight_gpu(self, rank: int):
        rmax = min(self.weight.shape)
        rank = min(rank, rmax)
        if rank >= rmax:
            self.disable_low_rank()
            return

        w = self.weight.detach().float()
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u = u[:, :rank]
        s = s[:rank]
        vh = vh[:rank, :]

        a = vh.to(dtype=torch.float32).contiguous()
        b = (u * s.unsqueeze(0)).to(dtype=torch.float32).contiguous()

        self.A = nn.Parameter(a)
        self.B = nn.Parameter(b)
        self.rank = rank

    def set_quant(self, qtype: str, group_size: int):
        self.quant_type = qtype
        self.quant_group_size = group_size

    def forward(self, x):
        if self.rank > 0 and self.A is not None and self.B is not None:
            a = fake_quant_ste(self.A, self.quant_type, self.quant_group_size)
            b = fake_quant_ste(self.B, self.quant_type, self.quant_group_size)
            # Use fp32 accumulation for stability; cast back to activation dtype.
            x32 = x.float()
            y = x32.matmul(a.float().transpose(0, 1))
            y = y.matmul(b.float().transpose(0, 1)).to(dtype=x.dtype)
            if self.bias is not None:
                y = y + self.bias
            return y
        return F.linear(x, self.weight, self.bias)


def replace_linear_with_factorized_qat(old: nn.Linear) -> FactorizedLinearQAT:
    new = FactorizedLinearQAT(
        old.in_features,
        old.out_features,
        bias=(old.bias is not None),
        device=old.weight.device,
        dtype=old.weight.dtype,
    )
    new.weight.data.copy_(old.weight.data)
    if old.bias is not None:
        new.bias.data.copy_(old.bias.data)
    return new


class InProjXOnly(nn.Module):
    def __init__(self, old: nn.Linear, z_dim: int, x_dim: int):
        super().__init__()
        assert isinstance(old, nn.Linear)
        assert 0 < z_dim and 0 < x_dim and (z_dim + x_dim) < old.out_features

        tail_dim = old.out_features - (z_dim + x_dim)

        self.z = nn.Linear(
            old.in_features,
            z_dim,
            bias=(old.bias is not None),
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        self.z.weight.data.copy_(old.weight.data[:z_dim])
        if old.bias is not None:
            self.z.bias.data.copy_(old.bias.data[:z_dim])

        x_lin = nn.Linear(
            old.in_features,
            x_dim,
            bias=(old.bias is not None),
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        x_lin.weight.data.copy_(old.weight.data[z_dim : z_dim + x_dim])
        if old.bias is not None:
            self.x_bias = True
            x_lin.bias.data.copy_(old.bias.data[z_dim : z_dim + x_dim])
        self.x = replace_linear_with_factorized_qat(x_lin)

        self.tail = nn.Linear(
            old.in_features,
            tail_dim,
            bias=(old.bias is not None),
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        self.tail.weight.data.copy_(old.weight.data[z_dim + x_dim :])
        if old.bias is not None:
            self.tail.bias.data.copy_(old.bias.data[z_dim + x_dim :])

    def forward(self, x):
        return torch.cat([self.z(x), self.x(x), self.tail(x)], dim=-1)


@dataclass
class Config:
    model_id: str = "state-spaces/mamba2-130m"
    tokenizer_id: str = "EleutherAI/gpt-neox-20b"

    dataset_id: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    train_split: str = "train"
    valid_split: str = "validation"

    seq_len: int = 1024
    train_max_tokens: Optional[int] = 2_000_000
    valid_max_tokens: Optional[int] = 200_000

    train_batch_size: int = 2
    eval_batch_size: int = 4

    in_layer_mode: str = "early_mid"
    out_layer_mode: str = "mid"
    in_rank: int = 384
    out_rank: int = 384

    quant_type: str = "int8"
    quant_group_size: int = 32
    qat_start_step: int = 200

    lr: float = 2e-5
    weight_decay: float = 0.0
    warmup_steps: int = 50
    max_steps: int = 1000
    grad_accum: int = 8
    grad_clip: float = 1.0
    anchor_lambda: float = 1e-4
    rollback_factor: float = 1.25
    rollback_lr_decay: float = 0.5
    eval_every: int = 100
    log_every: int = 10
    eval_batches: int = 200

    cache_dir: str = "./eval_cache"
    out_ckpt: str = "./ckpt_mamba130m_lowrank384_int8g32.pt"


@dataclass
class ActSVDState:
    total_time: float = 0.0


def _activation_svd(x: torch.Tensor, rank: int) -> torch.Tensor:
    if x.numel() == 0:
        return x
    orig_dtype = x.dtype
    x2 = x.float().reshape(-1, x.shape[-1])
    r = min(rank, min(x2.shape))
    if r >= min(x2.shape):
        return x
    u, s, vh = torch.linalg.svd(x2, full_matrices=False)
    x_rec = (u[:, :r] * s[:r].unsqueeze(0)).matmul(vh[:r, :])
    return x_rec.view_as(x).to(dtype=orig_dtype)


def _make_act_svd_hook(rank: int, state: ActSVDState):
    def hook(_module, inputs):
        x = inputs[0]
        if not torch.is_tensor(x):
            return inputs
        t0 = time.time()
        x_rec = _activation_svd(x, rank)
        state.total_time += (time.time() - t0)
        if len(inputs) == 1:
            return (x_rec,)
        return (x_rec,) + inputs[1:]

    return hook


def maybe_apply_act_svd_hooks(model: nn.Module, layers: Set[int], rank: Optional[int]):
    if rank is None:
        return ActSVDState(0.0), []
    state = ActSVDState(0.0)
    handles = []
    for i, layer in enumerate(model.backbone.layers):
        if i in layers:
            handles.append(layer.mixer.in_proj.register_forward_pre_hook(_make_act_svd_hook(rank, state)))
    return state, handles


def build_cache_paths(cfg: Config, split: str, max_tokens: Optional[int]) -> Tuple[str, str]:
    os.makedirs(cfg.cache_dir, exist_ok=True)
    ids_path = os.path.join(cfg.cache_dir, f"{cfg.dataset_id}_{cfg.dataset_config}_{split}_neox_ids.pt")
    blocks_path = os.path.join(cfg.cache_dir, f"{cfg.dataset_id}_{cfg.dataset_config}_{split}_blocks_{cfg.seq_len}_{max_tokens}.pt")
    return ids_path, blocks_path


def load_or_build_ids(cfg: Config, tokenizer: AutoTokenizer, split: str) -> torch.Tensor:
    ids_path, _ = build_cache_paths(cfg, split=split, max_tokens=None)
    if os.path.exists(ids_path):
        return torch.load(ids_path, map_location="cpu")
    ds = load_dataset(cfg.dataset_id, cfg.dataset_config, split=split)
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0].contiguous()
    torch.save(ids, ids_path)
    return ids


def load_or_build_blocks(cfg: Config, ids_1d: torch.Tensor, split: str, max_tokens: Optional[int]) -> torch.Tensor:
    _, blocks_path = build_cache_paths(cfg, split=split, max_tokens=max_tokens)
    if os.path.exists(blocks_path):
        return torch.load(blocks_path, map_location="cpu")

    if max_tokens is not None and ids_1d.numel() > max_tokens:
        ids_1d = ids_1d[:max_tokens]

    n_tokens = ids_1d.numel()
    n_blocks = (n_tokens - 1) // cfg.seq_len
    usable = n_blocks * cfg.seq_len
    ids_1d = ids_1d[: usable + 1]

    blocks = ids_1d.unfold(0, cfg.seq_len + 1, cfg.seq_len).contiguous()
    torch.save(blocks, blocks_path)
    return blocks


def patch_model(model: nn.Module):
    m0 = model.backbone.layers[0].mixer
    d_inner = m0.out_proj.in_features
    z_dim = d_inner
    x_dim = d_inner

    for layer in model.backbone.layers:
        mx = layer.mixer
        if not isinstance(mx.in_proj, InProjXOnly):
            mx.in_proj = InProjXOnly(mx.in_proj, z_dim=z_dim, x_dim=x_dim)
        if isinstance(mx.out_proj, nn.Linear) and not isinstance(mx.out_proj, FactorizedLinearQAT):
            mx.out_proj = replace_linear_with_factorized_qat(mx.out_proj)


def set_force_unfused(model: nn.Module, enabled: bool):
    for layer in model.backbone.layers:
        mixer = layer.mixer
        if hasattr(mixer, "use_mem_eff_path"):
            mixer.use_mem_eff_path = not enabled


@torch.no_grad()
def enable_lowrank(model: nn.Module, cfg: Config, in_layers: Set[int], out_layers: Set[int]):
    for i, layer in enumerate(model.backbone.layers):
        if i in in_layers:
            xproj = layer.mixer.in_proj.x
            assert isinstance(xproj, FactorizedLinearQAT)
            xproj.set_low_rank_from_weight_gpu(cfg.in_rank)
            xproj.set_quant(cfg.quant_type, cfg.quant_group_size)
        if i in out_layers:
            op = layer.mixer.out_proj
            assert isinstance(op, FactorizedLinearQAT)
            op.set_low_rank_from_weight_gpu(cfg.out_rank)
            op.set_quant(cfg.quant_type, cfg.quant_group_size)


@torch.no_grad()
def set_lowrank_quant(
    model: nn.Module,
    in_layers: Set[int],
    out_layers: Set[int],
    qtype: str,
    qgroup: int,
):
    for i, layer in enumerate(model.backbone.layers):
        if i in in_layers:
            xproj = layer.mixer.in_proj.x
            assert isinstance(xproj, FactorizedLinearQAT)
            xproj.set_quant(qtype, qgroup)
        if i in out_layers:
            op = layer.mixer.out_proj
            assert isinstance(op, FactorizedLinearQAT)
            op.set_quant(qtype, qgroup)


def freeze_all_params(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False


@torch.no_grad()
def load_trainable_from_checkpoint(model: nn.Module, ckpt_path: str, device: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("trainable_only", ckpt)
    model_state = model.state_dict()

    loaded = 0
    missing = 0
    for k, v in state.items():
        if k not in model_state:
            missing += 1
            continue
        model_state[k].copy_(v.to(device=device, dtype=model_state[k].dtype))
        loaded += 1

    print(f"resume: loaded {loaded} tensors from {ckpt_path} (missing={missing})")
    return ckpt if isinstance(ckpt, dict) else {}


def move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str):
    for st in optimizer.state.values():
        for k, v in st.items():
            if torch.is_tensor(v):
                st[k] = v.to(device=device, non_blocking=True)


def mark_trainable(
    model: nn.Module,
    in_layers: Set[int],
    out_layers: Set[int],
    train_in: bool = True,
    train_out: bool = True,
) -> int:
    n = 0
    for i, layer in enumerate(model.backbone.layers):
        if train_in and i in in_layers:
            xproj = layer.mixer.in_proj.x
            if xproj.A is not None:
                xproj.A.requires_grad = True
                n += xproj.A.numel()
            if xproj.B is not None:
                xproj.B.requires_grad = True
                n += xproj.B.numel()

        if train_out and i in out_layers:
            op = layer.mixer.out_proj
            if op.A is not None:
                op.A.requires_grad = True
                n += op.A.numel()
            if op.B is not None:
                op.B.requires_grad = True
                n += op.B.numel()
    return n


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * float(step + 1) / float(max(1, cfg.warmup_steps))
    progress = (step - cfg.warmup_steps) / float(max(1, cfg.max_steps - cfg.warmup_steps))
    progress = min(1.0, max(0.0, progress))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr * cosine


@torch.inference_mode()
def eval_ppl(model: nn.Module, blocks_cpu: torch.Tensor, batch_size: int, device: str, max_batches: Optional[int]) -> float:
    model.eval()
    total_nll = 0.0
    total_tok = 0
    bad_batches = 0
    n_blocks = blocks_cpu.size(0)
    if max_batches is not None:
        n_blocks = min(n_blocks, max_batches * batch_size)

    pbar = tqdm(range(0, n_blocks, batch_size), desc="Eval", leave=False)
    for i in pbar:
        batch = blocks_cpu[i : i + batch_size].to(device, non_blocking=True)
        x = batch[:, :-1]
        y = batch[:, 1:]

        logits = model(x).logits
        if not torch.isfinite(logits).all():
            bad_batches += 1
            continue
        logits = logits.float()
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")

        if torch.isfinite(loss):
            total_nll += loss.item()
            total_tok += y.numel()

    if total_tok == 0:
        return float("inf")
    if bad_batches > 0:
        print(f"eval warning: skipped {bad_batches} non-finite batch(es)")
    return math.exp(min(total_nll / total_tok, 80.0))


def next_train_batch(blocks_cpu: torch.Tensor, batch_size: int, device: str) -> torch.Tensor:
    idx = torch.randint(0, blocks_cpu.size(0), (batch_size,))
    return blocks_cpu[idx].to(device, non_blocking=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--quant-type", type=str, default="int8", choices=["none", "int8", "int4"])
    p.add_argument("--quant-group-size", type=int, default=32)
    p.add_argument("--qat-start-step", type=int, default=200, help="Global step to enable QAT quantization")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batches", type=int, default=200)
    p.add_argument("--train-max-tokens", type=int, default=2_000_000)
    p.add_argument("--valid-max-tokens", type=int, default=200_000)
    p.add_argument("--anchor-lambda", type=float, default=1e-4)
    p.add_argument("--rollback-factor", type=float, default=1.25, help="Rollback if eval ppl exceeds best * factor")
    p.add_argument("--rollback-lr-decay", type=float, default=0.5)
    p.add_argument("--out", type=str, default="./ckpt_mamba130m_lowrank384_int8g32.pt")
    p.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    p.add_argument("--eval-act-svd-rank", type=int, default=0, help="0 disables activation SVD hook at eval")
    p.add_argument("--train-in-only", action="store_true", help="Train only in_proj low-rank factors (out_proj stays frozen)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    dtype = get_dtype(device)

    cfg = Config(
        seq_len=args.seq_len,
        train_batch_size=args.batch_size,
        lr=args.lr,
        quant_type=args.quant_type,
        quant_group_size=args.quant_group_size,
        qat_start_step=args.qat_start_step,
        max_steps=args.steps,
        grad_accum=args.grad_accum,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        train_max_tokens=args.train_max_tokens,
        valid_max_tokens=args.valid_max_tokens,
        anchor_lambda=args.anchor_lambda,
        rollback_factor=args.rollback_factor,
        rollback_lr_decay=args.rollback_lr_decay,
        out_ckpt=args.out,
    )

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    print(f"Device={device} dtype={dtype} | loading tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_id, use_fast=True)

    train_ids = load_or_build_ids(cfg, tokenizer, split=cfg.train_split)
    valid_ids = load_or_build_ids(cfg, tokenizer, split=cfg.valid_split)
    train_blocks = load_or_build_blocks(cfg, train_ids, split=cfg.train_split, max_tokens=cfg.train_max_tokens)
    valid_blocks = load_or_build_blocks(cfg, valid_ids, split=cfg.valid_split, max_tokens=cfg.valid_max_tokens)

    if device == "cuda":
        train_blocks = train_blocks.pin_memory()
        valid_blocks = valid_blocks.pin_memory()

    model = MambaLMHeadModel.from_pretrained(cfg.model_id, device=device, dtype=dtype)
    patch_model(model)
    set_force_unfused(model, enabled=True)

    n_layers = len(model.backbone.layers)
    in_layers = layer_indices(cfg.in_layer_mode, n_layers)
    out_layers = layer_indices(cfg.out_layer_mode, n_layers)

    enable_lowrank(model, cfg, in_layers=in_layers, out_layers=out_layers)
    freeze_all_params(model)
    trainable_count = mark_trainable(
        model,
        in_layers=in_layers,
        out_layers=out_layers,
        train_in=True,
        train_out=(not args.train_in_only),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters were selected.")

    print(f"Trainable params: {trainable_count:,}")
    if args.train_in_only:
        print("Train mode: in_proj factors only (out_proj factors frozen)")
    print("Evaluating compressed init...")

    act_rank = args.eval_act_svd_rank if args.eval_act_svd_rank > 0 else None
    act_state, act_handles = maybe_apply_act_svd_hooks(model, layers=in_layers, rank=act_rank)
    init_ppl = eval_ppl(model, valid_blocks, batch_size=cfg.eval_batch_size, device=device, max_batches=cfg.eval_batches)
    for h in act_handles:
        h.remove()
    print(f"Init val PPL: {init_ppl:.4f} (act_svd_sec={act_state.total_time:.2f})")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=cfg.weight_decay)

    best_ppl = init_ppl
    best_state = None
    resume_step = 0
    lr_scale = 1.0
    last_qtype = None
    anchor = {n: p.detach().clone() for n, p in trainable_named}
    best_trainable = {n: p.detach().cpu().clone() for n, p in trainable_named}

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume path not found: {args.resume}")
        ckpt = load_trainable_from_checkpoint(model, args.resume, device=device)
        best_ppl = float(ckpt.get("ppl", best_ppl))
        resume_step = int(ckpt.get("step", 0))
        lr_scale = float(ckpt.get("lr_scale", 1.0))
        best_trainable = {n: p.detach().cpu().clone() for n, p in trainable_named}
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
            move_optimizer_state_to_device(optimizer, device=device)
            print("resume: optimizer state restored")
        print(f"resume: starting from global_step={resume_step}, best_ppl={best_ppl:.4f}, lr_scale={lr_scale:.3f}")

    model.train()
    t0 = time.time()
    for step in range(cfg.max_steps):
        step_idx = resume_step + step
        global_step = step_idx + 1
        qtype_now = cfg.quant_type if global_step >= cfg.qat_start_step else "none"
        if qtype_now != last_qtype:
            set_lowrank_quant(model, in_layers=in_layers, out_layers=out_layers, qtype=qtype_now, qgroup=cfg.quant_group_size)
            last_qtype = qtype_now
            print(f"step={global_step:5d} quant={qtype_now}")

        lr_now = get_lr(step_idx, cfg) * lr_scale
        for g in optimizer.param_groups:
            g["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        valid_micro = 0
        bad_micro = 0

        for _ in range(cfg.grad_accum):
            batch = next_train_batch(train_blocks, cfg.train_batch_size, device)
            x = batch[:, :-1]
            y = batch[:, 1:]

            logits = model(x).logits
            if not torch.isfinite(logits).all():
                bad_micro += 1
                continue
            logits = logits.float()
            ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="mean")

            if cfg.anchor_lambda > 0.0:
                reg = 0.0
                for n, p in trainable_named:
                    reg = reg + (p - anchor[n]).pow(2).mean()
                reg = reg / float(max(1, len(trainable_named)))
                loss = ce_loss + (cfg.anchor_lambda * reg)
            else:
                loss = ce_loss

            if torch.isfinite(loss):
                (loss / cfg.grad_accum).backward()
                accum_loss += loss.item()
                valid_micro += 1
            else:
                bad_micro += 1

        if valid_micro == 0:
            print(f"step={global_step:5d} skipped: all microbatches non-finite (bad_micro={bad_micro})")
            continue

        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
        optimizer.step()

        if global_step % cfg.log_every == 0:
            tok_per_step = cfg.train_batch_size * cfg.grad_accum * (cfg.seq_len)
            elapsed = time.time() - t0
            tok_s = (tok_per_step * (step + 1)) / max(1e-9, elapsed)
            print(
                f"step={global_step:5d} loss={accum_loss/valid_micro:.4f} "
                f"lr={lr_now:.2e} tok/s={tok_s:,.0f} bad_micro={bad_micro}"
            )

        if (global_step % cfg.eval_every) == 0 or (step + 1) == cfg.max_steps:
            model.eval()
            act_state, act_handles = maybe_apply_act_svd_hooks(model, layers=in_layers, rank=act_rank)
            ppl = eval_ppl(model, valid_blocks, batch_size=cfg.eval_batch_size, device=device, max_batches=cfg.eval_batches)
            for h in act_handles:
                h.remove()

            print(f"eval step={global_step:5d} ppl={ppl:.4f} best={best_ppl:.4f} act_svd_sec={act_state.total_time:.2f}")

            if ppl < best_ppl:
                best_ppl = ppl
                best_state = {
                    "step": global_step,
                    "ppl": best_ppl,
                    "config": cfg.__dict__,
                    "lr_scale": lr_scale,
                    "optimizer": optimizer.state_dict(),
                    "trainable_only": {k: v.detach().cpu() for k, v in model.state_dict().items() if ".A" in k or ".B" in k or k.endswith(".bias")},
                }
                torch.save(best_state, cfg.out_ckpt)
                best_trainable = {n: p.detach().cpu().clone() for n, p in trainable_named}
                print(f"saved best checkpoint: {cfg.out_ckpt}")
            elif ppl > (best_ppl * cfg.rollback_factor):
                with torch.no_grad():
                    for n, p in trainable_named:
                        p.copy_(best_trainable[n].to(device=device, dtype=p.dtype))
                lr_scale = max(0.05, lr_scale * cfg.rollback_lr_decay)
                print(
                    f"rollback: restored best params due to ppl spike "
                    f"(ppl={ppl:.4f} > {best_ppl * cfg.rollback_factor:.4f}); "
                    f"new lr_scale={lr_scale:.3f}"
                )
            model.train()

    print(f"Done. best_val_ppl={best_ppl:.4f} ckpt={cfg.out_ckpt}")


if __name__ == "__main__":
    main()
