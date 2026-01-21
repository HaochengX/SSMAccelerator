import csv
import gc
import itertools
import math
import os

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================
# Low-rank
# =========================
def apply_low_rank_(w, rank):
    if rank >= min(w.shape):
        return
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    w.copy_((U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank]).to(w.dtype))


# =========================
# Groupwise symmetric fake quant (quantize+dequant) along last dim
# =========================
def valid_group_size(d: int, desired: int) -> int:
    # If desired doesn't divide d, fall back to per-row/per-vector group
    return desired if (d % desired == 0) else d


def groupwise_fake_quant_lastdim_(w: torch.Tensor, bits: int, group_size: int):
    """
    In-place fake-quant along the last dimension. Groups along last dim.
    """
    qmax = (1 << (bits - 1)) - 1
    D = w.shape[-1]
    group_size = valid_group_size(D, group_size)
    if group_size == 0:
        return
    if D % group_size != 0:
        raise ValueError(f"Last dim {D} must be divisible by group_size {group_size}")

    orig_shape = w.shape
    rows = w.reshape(-1, D)  # [R, D]
    rows = rows.view(rows.shape[0], D // group_size, group_size)  # [R, G, gs]

    scale = rows.abs().amax(dim=-1, keepdim=True) / qmax
    scale.clamp_(min=1e-8)

    rows_q = (rows / scale).round().clamp(-qmax, qmax) * scale
    w.copy_(rows_q.view(orig_shape))


# =========================
# Fast Walsh-Hadamard Transform (FWHT) on last dim (power-of-two)
# =========================
def _fwht_inplace_lastdim_(x: torch.Tensor):
    n = x.shape[-1]
    if n & (n - 1) != 0:
        raise ValueError(f"FWHT requires power-of-two length, got {n}")

    orig_dtype = x.dtype
    y = x.float()

    h = 1
    while h < n:
        y = y.view(*y.shape[:-1], n // (2 * h), 2 * h)
        a = y[..., :, :h]
        b = y[..., :, h : 2 * h]

        tmp = a.clone()
        a.copy_(tmp + b)
        b.copy_(tmp - b)

        y = y.view(*y.shape[:-2], n)
        h *= 2

    y = y / math.sqrt(n)
    x.copy_(y.to(orig_dtype))


def choose_pow2_block_divisor(d: int, max_block: int = 256) -> int:
    b = 1 << int(math.floor(math.log2(max_block)))
    while b > 1:
        if d % b == 0:
            return b
        b //= 2
    return 1  # no-op


def hadamard_block_rotate_lastdim_(x: torch.Tensor, block: int):
    """
    Apply block-diagonal Hadamard rotation along last dim using FWHT per block.
    If block==1 -> no-op.
    """
    if block == 1:
        return x

    D = x.shape[-1]
    if D % block != 0:
        raise ValueError(f"Last dim {D} must be divisible by block {block}")

    orig_shape = x.shape
    y = x.reshape(-1, D).view(-1, D // block, block)  # [R, nb, block]
    y2 = y.reshape(-1, block)  # [R*nb, block]
    _fwht_inplace_lastdim_(y2)
    y.copy_(y2.view_as(y))
    x.copy_(y.view(orig_shape))
    return x


# =========================
# Hooks: optionally rotate + quantize activations before matmul
# =========================
def clear_hooks_(model):
    for mod in model.modules():
        if hasattr(mod, "_qq_handle"):
            try:
                mod._qq_handle.remove()
            except Exception:
                pass
            delattr(mod, "_qq_handle")
        for attr in ["_qq_bits", "_qq_gsize", "_qq_block", "_qq_do_hadamard"]:
            if hasattr(mod, attr):
                delattr(mod, attr)


def attach_act_prehook_(
    module, *, bits: int, gsize: int, do_hadamard: bool, block: int
):
    """
    Prehook that transforms the first positional input x:
      x -> (Hadamard(x) if do_hadamard) -> fake_quant(x)
    """
    if hasattr(module, "_qq_handle"):
        try:
            module._qq_handle.remove()
        except Exception:
            pass

    module._qq_bits = bits
    module._qq_gsize = gsize
    module._qq_do_hadamard = do_hadamard
    module._qq_block = block

    def _pre(mod, inputs):
        x = inputs[0]
        x2 = x.clone()
        if mod._qq_do_hadamard:
            hadamard_block_rotate_lastdim_(x2, mod._qq_block)
        groupwise_fake_quant_lastdim_(x2, mod._qq_bits, mod._qq_gsize)
        return (x2,) + tuple(inputs[1:])

    module._qq_handle = module.register_forward_pre_hook(_pre)


# =========================
# Layer selection
# =========================
def layer_indices(mode, n_layers):
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(2, n_layers // 3))
    if mode == "mid":
        return set(range(n_layers // 3, 2 * n_layers // 3))
    if mode == "late":
        return set(range(2 * n_layers // 3, n_layers))
    raise ValueError(mode)


# =========================
# Two modes:
#   (A) baseline quantized matmul: quantize x and W in original domain
#   (B) hadamard rotated quantized matmul: rotate x and W, quantize in rotated domain
# =========================
def prepare_mamba_mode_(
    model,
    rb,
    rc,
    rdt,
    bits,
    gsize,
    layers,
    mode: str,
    max_hadamard_block: int = 256,
    quantize_dt_part_in_xproj: bool = True,  # for "fully quantized matmul" fairness
):
    assert mode in ["baseline", "hadamard_rot"]

    for i, layer in enumerate(model.backbone.layers):
        m = layer.mixer

        # if not selected, ensure no hooks
        if i not in layers:
            for mod in [getattr(m, "x_proj", None), getattr(m, "dt_proj", None)]:
                if mod is not None and hasattr(mod, "_qq_handle"):
                    try:
                        mod._qq_handle.remove()
                    except Exception:
                        pass
                    for attr in [
                        "_qq_handle",
                        "_qq_bits",
                        "_qq_gsize",
                        "_qq_block",
                        "_qq_do_hadamard",
                    ]:
                        if hasattr(mod, attr):
                            delattr(mod, attr)
            continue

        DT = m.dt_proj.in_features
        DS = m.config.state_size

        # ---------- x_proj ----------
        W = m.x_proj.weight.data  # [DT+2*DS, d_model]
        d_model = W.shape[-1]
        block_x = choose_pow2_block_divisor(d_model, max_block=max_hadamard_block)

        dt_part = W[:DT, :]
        wb = W[DT : DT + DS, :]
        wc = W[DT + DS : DT + 2 * DS, :]

        # low-rank in original basis
        apply_low_rank_(wb, rb)
        apply_low_rank_(wc, rc)

        # baseline: quantize W in-place (original domain)
        if mode == "baseline":
            if quantize_dt_part_in_xproj:
                groupwise_fake_quant_lastdim_(dt_part, bits, gsize)
            groupwise_fake_quant_lastdim_(wb, bits, gsize)
            groupwise_fake_quant_lastdim_(wc, bits, gsize)

            # quantize activations (no hadamard)
            attach_act_prehook_(
                m.x_proj, bits=bits, gsize=gsize, do_hadamard=False, block=1
            )

        # hadamard_rot: rotate weights, quantize rotated weights, keep rotated
        else:
            W_rot = W.float().clone()
            hadamard_block_rotate_lastdim_(W_rot, block_x)

            if quantize_dt_part_in_xproj:
                groupwise_fake_quant_lastdim_(W_rot[:DT, :], bits, gsize)
            groupwise_fake_quant_lastdim_(W_rot[DT : DT + DS, :], bits, gsize)
            groupwise_fake_quant_lastdim_(W_rot[DT + DS : DT + 2 * DS, :], bits, gsize)

            m.x_proj.weight.data.copy_(W_rot.to(m.x_proj.weight.dtype))

            # rotate + quantize activations
            attach_act_prehook_(
                m.x_proj, bits=bits, gsize=gsize, do_hadamard=True, block=block_x
            )

        # ---------- dt_proj ----------
        Wdt = m.dt_proj.weight.data  # [out, DT]
        dt_in = Wdt.shape[-1]
        block_dt = choose_pow2_block_divisor(dt_in, max_block=dt_in)  # best possible
        g_dt = valid_group_size(dt_in, gsize)

        apply_low_rank_(Wdt, rdt)

        if mode == "baseline":
            groupwise_fake_quant_lastdim_(Wdt, bits, g_dt)
            attach_act_prehook_(
                m.dt_proj, bits=bits, gsize=g_dt, do_hadamard=False, block=1
            )
        else:
            Wdt_rot = Wdt.float().clone()
            hadamard_block_rotate_lastdim_(Wdt_rot, block_dt)
            groupwise_fake_quant_lastdim_(Wdt_rot, bits, g_dt)
            m.dt_proj.weight.data.copy_(Wdt_rot.to(m.dt_proj.weight.dtype))
            attach_act_prehook_(
                m.dt_proj, bits=bits, gsize=g_dt, do_hadamard=True, block=block_dt
            )


# =========================
# Eval
# =========================
def evaluate(model, ids):
    model.eval()
    nlls = []
    stride, seq_len = 512, 1024
    with torch.inference_mode():
        for i in range(0, min(ids.size(1), 5000), stride):
            end = min(i + seq_len, ids.size(1))
            if end < seq_len:
                continue
            x = ids[:, end - seq_len : end].to(DEVICE)
            y = x.clone()
            y[:, : -(end - i)] = -100
            logits = model(x).logits
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                y[..., 1:].reshape(-1),
            )
            nlls.append(loss)
    return torch.exp(torch.stack(nlls).mean()).item()


# =========================
# Main
# =========================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer(
        "\n\n".join(t for t in dataset["text"] if t.strip()),
        return_tensors="pt",
    ).input_ids

    print(f"--- Initializing: Loading {MODEL_ID} to CPU RAM ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    clean_state = {k: v.clone().cpu() for k, v in base_model.state_dict().items()}

    # Sweep Parameters
    ranks_c = [16]
    ranks_b = [16]
    ranks_dt = [160, 40]
    bits_list = [8, 4]
    group_sizes = [64]
    layer_modes = ["all", "mid", "late", "early"]

    modes = ["baseline", "hadamard_rot"]

    total = (
        len(modes)
        * len(bits_list)
        * len(group_sizes)
        * len(ranks_b)
        * len(ranks_c)
        * len(ranks_dt)
        * len(layer_modes)
    )
    csv_file = "compare_baseline_vs_hadamard_rot_quant_matmul_mamba2.8b.csv"

    with open(csv_file, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
            writer.writerow(
                ["mode", "bits", "group", "rb", "rc", "rdt", "layer_mode", "ppl"]
            )

        for mode, b, g, rb, rc, rdt, lm in tqdm(
            itertools.product(
                modes, bits_list, group_sizes, ranks_b, ranks_c, ranks_dt, layer_modes
            ),
            total=total,
        ):
            # Reset model weights + hooks
            clear_hooks_(base_model)
            base_model.load_state_dict(clean_state)

            model = base_model.to(DEVICE)

            layers = layer_indices(lm, len(model.backbone.layers))

            prepare_mamba_mode_(
                model=model,
                rb=rb,
                rc=rc,
                rdt=rdt,
                bits=b,
                gsize=g,
                layers=layers,
                mode=mode,
                max_hadamard_block=256,
                quantize_dt_part_in_xproj=True,  # makes "activation*weight matmul is quantized" literal
            )

            ppl = evaluate(model, ids)
            writer.writerow([mode, b, g, rb, rc, rdt, lm, f"{ppl:.4f}"])
            f.flush()

            # Cleanup
            model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

            gc.collect()


if __name__ == "__main__":
    main()
