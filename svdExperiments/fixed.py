import csv
import itertools
import os
import gc
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Enable better memory management for fragmented GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "state-spaces/mamba-2.8b-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- Low-rank utils ----------------
def apply_low_rank_(w: torch.Tensor, rank: int):
    if rank >= min(w.shape):
        return
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    w.copy_((U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank]).to(w.dtype))


# ---------------- HLS-style fixed-point (ap_fixed<W,I>) ----------------
def ap_fixed_quant_(w: torch.Tensor, W: int, I: int, rounding: str = "nearest"):
    """
    Simulate HLS ap_fixed<W,I> quantization IN-PLACE.

    ap_fixed<W,I> (signed):
      - W total bits
      - I integer bits INCLUDING sign bit
      - F = W - I fractional bits
      - range: [ -2^(I-1), 2^(I-1) - 2^-F ]
      - step: 2^-F

    This function:
      1) saturates to representable range
      2) rounds to the fixed grid (step = 2^-F)
      3) writes back to w (tensor dtype unchanged; values snapped)
    """
    assert W > 0 and 0 < I <= W
    Fbits = W - I

    min_val = -(2 ** (I - 1))
    max_val = (2 ** (I - 1)) - (2 ** (-Fbits))
    scale = 2**Fbits

    wf = w.float().clamp(min_val, max_val) * scale

    if rounding == "nearest":
        wf = wf.round()
    elif rounding == "floor":
        wf = wf.floor()
    elif rounding == "ceil":
        wf = wf.ceil()
    else:
        raise ValueError("rounding must be nearest/floor/ceil")

    wf = (wf / scale).clamp(min_val, max_val)
    w.copy_(wf.to(w.dtype))


def ap_fixed_quant_tensor(x: torch.Tensor, W: int, I: int):
    """
    Non-inplace quantization for activations (returns a snapped copy).
    Keeps accumulator/full-precision math INSIDE ops (matmul, etc.),
    but snaps tensor values at module boundaries.
    """
    if not torch.is_tensor(x):
        return x
    y = x.detach().clone()
    ap_fixed_quant_(y, W=W, I=I)
    return y


# ---------------- Activation quant hooks ----------------
def add_activation_quant_hooks(
    model,
    W: int,
    I: int,
    module_types=(torch.nn.Linear,),
):
    """
    Quantize *inputs and outputs* of selected module types using ap_fixed<W,I>.
    This keeps matmul/accumulation full precision inside kernels, but snaps
    activations at boundaries (HLS-ish, "keep accum full size").
    """
    hooks = []

    def pre_hook(module, inputs):
        if not inputs:
            return inputs
        x = inputs[0]
        if torch.is_tensor(x):
            xq = ap_fixed_quant_tensor(x, W, I)
            return (xq,) + tuple(inputs[1:])
        return inputs

    def post_hook(module, inputs, output):
        if torch.is_tensor(output):
            return ap_fixed_quant_tensor(output, W, I)
        return output

    for m in model.modules():
        if isinstance(m, module_types):
            hooks.append(m.register_forward_pre_hook(pre_hook))
            hooks.append(m.register_forward_hook(post_hook))

    return hooks


# ---------------- Layer selection ----------------
def layer_indices(mode: str, n_layers: int):
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(2, n_layers // 3))
    if mode == "mid":
        return set(range(n_layers // 3, 2 * n_layers // 3))
    if mode == "late":
        return set(range(2 * n_layers // 3, n_layers))
    raise ValueError(mode)


# ---------------- Compression + weight quant ----------------
def compress_mamba_fixed(
    model,
    rb: int,
    rc: int,
    rdt: int,
    fixed_W: int,
    fixed_I: int,
    layers: set,
):
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers:
            continue

        m = layer.mixer
        DT = m.dt_proj.in_features
        DS = m.config.state_size
        Wmat = m.x_proj.weight.data

        _, wb, wc = torch.split(Wmat, [DT, DS, DS], dim=0)

        # Low-rank SVD truncation
        apply_low_rank_(wb, rb)
        apply_low_rank_(wc, rc)
        apply_low_rank_(m.dt_proj.weight.data, rdt)

        # Weight fixed-point snap
        for tw in (wb, wc, m.dt_proj.weight.data):
            ap_fixed_quant_(tw, W=fixed_W, I=fixed_I)


# ---------------- Eval (perplexity) ----------------
def evaluate(model, ids: torch.Tensor):
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


# ---------------- Main ----------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer(
        "\n\n".join(t for t in dataset["text"] if t.strip()),
        return_tensors="pt",
    ).input_ids

    print(f"--- Initializing: Loading {MODEL_ID} to CPU RAM ---")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    clean_state = {k: v.clone().cpu() for k, v in base_model.state_dict().items()}

    # Sweep Parameters (keep low-rank sweep as-is)
    ranks_b = [16, 12, 8, 4]
    ranks_c = [16, 12, 8, 4]
    ranks_dt = [160, 120, 80, 40, 20]
    layer_modes = ["all", "mid", "late", "early"]

    # Fixed-point types to try: ap_fixed<W,I>
    fixed_types = [
        (16, 4),
        (32, 8),
    ]

    total = (
        len(fixed_types)
        * len(ranks_b)
        * len(ranks_c)
        * len(ranks_dt)
        * len(layer_modes)
    )

    csv_file = "apfixed_weights_acts_sweep_mamba2.8b.csv"

    with open(csv_file, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        if not os.path.exists(csv_file) or os.stat(csv_file).st_size == 0:
            writer.writerow(["W", "I", "rb", "rc", "rdt", "layer_mode", "ppl"])

        for (Wbits, Ibits), rb, rc, rdt, lm in tqdm(
            itertools.product(
                fixed_types,
                ranks_b,
                ranks_c,
                ranks_dt,
                layer_modes,
            ),
            total=total,
        ):
            # Reset weights
            base_model.load_state_dict(clean_state)
            model = base_model.to(DEVICE)

            # Quantize weights (and low-rank truncate)
            layers = layer_indices(lm, len(model.backbone.layers))
            compress_mamba_fixed(
                model=model,
                rb=rb,
                rc=rc,
                rdt=rdt,
                fixed_W=Wbits,
                fixed_I=Ibits,
                layers=layers,
            )

            # Quantize activations at module boundaries (keep accum full precision inside ops)
            hooks = add_activation_quant_hooks(
                model,
                W=Wbits,
                I=Ibits,
                module_types=(
                    torch.nn.Linear,
                ),  # start with Linear; expand later if needed
            )

            ppl = evaluate(model, ids)

            # Remove hooks
            for h in hooks:
                h.remove()

            writer.writerow([Wbits, Ibits, rb, rc, rdt, lm, f"{ppl:.4f}"])
            f.flush()

            # Cleanup
            model.to("cpu")
            torch.cuda.empty_cache()
            gc.collect()

    print(f"Done. Results appended to: {csv_file}")


if __name__ == "__main__":
    main()
