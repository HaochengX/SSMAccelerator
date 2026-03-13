import os, sys
import numpy as np

# ap_fixed<16,4> => total=16, int=4 => frac=12
FRAC_BITS = 12
TOTAL_BITS = 16
I16_MIN = -32768
I16_MAX =  32767

def float_to_apfixed16_4_bits(x: np.ndarray) -> np.ndarray:
    # ✅ FIX: use np.round (banker's rounding) to match gen_bins_raw.py / make_bins_raw.py
    scaled = x * (2 ** FRAC_BITS)
    rounded = np.round(scaled).astype(np.int32)
    clipped = np.clip(rounded, I16_MIN, I16_MAX).astype(np.int16)
    return clipped

def pack_one(in_path: str, out_path: str):
    a = np.fromfile(in_path, dtype=np.float32)
    bits = float_to_apfixed16_4_bits(a)
    bits.tofile(out_path)
    print(f"[PACK] {os.path.basename(in_path)}: f32[{a.size}] -> raw int16[{bits.size}] -> {out_path}")

def main():
    if len(sys.argv) != 3:
        print("Usage: py pack_dtypes.py <bins_f32_dir> <bins_raw_dir>")
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)

    # pack every *.bin in bins_f32
    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(".bin"):
            continue
        in_path = os.path.join(in_dir, fn)
        out_path = os.path.join(out_dir, fn.replace(".bin", ".raw.bin"))
        pack_one(in_path, out_path)

    print(f"[PY] wrote DUT raw bins to: {out_dir}")
    print("     Next: run CSIM testbench (tb_ssmu.cpp) reading *.raw.bin.")

if __name__ == "__main__":
    main()
