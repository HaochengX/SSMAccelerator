import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def big_model(csv_path):
    df = pd.read_csv(csv_path)

    print("-" * 10)
    key_cols = ["bits", "group", "rb", "rc", "rdt", "layer_mode"]
    pivot = df.pivot_table(index=key_cols, columns="quant", values="ppl").dropna()
    pivot["delta_fake_minus_hadamard"] = pivot["fake"] - pivot["hadamard"]
    print(pivot.describe())

    from scipy import stats

    # 1. Perform the Paired T-Test
    # This tests the null hypothesis that the mean difference is zero.
    t_stat, p_value = stats.ttest_rel(pivot["fake"], pivot["hadamard"])

    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value: {p_value:.4e}")
    print("-" * 10)
    key_cols = ["group", "rb", "rc", "rdt", "layer_mode", "quant"]
    pivot = df.pivot_table(index=key_cols, columns="bits", values="ppl").dropna()
    pivot["delta_4_minus_8"] = pivot[4] - pivot[8]
    print(pivot.describe())

    print("-" * 10)
    key_cols = ["group", "rb", "rc", "rdt", "quant", "bits"]
    pivot = df.pivot_table(index=key_cols, columns="layer_mode", values="ppl").dropna()
    print(pivot.describe())

    print("-" * 10)
    key_cols = ["layer_mode", "rb", "rc", "rdt", "quant", "bits"]
    pivot = df.pivot_table(index=key_cols, columns="group", values="ppl").dropna()
    print(pivot.describe())

    print("-" * 10)
    print("Effect of RDT when RC and RB are 16:")
    filtered_df = df[(df["rc"] == 16) & (df["rb"] == 16)]
    key_cols = ["bits", "group", "layer_mode", "quant"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rdt", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())

    print("-" * 10)
    print("Effect of RC when RDT is 160 and RB is 16:")
    filtered_df = df[(df["rb"] == 16) & (df["rdt"] == 160)]
    key_cols = ["bits", "group", "layer_mode", "quant"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rc", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())

    print("-" * 10)
    print("Effect of RB when RDT is 160 and RC is 16:")
    filtered_df = df[(df["rc"] == 16) & (df["rdt"] == 160)]
    key_cols = ["bits", "group", "layer_mode", "quant"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rb", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())


big_model("./hadamard_all_sweep_2.8b_optimized.csv")


def big_model_with_viz(csv_path):
    # Load data
    df = pd.read_csv(csv_path)
    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 22))
    axes = axes.flatten()

    # 1. Quantization Comparison (Delta: Fake - Hadamard)
    key_cols = ["bits", "group", "rb", "rc", "rdt", "layer_mode"]
    pivot_q = df.pivot_table(index=key_cols, columns="quant", values="ppl").dropna()
    pivot_q["delta_fake_minus_hadamard"] = pivot_q["fake"] - pivot_q["hadamard"]
    sns.boxplot(
        data=pivot_q, y="delta_fake_minus_hadamard", ax=axes[0], color="skyblue"
    )
    axes[0].set_title(
        "PPL Difference: Fake vs Hadamard\n(Higher Delta = Hadamard is better)",
        fontweight="bold",
    )
    axes[0].axhline(0, color="red", linestyle="--")

    # 2. Bits Comparison (Delta: 4-bit - 8-bit)
    key_cols = ["group", "rb", "rc", "rdt", "layer_mode", "quant"]
    pivot_b = df.pivot_table(index=key_cols, columns="bits", values="ppl").dropna()
    if 4 in pivot_b.columns and 8 in pivot_b.columns:
        pivot_b["delta_4_8"] = pivot_b[4] - pivot_b[8]
        sns.boxplot(data=pivot_b, y="delta_4_8", ax=axes[1], color="salmon")
        axes[1].set_title(
            "PPL Difference: 4-bit vs 8-bit\n(Precision Penalty)", fontweight="bold"
        )
        axes[1].axhline(0, color="red", linestyle="--")

    # 3. Layer Mode Effect
    sns.boxplot(x="layer_mode", y="ppl", data=df, ax=axes[2])
    axes[2].set_title("Impact of Layer Mode on PPL", fontweight="bold")

    # 4. Group Size Effect
    sns.boxplot(x="group", y="ppl", data=df, ax=axes[3])
    axes[3].set_title("Impact of Group Size on PPL", fontweight="bold")

    # 5. RDT effect when RC=16 and RB=16
    f_rdt = df[(df["rc"] == 16) & (df["rb"] == 16)]
    sns.lineplot(
        x="rdt",
        y="ppl",
        hue="bits",
        style="quant",
        data=f_rdt,
        ax=axes[4],
        marker="o",
        markersize=8,
    )
    axes[4].set_title("Effect of RDT (Fixed RC=16, RB=16)", fontweight="bold")

    # 6. RC effect when RDT=160 and RB=16
    f_rc = df[(df["rb"] == 16) & (df["rdt"] == 160)]
    sns.lineplot(
        x="rc",
        y="ppl",
        hue="bits",
        style="quant",
        data=f_rc,
        ax=axes[5],
        marker="s",
        markersize=8,
    )
    axes[5].set_title("Effect of RC (Fixed RDT=160, RB=16)", fontweight="bold")

    # 7. RB effect when RDT=160 and RC=16
    f_rb = df[(df["rc"] == 16) & (df["rdt"] == 160)]
    sns.lineplot(
        x="rb",
        y="ppl",
        hue="bits",
        style="quant",
        data=f_rb,
        ax=axes[6],
        marker="D",
        markersize=8,
    )
    axes[6].set_title("Effect of RB (Fixed RDT=160, RC=16)", fontweight="bold")

    # 8. Overall Distribution
    sns.violinplot(x="bits", y="ppl", hue="quant", data=df, split=True, ax=axes[7])
    axes[7].set_title("PPL Distribution by Bits and Quant", fontweight="bold")

    plt.tight_layout()
    plt.savefig("hadamard_analysis_summary.png", dpi=300)
    print("Analysis complete. Image saved as 'hadamard_analysis_summary.png'.")


# Execute
# big_model_with_viz("./hadamard_all_sweep_2.8b_optimized.csv")


def big_model_apfixed(csv_path):
    """
    For CSV with columns:
      W,I,rb,rc,rdt,layer_mode,ppl

    Prints pivot-based describe() summaries similar to your hadamard script.
    """

    df = pd.read_csv(csv_path)

    print("-" * 10)
    key_cols = ["rb", "rc", "rdt", "layer_mode"]
    pivot = df.pivot_table(index=key_cols, columns=["W", "I"], values="ppl").dropna()
    # optional delta vs (16,4) if present
    if (16, 4) in pivot.columns:
        for col in pivot.columns:
            if col != (16, 4):
                pivot[f"delta_{col[0]}_{col[1]}_minus_16_4"] = (
                    pivot[col] - pivot[(16, 4)]
                )
    print(pivot.describe())

    print("-" * 10)
    key_cols = ["W", "I", "rb", "rc", "rdt"]
    pivot = df.pivot_table(index=key_cols, columns="layer_mode", values="ppl").dropna()
    print(pivot.describe())

    print("-" * 10)
    key_cols = ["W", "I", "rb", "rc", "layer_mode"]
    pivot = df.pivot_table(index=key_cols, columns="rdt", values="ppl").dropna()
    print(pivot.describe())

    print("-" * 10)
    print("Effect of RDT when RC and RB are 16:")
    filtered_df = df[(df["rc"] == 16) & (df["rb"] == 16)]
    key_cols = ["W", "I", "layer_mode"]
    pivot_rdt = filtered_df.pivot_table(
        index=key_cols, columns="rdt", values="ppl"
    ).dropna()
    print(pivot_rdt.describe())

    print("-" * 10)
    print("Effect of RC when RDT is 160 and RB is 16:")
    filtered_df = df[(df["rb"] == 16) & (df["rdt"] == 160)]
    key_cols = ["W", "I", "layer_mode"]
    pivot_rc = filtered_df.pivot_table(
        index=key_cols, columns="rc", values="ppl"
    ).dropna()
    print(pivot_rc.describe())

    print("-" * 10)
    print("Effect of RB when RDT is 160 and RC is 16:")
    filtered_df = df[(df["rc"] == 16) & (df["rdt"] == 160)]
    key_cols = ["W", "I", "layer_mode"]
    pivot_rb = filtered_df.pivot_table(
        index=key_cols, columns="rb", values="ppl"
    ).dropna()
    print(pivot_rb.describe())


def big_model_apfixed_with_viz(csv_path, out_png="apfixed_analysis_summary.png"):
    """
    Makes a single 4x2 plot grid PNG, same style as your seaborn version.
    Works for CSV:
      W,I,rb,rc,rdt,layer_mode,ppl
    """
    df = pd.read_csv(csv_path)

    # formatting helpers
    df["W"] = df["W"].astype(int)
    df["I"] = df["I"].astype(int)
    df["rb"] = df["rb"].astype(int)
    df["rc"] = df["rc"].astype(int)
    df["rdt"] = df["rdt"].astype(int)
    df["layer_mode"] = df["layer_mode"].astype(str)
    df["ppl"] = df["ppl"].astype(float)

    df["apfixed"] = df.apply(lambda r: f"ap_fixed<{r['W']},{r['I']}>", axis=1)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 22))
    axes = axes.flatten()

    # 1) Fixed-point comparison (Delta vs baseline 16,4)
    key_cols = ["rb", "rc", "rdt", "layer_mode"]
    pivot_q = df.pivot_table(index=key_cols, columns=["W", "I"], values="ppl").dropna()

    if (16, 4) in pivot_q.columns and len(pivot_q.columns) > 1:
        # choose one comparison type for the "delta plot": compare all types vs baseline
        deltas = []
        labels = []
        for col in pivot_q.columns:
            if col == (16, 4):
                continue
            deltas.append((pivot_q[col] - pivot_q[(16, 4)]).values)
            labels.append(f"{col[0]},{col[1]}")

        # build a long-form df for seaborn boxplot
        delta_df = pd.DataFrame(
            {
                "delta_vs_16_4": np.concatenate(deltas),
                "type": np.repeat(labels, [len(d) for d in deltas]),
            }
        )
        sns.boxplot(
            data=delta_df, x="type", y="delta_vs_16_4", ax=axes[0], color="skyblue"
        )
        axes[0].set_title(
            "PPL Difference vs ap_fixed<16,4>\n(Negative = better than baseline)",
            fontweight="bold",
        )
        axes[0].axhline(0, color="red", linestyle="--")
        axes[0].set_xlabel("W,I")
        axes[0].set_ylabel("ppl - ppl(16,4)")
    else:
        sns.boxplot(x="apfixed", y="ppl", data=df, ax=axes[0], color="skyblue")
        axes[0].set_title("PPL by Fixed-Point Type", fontweight="bold")
        axes[0].tick_params(axis="x", rotation=30)

    # 2) RDT Comparison between fixed-point types (delta)
    # If you have multiple types, show delta (best type - baseline) per rdt
    if (16, 4) in pivot_q.columns and len(pivot_q.columns) > 1:
        # pick the "other" type with most samples (common case: (32,8))
        other_cols = [c for c in pivot_q.columns if c != (16, 4)]
        other = other_cols[0]

        tmp = df.pivot_table(
            index=["rb", "rc", "rdt", "layer_mode"], columns=["W", "I"], values="ppl"
        ).dropna()
        tmp = tmp.reset_index()
        tmp["delta_other_minus_16_4"] = tmp[other] - tmp[(16, 4)]

        sns.boxplot(
            x="rdt", y="delta_other_minus_16_4", data=tmp, ax=axes[1], color="salmon"
        )
        axes[1].set_title(
            f"PPL Delta vs 16,4 across RDT\n(other={other[0]},{other[1]})",
            fontweight="bold",
        )
        axes[1].axhline(0, color="red", linestyle="--")
        axes[1].set_xlabel("rdt")
        axes[1].set_ylabel("ppl(other) - ppl(16,4)")
        axes[1].tick_params(axis="x", rotation=30)
    else:
        axes[1].axis("off")
        axes[1].text(
            0.05,
            0.8,
            "Need >=2 fixed-point types\nand baseline (16,4) to show delta plots.",
            transform=axes[1].transAxes,
            fontsize=12,
        )

    # 3) Layer Mode Effect
    sns.boxplot(x="layer_mode", y="ppl", data=df, ax=axes[2])
    axes[2].set_title("Impact of Layer Mode on PPL", fontweight="bold")

    # 4) RB effect
    sns.boxplot(x="rb", y="ppl", hue="apfixed", data=df, ax=axes[3])
    axes[3].set_title("Impact of RB on PPL", fontweight="bold")
    axes[3].legend(title="ap_fixed", loc="best")

    # 5) RDT effect when RC=16 and RB=16
    f_rdt = df[(df["rc"] == 16) & (df["rb"] == 16)]
    if len(f_rdt) > 0:
        sns.lineplot(
            x="rdt",
            y="ppl",
            hue="apfixed",
            style="layer_mode",
            data=f_rdt,
            ax=axes[4],
            marker="o",
        )
        axes[4].set_title("Effect of RDT (Fixed RC=16, RB=16)", fontweight="bold")
    else:
        axes[4].axis("off")
        axes[4].text(
            0.05,
            0.8,
            "No rows with rc=16 & rb=16",
            transform=axes[4].transAxes,
            fontsize=12,
        )

    # 6) RC effect when RDT=160 and RB=16
    f_rc = df[(df["rb"] == 16) & (df["rdt"] == 160)]
    if len(f_rc) > 0:
        sns.lineplot(
            x="rc",
            y="ppl",
            hue="apfixed",
            style="layer_mode",
            data=f_rc,
            ax=axes[5],
            marker="s",
        )
        axes[5].set_title("Effect of RC (Fixed RDT=160, RB=16)", fontweight="bold")
    else:
        axes[5].axis("off")
        axes[5].text(
            0.05,
            0.8,
            "No rows with rb=16 & rdt=160",
            transform=axes[5].transAxes,
            fontsize=12,
        )

    # 7) RB effect when RDT=160 and RC=16
    f_rb = df[(df["rc"] == 16) & (df["rdt"] == 160)]
    if len(f_rb) > 0:
        sns.lineplot(
            x="rb",
            y="ppl",
            hue="apfixed",
            style="layer_mode",
            data=f_rb,
            ax=axes[6],
            marker="D",
        )
        axes[6].set_title("Effect of RB (Fixed RDT=160, RC=16)", fontweight="bold")
    else:
        axes[6].axis("off")
        axes[6].text(
            0.05,
            0.8,
            "No rows with rc=16 & rdt=160",
            transform=axes[6].transAxes,
            fontsize=12,
        )

    # 8) Overall Distribution (violin)
    sns.violinplot(x="apfixed", y="ppl", data=df, inner="quartile", ax=axes[7])
    axes[7].set_title("PPL Distribution by Fixed-Point Type", fontweight="bold")
    axes[7].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    print(f"Analysis complete. Image saved as '{out_png}'.")


# Execute like this:
# big_model_apfixed("./fixed.csv")
# big_model_apfixed_with_viz("./fixed.csv")
