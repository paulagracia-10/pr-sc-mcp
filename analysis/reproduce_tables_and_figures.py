"""
Reproduce Live Validation tables and Figure 2 for the PR-SC MCP study.

Expected repository layout:

results/live_validation/gpt_oss_120b/run_summaries/*.csv
results/live_validation/gemma_4_31b/run_summaries/*.csv
results/live_validation/zai_glm_4_7/run_summaries/*.csv

The script computes:
- Table 5 model-variant means and SDs.
- Table 6 Full PR-SC MCP deltas vs Exhaustive MCP Baseline.
- Table 7 paired run-level confidence intervals.
- Figure 2 token and latency deltas.

"""

from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "live_validation"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

MODEL_FOLDERS = {
    "gpt-oss-120b": RESULTS / "gpt_oss_120b" / "run_summaries",
    "gemma-4-31b": RESULTS / "gemma_4_31b" / "run_summaries",
    "zai-glm-4.7": RESULTS / "zai_glm_4_7" / "run_summaries",
}

VARIANT_MAP = {
    "Naive MCP": "Exhaustive MCP",
    "Exhaustive MCP Baseline": "Exhaustive MCP",
    "Routing + Packaging, no compression": "Routing + Packaging",
}

BASELINE = "Exhaustive MCP"
FULL = "Full PR-SC MCP"


def load_run_summaries():
    frames = []
    for model, folder in MODEL_FOLDERS.items():
        if not folder.exists():
            print(f"[WARN] Missing folder: {folder}")
            continue
        files = sorted(folder.glob("*.csv"))
        if not files:
            print(f"[WARN] No CSV files found in: {folder}")
            continue
        for idx, path in enumerate(files, start=1):
            df = pd.read_csv(path)
            df["source_file"] = path.name
            df["run_batch"] = idx
            df["model_folder"] = model
            if "variant" in df.columns:
                df["variant"] = df["variant"].replace(VARIANT_MAP)
            frames.append(df)
    if not frames:
        raise SystemExit("No run-summary CSV files found. Check the results/live_validation folder structure.")
    return pd.concat(frames, ignore_index=True)


def model_variant_summary(df):
    required = ["model", "variant", "total_tokens", "latency_ms"]
    for col in required:
        if col not in df.columns:
            raise SystemExit(f"Required column missing: {col}")
    # First aggregate by model, variant, run batch to match the paper's run-level SD logic.
    batch = (
        df.groupby(["model", "variant", "run_batch"], as_index=False)
          .agg(mean_total_tokens=("total_tokens", "mean"), mean_latency_ms=("latency_ms", "mean"))
    )
    table5 = (
        batch.groupby(["model", "variant"], as_index=False)
             .agg(
                 mean_total_tokens=("mean_total_tokens", "mean"),
                 sd_total_tokens=("mean_total_tokens", "std"),
                 mean_latency_ms=("mean_latency_ms", "mean"),
                 sd_latency_ms=("mean_latency_ms", "std"),
             )
    )
    table5.to_csv(OUT / "live_validation_table5_model_variant_summary.csv", index=False)
    return batch, table5


def full_vs_baseline_delta(table5):
    rows = []
    for model, g in table5.groupby("model"):
        full = g[g["variant"] == FULL]
        base = g[g["variant"] == BASELINE]
        if full.empty or base.empty:
            print(f"[WARN] Missing Full or baseline rows for {model}")
            continue
        f = full.iloc[0]
        b = base.iloc[0]
        rows.append({
            "model": model,
            "token_delta_pct": 100 * (f["mean_total_tokens"] - b["mean_total_tokens"]) / b["mean_total_tokens"],
            "latency_delta_pct": 100 * (f["mean_latency_ms"] - b["mean_latency_ms"]) / b["mean_latency_ms"],
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "live_validation_table6_full_vs_exhaustive_delta.csv", index=False)
    return out


def t_critical_95_df4():
    # Fixed because the paper uses five run-level paired differences: df = 4.
    return 2.7764451051977987


def confidence_intervals(batch):
    rows = []
    for model, g in batch.groupby("model"):
        full = g[g["variant"] == FULL].set_index("run_batch")
        base = g[g["variant"] == BASELINE].set_index("run_batch")
        common = sorted(set(full.index).intersection(set(base.index)))
        if len(common) < 2:
            print(f"[WARN] Not enough paired run batches for {model}")
            continue
        for metric, label in [("mean_latency_ms", "Latency difference, ms"), ("mean_total_tokens", "Total token difference")]:
            diffs = (full.loc[common, metric] - base.loc[common, metric]).astype(float)
            mean = diffs.mean()
            sd = diffs.std(ddof=1)
            se = sd / math.sqrt(len(diffs))
            tcrit = t_critical_95_df4() if len(diffs) == 5 else None
            if tcrit is None:
                ci_low = ci_high = float("nan")
            else:
                ci_low = mean - tcrit * se
                ci_high = mean + tcrit * se
            rows.append({
                "model": model,
                "metric": label,
                "n_run_batches": len(diffs),
                "mean_full_minus_exhaustive": mean,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "statistical_ci_run_level_differences.csv", index=False)
    return out


def plot_figure2(delta):
    if delta.empty:
        print("[WARN] No delta data; skipping Figure 2")
        return
    order = ["gpt-oss-120b", "gemma-4-31b", "zai-glm-4.7"]
    delta = delta.set_index("model").loc[[m for m in order if m in delta["model"].values]].reset_index()
    colors = ["#4E79A7", "#59A14F", "#F28E2B"][:len(delta)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=160)
    fig.suptitle("Full PR-SC MCP vs Exhaustive MCP Baseline, across three models", fontsize=13, fontweight="bold")

    for ax, col, title, ylabel, ylim in [
        (axes[0], "token_delta_pct", "Token delta", "Change in total tokens (%)", (-2, 4)),
        (axes[1], "latency_delta_pct", "Latency delta", "Change in mean latency (%)", (-35, 5)),
    ]:
        bars = ax.bar(delta["model"], delta[col], color=colors, edgecolor="#333333", linewidth=0.6)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        for b, v in zip(bars, delta[col]):
            ax.text(
                b.get_x() + b.get_width() / 2,
                v + (0.2 if v >= 0 else -1.0),
                f"{v:+.2f}%",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=9,
                fontweight="bold",
            )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "figure2_full_vs_exhaustive_mcp.png", bbox_inches="tight")
    plt.close(fig)


def main():
    df = load_run_summaries()
    batch, table5 = model_variant_summary(df)
    delta = full_vs_baseline_delta(table5)
    confidence_intervals(batch)
    plot_figure2(delta)
    print("[OK] Analysis outputs written to:", OUT)


if __name__ == "__main__":
    main()
