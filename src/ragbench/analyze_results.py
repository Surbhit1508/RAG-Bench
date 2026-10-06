"""Aggregate raw per-question results into per-config summary stats,
run pairwise significance tests against the baseline config, and
produce the ablation comparison plots.

Run as: `python -m ragbench.analyze_results`
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ragbench import config
from ragbench.stats_testing import bootstrap_ci, paired_bootstrap_test

sns.set_theme(style="whitegrid")

RETRIEVAL_METRICS = ["retrieval_hit", "retrieval_reciprocal_rank", "retrieval_ndcg", "retrieval_precision"]
GENERATION_METRICS = ["gen_exact_match", "gen_f1", "gen_groundedness", "gen_is_abstention"]
BASELINE_CONFIG = "baseline_dense_fixed200"


def load_results() -> pd.DataFrame:
    return pd.read_csv(config.RESULTS_CSV)


def summarize(df: pd.DataFrame) -> dict:
    """Per-config mean + 95% bootstrap CI for every metric."""
    summary = {}
    for cfg_name, group in df.groupby("config_name"):
        cfg_summary = {"n": len(group)}
        for metric in RETRIEVAL_METRICS + GENERATION_METRICS:
            values = group[metric].astype(float).tolist()
            cfg_summary[metric] = bootstrap_ci(values)
        summary[cfg_name] = cfg_summary
    return summary


def significance_vs_baseline(df: pd.DataFrame, baseline: str = BASELINE_CONFIG) -> pd.DataFrame:
    """Paired bootstrap test of every config against the baseline, on
    the metrics that matter most for an ablation story: retrieval hit
    rate and answer F1.
    """
    base = df[df["config_name"] == baseline].sort_values("qa_id")
    rows = []
    for cfg_name, group in df.groupby("config_name"):
        if cfg_name == baseline:
            continue
        group = group.sort_values("qa_id")
        # align on qa_id in case of any ordering drift
        merged = base.merge(group, on="qa_id", suffixes=("_base", "_cfg"))
        for metric in ["retrieval_hit", "gen_f1"]:
            test = paired_bootstrap_test(
                merged[f"{metric}_cfg"].astype(float).tolist(),
                merged[f"{metric}_base"].astype(float).tolist(),
            )
            rows.append({"config": cfg_name, "metric": metric, **test})
    return pd.DataFrame(rows)


def plot_metric_comparison(summary: dict, metrics: list[str], title: str, out_path):
    configs = list(summary.keys())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        means = [summary[c][metric]["mean"] for c in configs]
        los = [summary[c][metric]["mean"] - summary[c][metric]["ci_low"] for c in configs]
        his = [summary[c][metric]["ci_high"] - summary[c][metric]["mean"] for c in configs]
        ax.barh(configs, means, xerr=[los, his], color=sns.color_palette("crest", len(configs)))
        ax.set_title(metric)
        ax.set_xlim(0, max(1.0, max(means) * 1.2))
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run_analysis() -> None:
    df = load_results()
    summary = summarize(df)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {config.SUMMARY_JSON}")

    sig_df = significance_vs_baseline(df)
    sig_path = config.RESULTS_DIR / "significance_vs_baseline.csv"
    sig_df.to_csv(sig_path, index=False)
    print(f"Wrote significance tests -> {sig_path}")

    plot_metric_comparison(
        summary, RETRIEVAL_METRICS, "Retrieval quality by config (95% bootstrap CI)",
        config.RESULTS_DIR / "retrieval_comparison.png",
    )
    plot_metric_comparison(
        summary, GENERATION_METRICS, "Generation quality by config (95% bootstrap CI)",
        config.RESULTS_DIR / "generation_comparison.png",
    )
    print(f"Wrote plots -> {config.RESULTS_DIR}")

    print("\n=== Significant differences vs baseline (p < 0.05) ===")
    sig_only = sig_df[sig_df["significant_at_0.05"]]
    if sig_only.empty:
        print("(none found)")
    else:
        print(sig_only.to_string(index=False))


if __name__ == "__main__":
    run_analysis()
