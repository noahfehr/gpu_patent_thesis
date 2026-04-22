from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import LDAConfig, PROJECT_ROOT
from data import prepare_patent_corpus
from model import fit_lda_run
from stability import compute_topic_stability


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "code" / "topic_modeling" / "lda_pipeline" / "motivation" / "outputs" / "alpha_eta"
)


def parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def alpha_label(value: float) -> str:
    return f"alpha={value:g}"


def eta_label(value: float) -> str:
    return f"eta={value:g}"


def config_label(alpha: float, eta: float) -> str:
    return f"a={alpha:g}, e={eta:g}"


def run_alpha_topic_sparsity_analysis(
    token_lists: list[list[str]],
    lens_ids: list[str],
    base_config: LDAConfig,
    alpha_values: list[float],
    topic_threshold: float,
    analysis_seed: int,
) -> pd.DataFrame:
    rows = []

    for alpha in alpha_values:
        run_config = replace(base_config, alpha=alpha, seed=analysis_seed)
        run = fit_lda_run(token_lists=token_lists, lens_ids=lens_ids, config=run_config)
        doc_topic_df = run["doc_topic_df"]
        topic_cols = [c for c in doc_topic_df.columns if c.startswith("topic_")]
        active_topic_counts = (doc_topic_df[topic_cols] > topic_threshold).sum(axis=1)

        for active_count in active_topic_counts.tolist():
            rows.append(
                {
                    "alpha": alpha,
                    "active_topics_above_threshold": int(active_count),
                    "topic_threshold": topic_threshold,
                }
            )

    return pd.DataFrame(rows)


def plot_alpha_topic_sparsity(
    sparsity_df: pd.DataFrame,
    output_path: Path,
    chosen_alpha: float,
    topic_threshold: float,
) -> None:
    pivot = (
        sparsity_df.groupby(["alpha", "active_topics_above_threshold"])
        .size()
        .rename("frequency")
        .reset_index()
    )

    plt.figure(figsize=(12, 7))
    for alpha in sorted(pivot["alpha"].unique()):
        subset = pivot[pivot["alpha"] == alpha].sort_values("active_topics_above_threshold")
        linewidth = 3 if alpha == chosen_alpha else 1.8
        alpha_value = 1.0 if alpha == chosen_alpha else 0.8
        plt.plot(
            subset["active_topics_above_threshold"],
            subset["frequency"],
            marker="o",
            linewidth=linewidth,
            alpha=alpha_value,
            label=alpha_label(alpha),
        )

    plt.xlabel(f"Number of Topics per Document with Weight > {topic_threshold}")
    plt.ylabel("Document Frequency")
    plt.title("Alpha Motivation: Topic Sparsity per Document")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_eta_topic_concentration_analysis(
    token_lists: list[list[str]],
    lens_ids: list[str],
    base_config: LDAConfig,
    eta_values: list[float],
    top_n_words: int,
    analysis_seed: int,
) -> pd.DataFrame:
    rows = []

    for eta in eta_values:
        run_config = replace(base_config, eta=eta, seed=analysis_seed)
        run = fit_lda_run(token_lists=token_lists, lens_ids=lens_ids, config=run_config)
        model = run["model"]

        for topic_id in range(model.k):
            top_words = model.get_topic_words(topic_id, top_n=top_n_words)
            cumulative_prob = float(sum(prob for _, prob in top_words))
            rows.append(
                {
                    "eta": eta,
                    "topic_id": topic_id,
                    "top_n_words": top_n_words,
                    "cumulative_top_word_probability": cumulative_prob,
                }
            )

    return pd.DataFrame(rows)


def plot_eta_topic_concentration(
    concentration_df: pd.DataFrame,
    output_path: Path,
    chosen_eta: float,
    top_n_words: int,
) -> None:
    eta_values = sorted(concentration_df["eta"].unique())
    grouped = [
        concentration_df.loc[
            concentration_df["eta"] == eta, "cumulative_top_word_probability"
        ].tolist()
        for eta in eta_values
    ]

    plt.figure(figsize=(12, 7))
    box = plt.boxplot(grouped, patch_artist=True, labels=[f"{eta:g}" for eta in eta_values])
    for patch, eta in zip(box["boxes"], eta_values):
        if eta == chosen_eta:
            patch.set_facecolor("#2E86C1")
            patch.set_alpha(0.9)
        else:
            patch.set_facecolor("#AED6F1")
            patch.set_alpha(0.8)

    means = concentration_df.groupby("eta")["cumulative_top_word_probability"].mean()
    plt.plot(
        range(1, len(eta_values) + 1),
        [means[eta] for eta in eta_values],
        color="#1B4F72",
        marker="o",
        linewidth=2,
        label="Mean",
    )

    plt.xlabel("Eta")
    plt.ylabel(f"Cumulative Probability of Top {top_n_words} Words")
    plt.title("Eta Motivation: Topic Concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_alpha_eta_stability_sweep(
    token_lists: list[list[str]],
    lens_ids: list[str],
    base_config: LDAConfig,
    alpha_values: list[float],
    eta_values: list[float],
    stability_seeds: list[int],
) -> pd.DataFrame:
    rows = []

    for alpha in alpha_values:
        for eta in eta_values:
            sweep_config = replace(base_config, alpha=alpha, eta=eta)
            results_df, _ = compute_topic_stability(
                tokenized_docs=token_lists,
                lens_ids=lens_ids,
                base_config=sweep_config,
                k_values=[base_config.k],
                seeds=stability_seeds,
                vocab_min_df=base_config.min_df,
            )
            result = results_df.iloc[0].to_dict()
            result["alpha"] = alpha
            result["eta"] = eta
            result["config_label"] = config_label(alpha, eta)
            rows.append(result)

    return pd.DataFrame(rows)


def plot_stability_bar(
    stability_df: pd.DataFrame,
    output_path: Path,
    chosen_alpha: float,
    chosen_eta: float,
) -> None:
    plot_df = stability_df.sort_values(["alpha", "eta"]).reset_index(drop=True)
    labels = plot_df["config_label"].tolist()
    colors = []
    for _, row in plot_df.iterrows():
        if row["alpha"] == chosen_alpha and row["eta"] == chosen_eta:
            colors.append("#C0392B")
        else:
            colors.append("#5DADE2")

    plt.figure(figsize=(14, 7))
    plt.bar(range(len(plot_df)), plot_df["omega"], color=colors)
    plt.xticks(range(len(plot_df)), labels, rotation=45, ha="right")
    plt.ylabel("McDonald's Omega")
    plt.xlabel("(Alpha, Eta) Configuration")
    plt.title("Stability Motivation: Omega Across Alpha/Eta Configurations")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_stability_heatmap(
    stability_df: pd.DataFrame,
    output_path: Path,
    chosen_alpha: float,
    chosen_eta: float,
) -> None:
    pivot = stability_df.pivot(index="alpha", columns="eta", values="omega").sort_index().sort_index(axis=1)
    alpha_values = pivot.index.tolist()
    eta_values = pivot.columns.tolist()
    matrix = pivot.to_numpy(dtype=float)

    plt.figure(figsize=(9, 7))
    im = plt.imshow(matrix, cmap="YlGnBu", aspect="auto")
    plt.colorbar(im, label="McDonald's Omega")
    plt.xticks(range(len(eta_values)), [f"{eta:g}" for eta in eta_values])
    plt.yticks(range(len(alpha_values)), [f"{alpha:g}" for alpha in alpha_values])
    plt.xlabel("Eta")
    plt.ylabel("Alpha")
    plt.title("Stability Heatmap Across Alpha/Eta Configurations")

    for i, alpha in enumerate(alpha_values):
        for j, eta in enumerate(eta_values):
            value = matrix[i, j]
            marker = " *" if alpha == chosen_alpha and eta == chosen_eta else ""
            plt.text(j, i, f"{value:.3f}{marker}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_summary(
    base_config: LDAConfig,
    n_docs: int,
    alpha_values: list[float],
    eta_values: list[float],
    stability_seeds: list[int],
    topic_threshold: float,
    top_n_words: int,
    alpha_df: pd.DataFrame,
    eta_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: Path,
) -> dict:
    alpha_summary = (
        alpha_df.groupby("alpha")["active_topics_above_threshold"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    eta_summary = (
        eta_df.groupby("eta")["cumulative_top_word_probability"]
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )
    best_row = stability_df.sort_values("omega", ascending=False).iloc[0]

    return {
        "n_docs": int(n_docs),
        "k": int(base_config.k),
        "min_df": int(base_config.min_df),
        "max_df": base_config.max_df,
        "iterations": int(base_config.iterations),
        "alpha_values": alpha_values,
        "eta_values": eta_values,
        "stability_seeds": stability_seeds,
        "topic_threshold": topic_threshold,
        "top_n_words": top_n_words,
        "chosen_alpha": base_config.alpha,
        "chosen_eta": base_config.eta,
        "alpha_topic_sparsity_summary": alpha_summary.to_dict(orient="records"),
        "eta_topic_concentration_summary": eta_summary.to_dict(orient="records"),
        "best_stability_config": {
            "alpha": float(best_row["alpha"]),
            "eta": float(best_row["eta"]),
            "omega": float(best_row["omega"]),
        },
        "caption_logic": {
            "alpha": (
                "We select alpha to reflect the assumption that patents are centered "
                "on a small number of architectural concepts, rather than diffuse mixtures."
            ),
            "eta": (
                "Lower eta yields sharper, more interpretable topics corresponding "
                "to distinct architectural primitives."
            ),
            "stability": (
                "We select hyperparameters that maximize topic stability across runs."
            ),
        },
        "outputs_dir": str(output_dir),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run motivation analyses for alpha and eta: document topic sparsity, "
            "topic concentration, and stability sweeps."
        )
    )
    defaults = LDAConfig()
    parser.add_argument("--version-prefix", default=defaults.version_prefix)
    parser.add_argument("--predictions-path", type=Path, default=defaults.predictions_path)
    parser.add_argument("--base-data-dir", type=Path, default=defaults.base_data_dir)
    parser.add_argument("--text-column", default=defaults.text_column)
    parser.add_argument("--id-column", default=defaults.id_column)
    parser.add_argument("--stopwords-path", type=Path, default=defaults.stopwords_path)
    parser.add_argument("--k", type=int, default=defaults.k)
    parser.add_argument("--min-bigram-count", type=int, default=defaults.min_bigram_count)
    parser.add_argument("--min-df", type=int, default=defaults.min_df)
    parser.add_argument("--max-df", type=float, default=defaults.max_df)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--top-words-n", type=int, default=defaults.top_words_n)

    parser.add_argument("--alpha-values", default="0.05,0.1,0.3,0.5")
    parser.add_argument("--eta-values", default="0.005,0.01,0.05,0.1")
    parser.add_argument("--analysis-seed", type=int, default=defaults.seed)
    parser.add_argument("--stability-seeds", default="0,1,2,3,4")
    parser.add_argument("--topic-threshold", type=float, default=0.05)
    parser.add_argument("--top-n-words", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    alpha_values = parse_float_list(args.alpha_values)
    eta_values = parse_float_list(args.eta_values)
    stability_seeds = parse_int_list(args.stability_seeds)

    base_config = LDAConfig(
        version_prefix=args.version_prefix,
        predictions_path=args.predictions_path,
        stopwords_path=args.stopwords_path,
        base_data_dir=args.base_data_dir,
        text_column=args.text_column,
        id_column=args.id_column,
        min_bigram_count=args.min_bigram_count,
        k=args.k,
        alpha=min(alpha_values, key=lambda value: abs(value - LDAConfig().alpha)),
        eta=min(eta_values, key=lambda value: abs(value - LDAConfig().eta)),
        min_df=args.min_df,
        max_df=args.max_df,
        seed=args.analysis_seed,
        iterations=args.iterations,
        top_words_n=args.top_words_n,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing corpus for alpha/eta motivation...")
    df_docs = prepare_patent_corpus(base_config)
    token_lists = df_docs["tokens"].tolist()
    lens_ids = df_docs["lens_id"].tolist()

    print("\nRunning alpha topic sparsity analysis...")
    alpha_df = run_alpha_topic_sparsity_analysis(
        token_lists=token_lists,
        lens_ids=lens_ids,
        base_config=base_config,
        alpha_values=alpha_values,
        topic_threshold=args.topic_threshold,
        analysis_seed=args.analysis_seed,
    )

    print("\nRunning eta topic concentration analysis...")
    eta_df = run_eta_topic_concentration_analysis(
        token_lists=token_lists,
        lens_ids=lens_ids,
        base_config=base_config,
        eta_values=eta_values,
        top_n_words=args.top_n_words,
        analysis_seed=args.analysis_seed,
    )

    print("\nRunning alpha/eta stability sweep...")
    stability_df = run_alpha_eta_stability_sweep(
        token_lists=token_lists,
        lens_ids=lens_ids,
        base_config=base_config,
        alpha_values=alpha_values,
        eta_values=eta_values,
        stability_seeds=stability_seeds,
    )

    alpha_csv = output_dir / "alpha_topic_sparsity.csv"
    eta_csv = output_dir / "eta_topic_concentration.csv"
    stability_csv = output_dir / "alpha_eta_stability.csv"
    summary_json = output_dir / "alpha_eta_motivation_summary.json"

    alpha_plot = output_dir / "alpha_topic_sparsity.png"
    eta_plot = output_dir / "eta_topic_concentration.png"
    stability_bar = output_dir / "alpha_eta_stability_bar.png"
    stability_heatmap = output_dir / "alpha_eta_stability_heatmap.png"

    alpha_df.to_csv(alpha_csv, index=False)
    eta_df.to_csv(eta_csv, index=False)
    stability_df.to_csv(stability_csv, index=False)

    plot_alpha_topic_sparsity(
        sparsity_df=alpha_df,
        output_path=alpha_plot,
        chosen_alpha=base_config.alpha,
        topic_threshold=args.topic_threshold,
    )
    plot_eta_topic_concentration(
        concentration_df=eta_df,
        output_path=eta_plot,
        chosen_eta=base_config.eta,
        top_n_words=args.top_n_words,
    )
    plot_stability_bar(
        stability_df=stability_df,
        output_path=stability_bar,
        chosen_alpha=base_config.alpha,
        chosen_eta=base_config.eta,
    )
    plot_stability_heatmap(
        stability_df=stability_df,
        output_path=stability_heatmap,
        chosen_alpha=base_config.alpha,
        chosen_eta=base_config.eta,
    )

    summary = build_summary(
        base_config=base_config,
        n_docs=len(df_docs),
        alpha_values=alpha_values,
        eta_values=eta_values,
        stability_seeds=stability_seeds,
        topic_threshold=args.topic_threshold,
        top_n_words=args.top_n_words,
        alpha_df=alpha_df,
        eta_df=eta_df,
        stability_df=stability_df,
        output_dir=output_dir,
    )
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Patent count analyzed: {len(df_docs)}")
    print(f"Alpha sparsity plot: {alpha_plot}")
    print(f"Eta concentration plot: {eta_plot}")
    print(f"Stability bar plot: {stability_bar}")
    print(f"Stability heatmap: {stability_heatmap}")
    print(f"Summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
