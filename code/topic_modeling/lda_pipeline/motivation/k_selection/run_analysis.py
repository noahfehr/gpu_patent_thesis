from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from itertools import combinations
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import LDAConfig
from data import prepare_patent_corpus
from model import fit_lda_run
from stability import (
    build_fixed_vocab,
    compute_topic_similarity_matrix,
    doc_topic_matrix_from_df,
    summarize_stability_from_runs,
    summarize_topic_alignment,
    topic_word_matrix_from_model,
)
from motivation.k_selection.config import KSweepConfig


def parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def build_parser() -> argparse.ArgumentParser:
    defaults = LDAConfig()
    ksweep_defaults = KSweepConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Run a k sweep for LDA, save per-run artifacts, compute pairwise topic "
            "alignment metrics, and generate k-selection motivation outputs."
        )
    )
    parser.add_argument("--version-prefix", default=defaults.version_prefix)
    parser.add_argument("--predictions-path", type=Path, default=defaults.predictions_path)
    parser.add_argument("--base-data-dir", type=Path, default=defaults.base_data_dir)
    parser.add_argument("--stopwords-path", type=Path, default=defaults.stopwords_path)
    parser.add_argument("--text-column", default=defaults.text_column)
    parser.add_argument("--id-column", default=defaults.id_column)
    parser.add_argument("--min-bigram-count", type=int, default=ksweep_defaults.min_bigram_count)
    parser.add_argument("--alpha", type=float, default=ksweep_defaults.alpha)
    parser.add_argument("--eta", type=float, default=ksweep_defaults.eta)
    parser.add_argument("--min-df", type=int, default=ksweep_defaults.min_df)
    parser.add_argument("--max-df", type=float, default=ksweep_defaults.max_df)
    parser.add_argument("--iterations", type=int, default=ksweep_defaults.iterations)
    parser.add_argument("--top-n-words", type=int, default=ksweep_defaults.top_n_words)
    parser.add_argument("--k-values", default="20,30,40,50,60")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--heatmap-k-values", default="30,40")
    parser.add_argument("--qualitative-k-values", default="30,40")
    parser.add_argument("--top-split-examples", type=int, default=ksweep_defaults.top_split_examples)
    parser.add_argument("--output-dir", type=Path, default=ksweep_defaults.output_dir)
    return parser


def choose_pair_for_heatmap(pairwise_df: pd.DataFrame) -> tuple[int, int]:
    if pairwise_df.empty:
        raise ValueError("No pairwise alignment rows available.")
    ranked = pairwise_df.assign(split_signal=pairwise_df["top3_mean"] - pairwise_df["top1_mean"])
    best = ranked.sort_values(["split_signal", "top3_mean"], ascending=False).iloc[0]
    return int(best["seed_a"]), int(best["seed_b"])


def save_similarity_heatmap(sim_matrix: np.ndarray, output_base: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(sim_matrix, cmap="YlGnBu", aspect="auto")
    fig.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_xlabel("Run B Topic ID")
    ax.set_ylabel("Run A Topic ID")
    ax.set_title(title)
    ax.set_xticks(range(sim_matrix.shape[1]))
    ax.set_yticks(range(sim_matrix.shape[0]))
    plt.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=200)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def save_stability_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.sort_values("k")
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        plot_df["k"],
        plot_df["omega_mean"],
        yerr=plot_df["omega_se_mean"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#1F618D",
    )
    plt.xticks(plot_df["k"])
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("McDonald's Omega")
    plt.title("Stability Across Candidate k Values")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_alignment_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.sort_values("k")
    plt.figure(figsize=(10, 6))
    plt.plot(plot_df["k"], plot_df["top1_mean"], marker="o", linewidth=2, label="top1")
    plt.plot(plot_df["k"], plot_df["top2_mean"], marker="o", linewidth=2, label="top2_sum")
    plt.plot(plot_df["k"], plot_df["top3_mean"], marker="o", linewidth=2, label="top3_sum")
    plt.xticks(plot_df["k"])
    plt.xlabel("Number of Topics (k)")
    plt.ylabel("Alignment Score")
    plt.title("Alignment Structure Across Candidate k Values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_tradeoff_plot(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.sort_values("k")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].errorbar(
        plot_df["k"],
        plot_df["omega_mean"],
        yerr=plot_df["omega_se_mean"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#7D3C98",
    )
    axes[0].set_ylabel("Omega")
    axes[0].set_title("Stability and Alignment Tradeoff Across k")

    axes[1].plot(plot_df["k"], plot_df["top1_mean"], marker="o", linewidth=2, label="top1")
    axes[1].plot(plot_df["k"], plot_df["top2_mean"], marker="o", linewidth=2, label="top2_sum")
    axes[1].plot(plot_df["k"], plot_df["top3_mean"], marker="o", linewidth=2, label="top3_sum")
    axes[1].set_xlabel("Number of Topics (k)")
    axes[1].set_ylabel("Alignment")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close(fig)


def build_run_artifact(
    run: dict,
    phi: np.ndarray,
    theta: np.ndarray,
    run_dir: Path,
    seed: int,
    k: int,
    lens_ids: list[str],
) -> dict:
    np.save(run_dir / "phi.npy", phi)
    np.save(run_dir / "theta.npy", theta)
    save_json(run_dir / "top_words.json", run["topic_words"])
    save_json(
        run_dir / "model_meta.json",
        {
            "seed": seed,
            "k": k,
            "metrics": run["metrics"],
            "n_docs": len(lens_ids),
            "topic_ids": list(range(k)),
        },
    )
    save_json(run_dir / "doc_ids.json", {"lens_ids": lens_ids})

    return {
        "seed": seed,
        "run_dir": run_dir,
        "model": run["model"],
        "doc_topic_df": run["doc_topic_df"],
        "phi": phi,
        "theta": theta,
        "top_words": run["topic_words"],
        "metrics": run["metrics"],
    }


def compute_pairwise_alignment_rows(
    k: int,
    run_records: list[dict],
    k_dir: Path,
) -> tuple[pd.DataFrame, dict[tuple[int, int], dict]]:
    pair_rows = []
    pair_artifacts: dict[tuple[int, int], dict] = {}
    pairs_dir = k_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)

    for run_a, run_b in combinations(run_records, 2):
        seed_a = int(run_a["seed"])
        seed_b = int(run_b["seed"])
        pair_dir = pairs_dir / f"seed_{seed_a}__seed_{seed_b}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        sim_matrix = compute_topic_similarity_matrix(run_a["phi"], run_b["phi"])
        per_topic_df, summary = summarize_topic_alignment(sim_matrix)
        np.save(pair_dir / "sim_matrix.npy", sim_matrix)
        per_topic_df.to_csv(pair_dir / "per_topic_alignment.csv", index=False)
        save_json(pair_dir / "pair_metrics.json", summary)

        pair_row = {"k": k, "seed_a": seed_a, "seed_b": seed_b, **summary}
        pair_rows.append(pair_row)
        pair_artifacts[(seed_a, seed_b)] = {
            "sim_matrix": sim_matrix,
            "per_topic_df": per_topic_df,
            "summary": summary,
            "run_a": run_a,
            "run_b": run_b,
            "pair_dir": pair_dir,
        }

    pairwise_df = pd.DataFrame(pair_rows)
    if not pairwise_df.empty:
        pairwise_df.to_csv(k_dir / "pairwise_alignment_summary.csv", index=False)

    return pairwise_df, pair_artifacts


def aggregate_pairwise_metrics(pairwise_df: pd.DataFrame) -> dict[str, float]:
    if pairwise_df.empty:
        return {
            "top1_mean": float("nan"),
            "top1_std": float("nan"),
            "top2_mean": float("nan"),
            "top2_std": float("nan"),
            "top3_mean": float("nan"),
            "top3_std": float("nan"),
            "max_cosine_mean": float("nan"),
            "mean_cosine_mean": float("nan"),
        }

    return {
        "top1_mean": float(pairwise_df["top1_mean"].mean()),
        "top1_std": float(pairwise_df["top1_mean"].std(ddof=0)),
        "top2_mean": float(pairwise_df["top2_mean"].mean()),
        "top2_std": float(pairwise_df["top2_mean"].std(ddof=0)),
        "top3_mean": float(pairwise_df["top3_mean"].mean()),
        "top3_std": float(pairwise_df["top3_mean"].std(ddof=0)),
        "max_cosine_mean": float(pairwise_df["max_cosine_mean"].mean()),
        "mean_cosine_mean": float(pairwise_df["mean_cosine_mean"].mean()),
    }


def build_topic_split_examples(
    k: int,
    pair_artifact: dict,
    top_n_words: int,
    top_split_examples: int,
) -> list[dict]:
    per_topic_df = pair_artifact["per_topic_df"].copy()
    sim_matrix = pair_artifact["sim_matrix"]
    run_a = pair_artifact["run_a"]
    run_b = pair_artifact["run_b"]

    per_topic_df["split_score"] = per_topic_df["top3_sum"] - per_topic_df["top1"]
    candidate_df = per_topic_df[
        (per_topic_df["top1"] < 0.75) & (per_topic_df["top3_sum"] > 0.85)
    ].copy()
    if candidate_df.empty:
        candidate_df = per_topic_df.copy()

    candidate_df = candidate_df.sort_values(["split_score", "top3_sum"], ascending=False).head(top_split_examples)

    examples = []
    for _, row in candidate_df.iterrows():
        topic_a_id = int(row["topic_id"])
        match_ids = row["top3_match_topic_ids"]
        matches = []
        for topic_b_id in match_ids:
            topic_b_id = int(topic_b_id)
            matches.append(
                {
                    "topic_b_id": topic_b_id,
                    "similarity": float(sim_matrix[topic_a_id, topic_b_id]),
                    "top_words": run_b["top_words"][topic_b_id][:top_n_words],
                }
            )

        examples.append(
            {
                "topic_a_id": topic_a_id,
                "topic_a_top_words": run_a["top_words"][topic_a_id][:top_n_words],
                "top1": float(row["top1"]),
                "top2_sum": float(row["top2_sum"]),
                "top3_sum": float(row["top3_sum"]),
                "split_score": float(row["split_score"]),
                "matches": matches,
            }
        )

    return examples


def write_split_report_markdown(
    k: int,
    seed_a: int,
    seed_b: int,
    examples: list[dict],
    output_path: Path,
) -> None:
    lines = [
        f"# Topic Splitting Examples for k={k}",
        "",
        f"Run pair: seed {seed_a} vs seed {seed_b}",
        "",
    ]

    for ex in examples:
        lines.append(f"## Topic A {ex['topic_a_id']}")
        lines.append("")
        lines.append(f"- Top words: {', '.join(ex['topic_a_top_words'])}")
        lines.append(f"- top1: {ex['top1']:.3f}")
        lines.append(f"- top2_sum: {ex['top2_sum']:.3f}")
        lines.append(f"- top3_sum: {ex['top3_sum']:.3f}")
        lines.append(f"- split_score: {ex['split_score']:.3f}")
        lines.append("")
        lines.append("Matches:")
        for match in ex["matches"]:
            lines.append(
                f"- Topic B {match['topic_b_id']} | sim={match['similarity']:.3f} | "
                f"{', '.join(match['top_words'])}"
            )
        lines.append("")

    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    k_values = parse_int_list(args.k_values)
    seeds = parse_int_list(args.seeds)
    heatmap_k_values = parse_int_list(args.heatmap_k_values)
    qualitative_k_values = parse_int_list(args.qualitative_k_values)

    ksweep_config = KSweepConfig(
        k_values=k_values,
        n_seeds=len(seeds),
        alpha=args.alpha,
        eta=args.eta,
        min_df=args.min_df,
        max_df=args.max_df,
        min_bigram_count=args.min_bigram_count,
        iterations=args.iterations,
        top_n_words=args.top_n_words,
        heatmap_k_values=heatmap_k_values,
        qualitative_k_values=qualitative_k_values,
        top_split_examples=args.top_split_examples,
        output_dir=args.output_dir,
    )

    base_config = LDAConfig(
        version_prefix=args.version_prefix,
        predictions_path=args.predictions_path,
        stopwords_path=args.stopwords_path,
        base_data_dir=args.base_data_dir,
        text_column=args.text_column,
        id_column=args.id_column,
        min_bigram_count=args.min_bigram_count,
        k=k_values[0],
        alpha=args.alpha,
        eta=args.eta,
        min_df=args.min_df,
        max_df=args.max_df,
        seed=seeds[0],
        iterations=args.iterations,
        top_words_n=args.top_n_words,
    )

    output_dir = ksweep_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", {"k_sweep_config": asdict(ksweep_config), "base_lda_config": asdict(base_config), "seeds": seeds})

    print("Preparing corpus for k selection analysis...")
    df_docs = prepare_patent_corpus(base_config)
    token_lists = df_docs["tokens"].tolist()
    lens_ids = df_docs["lens_id"].tolist()
    shared_vocab_list, shared_vocab_index = build_fixed_vocab(token_lists, min_df=base_config.min_df)
    avg_tokens_per_doc = float(np.mean([len(tokens) for tokens in token_lists]))

    overall_rows = []
    k_pair_artifacts: dict[int, dict[tuple[int, int], dict]] = {}

    for k in k_values:
        print(f"\n=== Running k={k} ===")
        k_dir = output_dir / f"k_{k}"
        k_dir.mkdir(parents=True, exist_ok=True)

        k_start = time.time()
        run_records = []

        for seed in seeds:
            print(f"Fitting seed {seed} for k={k}")
            run_config = replace(base_config, k=k, seed=seed)
            run = fit_lda_run(token_lists=token_lists, lens_ids=lens_ids, config=run_config)

            phi = topic_word_matrix_from_model(run["model"], shared_vocab_index)
            theta = doc_topic_matrix_from_df(run["doc_topic_df"])
            run_dir = k_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            run_record = build_run_artifact(
                run=run,
                phi=phi,
                theta=theta,
                run_dir=run_dir,
                seed=seed,
                k=k,
                lens_ids=lens_ids,
            )
            run_records.append(run_record)

        pairwise_df, pair_artifacts = compute_pairwise_alignment_rows(k=k, run_records=run_records, k_dir=k_dir)
        k_pair_artifacts[k] = pair_artifacts

        stability_metrics = summarize_stability_from_runs(
            run_records=run_records,
            seeds=seeds,
            n_topics=k,
            vocab_index=shared_vocab_index,
        )
        alignment_metrics = aggregate_pairwise_metrics(pairwise_df)
        runtime_sec = time.time() - k_start

        summary_row = {
            "k": k,
            "n_seeds": len(seeds),
            "omega_mean": stability_metrics["omega"],
            "omega_se_mean": stability_metrics["omega_se"],
            "top1_mean": alignment_metrics["top1_mean"],
            "top1_std": alignment_metrics["top1_std"],
            "top2_mean": alignment_metrics["top2_mean"],
            "top2_std": alignment_metrics["top2_std"],
            "top3_mean": alignment_metrics["top3_mean"],
            "top3_std": alignment_metrics["top3_std"],
            "max_cosine_mean": alignment_metrics["max_cosine_mean"],
            "mean_cosine_mean": alignment_metrics["mean_cosine_mean"],
            "omega_theta": stability_metrics["omega_theta"],
            "omega_words": stability_metrics["omega_words"],
            "vocab_size": len(shared_vocab_list),
            "average_tokens_per_doc": avg_tokens_per_doc,
            "alpha": base_config.alpha,
            "eta": base_config.eta,
            "runtime_sec": runtime_sec,
        }
        overall_rows.append(summary_row)
        save_json(k_dir / "summary.json", summary_row)

    overall_summary_df = pd.DataFrame(overall_rows).sort_values("k").reset_index(drop=True)
    overall_summary_df.to_csv(output_dir / "overall_summary.csv", index=False)
    save_json(output_dir / "overall_summary.json", overall_summary_df.to_dict(orient="records"))

    save_stability_plot(overall_summary_df, output_dir / "stability_vs_k.png")
    save_alignment_plot(overall_summary_df, output_dir / "alignment_vs_k.png")
    save_tradeoff_plot(overall_summary_df, output_dir / "stability_alignment_tradeoff.png")

    heatmap_manifest = []
    qualitative_manifest = []

    for k in k_values:
        pair_artifacts = k_pair_artifacts[k]
        if not pair_artifacts:
            continue

        pairwise_df = pd.read_csv(output_dir / f"k_{k}" / "pairwise_alignment_summary.csv")
        seed_a, seed_b = choose_pair_for_heatmap(pairwise_df)
        pair_artifact = pair_artifacts[(seed_a, seed_b)]

        if k in heatmap_k_values:
            heatmap_base = output_dir / f"k_{k}" / f"topic_similarity_heatmap_seed_{seed_a}__seed_{seed_b}"
            save_similarity_heatmap(
                pair_artifact["sim_matrix"],
                heatmap_base,
                title=f"Topic Similarity Heatmap for k={k} (seed {seed_a} vs {seed_b})",
            )
            heatmap_manifest.append(
                {
                    "k": k,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "png": str(heatmap_base.with_suffix(".png")),
                    "pdf": str(heatmap_base.with_suffix(".pdf")),
                }
            )

        if k in qualitative_k_values:
            examples = build_topic_split_examples(
                k=k,
                pair_artifact=pair_artifact,
                top_n_words=base_config.top_words_n,
                top_split_examples=ksweep_config.top_split_examples,
            )
            report = {
                "k": k,
                "seed_a": seed_a,
                "seed_b": seed_b,
                "examples": examples,
            }
            json_path = output_dir / f"k_{k}" / f"topic_split_examples_seed_{seed_a}__seed_{seed_b}.json"
            md_path = output_dir / f"k_{k}" / f"topic_split_examples_seed_{seed_a}__seed_{seed_b}.md"
            save_json(json_path, report)
            write_split_report_markdown(k=k, seed_a=seed_a, seed_b=seed_b, examples=examples, output_path=md_path)
            qualitative_manifest.append(
                {
                    "k": k,
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "json": str(json_path),
                    "markdown": str(md_path),
                }
            )

    save_json(
        output_dir / "manifest.json",
        {
            "overall_summary_csv": str(output_dir / "overall_summary.csv"),
            "stability_plot": str(output_dir / "stability_vs_k.png"),
            "alignment_plot": str(output_dir / "alignment_vs_k.png"),
            "tradeoff_plot": str(output_dir / "stability_alignment_tradeoff.png"),
            "heatmaps": heatmap_manifest,
            "qualitative_reports": qualitative_manifest,
        },
    )

    print(f"Patent count analyzed: {len(df_docs)}")
    print(f"Shared vocab size: {len(shared_vocab_list)}")
    print(f"Overall summary: {output_dir / 'overall_summary.csv'}")
    print(f"Stability plot: {output_dir / 'stability_vs_k.png'}")
    print(f"Alignment plot: {output_dir / 'alignment_vs_k.png'}")
    print(f"Manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
