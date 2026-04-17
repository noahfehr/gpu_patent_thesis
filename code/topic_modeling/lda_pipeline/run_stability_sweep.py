from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from stability import compute_topic_stability
from config import LDAConfig
from data import prepare_patent_corpus


def build_run_name(k_values: list[int], seeds: list[int]) -> str:
    """
    Create a readable run name for the output folder.
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    k_part = "-".join(str(k) for k in k_values)
    seed_part = f"{len(seeds)}seeds"
    return f"run_{date_str}_k-{k_part}_{seed_part}"


def summarize_stability_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a compact summary table by k.
    """
    if "k" not in results_df.columns:
        raise KeyError("Expected a 'k' column in stability results.")

    numeric_cols = results_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "k"]

    if not numeric_cols:
        return results_df[["k"]].drop_duplicates().sort_values("k").reset_index(drop=True)

    grouped = results_df.groupby("k")[numeric_cols].agg(["mean", "std", "min", "max"])
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    grouped = grouped.reset_index().sort_values("k")

    return grouped


def main() -> None:
    config = LDAConfig(
        version_prefix="v6",
        predictions_path=Path("data/analysis/runs/v6__full__two_stage__ts1__predictions.jsonl"),
        stopwords_path=Path("code/topic_modeling/lda_pipeline/custom_stopwords.txt"),
        base_data_dir=Path("data/claims_added"),
        text_column="claims",
        id_column="lens_id",
        min_bigram_count=15,
        k=30,              # placeholder; actual sweep uses k_values below
        alpha=0.3,
        eta=0.01,
        min_df=10,
        rm_top=70,
        seed=42,           # placeholder; actual sweep uses seeds below
        iterations=100,
        top_words_n=15,
        top_docs_n=5,
    )

    k_values = [20, 30, 40, 50]
    seeds = list(range(20))

    project_root = Path.cwd()
    output_root = project_root / "outputs" / "lda" / "runs"
    run_name = build_run_name(k_values, seeds)
    run_dir = output_root / run_name

    stability_dir = run_dir / "stability"
    doc_topic_dir = run_dir / "doc_topics"
    topic_words_dir = run_dir / "topic_words"
    metadata_dir = run_dir / "metadata"

    for d in [run_dir, stability_dir, doc_topic_dir, topic_words_dir, metadata_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Writing outputs to: {run_dir}")

    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "lda_config": asdict(config),
                "k_values": k_values,
                "seeds": seeds,
                "run_name": run_name,
            },
            f,
            indent=2,
            default=str,
        )

    print("\nPreparing corpus...")
    df_docs = prepare_patent_corpus(config)
    token_lists = df_docs["tokens"].tolist()
    lens_ids = df_docs["lens_id"].tolist()

    print(f"Prepared {len(token_lists)} tokenized documents")

    df_docs[["lens_id"]].reset_index(drop=True).to_parquet(
        metadata_dir / "doc_index.parquet",
        index=False,
    )

    print("\nRunning stability sweep...")
    results_df, artifacts = compute_topic_stability(
        tokenized_docs=token_lists,
        lens_ids=lens_ids,
        base_config=config,
        k_values=k_values,
        seeds=seeds,
        vocab_min_df=config.min_df,
    )

    if not isinstance(results_df, pd.DataFrame):
        results_df = pd.DataFrame(results_df)

    summary_df = summarize_stability_results(results_df)

    results_df.to_csv(stability_dir / "stability_results.csv", index=False)
    results_df.to_parquet(stability_dir / "stability_results.parquet", index=False)

    summary_df.to_csv(stability_dir / "summary_by_k.csv", index=False)
    summary_df.to_parquet(stability_dir / "summary_by_k.parquet", index=False)

    print("\nSaving doc-topic matrices and topic words...")

    for (k, seed), artifact in artifacts.items():
        doc_topic_df = artifact["doc_topic_df"]
        topic_words = artifact["topic_words"]

        doc_topic_df.to_parquet(
            doc_topic_dir / f"k{k}_seed{seed}.parquet",
            index=False,
        )

        with open(topic_words_dir / f"k{k}_seed{seed}.json", "w") as f:
            json.dump(topic_words, f, indent=2)

    print("\n===== RAW RESULTS =====")
    print(results_df)

    print("\n===== SUMMARY BY K =====")
    print(summary_df.sort_values("k"))

    candidate_score_cols = [
        "omega",
        "omega_theta",
        "omega_words",
        "max_cosine",
        "mean_cosine",
    ]
    available = [c for c in candidate_score_cols if c in results_df.columns]

    if available:
        score_col = available[0]
        best_row = results_df.sort_values(score_col, ascending=False).iloc[0]
        print("\n===== BEST K (by first available score column) =====")
        print(f"{score_col}: best k = {best_row['k']} ({best_row[score_col]:.4f})")

    print(f"\nDone. Outputs written to: {run_dir}")


if __name__ == "__main__":
    main()