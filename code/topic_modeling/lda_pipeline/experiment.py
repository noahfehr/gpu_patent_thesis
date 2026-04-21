from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import LDAConfig
from data import prepare_patent_corpus
from label_topics import (
    TopicLabelingConfig,
    label_all_topics,
    save_topic_labeling_outputs,
)
from stability import compute_topic_stability


def _format_float_for_name(value: float) -> str:
    return str(value).replace(".", "p")


def build_single_config_run_name(
    config: LDAConfig,
    seeds: list[int],
    label_seed: int,
) -> str:
    """
    Create a readable run name for a fixed-parameter experiment.
    """
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return (
        f"run_{date_str}"
        f"_{config.version_prefix}"
        f"_k-{config.k}"
        f"_a-{_format_float_for_name(config.alpha)}"
        f"_e-{_format_float_for_name(config.eta)}"
        f"_{len(seeds)}seeds"
        f"_label-seed{label_seed}"
    )


def _write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _run_labeled_stability_experiment_from_prepared_docs(
    df_docs: pd.DataFrame,
    lda_config: LDAConfig,
    seeds: list[int],
    label_seed: int,
    label_config: TopicLabelingConfig,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    metadata: dict | None = None,
) -> dict:
    if not seeds:
        raise ValueError("Expected at least one seed.")

    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    if label_seed not in seeds:
        raise ValueError(
            f"label_seed={label_seed} must be included in seeds={seeds}"
        )

    project_root = Path.cwd()
    output_root = Path(output_root) if output_root is not None else project_root / "outputs" / "lda" / "runs"
    run_name = run_name or build_single_config_run_name(lda_config, seeds, label_seed)
    run_dir = output_root / run_name

    stability_dir = run_dir / "stability"
    doc_topic_dir = run_dir / "doc_topics"
    topic_words_dir = run_dir / "topic_words"
    metadata_dir = run_dir / "metadata"
    label_output_dir = run_dir / "topic_labels" / f"k{lda_config.k}_seed{label_seed}"

    for d in [run_dir, stability_dir, doc_topic_dir, topic_words_dir, metadata_dir, label_output_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Writing outputs to: {run_dir}")

    _write_json(
        run_dir / "config.json",
        {
            "lda_config": asdict(lda_config),
            "label_config": asdict(label_config),
            "seeds": seeds,
            "label_seed": label_seed,
            "run_name": run_name,
            "metadata": metadata or {},
        },
    )

    token_lists = df_docs["tokens"].tolist()
    lens_ids = df_docs["lens_id"].tolist()

    print(f"Prepared {len(token_lists)} tokenized documents")

    df_docs[["lens_id"]].reset_index(drop=True).to_parquet(
        metadata_dir / "doc_index.parquet",
        index=False,
    )

    print("\nRunning stability analysis...")
    results_df, artifacts = compute_topic_stability(
        tokenized_docs=token_lists,
        lens_ids=lens_ids,
        base_config=lda_config,
        k_values=[lda_config.k],
        seeds=seeds,
        vocab_min_df=lda_config.min_df,
    )

    if not isinstance(results_df, pd.DataFrame):
        results_df = pd.DataFrame(results_df)

    results_df.to_csv(stability_dir / "stability_results.csv", index=False)
    results_df.to_parquet(stability_dir / "stability_results.parquet", index=False)
    _write_json(
        stability_dir / "stability_results.json",
        {"rows": results_df.to_dict(orient="records")},
    )

    print("\nSaving doc-topic matrices and topic words...")
    for (k, seed), artifact in artifacts.items():
        artifact["doc_topic_df"].to_parquet(
            doc_topic_dir / f"k{k}_seed{seed}.parquet",
            index=False,
        )
        _write_json(
            topic_words_dir / f"k{k}_seed{seed}.json",
            artifact["topic_words"],
        )

    print(f"\nLabeling topics for seed {label_seed}...")
    label_artifact = artifacts[(lda_config.k, label_seed)]
    label_results = label_all_topics(
        df_docs=df_docs,
        doc_topic_df=label_artifact["doc_topic_df"],
        topic_words=label_artifact["topic_words"],
        config=label_config,
    )
    save_topic_labeling_outputs(
        results=label_results,
        output_dir=label_output_dir,
    )

    print("\n===== STABILITY RESULTS =====")
    print(results_df)
    print(f"\nDone. Outputs written to: {run_dir}")

    return {
        "run_dir": run_dir,
        "results_df": results_df,
        "artifacts": artifacts,
        "label_results": label_results,
        "label_output_dir": label_output_dir,
    }


def run_labeled_stability_experiment(
    lda_config: LDAConfig,
    seeds: list[int],
    label_seed: int,
    label_config: TopicLabelingConfig | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
) -> dict:
    """
    Run a fixed LDA configuration across multiple seeds, report stability,
    and label one selected seed's topics.
    """
    label_config = label_config or TopicLabelingConfig()

    print("\nPreparing corpus...")
    df_docs = prepare_patent_corpus(lda_config)

    return _run_labeled_stability_experiment_from_prepared_docs(
        df_docs=df_docs,
        lda_config=lda_config,
        seeds=seeds,
        label_seed=label_seed,
        label_config=label_config,
        output_root=output_root,
        run_name=run_name,
    )


def run_labeled_stability_experiment_on_prepared_docs(
    df_docs: pd.DataFrame,
    lda_config: LDAConfig,
    seeds: list[int],
    label_seed: int,
    label_config: TopicLabelingConfig | None = None,
    output_root: str | Path | None = None,
    run_name: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Run the fixed-parameter experiment on an already prepared dataframe that
    contains at least `lens_id` and `tokens`.
    """
    label_config = label_config or TopicLabelingConfig()

    return _run_labeled_stability_experiment_from_prepared_docs(
        df_docs=df_docs,
        lda_config=lda_config,
        seeds=seeds,
        label_seed=label_seed,
        label_config=label_config,
        output_root=output_root,
        run_name=run_name,
        metadata=metadata,
    )
