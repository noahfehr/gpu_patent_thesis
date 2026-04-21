from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from config import LDAConfig
from data import (
    add_bigrams,
    build_bigram_vocab,
    load_patent_dataframe,
    load_stopwords,
    clean_text,
    tokenize_unigrams,
)
from experiment import run_labeled_stability_experiment_on_prepared_docs
from label_topics import TopicLabelingConfig


def _parse_seeds(raw_seeds: str | None, n_seeds: int) -> list[int]:
    if raw_seeds:
        return [int(part.strip()) for part in raw_seeds.split(",") if part.strip()]
    return list(range(n_seeds))


def load_predictions_df(predictions_path: Path) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    records = []
    with open(predictions_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status") != "success":
                continue
            records.append(
                {
                    "lens_id": str(record["lens_id"]),
                    "is_accelerator_relevant": bool(record.get("is_accelerator_relevant", False)),
                    "is_accelerator_hardware": bool(record.get("is_accelerator_hardware", False)),
                    "is_accelerator_hardware_design_patent": bool(
                        record.get("is_accelerator_hardware_design_patent", False)
                    ),
                }
            )

    if not records:
        raise ValueError("No successful prediction records found.")

    return pd.DataFrame(records).drop_duplicates(subset=["lens_id"])


def prepare_subset_corpus(
    base_config: LDAConfig,
    subset_name: str,
    subset_mask,
) -> tuple[pd.DataFrame, dict]:
    df = load_patent_dataframe(base_config)
    pred_df = load_predictions_df(base_config.predictions_path)

    df = df.merge(pred_df, on="lens_id", how="inner")
    full_count = len(df)
    subset_count = int(subset_mask(df).sum())
    df = df[subset_mask(df)].copy()

    print(f"\n=== Preparing subset: {subset_name} ===")
    print(f"Subset size before token filtering: {len(df)} / {full_count}")

    stop_words = load_stopwords(base_config.stopwords_path)

    df["text_clean"] = df["text"].map(clean_text)
    df["tokens_unigram"] = df["text"].map(lambda s: tokenize_unigrams(s, stop_words))
    df = df[df["tokens_unigram"].map(len) > 0].copy()

    bigram_vocab = build_bigram_vocab(
        df["tokens_unigram"].tolist(),
        min_count=base_config.min_bigram_count,
    )

    print(f"Frequent bigrams kept for {subset_name}: {len(bigram_vocab)}")
    print(
        "Sample bigrams:",
        [f"{a}_{b}" for a, b in list(sorted(bigram_vocab))[:20]],
    )

    df["tokens"] = df["tokens_unigram"].map(lambda toks: add_bigrams(toks, bigram_vocab))
    df = df[df["tokens"].map(len) > 0].copy()

    print(f"Usable documents in {subset_name}: {len(df)}")
    print(f"Average unigram tokens per doc: {df['tokens_unigram'].map(len).mean():.1f}")
    print(f"Average tokens per doc after bigrams: {df['tokens'].map(len).mean():.1f}")

    metadata = {
        "subset_name": subset_name,
        "docs_with_successful_predictions": int(full_count),
        "docs_in_subset_before_token_filtering": subset_count,
        "docs_in_subset_after_token_filtering": int(len(df)),
    }
    return df, metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full topic modeling plus labeling on the two-stage reject corpora: "
            "stage-1 rejects and stage-2 rejects."
        )
    )

    parser.add_argument("--version-prefix", default="v6")
    parser.add_argument(
        "--predictions-path",
        default="data/analysis/runs/v6__full__two_stage__ts1__predictions.jsonl",
    )
    parser.add_argument(
        "--stopwords-path",
        default="code/topic_modeling/lda_pipeline/custom_stopwords.txt",
    )
    parser.add_argument("--base-data-dir", default="data/claims_added")
    parser.add_argument("--text-column", default="claims")
    parser.add_argument("--id-column", default="lens_id")

    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--min-bigram-count", type=int, default=15)
    parser.add_argument("--min-df", type=int, default=10)
    parser.add_argument("--rm-top", type=int, default=70)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--top-words-n", type=int, default=15)

    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--label-seed", type=int, default=None)

    parser.add_argument("--label-model", default="gpt-5-mini")
    parser.add_argument("--label-top-n-docs", type=int, default=3)
    parser.add_argument("--label-n-passes", type=int, default=3)
    parser.add_argument("--label-excerpt-chars", type=int, default=1500)
    parser.add_argument("--label-max-words", type=int, default=6)
    parser.add_argument("--label-max-explanation-sentences", type=int, default=3)

    parser.add_argument("--output-root", default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds, args.n_seeds)
    if not seeds:
        raise ValueError("At least one seed is required.")
    label_seed = args.label_seed if args.label_seed is not None else seeds[0]

    lda_config = LDAConfig(
        version_prefix=args.version_prefix,
        predictions_path=Path(args.predictions_path),
        stopwords_path=Path(args.stopwords_path),
        base_data_dir=Path(args.base_data_dir),
        text_column=args.text_column,
        id_column=args.id_column,
        min_bigram_count=args.min_bigram_count,
        k=args.k,
        alpha=args.alpha,
        eta=args.eta,
        min_df=args.min_df,
        rm_top=args.rm_top,
        seed=seeds[0],
        iterations=args.iterations,
        top_words_n=args.top_words_n,
    )

    label_config = TopicLabelingConfig(
        model=args.label_model,
        top_n_docs=args.label_top_n_docs,
        excerpt_chars=args.label_excerpt_chars,
        n_passes=args.label_n_passes,
        max_label_words=args.label_max_words,
        max_explanation_sentences=args.label_max_explanation_sentences,
    )

    subset_specs = [
        (
            "stage1_rejects_not_accelerator_related",
            lambda df: ~df["is_accelerator_relevant"],
            "All v6 patents rejected at stage 1 as not accelerator related.",
        ),
        (
            "stage2_rejects_accelerator_related_not_hardware",
            lambda df: df["is_accelerator_relevant"] & ~df["is_accelerator_hardware"],
            "All v6 patents that passed stage 1 but were rejected at stage 2 as not hardware related.",
        ),
    ]

    for subset_name, subset_mask, subset_description in subset_specs:
        df_subset, subset_metadata = prepare_subset_corpus(
            base_config=lda_config,
            subset_name=subset_name,
            subset_mask=subset_mask,
        )

        run_name = (
            f"run_{args.version_prefix}_{subset_name}"
            f"_k-{args.k}_a-{str(args.alpha).replace('.', 'p')}"
            f"_e-{str(args.eta).replace('.', 'p')}"
            f"_{len(seeds)}seeds_label-seed{label_seed}"
        )

        result = run_labeled_stability_experiment_on_prepared_docs(
            df_docs=df_subset,
            lda_config=lda_config,
            seeds=seeds,
            label_seed=label_seed,
            label_config=label_config,
            output_root=args.output_root,
            run_name=run_name,
            metadata={
                **subset_metadata,
                "subset_name": subset_name,
                "subset_description": subset_description,
            },
        )

        print(f"\nCompleted subset: {subset_name}")
        print(f"Run directory: {result['run_dir']}")
        print(f"Topic labels saved to: {result['label_output_dir']}")


if __name__ == "__main__":
    main()
