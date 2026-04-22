from __future__ import annotations

import argparse
from pathlib import Path

from config import LDAConfig
from experiment import run_labeled_stability_experiment
from label_topics import TopicLabelingConfig


def _parse_seeds(raw_seeds: str | None, n_seeds: int) -> list[int]:
    if raw_seeds:
        return [int(part.strip()) for part in raw_seeds.split(",") if part.strip()]
    return list(range(n_seeds))


def build_parser() -> argparse.ArgumentParser:
    default_config = LDAConfig()

    parser = argparse.ArgumentParser(
        description=(
            "Run one LDA configuration across multiple seeds, compute stability, "
            "and label one selected seed."
        )
    )

    parser.add_argument("--version-prefix", default=default_config.version_prefix)
    parser.add_argument(
        "--predictions-path",
        default=str(default_config.predictions_path),
    )
    parser.add_argument(
        "--stopwords-path",
        default=str(default_config.stopwords_path),
    )
    parser.add_argument("--base-data-dir", default=str(default_config.base_data_dir))
    parser.add_argument("--text-column", default=default_config.text_column)
    parser.add_argument("--id-column", default=default_config.id_column)

    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--eta", type=float, required=True)
    parser.add_argument("--min-bigram-count", type=int, default=default_config.min_bigram_count)
    parser.add_argument("--min-df", type=int, default=default_config.min_df)
    parser.add_argument("--max-df", type=float, default=default_config.max_df)
    parser.add_argument("--iterations", type=int, default=default_config.iterations)
    parser.add_argument("--top-words-n", type=int, default=default_config.top_words_n)

    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list. Overrides --n-seeds if provided.",
    )
    parser.add_argument(
        "--label-seed",
        type=int,
        default=None,
        help="Seed to label. Defaults to the first seed in the run.",
    )

    parser.add_argument("--label-model", default="gpt-5-mini")
    parser.add_argument("--label-top-n-docs", type=int, default=3)
    parser.add_argument("--label-n-passes", type=int, default=3)
    parser.add_argument("--label-excerpt-chars", type=int, default=1500)
    parser.add_argument("--label-max-words", type=int, default=6)
    parser.add_argument("--label-max-explanation-sentences", type=int, default=3)

    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-name", default=None)

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
        max_df=args.max_df,
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

    result = run_labeled_stability_experiment(
        lda_config=lda_config,
        seeds=seeds,
        label_seed=label_seed,
        label_config=label_config,
        output_root=args.output_root,
        run_name=args.run_name,
    )

    print(f"\nRun directory: {result['run_dir']}")
    print(f"Topic labels saved to: {result['label_output_dir']}")


if __name__ == "__main__":
    main()
