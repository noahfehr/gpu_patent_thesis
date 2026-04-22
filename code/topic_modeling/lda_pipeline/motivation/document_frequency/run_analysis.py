from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import LDAConfig, PROJECT_ROOT
from data import (
    filter_tokens_by_document_frequency,
    load_patent_dataframe,
    load_stopwords,
    load_keep_lens_ids,
    filter_patents_to_design_set,
    tokenize_patent_dataframe,
)


DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "code" / "topic_modeling" / "lda_pipeline" / "motivation" / "document_frequency" / "outputs"
)


def prepare_corpus_before_df_filter(config: LDAConfig) -> pd.DataFrame:
    """
    Prepare the corpus through bigram construction, but stop before applying
    the final min_df/max_df vocabulary filtering so the document-frequency
    distribution reflects the full candidate vocabulary.
    """
    df = load_patent_dataframe(config)
    keep_lens_ids = load_keep_lens_ids(config.predictions_path)
    df = filter_patents_to_design_set(df, keep_lens_ids, config.id_column)

    stop_words = load_stopwords(config.stopwords_path)
    df, token_stats = tokenize_patent_dataframe(
        df=df,
        config=config,
        stop_words=stop_words,
        apply_df_filters=False,
        verbose=False,
    )
    print(f"Frequent bigrams kept: {token_stats['bigram_vocab_size']}")
    return df


def compute_document_frequency(token_lists: list[list[str]]) -> Counter[str]:
    """
    Count in how many documents each token appears.
    """
    doc_freq: Counter[str] = Counter()
    for tokens in token_lists:
        for token in set(tokens):
            doc_freq[token] += 1
    return doc_freq


def count_unique_tokens(token_lists: list[list[str]]) -> int:
    vocab = set()
    for tokens in token_lists:
        vocab.update(tokens)
    return len(vocab)


def max_df_to_count(max_df: float | int | None, n_docs: int) -> int | None:
    if max_df is None:
        return None
    if isinstance(max_df, float) and 0 < max_df <= 1:
        return max(1, int(n_docs * max_df))
    return int(max_df)


def parse_max_df(raw: str) -> float | int:
    value = float(raw)
    if value.is_integer() and value > 1:
        return int(value)
    return value


def save_doc_frequency_json(doc_freq: Counter[str], output_path: Path) -> None:
    """
    Save token -> document count as a JSON dictionary, sorted by descending df.
    """
    sorted_items = sorted(doc_freq.items(), key=lambda item: (-item[1], item[0]))
    payload = {token: count for token, count in sorted_items}

    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)


def save_document_frequency_histogram(
    doc_freq: Counter[str],
    output_path: Path,
    min_df: int,
    max_df_count: int | None,
    version_prefix: str,
) -> None:
    """
    Plot a histogram where:
    - x-axis = number of documents containing a word
    - y-axis = number of words with that document count
    """
    counts = list(doc_freq.values())
    if not counts:
        raise ValueError("No document-frequency counts available to plot.")

    freq_of_freq = Counter(counts)
    x_values = sorted(freq_of_freq)
    y_values = [freq_of_freq[x] for x in x_values]

    zoom_x_max = min(
        max(x_values),
        max(
            min_df * 4,
            max_df_count if max_df_count is not None else 0,
            int(sorted(counts)[int(0.99 * len(counts))]),
        ),
    )
    zoom_x_max = max(zoom_x_max, min_df + 5)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 1.2])

    axes[0].bar(x_values, y_values, width=1.0, color="#2E5EAA", edgecolor="#2E5EAA")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Document Count")
    axes[0].set_ylabel("Number of Words (log scale)")
    axes[0].set_title(
        f"Vocabulary Document-Frequency Distribution ({version_prefix})\n"
        "Full range, computed before final min_df / max_df filtering"
    )

    axes[1].bar(x_values, y_values, width=1.0, color="#2E5EAA", edgecolor="#2E5EAA")
    axes[1].set_xlim(1, zoom_x_max)
    axes[1].set_xlabel("Document Count")
    axes[1].set_ylabel("Number of Words")
    axes[1].set_title("Zoomed view of low-to-mid document frequencies")

    for ax in axes:
        ax.axvline(
            min_df,
            color="#C0392B",
            linestyle="--",
            linewidth=2,
            label=f"min_df = {min_df}",
        )

        if max_df_count is not None:
            ax.axvline(
                max_df_count,
                color="#1E8449",
                linestyle="--",
                linewidth=2,
                label=f"max_df cutoff = {max_df_count}",
            )

    axes[0].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_low_df_zoom_histogram(
    doc_freq: Counter[str],
    output_path: Path,
    min_df: int,
    zoom_limit: int = 20,
    version_prefix: str = "v6",
) -> None:
    """
    Plot a zoomed histogram for tokens appearing in fewer than zoom_limit docs.
    """
    low_counts = [count for count in doc_freq.values() if count < zoom_limit]
    if not low_counts:
        raise ValueError("No low document-frequency counts available to plot.")

    freq_of_freq = Counter(low_counts)
    x_values = sorted(freq_of_freq)
    y_values = [freq_of_freq[x] for x in x_values]

    plt.figure(figsize=(12, 7))
    plt.bar(x_values, y_values, width=0.9, color="#B9770E", edgecolor="#B9770E")
    plt.xlim(1, zoom_limit - 1)
    plt.xlabel("Document Count")
    plt.ylabel("Number of Words")
    plt.title(
        f"Vocabulary Document-Frequency Distribution ({version_prefix})\n"
        f"Zoomed view for words appearing in fewer than {zoom_limit} documents"
    )
    plt.axvline(
        min_df,
        color="#C0392B",
        linestyle="--",
        linewidth=2,
        label=f"min_df = {min_df}",
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze token document-frequency distribution for the LDA corpus "
            "and save a histogram plus token->document-count JSON."
        )
    )
    parser.add_argument("--version-prefix", default="v6")
    parser.add_argument("--predictions-path", type=Path, default=LDAConfig().predictions_path)
    parser.add_argument("--base-data-dir", type=Path, default=LDAConfig().base_data_dir)
    parser.add_argument("--text-column", default="claims")
    parser.add_argument("--id-column", default="lens_id")
    parser.add_argument("--min-bigram-count", type=int, default=LDAConfig().min_bigram_count)
    parser.add_argument("--min-df", type=int, default=LDAConfig().min_df)
    parser.add_argument("--max-df", type=parse_max_df, default=LDAConfig().max_df)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    config = LDAConfig(
        version_prefix=args.version_prefix,
        predictions_path=args.predictions_path,
        base_data_dir=args.base_data_dir,
        text_column=args.text_column,
        id_column=args.id_column,
        min_bigram_count=args.min_bigram_count,
        min_df=args.min_df,
        max_df=args.max_df,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df_docs = prepare_corpus_before_df_filter(config)
    pre_filter_token_lists = df_docs["tokens"].tolist()
    doc_freq = compute_document_frequency(pre_filter_token_lists)
    max_df_count = max_df_to_count(config.max_df, len(df_docs))

    post_filter_token_lists, removed_vocab = filter_tokens_by_document_frequency(
        pre_filter_token_lists,
        min_df=config.min_df,
        max_df=config.max_df,
    )
    post_filter_token_lists = [tokens for tokens in post_filter_token_lists if tokens]
    vocab_size_passed_to_lda = count_unique_tokens(post_filter_token_lists)
    avg_tokens_per_doc_passed_to_lda = (
        sum(len(tokens) for tokens in post_filter_token_lists) / len(post_filter_token_lists)
        if post_filter_token_lists
        else 0.0
    )

    json_path = output_dir / f"{config.version_prefix}_document_frequency.json"
    png_path = output_dir / f"{config.version_prefix}_document_frequency_histogram.png"
    zoom_png_path = output_dir / f"{config.version_prefix}_document_frequency_histogram_lt20docs.png"
    summary_path = output_dir / f"{config.version_prefix}_document_frequency_summary.json"

    save_doc_frequency_json(doc_freq, json_path)
    save_document_frequency_histogram(
        doc_freq=doc_freq,
        output_path=png_path,
        min_df=config.min_df,
        max_df_count=max_df_count,
        version_prefix=config.version_prefix,
    )
    save_low_df_zoom_histogram(
        doc_freq=doc_freq,
        output_path=zoom_png_path,
        min_df=config.min_df,
        zoom_limit=20,
        version_prefix=config.version_prefix,
    )

    summary = {
        "version_prefix": config.version_prefix,
        "n_docs": int(len(df_docs)),
        "n_unique_tokens_before_df_filter": int(len(doc_freq)),
        "avg_tokens_per_doc_before_df_filter": float(df_docs["tokens"].map(len).mean()),
        "n_docs_after_df_filter": int(len(post_filter_token_lists)),
        "n_unique_tokens_passed_to_lda": int(vocab_size_passed_to_lda),
        "avg_tokens_per_doc_passed_to_lda": float(avg_tokens_per_doc_passed_to_lda),
        "vocabulary_items_removed_by_df_filters": int(removed_vocab),
        "min_df": int(config.min_df),
        "max_df": config.max_df,
        "max_df_count": max_df_count,
        "min_bigram_count": int(config.min_bigram_count),
        "json_output": str(json_path),
        "histogram_output": str(png_path),
        "low_df_zoom_histogram_output": str(zoom_png_path),
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Patent count: {summary['n_docs']}")
    print(f"Total vocabulary size: {summary['n_unique_tokens_before_df_filter']}")
    print(f"Average tokens per doc: {summary['avg_tokens_per_doc_before_df_filter']:.1f}")
    print(f"Patent count passed to LDA: {summary['n_docs_after_df_filter']}")
    print(f"Vocabulary size passed to LDA: {summary['n_unique_tokens_passed_to_lda']}")
    print(f"Average tokens per doc passed to LDA: {summary['avg_tokens_per_doc_passed_to_lda']:.1f}")
    print(f"Document-frequency JSON written to: {json_path}")
    print(f"Histogram written to: {png_path}")
    print(f"Low-df zoom histogram written to: {zoom_png_path}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
