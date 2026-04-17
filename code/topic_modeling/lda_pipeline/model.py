from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import tomotopy as tp

from config import LDAConfig
from data import prepare_patent_corpus


def train_lda(token_lists: list[list[str]], config: LDAConfig) -> tp.LDAModel:
    """
    Train a tomotopy LDA model from tokenized documents.
    """
    mdl = tp.LDAModel(
        k=config.k,
        alpha=config.alpha,
        eta=config.eta,
        min_df=config.min_df,
        rm_top=config.rm_top,
        seed=config.seed,
    )

    for tokens in token_lists:
        mdl.add_doc(tokens)

    print(f"Num docs in model: {len(mdl.docs)}")
    print(f"Vocab size: {mdl.num_vocabs}")

    for i in range(0, config.iterations, 10):
        mdl.train(10)
        if (i + 10) % 100 == 0 or i == 0:
            print(
                f"Iteration: {i + 10:04d} "
                f"LL per word: {mdl.ll_per_word:.4f}"
            )

    return mdl


def get_topic_words(mdl: tp.LDAModel, top_n: int = 15) -> dict[int, list[str]]:
    """
    Return top words for each topic as a dictionary.
    """
    topic_words: dict[int, list[str]] = {}

    for k in range(mdl.k):
        words = [word for word, _ in mdl.get_topic_words(k, top_n=top_n)]
        topic_words[k] = words

    return topic_words


def topic_words_df(mdl: tp.LDAModel, top_n: int = 15) -> pd.DataFrame:
    """
    Return top topic words in long dataframe format.
    """
    rows = []

    for topic_id in range(mdl.k):
        for rank, (word, prob) in enumerate(mdl.get_topic_words(topic_id, top_n=top_n), start=1):
            rows.append(
                {
                    "topic_id": topic_id,
                    "rank": rank,
                    "word": word,
                    "prob": prob,
                }
            )

    return pd.DataFrame(rows)


def print_topics(mdl: tp.LDAModel, top_n: int = 15) -> None:
    """
    Print top words for each topic.
    """
    for topic_id in range(mdl.k):
        words = [word for word, _ in mdl.get_topic_words(topic_id, top_n=top_n)]
        print(f"Topic {topic_id:02d}: {', '.join(words)}")


def get_doc_topic_matrix(mdl: tp.LDAModel, lens_ids: list[str]) -> pd.DataFrame:
    """
    Build a dataframe with one row per document and one column per topic weight.
    """
    if len(lens_ids) != len(mdl.docs):
        raise ValueError(
            f"lens_ids length ({len(lens_ids)}) does not match model docs ({len(mdl.docs)})"
        )

    rows = []
    for lens_id, doc in zip(lens_ids, mdl.docs):
        topic_dist = doc.get_topic_dist()
        row = {"lens_id": lens_id}
        for topic_id, weight in enumerate(topic_dist):
            row[f"topic_{topic_id}"] = float(weight)
        rows.append(row)

    return pd.DataFrame(rows)


def compute_basic_run_metrics(mdl: tp.LDAModel) -> dict:
    """
    Lightweight run-level metrics extracted from a fitted model.
    """
    return {
        "k": int(mdl.k),
        "num_docs": int(len(mdl.docs)),
        "vocab_size": int(mdl.num_vocabs),
        "ll_per_word": float(mdl.ll_per_word),
    }


def fit_lda_run(
    token_lists: list[list[str]],
    lens_ids: list[str],
    config: LDAConfig,
) -> dict:
    """
    Fit one LDA run and return the core artifacts needed downstream.

    Returns a dict with:
    - model
    - metrics
    - topic_words
    - doc_topic_df
    """
    mdl = train_lda(token_lists, config)
    topic_words = get_topic_words(mdl, top_n=config.top_words_n)
    doc_topic_df = get_doc_topic_matrix(mdl, lens_ids)
    metrics = compute_basic_run_metrics(mdl)

    return {
        "model": mdl,
        "metrics": metrics,
        "topic_words": topic_words,
        "doc_topic_df": doc_topic_df,
    }


def get_top_docs_for_topic(
    df_docs: pd.DataFrame,
    doc_topic_df: pd.DataFrame,
    topic_id: int,
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Return the documents with the highest weight for a given topic.
    """
    topic_col = f"topic_{topic_id}"
    if topic_col not in doc_topic_df.columns:
        raise KeyError(f"missing topic column: {topic_col}")

    merged = df_docs.merge(doc_topic_df[["lens_id", topic_col]], on="lens_id", how="inner")
    merged = merged.sort_values(topic_col, ascending=False).head(top_n).copy()

    return merged[["lens_id", "text", topic_col]]


def build_topic_labeling_payload(
    df_docs: pd.DataFrame,
    doc_topic_df: pd.DataFrame,
    topic_words: dict[int, list[str]],
    topic_id: int,
    top_n_docs: int = 3,
    excerpt_chars: int = 1500,
) -> dict:
    """
    Build a compact payload for LLM-based topic labeling:
    - top words
    - top documents for the topic
    - truncated excerpts
    """
    top_docs = get_top_docs_for_topic(
        df_docs=df_docs,
        doc_topic_df=doc_topic_df,
        topic_id=topic_id,
        top_n=top_n_docs,
    )

    topic_col = f"topic_{topic_id}"
    docs_payload = []
    for _, row in top_docs.iterrows():
        docs_payload.append(
            {
                "lens_id": row["lens_id"],
                "topic_weight": float(row[topic_col]),
                "text_excerpt": str(row["text"])[:excerpt_chars],
            }
        )

    return {
        "topic_id": topic_id,
        "top_words": topic_words[topic_id],
        "top_docs": docs_payload,
    }


def build_all_topic_labeling_payloads(
    df_docs: pd.DataFrame,
    doc_topic_df: pd.DataFrame,
    topic_words: dict[int, list[str]],
    top_n_docs: int = 3,
    excerpt_chars: int = 1500,
) -> list[dict]:
    """
    Build labeling payloads for all topics.
    """
    payloads = []
    for topic_id in sorted(topic_words):
        payloads.append(
            build_topic_labeling_payload(
                df_docs=df_docs,
                doc_topic_df=doc_topic_df,
                topic_words=topic_words,
                topic_id=topic_id,
                top_n_docs=top_n_docs,
                excerpt_chars=excerpt_chars,
            )
        )
    return payloads


def run_lda_experiment(config: LDAConfig) -> tuple[pd.DataFrame, tp.LDAModel, pd.DataFrame, dict[int, list[str]]]:
    """
    End-to-end helper:
    - prepare corpus
    - train model
    - extract topic words
    - build doc-topic matrix
    """
    df_docs = prepare_patent_corpus(config)
    run = fit_lda_run(
        token_lists=df_docs["tokens"].tolist(),
        lens_ids=df_docs["lens_id"].tolist(),
        config=config,
    )

    return df_docs, run["model"], run["doc_topic_df"], run["topic_words"]


def save_run_outputs(
    output_dir: Path,
    config: LDAConfig,
    mdl: tp.LDAModel,
    doc_topic_df: pd.DataFrame,
    topic_words: dict[int, list[str]],
) -> None:
    """
    Save lightweight run artifacts to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2, default=str)

    with open(output_dir / "topic_words.json", "w") as f:
        json.dump(topic_words, f, indent=2)

    doc_topic_df.to_parquet(output_dir / "doc_topic.parquet", index=False)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "k": mdl.k,
                "num_docs": len(mdl.docs),
                "vocab_size": mdl.num_vocabs,
                "ll_per_word": mdl.ll_per_word,
            },
            f,
            indent=2,
        )