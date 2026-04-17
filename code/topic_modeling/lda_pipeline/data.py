from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from config import LDAConfig


def load_patent_dataframe(config: LDAConfig) -> pd.DataFrame:
    """
    Load the processed patent CSV for the selected corpus version.
    """
    data_path = config.base_data_dir / f"{config.version_prefix}_processed.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"patent data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"Dataset loaded: {df.shape[0]} patents")

    if config.id_column not in df.columns:
        raise KeyError(f"missing id column '{config.id_column}' in patent dataframe")

    if config.text_column not in df.columns:
        raise KeyError(f"missing text column '{config.text_column}' in patent dataframe")

    df[config.id_column] = df[config.id_column].astype(str)
    df["text"] = df[config.text_column]

    return df


def load_keep_lens_ids(predictions_path: Path) -> set[str]:
    """
    Read the classification JSONL and return the set of lens_ids
    classified as accelerator hardware design patents.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"predictions file not found: {predictions_path}")

    keep_lens_ids: set[str] = set()

    with open(predictions_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if (
                record.get("status") == "success"
                and record.get("is_accelerator_hardware_design_patent") is True
            ):
                keep_lens_ids.add(str(record["lens_id"]))

    print(f"Predicted hardware design patents: {len(keep_lens_ids)}")
    return keep_lens_ids


def filter_patents_to_design_set(df: pd.DataFrame, keep_lens_ids: set[str], id_column: str) -> pd.DataFrame:
    """
    Filter the patent dataframe to only the patents whose ids are in keep_lens_ids.
    """
    df = df[df[id_column].isin(keep_lens_ids)].copy()
    print(f"Filtered dataset: {df.shape[0]} patents")
    return df


def load_stopwords(stopfile: Path) -> set[str]:
    """
    Combine sklearn English stopwords with custom stopwords from file.
    """
    if not stopfile.exists():
        raise FileNotFoundError(f"custom stopwords file not found: {stopfile}")

    with open(stopfile, "r") as f:
        custom_stopwords = [line.strip() for line in f if line.strip()]

    stop_words = set(ENGLISH_STOP_WORDS)
    for w in custom_stopwords:
        for part in w.split():
            stop_words.add(part.lower())

    return stop_words


def clean_text(s: str) -> str:
    """
    Lowercase and normalize text while keeping letters, digits, underscores, hyphens.
    Also removes standalone 1-2 digit numbers.
    """
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9_\s\-]", " ", s)
    s = re.sub(r"\b\d{1,2}\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize_unigrams(s: str, stop_words: set[str]) -> list[str]:
    """
    Tokenize cleaned text into unigram tokens, then remove stopwords.
    """
    s = clean_text(s)
    tokens = re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]{1,}\b", s)
    tokens = [t for t in tokens if t not in stop_words]
    return tokens


def build_bigram_vocab(token_lists: list[list[str]], min_count: int = 100) -> set[tuple[str, str]]:
    """
    Build a vocabulary of frequent adjacent bigrams across the corpus.
    """
    bigram_counts: Counter[tuple[str, str]] = Counter()

    for tokens in token_lists:
        for i in range(len(tokens) - 1):
            bigram_counts[(tokens[i], tokens[i + 1])] += 1

    return {bg for bg, count in bigram_counts.items() if count >= min_count}


def add_bigrams(tokens: list[str], bigram_vocab: set[tuple[str, str]]) -> list[str]:
    """
    Keep unigrams and append frequent bigrams as underscore-joined tokens.
    """
    out = list(tokens)

    for i in range(len(tokens) - 1):
        bg = (tokens[i], tokens[i + 1])
        if bg in bigram_vocab:
            out.append(tokens[i] + "_" + tokens[i + 1])

    return out


def prepare_patent_corpus(config: LDAConfig) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - load patent CSV
    - filter to design patents using JSONL predictions
    - load stopwords
    - clean text
    - tokenize unigrams
    - build and add corpus-wide bigrams
    - return dataframe with text/tokens columns
    """
    df = load_patent_dataframe(config)

    keep_lens_ids = load_keep_lens_ids(config.predictions_path)
    df = filter_patents_to_design_set(df, keep_lens_ids, config.id_column)

    tensor_count = df["text"].str.contains("tensor", case=False, na=False).sum()
    print(f"Patents containing 'tensor': {tensor_count}")

    stop_words = load_stopwords(config.stopwords_path)

    df["text_clean"] = df["text"].map(clean_text)
    df["tokens_unigram"] = df["text"].map(lambda s: tokenize_unigrams(s, stop_words))

    df = df[df["tokens_unigram"].map(len) > 0].copy()

    bigram_vocab = build_bigram_vocab(
        df["tokens_unigram"].tolist(),
        min_count=config.min_bigram_count,
    )

    print(f"Frequent bigrams kept: {len(bigram_vocab)}")
    print("Sample bigrams:", [f"{a}_{b}" for a, b in list(sorted(bigram_vocab))[:20]])

    df["tokens"] = df["tokens_unigram"].map(lambda toks: add_bigrams(toks, bigram_vocab))
    df = df[df["tokens"].map(len) > 0].copy()

    print(f"Usable documents: {len(df)}")
    print(f"Average unigram tokens per doc: {df['tokens_unigram'].map(len).mean():.1f}")
    print(f"Average tokens per doc after bigrams: {df['tokens'].map(len).mean():.1f}")

    for i in range(min(3, len(df))):
        print(f"\nDoc {i} sample tokens:")
        print(df.iloc[i]["tokens"][:40])

    return df