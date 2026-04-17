from pathlib import Path

from config import LDAConfig
from data import prepare_patent_corpus
from model import fit_lda_run


def main():
    print("\n=== DEBUG SINGLE RUN ===")

    # -----------------------------------
    # Small config for quick test
    # -----------------------------------
    config = LDAConfig(
        version_prefix="v6",
        predictions_path=Path("data/analysis/runs/v6__full__two_stage__ts1__predictions.jsonl"),
        stopwords_path=Path("code/topic_modeling/lda_pipeline/custom_stopwords.txt"),
        base_data_dir=Path("data/claims_added"),
        text_column="claims",
        id_column="lens_id",
        min_bigram_count=15,
        k=20,              # small test
        alpha=0.3,
        eta=0.01,
        min_df=10,
        rm_top=70,
        seed=0,
        iterations=50,     # shorter for speed
        top_words_n=10,
        top_docs_n=3,
    )

    # -----------------------------------
    # Load + preprocess data
    # -----------------------------------
    print("\nLoading and preparing data...")
    df_docs = prepare_patent_corpus(config)

    print(f"Documents: {len(df_docs)}")
    print(f"Sample lens_id: {df_docs['lens_id'].iloc[0]}")

    token_lists = df_docs["tokens"].tolist()
    lens_ids = df_docs["lens_id"].tolist()

    # -----------------------------------
    # Run one LDA fit
    # -----------------------------------
    print("\nRunning single LDA fit...")
    run = fit_lda_run(
        token_lists=token_lists,
        lens_ids=lens_ids,
        config=config,
    )

    mdl = run["model"]
    doc_topic_df = run["doc_topic_df"]
    topic_words = run["topic_words"]
    metrics = run["metrics"]

    # -----------------------------------
    # Basic checks
    # -----------------------------------
    print("\n=== BASIC CHECKS ===")

    print(f"Model topics (k): {mdl.k}")
    print(f"Doc-topic shape: {doc_topic_df.shape}")

    assert len(doc_topic_df) == len(df_docs), "Mismatch: docs vs doc-topic rows"
    assert "lens_id" in doc_topic_df.columns, "Missing lens_id column"

    topic_cols = [c for c in doc_topic_df.columns if c.startswith("topic_")]
    print(f"Number of topic columns: {len(topic_cols)}")

    assert len(topic_cols) == config.k, "Mismatch: topic columns vs k"

    # Check distributions sum to ~1
    row_sums = doc_topic_df[topic_cols].sum(axis=1)
    print(f"Mean topic distribution sum: {row_sums.mean():.4f}")

    # -----------------------------------
    # Print sample topics
    # -----------------------------------
    print("\n=== SAMPLE TOPICS ===")
    for k in range(min(5, config.k)):
        print(f"Topic {k:02d}: {', '.join(topic_words[k])}")

    # -----------------------------------
    # Print sample doc-topic rows
    # -----------------------------------
    print("\n=== SAMPLE DOC-TOPIC ROW ===")
    print(doc_topic_df.head(1).T.head(15))

    # -----------------------------------
    # Metrics
    # -----------------------------------
    print("\n=== METRICS ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print("\n=== DEBUG SUCCESSFUL ===")


if __name__ == "__main__":
    main()