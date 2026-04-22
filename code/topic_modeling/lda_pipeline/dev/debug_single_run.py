from pathlib import Path
import sys

PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from config import LDAConfig
from data import prepare_patent_corpus
from model import fit_lda_run

from label_topics import (
    TopicLabelingConfig,
    label_saved_run_outputs,
)

def topic_labeling_test():
    print("\n=== TOPIC LABELING DEMO ===")

    # -----------------------------------
    # LDA CONFIG (same as your pipeline)
    # -----------------------------------
    lda_config = LDAConfig()

    # -----------------------------------
    # LOAD CORPUS (needed for payloads)
    # -----------------------------------
    print("\nLoading corpus...")
    df_docs = prepare_patent_corpus(lda_config)

    print(f"Docs loaded: {len(df_docs)}")

    # -----------------------------------
    # SELECT ONE RUN (keep this small for testing)
    # -----------------------------------
    run_dir = Path("outputs/lda/runs")

    # 👇 update this to your actual run
    run_name = sorted(run_dir.glob("run_*"))[-1]
    print(f"\nUsing run: {run_name}")

    # pick ONE config to test
    k = 20
    seed = 0

    doc_topic_path = run_name / "doc_topics" / f"k{k}_seed{seed}.parquet"
    topic_words_path = run_name / "topic_words" / f"k{k}_seed{seed}.json"

    assert doc_topic_path.exists(), f"Missing: {doc_topic_path}"
    assert topic_words_path.exists(), f"Missing: {topic_words_path}"

    # -----------------------------------
    # LABELING CONFIG (your simplified setup)
    # -----------------------------------
    label_config = TopicLabelingConfig(
        model="gpt-5-mini",
        n_passes=3,
        top_n_docs=3,
        excerpt_chars=1500,
    )

    # -----------------------------------
    # RUN LABELING
    # -----------------------------------
    print("\nRunning topic labeling...")

    results = label_saved_run_outputs(
        df_docs=df_docs,
        doc_topic_path=doc_topic_path,
        topic_words_path=topic_words_path,
        output_dir=run_name / "topic_labels" / f"k{k}_seed{seed}",
        config=label_config,
    )

    # -----------------------------------
    # QUICK INSPECTION
    # -----------------------------------
    print("\n=== SAMPLE OUTPUT ===")

    for r in results[:5]:
        print("\n-------------------------")
        print(f"Topic {r['topic_id']}")
        print(f"Label: {r['label']}")
        print(f"Candidates: {r['candidate_labels']}")
        print(f"Explanation: {r['explanation'][:200]}...")

    print("\n=== DONE ===")
def lda_run_test():
    print("\n=== DEBUG SINGLE RUN ===")

    # -----------------------------------
    # Small config for quick test
    # -----------------------------------
    default_config = LDAConfig()
    config = LDAConfig(
        version_prefix=default_config.version_prefix,
        predictions_path=default_config.predictions_path,
        stopwords_path=default_config.stopwords_path,
        base_data_dir=default_config.base_data_dir,
        text_column=default_config.text_column,
        id_column=default_config.id_column,
        min_bigram_count=15,
        k=20,              # small test
        alpha=0.3,
        eta=0.01,
        min_df=10,
        max_df=0.7,
        seed=0,
        iterations=50,     # shorter for speed
        top_words_n=10,
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
    topic_labeling_test()
