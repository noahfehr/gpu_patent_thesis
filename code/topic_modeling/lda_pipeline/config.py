from dataclasses import dataclass
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PIPELINE_DIR.parents[2]


@dataclass
class LDAConfig:
    """
    Configuration object for one LDA experiment run.

    This keeps all key settings in one place so you can:
    - rerun the same setup consistently
    - sweep over topic counts / hyperparameters
    - save configs alongside outputs
    """

    # Short corpus identifier used to construct the processed CSV filename.
    # Example: version_prefix="v6" -> "../../data/claims_added/v6_processed.csv"
    version_prefix: str = "v6"

    # Path to the JSONL file containing the two-stage classification outputs.
    # This file is used to filter the patent corpus down to only the patents
    # classified as accelerator hardware design patents.
    predictions_path: Path = Path(
        PROJECT_ROOT / "data/analysis/runs/v6__full__two_stage__ts1__predictions.jsonl"
    )

    # Path to the custom stopword list.
    # These are added on top of sklearn's built-in English stopwords.
    stopwords_path: Path = PIPELINE_DIR / "custom_stopwords.txt"

    # Base directory containing processed patent CSV files.
    # The actual CSV loaded is:
    #   base_data_dir / f"{version_prefix}_processed.csv"
    base_data_dir: Path = PROJECT_ROOT / "data/claims_added"

    # Name of the raw text column to use as the main input text.
    # In your current workflow this is "claims".
    text_column: str = "claims"

    # Column used to link the CSV to the classification JSONL.
    # Must exist in the patent dataframe and in the JSONL records.
    id_column: str = "lens_id"

    # Minimum corpus-wide count required for a bigram to be kept.
    # Example: min_bigram_count=15 means only adjacent token pairs seen
    # at least 15 times across the corpus will be added as bigrams.
    min_bigram_count: int = 20

    # Number of latent topics in the LDA model.
    k: int = 30

    # Dirichlet prior over document-topic distributions.
    # Lower values -> documents concentrate on fewer topics.
    alpha: float = 0.5

    # Dirichlet prior over topic-word distributions.
    # Lower values -> topics concentrate on fewer, more distinctive words.
    eta: float = 0.01

    # Ignore vocabulary items that appear in fewer than this many documents.
    # This helps remove very rare tokens.
    min_df: int = 20

    # Ignore vocabulary items that appear in more than this share of documents.
    # Helps remove overly common, low-information tokens across the corpus.
    max_df: float = 0.18

    # Random seed for reproducibility.
    seed: int = 42

    # Number of training iterations for tomotopy LDA.
    iterations: int = 1500

    # Number of top words to retrieve per topic for inspection/export.
    top_words_n: int = 15
