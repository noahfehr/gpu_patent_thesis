# Data Pipeline Folder Structure

This directory contains the data pipeline for GPU patent analysis using lens.org API.

## v1_gpu_kw_cpc/

Patent data fetched based on GPU-related keywords and CPC (Cooperative Patent Classification) codes, filtered for US jurisdiction.

### Folder Structure

- **raw/**: Contains exactly what was fetched from lens.org API (compressed format)
- **parsed/**: Normalized schema in parquet or jsonl format
- **text_clean/**: Cleaned text fields ready for embedding
- **embeddings/**: Vector embeddings stored as .npy or parquet files, along with patent IDs
- **logs/**: Run logs and query specifications for reproducibility
