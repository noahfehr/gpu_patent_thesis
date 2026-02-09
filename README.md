# GPU Patent Thesis

A research project for analyzing GPU-related patents using the lens.org API.

## Project Structure

```
gpu_patent_thesis/
├── data/
│   └── patents/
│       └── v1_gpu_kw_cpc/          # GPU patents by CPC codes
│           ├── raw/                 # Raw API responses (compressed)
│           ├── parsed/              # Normalized parquet/jsonl files
│           ├── text_clean/          # Cleaned text fields for embedding
│           ├── embeddings/          # Vector embeddings (.npy or parquet)
│           ├── logs/                # Pipeline run logs and query specs
│           └── README.md            # Detailed folder documentation
└── code/
    └── v1_sample.ipynb              # Patent data pipeline notebook
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install requests pandas pyarrow jupyter
```

### Setup

1. Get a lens.org API token from https://www.lens.org/lens/user/subscriptions
2. Set your API token as an environment variable:
   ```bash
   export LENS_API_TOKEN="your-token-here"
   ```

### Running the Pipeline

1. Navigate to the code directory:
   ```bash
   cd code
   ```

2. Start Jupyter notebook:
   ```bash
   jupyter notebook v1_sample.ipynb
   ```

3. Run the cells to:
   - Fetch GPU-related patents from lens.org (US jurisdiction)
   - Parse and normalize the data
   - Clean text fields for analysis
   - Optionally generate embeddings
   - Log all operations

## Pipeline Overview

The `v1_sample.ipynb` notebook implements a complete data pipeline:

1. **Fetch**: Query lens.org API with GPU-related CPC codes
2. **Store Raw**: Save compressed JSON responses
3. **Parse**: Normalize schema to parquet format
4. **Clean**: Prepare text fields for embedding
5. **Embed**: Generate vector embeddings (optional)
6. **Log**: Track all operations for reproducibility

## CPC Codes Used

The pipeline focuses on GPU-related patents using these CPC classifications:
- **G06F3/14**: Graphics input/output
- **G06T1/20**: Parallel data processing
- **G06T1/60**: GPU architecture
- **G09G5/36**: Graphics processing

## Data Format

### Raw Data
- Compressed JSON files from lens.org API
- Contains full patent records with all metadata

### Parsed Data
- Parquet files with normalized schema
- Fields: lens_id, title, abstract, description, dates, applicants, inventors, CPC codes

### Cleaned Text
- Parquet files with cleaned text ready for embedding
- Combined text fields optimized for semantic analysis

### Embeddings
- NumPy arrays (.npy) with vector embeddings
- Separate ID files for mapping

## License

This project is for research purposes.