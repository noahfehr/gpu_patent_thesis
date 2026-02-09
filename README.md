# GPU Patent Thesis

A research project for analyzing GPU-related patents using the lens.org API.

## Project Structure

```
gpu_patent_thesis/
├── data/
│   └── patents/
│       ├── v1_gpu_kw_cpc/           # GPU patents by CPC codes
│       │   ├── raw/                 # Raw API responses (compressed)
│       │   ├── parsed/              # Normalized parquet/jsonl files
│       │   ├── text_clean/          # Cleaned text fields for embedding
│       │   ├── embeddings/          # Vector embeddings (.npy or parquet)
│       │   ├── logs/                # Pipeline run logs and query specs
│       │   └── README.md            # Detailed folder documentation
│       └── v2_core_expansion/       # Three-dataset pipeline
│           ├── core/                # Core dataset (9 CPC codes)
│           ├── expansion/           # Expansion dataset (3 CPC codes)
│           ├── expansionxvocab/     # Expansion + keyword filter
│           └── README.md            # Dataset documentation
└── code/
    ├── v1_sample.ipynb              # Original GPU patent pipeline
    └── v2_core_expansion_pipeline.ipynb  # Three-dataset pipeline
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

### Running the Pipelines

#### v1: Original GPU Patent Pipeline

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

#### v2: Three-Dataset Pipeline (Core, Expansion, ExpansionXVocab)

1. Navigate to the code directory:
   ```bash
   cd code
   ```

2. Start Jupyter notebook:
   ```bash
   jupyter notebook v2_core_expansion_pipeline.ipynb
   ```

3. Run the cells to fetch and process three datasets:
   - **Core**: Patents with 9 CPC codes (parallel processing, memory, buses)
   - **Expansion**: Patents with 3 CPC codes (multiprocessor systems, neural networks)
   - **ExpansionXVocab**: Expansion patents filtered by keywords (gpu, hpc, high-performance compute)

## Pipelines Overview

### v1_sample.ipynb - Original Pipeline

A complete data pipeline for GPU-related patents:

1. **Fetch**: Query lens.org API with GPU-related CPC codes
2. **Store Raw**: Save compressed JSON responses
3. **Parse**: Normalize schema to parquet format
4. **Clean**: Prepare text fields for embedding
5. **Embed**: Generate vector embeddings (optional)
6. **Log**: Track all operations for reproducibility

**CPC Codes Used:**
- **G06F3/14**: Graphics input/output
- **G06T1/20**: Parallel data processing
- **G06T1/60**: GPU architecture
- **G09G5/36**: Graphics processing

### v2_core_expansion_pipeline.ipynb - Three-Dataset Pipeline

Implements three related patent datasets:

1. **Core Dataset** (9 CPC codes):
   - G06F 9/3887, G06F 9/3888, G06F 9/38885 (Parallel processing)
   - G06F 9/3009 (Multiprocessing arrangements)
   - G06F 12/0842, G06F 12/0844 (Cache memory)
   - G06F 13/42, G06F 13/14, G06F 13/16 (Bus architectures)

2. **Expansion Dataset** (3 CPC codes):
   - G06F 15/8007, G06F 15/8053 (Multiprocessor systems)
   - G06N 3/06 (Neural networks)

3. **ExpansionXVocab Dataset**:
   - Subset of Expansion filtered by keywords: gpu, high-performance compute, hpc
   - Searches in title, abstract, and description fields

See [data/patents/v2_core_expansion/README.md](data/patents/v2_core_expansion/README.md) for detailed documentation.

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