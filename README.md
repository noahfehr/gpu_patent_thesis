# GPU Patent Thesis

A research project for analyzing GPU-related patents using the lens.org API.

## Project Structure

```
gpu_patent_thesis/
├── data/
│   └── patents/
│       └── v1_core_expansion/       # Three-dataset pipeline
│           ├── core/                # Core dataset (9 CPC codes)
│           ├── expansion/           # Expansion dataset (3 CPC codes)
│           ├── expansionxvocab/     # Expansion + keyword filter
│           └── README.md            # Dataset documentation
└── code/
    ├── v1_pipeline.ipynb            # Three-dataset pipeline notebook
    ├── v1_pipeline.py               # Command-line script version
    └── test_v1_pipeline.py          # Validation tests
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

2. **Option 1: Jupyter Notebook**
   ```bash
   jupyter notebook v1_pipeline.ipynb
   ```

3. **Option 2: Command-line Script**
   ```bash
   python3 v1_pipeline.py
   ```

4. **Option 3: Run Validation Tests**
   ```bash
   python3 test_v1_pipeline.py
   ```

The pipeline will fetch and process three datasets:
- **Core**: Patents with 9 CPC codes (parallel processing, memory, buses)
- **Expansion**: Patents with 3 CPC codes (multiprocessor systems, neural networks)
- **ExpansionXVocab**: Expansion patents filtered by keywords (gpu, hpc, high-performance compute)

## Pipeline Overview

The `v1_pipeline` implements a three-dataset pipeline for GPU and parallel computing patent analysis:

1. **Fetch**: Query lens.org API with specific CPC codes for each dataset
2. **Store Raw**: Save compressed JSON responses
3. **Parse**: Normalize schema to parquet format
4. **Filter**: Apply keyword filtering for ExpansionXVocab dataset
5. **Log**: Track all operations for reproducibility

### Three Datasets

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

See [data/patents/v1_core_expansion/README.md](data/patents/v1_core_expansion/README.md) for detailed documentation.

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