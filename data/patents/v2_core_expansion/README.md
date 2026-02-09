# v2_core_expansion: Three-Dataset Patent Pipeline

This directory contains three related patent datasets for GPU and parallel computing analysis.

## Dataset Definitions

### 1. Core Dataset (`core/`)

Patents with CPC codes related to fundamental parallel processing, memory management, and bus architectures:

- **G06F 9/3887** - Parallel processing (vector processors)
- **G06F 9/3888** - Parallel processing (SIMD)
- **G06F 9/38885** - Parallel processing (specific implementations)
- **G06F 9/3009** - Multiprocessing arrangements
- **G06F 12/0842** - Cache memory (allocation policy)
- **G06F 12/0844** - Cache memory (replacement policy)
- **G06F 13/42** - Bus transfer protocol
- **G06F 13/14** - Handling requests for interconnection or transfer for access to input/output bus
- **G06F 13/16** - Handling requests for interconnection or transfer for access to memory bus

**Purpose**: Core patents directly related to GPU-relevant computing architectures and memory systems.

### 2. Expansion Dataset (`expansion/`)

Patents with CPC codes related to multiprocessor systems and neural networks:

- **G06F 15/8007** - Multiprocessor systems; Multiprocessing systems
- **G06F 15/8053** - Multiprocessor systems with vector processor
- **G06N 3/06** - Physical realisation of neural networks (including hardware implementations)

**Purpose**: Broader set of patents covering multiprocessing and neural network hardware that may relate to GPU computing.

### 3. ExpansionXVocab Dataset (`expansionxvocab/`)

A filtered subset of the Expansion dataset containing only patents that mention specific GPU/HPC terminology:

**Keywords searched** (case-insensitive, in title, abstract, or description):
- `gpu`
- `high-performance compute`
- `hpc`

**Purpose**: Most relevant patents from the expansion set that explicitly discuss GPU or high-performance computing.

## Pipeline Workflow

The `v2_core_expansion_pipeline.ipynb` notebook implements the complete pipeline:

1. **Fetch Core Dataset**
   - Query lens.org API with core CPC codes
   - Filter by US jurisdiction
   - Save raw and parsed data

2. **Fetch Expansion Dataset**
   - Query lens.org API with expansion CPC codes
   - Filter by US jurisdiction
   - Save raw and parsed data

3. **Filter ExpansionXVocab Dataset**
   - Filter expansion patents by keywords
   - Create subset with only keyword-matching patents
   - Save filtered raw and parsed data

## Directory Structure

Each dataset has the following subdirectories:

```
{dataset_name}/
├── raw/            # Compressed JSON from lens.org API (*.json.gz)
├── parsed/         # Normalized parquet files (*.parquet)
├── text_clean/     # Cleaned text for embeddings
├── embeddings/     # Vector embeddings (if generated)
└── logs/           # Pipeline execution logs (*.log.json)
```

## Data Format

### Raw Data
- Format: Compressed JSON (`.json.gz`)
- Content: Complete patent records from lens.org API
- Naming: `{dataset}_{timestamp}.json.gz`

### Parsed Data
- Format: Parquet (`.parquet`)
- Schema:
  - `lens_id`: Unique patent identifier
  - `title`: Patent title
  - `abstract`: Patent abstract
  - `description`: Full patent description
  - `date_published`: Publication date
  - `jurisdiction`: Patent jurisdiction (US)
  - `applicants`: JSON array of applicants
  - `inventors`: JSON array of inventors
  - `cpc_codes`: JSON array of CPC classification codes
  - `claims_count`: Number of claims
  - `first_claim`: Text of first claim
- Naming: `{dataset}_{timestamp}.parquet`

### Logs
- Format: JSON (`.log.json`)
- Content: Query specifications, result counts, file paths, timestamps
- Naming: `{dataset}_{timestamp}.log.json`

## Usage

Run the pipeline notebook:

```bash
cd code
jupyter notebook v2_core_expansion_pipeline.ipynb
```

Or run the command-line script:

```bash
cd code
python3 v2_core_expansion_pipeline.py
```

Ensure you have:
1. Set the `LENS_API_TOKEN` environment variable
2. Installed required packages: `pip install -r ../requirements.txt`

## Dataset Relationships

```
Core Dataset (9 CPC codes)
└─ Fundamental GPU-relevant architectures

Expansion Dataset (3 CPC codes)
└─ Multiprocessor and neural network hardware
   │
   └─ ExpansionXVocab (keyword-filtered)
      └─ Explicitly mentions GPU/HPC terms
```

## Notes

- All datasets are filtered for US jurisdiction (`jurisdiction=US`)
- The maximum results per query can be adjusted in the notebook configuration
- ExpansionXVocab is always a subset of Expansion
- CPC codes follow the Cooperative Patent Classification system
- Keywords are matched case-insensitively across title, abstract, and description fields
