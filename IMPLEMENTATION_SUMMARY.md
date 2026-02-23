# Three-Dataset Pipeline Implementation Summary

## Requirements (from problem statement)

The pipeline was requested for three datasets:

1. **Core**: Patents with CPC codes:
   - G06F 9/3887, G06F 9/3888, G06F 9/38885
   - G06F 9/3009
   - G06F 12/0842, G06F 12/0844
   - G06F 13/42, G06F 13/14, G06F 13/16

2. **Expansion**: Patents with CPC codes:
   - G06F 15/8007, G06F 15/8053
   - G06N 3/06

3. **Expansionxvocab**: Patents from Expansion that contain keywords:
   - gpu
   - high-performance compute
   - hpc

## Implementation

### Files Created

1. **`code/v1_pipeline.ipynb`** - Jupyter notebook implementing the full pipeline
2. **`code/v1_pipeline.py`** - Command-line script version
3. **`code/test_v1_pipeline.py`** - Validation tests for the pipeline
4. **`data/patents/v1_core_expansion/README.md`** - Comprehensive documentation
5. **Directory structure** - Created all necessary directories with .gitkeep files

### Features Implemented

✅ **Core Dataset Fetching**
- Queries lens.org API with 9 CPC codes
- Filters by US jurisdiction
- Saves raw compressed JSON data
- Parses and normalizes to parquet format
- Logs all operations

✅ **Expansion Dataset Fetching**
- Queries lens.org API with 3 CPC codes
- Filters by US jurisdiction
- Saves raw compressed JSON data
- Parses and normalizes to parquet format
- Logs all operations

✅ **ExpansionXVocab Filtering**
- Filters expansion dataset by keywords
- Case-insensitive keyword matching
- Searches in title, abstract, and description fields
- Creates subset with filtered raw and parsed data
- Logs filtering operations

✅ **Data Pipeline**
- Modular, reusable functions
- Comprehensive error handling
- Progress reporting with emojis
- Detailed logging with JSON format
- Parquet format for efficient storage
- Directory structure following v1 conventions

✅ **Testing & Validation**
- Automated tests for all key functions
- CPC code validation (9 core + 3 expansion)
- Keyword filtering validation
- Query structure validation
- Directory structure validation
- Documentation validation
- All tests pass ✓

✅ **Documentation**
- Main README updated with v2 pipeline info
- Detailed v1_core_expansion README
- Inline code documentation
- Usage instructions for both notebook and script
- Dataset relationship diagrams

### CPC Code Verification

**Core (9 codes):**
- G06F9/3887 ✓
- G06F9/3888 ✓
- G06F9/38885 ✓
- G06F9/3009 ✓
- G06F12/0842 ✓
- G06F12/0844 ✓
- G06F13/42 ✓
- G06F13/14 ✓
- G06F13/16 ✓

**Expansion (3 codes):**
- G06F15/8007 ✓
- G06F15/8053 ✓
- G06N3/06 ✓

**Keywords (3):**
- gpu ✓
- high-performance compute ✓
- hpc ✓

### Usage

**Option 1: Jupyter Notebook**
```bash
cd code
jupyter notebook v1_pipeline.ipynb
```

**Option 2: Command-Line Script**
```bash
cd code
python3 v1_pipeline.py
```

**Option 3: Run Tests**
```bash
cd code
python3 test_v1_pipeline.py
```

### Data Output Structure

```
data/patents/v1_core_expansion/
├── core/
│   ├── raw/          # core_YYYYMMDD_HHMMSS.json.gz
│   ├── parsed/       # core_YYYYMMDD_HHMMSS.parquet
│   └── logs/         # core_YYYYMMDD_HHMMSS.log.json
├── expansion/
│   ├── raw/          # expansion_YYYYMMDD_HHMMSS.json.gz
│   ├── parsed/       # expansion_YYYYMMDD_HHMMSS.parquet
│   └── logs/         # expansion_YYYYMMDD_HHMMSS.log.json
└── expansionxvocab/
    ├── raw/          # expansionxvocab_YYYYMMDD_HHMMSS.json.gz
    ├── parsed/       # expansionxvocab_YYYYMMDD_HHMMSS.parquet
    └── logs/         # expansionxvocab_YYYYMMDD_HHMMSS.log.json
```

## Testing Results

All validation tests passed:
- ✓ CPC codes configuration (9 core + 3 expansion)
- ✓ Keyword filtering logic
- ✓ Query structure
- ✓ Directory structure
- ✓ Documentation completeness

## Next Steps

To use the pipeline:

1. Set your lens.org API token:
   ```bash
   export LENS_API_TOKEN="your-token-here"
   ```

2. Install dependencies (if not already installed):
   ```bash
   pip install -r requirements.txt
   ```

3. Run the pipeline (choose one):
   - Jupyter: `jupyter notebook code/v1_pipeline.ipynb`
   - CLI: `python3 code/v1_pipeline.py`

The pipeline will:
- Fetch core patents (9 CPC codes)
- Fetch expansion patents (3 CPC codes)
- Filter expansion by keywords to create expansionxvocab
- Save all data in compressed and parquet formats
- Log all operations for reproducibility
