# Patent Claims Extraction Pipeline

This directory contains tools to extract detailed patent information, including claims, from Lens.org CSV exports using the Lens API.

## Overview

You have a CSV export from Lens with these fields:
```
"#",Jurisdiction,Kind,Display Key,Lens ID,Publication Date,Publication Year,Application Number,Application Date,Priority Numbers,Earliest Priority Date,Title,Abstract,Applicants,Inventors,Owners,URL,Document Type,Has Full Text,Cites Patent Count,Cited by Patent Count,Simple Family Size,Simple Family Members,Simple Family Member Jurisdictions,Extended Family Size,Extended Family Members,Extended Family Member Jurisdictions,Sequence Count,CPC Classifications,IPCR Classifications,US Classifications,NPL Citation Count,NPL Resolved Citation Count,NPL Resolved Lens ID(s),NPL Resolved External ID(s),NPL Citations,Legal Status
```

The pipeline will use your Lens IDs to query the Lens API and extract detailed patent information including **claims** that are not available in the basic export.

## Files

- `lens_id_extract.py` - Queries Lens API to get detailed patent data
- `lens_fill_df.py` - Processes JSON API responses into structured DataFrame
- `claims_pipeline.py` - Complete pipeline combining both functions
- `example_usage.py` - Example showing how to use the pipeline


**Option A: Command Line**
```bash
python claims_pipeline.py \
  --csv_input /path/to/your/lens_export.csv \
  --token your_lens_api_token \
  --output_dir ./output
```

**Option B: Python Script**
```python
from claims_pipeline import create_claims_pipeline

result_file = create_claims_pipeline(
    csv_input="your_lens_export.csv",
    lens_token="your_api_token",
    output_dir="./output"
)
```

## Output

The pipeline creates several files in your output directory:

1. `lens_ids.txt` - Extracted Lens IDs from your CSV
2. `lens_api_data.json` - Raw API response from Lens
3. `patents_with_claims.csv` - **Final structured dataset with claims**

## Final Dataset Fields

Your final CSV will include all this information:

| Field | Description |
|-------|-------------|
| `lens_id` | Unique Lens identifier |
| `jurisdiction` | Patent jurisdiction |
| `doc_number` | Document number |
| `kind_code` | Patent kind code |
| `publication_type` | Type of publication |
| `priority_date` | Priority date |
| `priority_jurisdiction` | Priority jurisdiction |
| `publish_date` | Publication date |
| `title` | Patent title |
| `abstract` | Patent abstract |
| `**claims**` | **Patent claims text** |
| `applicant` | Patent applicants |
| `inventor_jurisdiction` | Inventor jurisdictions |
| `cpc_codes` | CPC classification codes |
| `simp_famil_lens_ids` | Simple family member IDs |
| `for_cite_lens_ids` | Forward citation IDs |
| `back_cite_lens_ids` | Backward citation IDs |
| `collection_file` | Source filename |
