#!/usr/bin/env python3
"""
GPU Patent Data Pipeline v2: Core, Expansion, and ExpansionXVocab Datasets

Command-line script version of the v2_core_expansion_pipeline notebook.
Run this script to fetch and process all three datasets.

Usage:
    export LENS_API_TOKEN="your-api-token"
    python3 v2_core_expansion_pipeline.py
"""

import os
import json
import gzip
import re
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys

# Configuration
API_TOKEN = os.getenv('LENS_API_TOKEN')
if not API_TOKEN or API_TOKEN == 'your-api-token-here':
    print("ERROR: LENS_API_TOKEN environment variable not set")
    print("Please set it with: export LENS_API_TOKEN='your-token'")
    sys.exit(1)

API_URL = 'https://api.lens.org/patent/search'
BASE_PATH = Path(__file__).parent.parent / 'data' / 'patents' / 'v2_core_expansion'

# Dataset-specific CPC codes
CORE_CPC_CODES = [
    'G06F9/3887', 'G06F9/3888', 'G06F9/38885',  # Parallel processing
    'G06F9/3009',                                 # Multiprocessing
    'G06F12/0842', 'G06F12/0844',                # Cache memory
    'G06F13/42', 'G06F13/14', 'G06F13/16',       # Bus architectures
]

EXPANSION_CPC_CODES = [
    'G06F15/8007', 'G06F15/8053',  # Multiprocessor systems
    'G06N3/06',                     # Neural networks
]

VOCAB_KEYWORDS = ['gpu', 'high-performance compute', 'hpc']
JURISDICTION = 'US'
MAX_RESULTS = 1000


def create_directories(base_path, dataset_name):
    """Create directory structure for a dataset."""
    dataset_path = base_path / dataset_name
    for subdir in ['raw', 'parsed', 'text_clean', 'embeddings', 'logs']:
        (dataset_path / subdir).mkdir(parents=True, exist_ok=True)
    return dataset_path


def build_cpc_query(cpc_codes, jurisdiction, max_results=1000):
    """Build lens.org API query for patents with specific CPC codes."""
    query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {"term": {"classification_cpc.classification_id": code}}
                                for code in cpc_codes
                            ],
                            "minimum_should_match": 1
                        }
                    },
                    {"term": {"jurisdiction": jurisdiction}}
                ]
            }
        },
        "size": max_results,
        "include": [
            "lens_id", "title", "abstract", "description", "claims",
            "date_published", "jurisdiction", "applicants", "inventors",
            "classification_cpc", "biblio"
        ]
    }
    return query


def fetch_patents(api_token, query):
    """Fetch patents from lens.org API."""
    headers = {
        'Authorization': f'Bearer {api_token}',
        'Content-Type': 'application/json'
    }
    response = requests.post(API_URL, json=query, headers=headers)
    response.raise_for_status()
    return response.json()


def save_raw_data(raw_data, output_path, dataset_name, timestamp):
    """Save raw data as compressed JSON."""
    raw_file = output_path / 'raw' / f'{dataset_name}_{timestamp}.json.gz'
    with gzip.open(raw_file, 'wt', encoding='utf-8') as f:
        json.dump(raw_data, f, indent=2)
    return raw_file


def parse_patents(raw_data):
    """Parse raw patent data into normalized schema."""
    patents = raw_data.get('data', [])
    parsed_records = []
    for patent in patents:
        record = {
            'lens_id': patent.get('lens_id'),
            'title': patent.get('title', ''),
            'abstract': patent.get('abstract', ''),
            'description': patent.get('description', ''),
            'date_published': patent.get('date_published'),
            'jurisdiction': patent.get('jurisdiction'),
            'applicants': json.dumps(patent.get('applicants', [])),
            'inventors': json.dumps(patent.get('inventors', [])),
            'cpc_codes': json.dumps([c.get('classification_id') for c in patent.get('classification_cpc', [])]),
            'claims_count': len(patent.get('claims', [])),
            'first_claim': patent.get('claims', [{}])[0].get('claim_text') if patent.get('claims') else None
        }
        parsed_records.append(record)
    return pd.DataFrame(parsed_records)


def save_parsed_data(df, output_path, dataset_name, timestamp):
    """Save parsed data as parquet."""
    parsed_file = output_path / 'parsed' / f'{dataset_name}_{timestamp}.parquet'
    df.to_parquet(parsed_file, index=False)
    return parsed_file


def contains_keywords(text, keywords):
    """Check if text contains any of the specified keywords (case-insensitive)."""
    if not text or pd.isna(text):
        return False
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in keywords)


def filter_by_keywords(df, keywords):
    """Filter dataframe to include only patents containing specified keywords."""
    mask = df.apply(
        lambda row: (
            contains_keywords(row['title'], keywords) or
            contains_keywords(row['abstract'], keywords) or
            contains_keywords(row['description'], keywords)
        ),
        axis=1
    )
    return df[mask].copy()


def log_pipeline_run(output_path, dataset_name, timestamp, query, results_count, file_info):
    """Log pipeline execution details."""
    log_file = output_path / 'logs' / f'{dataset_name}_{timestamp}.log.json'
    log_data = {
        'timestamp': timestamp,
        'dataset': dataset_name,
        'query': query,
        'results_count': results_count,
        'files': file_info
    }
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    return log_file


def process_core_dataset(timestamp):
    """Fetch and process core dataset."""
    print("=" * 80)
    print("CORE DATASET PIPELINE")
    print("=" * 80)
    
    core_path = create_directories(BASE_PATH, 'core')
    print(f"\n✓ Directories created at: {core_path}")
    
    core_query = build_cpc_query(CORE_CPC_CODES, JURISDICTION, MAX_RESULTS)
    print(f"\n📋 Query: {len(CORE_CPC_CODES)} CPC codes, Jurisdiction: {JURISDICTION}")
    
    print(f"\n🔍 Fetching core patents from lens.org...")
    core_raw_data = fetch_patents(API_TOKEN, core_query)
    core_count = len(core_raw_data.get('data', []))
    print(f"✓ Fetched {core_count} patents (total available: {core_raw_data.get('total', 0)})")
    
    core_raw_file = save_raw_data(core_raw_data, core_path, 'core', timestamp)
    print(f"✓ Raw data saved: {core_raw_file.stat().st_size / 1024:.2f} KB")
    
    df_core = parse_patents(core_raw_data)
    core_parsed_file = save_parsed_data(df_core, core_path, 'core', timestamp)
    print(f"✓ Parsed data saved: {len(df_core)} records")
    
    log_pipeline_run(core_path, 'core', timestamp, core_query, core_count,
                    {'raw': str(core_raw_file), 'parsed': str(core_parsed_file)})
    
    return df_core, core_raw_data


def process_expansion_dataset(timestamp):
    """Fetch and process expansion dataset."""
    print("\n" + "=" * 80)
    print("EXPANSION DATASET PIPELINE")
    print("=" * 80)
    
    expansion_path = create_directories(BASE_PATH, 'expansion')
    print(f"\n✓ Directories created at: {expansion_path}")
    
    expansion_query = build_cpc_query(EXPANSION_CPC_CODES, JURISDICTION, MAX_RESULTS)
    print(f"\n📋 Query: {len(EXPANSION_CPC_CODES)} CPC codes, Jurisdiction: {JURISDICTION}")
    
    print(f"\n🔍 Fetching expansion patents from lens.org...")
    expansion_raw_data = fetch_patents(API_TOKEN, expansion_query)
    expansion_count = len(expansion_raw_data.get('data', []))
    print(f"✓ Fetched {expansion_count} patents (total available: {expansion_raw_data.get('total', 0)})")
    
    expansion_raw_file = save_raw_data(expansion_raw_data, expansion_path, 'expansion', timestamp)
    print(f"✓ Raw data saved: {expansion_raw_file.stat().st_size / 1024:.2f} KB")
    
    df_expansion = parse_patents(expansion_raw_data)
    expansion_parsed_file = save_parsed_data(df_expansion, expansion_path, 'expansion', timestamp)
    print(f"✓ Parsed data saved: {len(df_expansion)} records")
    
    log_pipeline_run(expansion_path, 'expansion', timestamp, expansion_query, expansion_count,
                    {'raw': str(expansion_raw_file), 'parsed': str(expansion_parsed_file)})
    
    return df_expansion, expansion_raw_data


def process_expansionxvocab_dataset(df_expansion, expansion_raw_data, timestamp):
    """Filter and process expansionxvocab dataset."""
    print("\n" + "=" * 80)
    print("EXPANSIONXVOCAB DATASET PIPELINE")
    print("=" * 80)
    
    expansionxvocab_path = create_directories(BASE_PATH, 'expansionxvocab')
    print(f"\n✓ Directories created at: {expansionxvocab_path}")
    
    print(f"\n📋 Filtering {len(df_expansion)} expansion patents by keywords: {', '.join(VOCAB_KEYWORDS)}")
    
    df_expansionxvocab = filter_by_keywords(df_expansion, VOCAB_KEYWORDS)
    vocab_count = len(df_expansionxvocab)
    print(f"✓ Filtered to {vocab_count} patents ({vocab_count/len(df_expansion)*100:.1f}% of expansion)")
    
    if vocab_count > 0:
        filtered_raw_data = {
            'total': vocab_count,
            'data': [
                patent for patent in expansion_raw_data.get('data', [])
                if patent.get('lens_id') in df_expansionxvocab['lens_id'].values
            ]
        }
        
        vocab_raw_file = save_raw_data(filtered_raw_data, expansionxvocab_path, 'expansionxvocab', timestamp)
        print(f"✓ Raw data saved: {vocab_raw_file.stat().st_size / 1024:.2f} KB")
        
        vocab_parsed_file = save_parsed_data(df_expansionxvocab, expansionxvocab_path, 'expansionxvocab', timestamp)
        print(f"✓ Parsed data saved: {len(df_expansionxvocab)} records")
        
        log_pipeline_run(expansionxvocab_path, 'expansionxvocab', timestamp,
                        {'source': 'expansion', 'keywords': VOCAB_KEYWORDS}, vocab_count,
                        {'raw': str(vocab_raw_file), 'parsed': str(vocab_parsed_file)})
    else:
        print("\n⚠ No patents matched the keyword filter")
    
    return df_expansionxvocab


def main():
    """Main pipeline execution."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        df_core, _ = process_core_dataset(timestamp)
        df_expansion, expansion_raw_data = process_expansion_dataset(timestamp)
        df_expansionxvocab = process_expansionxvocab_dataset(df_expansion, expansion_raw_data, timestamp)
        
        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"\nTimestamp: {timestamp}")
        print(f"\nDataset Statistics:")
        print(f"  Core Dataset:          {len(df_core):5d} patents")
        print(f"  Expansion Dataset:     {len(df_expansion):5d} patents")
        print(f"  ExpansionXVocab:       {len(df_expansionxvocab):5d} patents")
        print(f"\n✓ Pipeline execution complete!")
        print(f"\nAll data saved to: {BASE_PATH}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
