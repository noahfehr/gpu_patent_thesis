#!/usr/bin/env python3
"""
Patent Claims Extraction Pipeline

This script processes a CSV export from Lens to extract detailed patent information
including claims using the Lens API. It creates a complete pipeline that:
1. Extracts Lens IDs from your CSV export
2. Queries the Lens API for detailed patent data
3. Processes the results into a structured DataFrame with claims

Input: ../../../data/raw/{prefix}_lens_export.csv
Output: ../../../claims_added/{prefix}_processed.csv

The Lens API token should be set via LENS_API_TOKEN environment variable.

Usage:
    # Command-line interface
    python claims_pipeline.py v1
    python claims_pipeline.py v2
    
    # As a module
    from claims_pipeline import create_claims_pipeline
    create_claims_pipeline("v1")
"""

import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from lens_api.util_lens_id_extract import lens_id_extract
from lens_api.util_lens_fill_df import lens_fill_df


def extract_lens_ids_from_csv(csv_path: str, output_path: str) -> int:
    print(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # The column might be named "Lens ID" with a space
    lens_id_column = None
    for col in df.columns:
        if 'lens' in col.lower() and 'id' in col.lower():
            lens_id_column = col
            break
    
    if lens_id_column is None:
        raise ValueError(f"Could not find Lens ID column. Available columns: {list(df.columns)}")
    
    print(f"Found Lens ID column: '{lens_id_column}'")
    
    # Extract unique Lens IDs and remove any NaN values
    lens_ids = df[lens_id_column].dropna().unique()
    lens_ids = [str(lid).strip() for lid in lens_ids if str(lid).strip() != 'nan']
    
    print(f"Found {len(lens_ids)} unique Lens IDs")
    
    # Save to text file, one ID per line
    with open(output_path, 'w') as f:
        for lid in lens_ids:
            f.write(f"{lid}\n")
    
    print(f"Lens IDs saved to: {output_path}")
    return len(lens_ids)


def create_claims_pipeline(prefix: str):
    """
    Complete pipeline to extract claims from Lens CSV export.
    
    Parameters:
        prefix (str): Version prefix (e.g., 'v1', 'v2', 'v3', 'v4')
    """
    
    # Load token from environment
    load_dotenv()
    lens_token = os.getenv('LENS_API_TOKEN')
    
    if not lens_token:
        raise ValueError("LENS_API_TOKEN environment variable not set")
    
    # Build file paths from home directory (assumes called from home dir)
    data_dir = "data"
    
    csv_input = os.path.join(data_dir, "raw", f"{prefix}_lens_export.csv")
    output_dir = os.path.join(data_dir, "claims_added")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define intermediate file paths
    lens_ids_file = os.path.join(output_dir, f"{prefix}_lens_ids.txt")
    json_output = os.path.join(output_dir, f"{prefix}_lens_api_data.json")
    final_csv = os.path.join(output_dir, f"{prefix}_processed.csv")
    
    print("="*60)
    print(f"PATENT CLAIMS EXTRACTION PIPELINE - {prefix.upper()}")
    print("="*60)
    
    # Step 1: Extract Lens IDs from CSV
    print("\nSTEP 1: Extracting Lens IDs from CSV...")
    if not os.path.exists(csv_input):
        raise FileNotFoundError(f"Input file not found: {csv_input}")
    num_ids = extract_lens_ids_from_csv(csv_input, lens_ids_file)
    
    # Step 2: Query Lens API for detailed data
    print(f"\nSTEP 2: Querying Lens API for {num_ids} patents...")
    print("This may take several minutes depending on the number of patents...")
    lens_id_extract(lens_token, lens_ids_file, json_output)
    
    # Step 3: Process JSON into structured DataFrame with claims
    print("\nSTEP 3: Processing API response into structured DataFrame...")
    
    # Handle chunked output files (lens_id_extract creates files with _0, _1, etc. suffixes)
    base_path, ext = os.path.splitext(json_output)
    chunk_files = []
    
    # Find all chunk files (they are named with starting indices, not sequential)
    # Check for files named with chunk starting indices (0, 10000, 20000, etc.)
    chunk_index = 0
    max_chunk_size = 10000  # Should match the chunk size in lens_id_extract.py
    
    while True:
        chunk_file = f"{base_path}_{chunk_index}{ext}"
        if os.path.exists(chunk_file):
            chunk_files.append(chunk_file)
            chunk_index += max_chunk_size  # Increment by chunk size, not by 1
        else:
            break
    
    if not chunk_files:
        # Fallback: check if the original filename exists
        if os.path.exists(json_output):
            chunk_files = [json_output]
        else:
            raise FileNotFoundError(f"No API data files found. Expected {json_output} or {base_path}_0{ext}")
    
    print(f"Found {len(chunk_files)} API data file(s) to process")
    
    # Process each chunk file
    for i, chunk_file in enumerate(chunk_files):
        print(f"Processing chunk {i+1}/{len(chunk_files)}: {os.path.basename(chunk_file)}")
        lens_fill_df(chunk_file, final_csv)
        # Remove the intermediate API JSON file written by lens_id_extract
        try:
            os.remove(chunk_file)
            print(f"Removed API data file: {chunk_file}")
        except Exception as e:
            print(f"Warning: could not remove {chunk_file}: {e}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Final output with claims: {final_csv}")
    
    # Load and display summary
    df_final = pd.read_csv(final_csv)
    print(f"\nSUMMARY:")
    print(f"- Total patents processed: {len(df_final)}")
    print(f"- Patents with claims: {len(df_final[df_final['claims'].notna() & (df_final['claims'] != '[]')])}")
    print(f"- Available fields: {list(df_final.columns)}")
    
    return final_csv


def main():
    parser = argparse.ArgumentParser(
        description="Extract patent claims from Lens CSV export",
        epilog="Example: python claims_pipeline.py v1"
    )
    parser.add_argument("prefix", help="Version prefix (e.g., v1, v2, v3, v4)")
    
    args = parser.parse_args()
    
    try:
        create_claims_pipeline(args.prefix)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()