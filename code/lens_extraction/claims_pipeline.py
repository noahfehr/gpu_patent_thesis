#!/usr/bin/env python3
"""
Patent Claims Extraction Pipeline

This script processes a CSV export from Lens to extract detailed patent information
including claims using the Lens API. It creates a complete pipeline that:
1. Extracts Lens IDs from your CSV export
2. Queries the Lens API for detailed patent data
3. Processes the results into a structured DataFrame with claims

Usage:
    python claims_pipeline.py --csv_input path/to/your/lens_export.csv \
                             --token your_lens_api_token \
                             --output_dir path/to/output/directory
"""

import pandas as pd
import os
import argparse
from dotenv import load_dotenv
from lens_id_extract import lens_id_extract
from lens_fill_df import lens_fill_df


def extract_lens_ids_from_csv(csv_path: str, output_path: str) -> int:
    """
    Extract Lens IDs from the CSV export and save them to a text file.
    
    Parameters:
        csv_path (str): Path to the CSV file with patent data
        output_path (str): Path where to save the extracted Lens IDs
        
    Returns:
        int: Number of unique Lens IDs extracted
    """
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


def create_claims_pipeline(csv_input: str, lens_token: str, output_dir: str):
    """
    Complete pipeline to extract claims from Lens CSV export.
    
    Parameters:
        csv_input (str): Path to the input CSV file
        lens_token (str): Lens API token
        output_dir (str): Directory where to save all outputs
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define intermediate file paths
    lens_ids_file = os.path.join(output_dir, "lens_ids.txt")
    json_output = os.path.join(output_dir, "lens_api_data.json")
    final_csv = os.path.join(output_dir, "patents_with_claims.csv")
    
    print("="*60)
    print("PATENT CLAIMS EXTRACTION PIPELINE")
    print("="*60)
    
    # Step 1: Extract Lens IDs from CSV
    print("\nSTEP 1: Extracting Lens IDs from CSV...")
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


def load_lens_token_cli(provided_token=None):
    """Load Lens API token from various sources"""
    if provided_token:
        return provided_token
    
    # Try to load from .env file
    load_dotenv()
    
    # Get from environment variable
    token = os.getenv('LENS_API_TOKEN')
    
    if not token:
        print("❌ Lens API token not found!")
        print("Please provide token in one of these ways:")
        print("  1. Command line: --token your_token")
        print("  2. Environment variable: export LENS_API_TOKEN='your_token'")
        print("  3. Create a .env file with: LENS_API_TOKEN=your_token")
        raise ValueError("No Lens API token provided")
    
    return token

def main():
    parser = argparse.ArgumentParser(description="Extract patent claims from Lens CSV export")
    parser.add_argument("--csv_input", required=True, help="Path to input CSV file")
    parser.add_argument("--token", help="Lens API token (or set LENS_API_TOKEN env var)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_input):
        print(f"Error: Input file not found: {args.csv_input}")
        return
    
    try:
        # Load token from environment if not provided
        lens_token = load_lens_token_cli(args.token)
        create_claims_pipeline(args.csv_input, lens_token, args.output_dir)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        raise


if __name__ == "__main__":
    main()