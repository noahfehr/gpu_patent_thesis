#!/usr/bin/env python3
"""
Example script showing how to use the claims extraction pipeline
with your Lens CSV export data.

Before running, make sure you have:
1. Your Lens API token
2. Your CSV file with the headers you provided
3. The lens_extraction functions in the same directory
"""

from claims_pipeline import create_claims_pipeline
import os
from dotenv import load_dotenv

def load_lens_token():
    """Load Lens API token from environment variables or .env file"""
    # Try to load from .env file first
    load_dotenv()
    
    # Get token from environment variable
    token = os.getenv('LENS_API_TOKEN')
    
    if not token:
        print("❌ Lens API token not found!")
        print("Please set your token in one of these ways:")
        print("  1. Environment variable: export LENS_API_TOKEN='your_token'")
        print("  2. Create a .env file with: LENS_API_TOKEN=your_token")
        print("  3. Set it in your shell profile (.bashrc, .zshrc, etc.)")
        return None
    
    return token

def run_example():
    """
    Example of how to run the claims extraction pipeline.
    The API token is loaded securely from environment variables.
    """
    
    # Load API token from environment
    lens_api_token = load_lens_token()
    if not lens_api_token:
        return
    
    # UPDATE THESE PATHS WITH YOUR ACTUAL FILES
    csv_input_path = "../../data/patents/v1_core_expansion/core/raw/feb25_lens_export.csv"  # Your CSV with the headers you listed
    output_directory = "../../data/patents/v1_core_expansion/core/claims_added"    # Where to save results
    
    # Check if input file exists
    if not os.path.exists(csv_input_path):
        print(f"❌ CSV file not found: {csv_input_path}")
        print(f"Please update csv_input_path to point to your actual CSV file")
        return
    
    print(f"✅ Found CSV file: {csv_input_path}")
    print(f"✅ API token loaded from environment")
    
    # Run the pipeline
    try:
        final_output = create_claims_pipeline(
            csv_input=csv_input_path,
            lens_token=lens_api_token,
            output_dir=output_directory
        )
        
        print(f"\n✅ Success! Your patent data with claims is saved to:")
        print(f"   {final_output}")
        
        # Optional: Load and examine the results
        import pandas as pd
        df = pd.read_csv(final_output)
        
        print(f"\n📊 Results Summary:")
        print(f"   • Total patents: {len(df)}")
        print(f"   • Patents with claims: {len(df[df['claims'].str.len() > 2])}")  # More than just "[]"
        print(f"   • Available fields: {len(df.columns)}")
        
        # Show a sample of available fields
        print(f"\n📋 Available fields in your final dataset:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col}")
        
        # Show example of claims data
        claims_sample = df[df['claims'].str.len() > 2]['claims'].iloc[0] if any(df['claims'].str.len() > 2) else None
        if claims_sample:
            print(f"\n📝 Example claims data (first 200 characters):")
            print(f"   {str(claims_sample)[:200]}...")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")


if __name__ == "__main__":
    print("🚀 Patent Claims Extraction Example")
    print("=" * 50)
    run_example()