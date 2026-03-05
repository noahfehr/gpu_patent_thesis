import pandas as pd


# CPC prefixes we want to keep
PREFIXES = ("G06N", "G06T", "G06V", "G06F")


def has_target_cpc(cpc_string):
    """Check if CPC string contains any of the target prefixes."""
    if pd.isna(cpc_string):
        return False
    
    codes = [c.strip() for c in str(cpc_string).split(";;")]
    
    for code in codes:
        if code.startswith(PREFIXES):
            return True
    return False


def filter_cpc_csv(csv_path):
    """Filter a CSV by CPC classifications and overwrite the original file.
    
    Args:
        csv_path (str): Path to the CSV file to filter
    """
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter rows
    filtered_df = df[df["CPC Classifications"].apply(has_target_cpc)]
    
    # Write output, overwriting original
    filtered_df.to_csv(csv_path, index=False)
    
    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")
    print(f"Saved to: {csv_path}")