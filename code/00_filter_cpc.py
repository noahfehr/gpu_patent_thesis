import pandas as pd

# Input / output files
input_file = "./data/patents/v1_core_expansion/core/raw/v3_lens_export.csv"
output_file = "./data/patents/v1_core_expansion/core/raw/v3_lens_export.csv"

# CPC prefixes we want to keep
prefixes = ("G06N", "G06T", "G06V", "G06F")

# Load CSV
df = pd.read_csv(input_file)

def has_target_cpc(cpc_string):
    if pd.isna(cpc_string):
        return False
    
    codes = [c.strip() for c in str(cpc_string).split(";;")]
    
    for code in codes:
        if code.startswith(prefixes):
            return True
    return False

# Filter rows
filtered_df = df[df["CPC Classifications"].apply(has_target_cpc)]

# Write output
filtered_df.to_csv(output_file, index=False)

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered_df)}")
print(f"Saved to: {output_file}")