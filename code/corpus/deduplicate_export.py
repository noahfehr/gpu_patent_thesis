import pandas as pd
## This file is just for when you export in multiple chunks from lenge and merge together; removes duplicte entries that got split across exports

INPUT_FILE = "../../data/raw/v6_lens_export.csv"
OUTPUT_FILE = "../../data/raw/v6_lens_export.csv"


def norm(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def kind_rank(kind):
    k = norm(kind).upper()
    if k.startswith("C"):
        return 0
    if k.startswith("B"):
        return 1
    if k.startswith("A"):
        return 2
    return 3


# load
df = pd.read_csv(INPUT_FILE, dtype=str, keep_default_na=False)

# normalize
df["_jurisdiction"] = df["Jurisdiction"].map(norm).str.upper()
df["_family"] = df["Simple Family Members"].map(norm)
df["_appnum"] = df["Application Number"].map(norm)
df["_kind_rank"] = df["Kind"].map(kind_rank)
df["_pub_date"] = pd.to_datetime(df["Publication Date"], errors="coerce")

# grouping key
def make_key(row):
    if row["_family"]:
        return f'FAM|{row["_jurisdiction"]}|{row["_family"]}'
    return f'APP|{row["_jurisdiction"]}|{row["_appnum"]}'

df["_key"] = df.apply(make_key, axis=1)

# sort: best first
df = df.sort_values(
    by=["_key", "_kind_rank", "_pub_date"],
    ascending=[True, True, False],
    na_position="last"
)

# dedupe
deduped = df.drop_duplicates(subset=["_key"], keep="first").copy()

# drop helper cols
deduped = deduped.drop(columns=[c for c in deduped.columns if c.startswith("_")])

# overwrite
deduped.to_csv(OUTPUT_FILE, index=False)

print(f"Before: {len(df):,}")
print(f"After:  {len(deduped):,}")