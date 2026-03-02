
import os
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

# ==============================
# Config
# ==============================

BASE_DIR = Path("data/patents/v1_core_expansion/core")
DATA_DIR = BASE_DIR / "claims_added"
ANALYSIS_DIR = BASE_DIR / "analysis"

FILES = {
    "v1": DATA_DIR / "v1_processed.csv",  # big
    "v2": DATA_DIR / "v2_processed.csv",  # reduced
    "v3": DATA_DIR / "v3_processed.csv",  # small
}

ID_COL = "lens_id"
ABSTRACT_COL = "abstract"
CLAIMS_COL = "claims"


# ==============================
# Data structures
# ==============================

@dataclass
class PatentText:
    lens_id: str
    abstract: str
    claims: str


# ==============================
# IO helpers
# ==============================

def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    missing = [c for c in [ID_COL, ABSTRACT_COL, CLAIMS_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df[ID_COL] = df[ID_COL].astype(str).str.strip()
    df[ABSTRACT_COL] = df[ABSTRACT_COL].fillna("").astype(str)
    df[CLAIMS_COL] = df[CLAIMS_COL].fillna("").astype(str)

    df = df[df[ID_COL].notna() & (df[ID_COL].str.len() > 0)].copy()
    return df


def ids_from_df(df: pd.DataFrame) -> Set[str]:
    return set(df[ID_COL].astype(str).str.strip())


def build_text_lookup(v1_df: pd.DataFrame, v2_df: pd.DataFrame, v3_df: pd.DataFrame) -> Dict[str, PatentText]:
    """
    Build lens_id -> (abstract, claims) lookup.
    Priority: v3 -> v2 -> v1.
    """
    lookup: Dict[str, PatentText] = {}

    for df in [v3_df, v2_df, v1_df]:
        for _, row in df.iterrows():
            lid = str(row[ID_COL]).strip()
            if not lid:
                continue
            if lid not in lookup:
                lookup[lid] = PatentText(
                    lens_id=lid,
                    abstract=str(row[ABSTRACT_COL]).strip(),
                    claims=str(row[CLAIMS_COL]).strip(),
                )
    return lookup


# ==============================
# Set math + reporting
# ==============================

def overlap_tables(sets: Dict[str, Set[str]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      counts[i,j] = |Vi ∩ Vj|
      pct_row[i,j] = |Vi ∩ Vj| / |Vi|
    """
    names = list(sets.keys())
    counts = pd.DataFrame(index=names, columns=names, dtype=int)
    pct_row = pd.DataFrame(index=names, columns=names, dtype=float)

    for i in names:
        for j in names:
            inter = sets[i] & sets[j]
            counts.loc[i, j] = len(inter)
            denom = len(sets[i]) if len(sets[i]) else 1
            pct_row.loc[i, j] = len(inter) / denom

    return counts, pct_row


def make_buckets(v1: Set[str], v2: Set[str], v3: Set[str]) -> Dict[str, Set[str]]:
    """
    a) patents in v3
    b) patents in v2 which are not in v3
    c) patents in v1 which are not in v2 or v3
    """
    return {
        "a_v3": set(v3),
        "b_v2_minus_v3": set(v2) - set(v3),
        "c_v1_minus_v2_minus_v3": set(v1) - set(v2) - set(v3),
    }


def sample_ids(ids: Set[str], n: int, seed: int) -> List[str]:
    ids_list = list(ids)
    if not ids_list:
        return []
    rng = random.Random(seed)
    if len(ids_list) <= n:
        rng.shuffle(ids_list)
        return ids_list
    return rng.sample(ids_list, n)


# ==============================
# OpenAI classification
# ==============================

class GPUArchitectureLabel(BaseModel):
    is_gpu_architecture_design_patent: bool


def classify_gpu_architecture_design(client: OpenAI, patent: PatentText) -> bool:
    prompt = (
        "You are labeling patents.\n\n"
        "Task: Decide if this patent is a GPU architecture design patent.\n\n"
        "Operational definition:\n"
        "Return TRUE if the invention introduces or modifies GPU hardware architecture or microarchitecture, "
        "including architectural support for specific workloads (e.g., ML/NN), as long as the novelty is in GPU "
        "hardware mechanisms (blocks, datapaths, on-chip structures, execution support, control, memory movement, "
        "or other architectural features inside the GPU).\n\n"
        "Return FALSE if the contribution is primarily software/algorithmic, application-layer, training/inference "
        "methods, or a generic accelerator/computing idea where the GPU is merely a platform/target rather than the "
        "object being architected.\n\n"
        "Decision rule:\n"
        "- Focus on what is actually *claimed* as the inventive contribution.\n"
        "- If the claims describe GPU-internal hardware changes to support computation or data movement, return TRUE.\n"
        "- If the claims mostly describe how to run workloads on GPUs without GPU hardware changes, return FALSE.\n\n"
        "Use both the abstract and the claims. If genuinely ambiguous, lean FALSE.\n"
    )

    content = (
        f"lens_id: {patent.lens_id}\n\n"
        f"ABSTRACT:\n{patent.abstract}\n\n"
        f"CLAIMS:\n{patent.claims}\n"
    )

    resp = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Return only the structured boolean."},
            {"role": "user", "content": prompt + "\n\n" + content},
        ],
        text_format=GPUArchitectureLabel,
    )

    return resp.output_parsed.is_gpu_architecture_design_patent
# ==============================
# JSON output
# ==============================

def write_bucket_json(bucket_name: str, sampled_ids: List[str], rows: List[Dict], meta: Dict) -> Path:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ANALYSIS_DIR / f"{bucket_name}_sample_and_classification.json"

    payload = {
        "bucket": bucket_name,
        "meta": meta,
        "sampled_ids": sampled_ids,
        "rows": rows,
    }

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    return out_path


# ==============================
# Main
# ==============================

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()

    print("Loading datasets from claims_added/ ...")
    v1_df = load_df(FILES["v1"])
    v2_df = load_df(FILES["v2"])
    v3_df = load_df(FILES["v3"])

    v1 = ids_from_df(v1_df)
    v2 = ids_from_df(v2_df)
    v3 = ids_from_df(v3_df)

    print("\n==============================")
    print("Dataset Sizes")
    print("==============================")
    print(f"v1: {len(v1)}")
    print(f"v2: {len(v2)}")
    print(f"v3: {len(v3)}")

    print("\n==============================")
    print("3x3 Overlap Table (Counts: |Vi ∩ Vj|)")
    print("==============================")
    counts, pct_row = overlap_tables({"v1": v1, "v2": v2, "v3": v3})
    print(counts.to_string())

    print("\n==============================")
    print("3x3 Overlap Table (Row coverage: |Vi ∩ Vj| / |Vi|)")
    print("==============================")
    pct_disp = (pct_row * 100).round(2)
    print(pct_disp.to_string() + "  (%)")

    lookup = build_text_lookup(v1_df, v2_df, v3_df)
    print(f"\nText lookup size (unique lens_id with text): {len(lookup)}")

    buckets = make_buckets(v1, v2, v3)

    print("\n==============================")
    print("Bucket Sizes")
    print("==============================")
    for k, s in buckets.items():
        print(f"{k}: {len(s)}")

    N = 100
    seed = 7
    run_ts = datetime.now(timezone.utc).isoformat()

    results: Dict[str, Dict] = {}

    for bucket_name, idset in buckets.items():
        sampled = sample_ids(idset, N, seed=seed)
        rows_out: List[Dict] = []

        scored = 0
        true_count = 0

        print(f"\n[{bucket_name}] Sampled {len(sampled)}")

        for lid in sampled:
            row = {
                "lens_id": lid,
                "abstract": None,
                "claims": None,
                "has_text": False,
                "is_gpu_architecture_design_patent": None,
                "error": None,
            }

            pt = lookup.get(lid)
            if pt is None:
                rows_out.append(row)
                continue

            row["abstract"] = pt.abstract
            row["claims"] = pt.claims
            row["has_text"] = True

            try:
                label = classify_gpu_architecture_design(client, pt)
                row["is_gpu_architecture_design_patent"] = bool(label)
                scored += 1
                true_count += int(label)
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"

            rows_out.append(row)

            if scored and scored % 10 == 0:
                print(f"  Progress: {scored} scored")

        pct_true = (true_count / scored) * 100 if scored else float("nan")

        results[bucket_name] = {
            "n_sampled": len(sampled),
            "n_with_text": sum(1 for r in rows_out if r["has_text"]),
            "n_scored": scored,
            "n_true": true_count,
            "pct_true": round(pct_true, 2) if pct_true == pct_true else pct_true,  # keep nan as nan
        }

        meta = {
            "run_timestamp_utc": run_ts,
            "source_files": {k: str(v) for k, v in FILES.items()},
            "columns": {"lens_id": ID_COL, "abstract": ABSTRACT_COL, "claims": CLAIMS_COL},
            "seed": seed,
            "target_sample_size": N,
            "bucket_counts": results[bucket_name],
        }

        out_path = write_bucket_json(bucket_name, sampled, rows_out, meta)
        print(f"  Wrote: {out_path}")

    print("\n==============================")
    print("GPU Architecture Design % by Bucket")
    print("==============================")
    out_df = pd.DataFrame(results).T
    out_df = out_df.loc[["a_v3", "b_v2_minus_v3", "c_v1_minus_v2_minus_v3"]]
    print(out_df.to_string())


if __name__ == "__main__":
    main()