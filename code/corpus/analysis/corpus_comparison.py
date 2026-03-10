
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

BASE_DIR = Path("data")
DATA_DIR = BASE_DIR / "claims_added"
ANALYSIS_DIR = BASE_DIR / "analysis"

FILES = {
    "v1": DATA_DIR / "v1_processed.csv",  # big
    "v2": DATA_DIR / "v2_processed.csv",  # reduced
    "v3": DATA_DIR / "v3_processed.csv",  # small
    "v4": DATA_DIR / "v4_processed.csv",  # new, unprocessed, for potential future use
}

ID_COL = "lens_id"
ABSTRACT_COL = "abstract"
CLAIMS_COL = "claims"

@dataclass
class PatentText:
    lens_id: str
    abstract: str
    claims: str

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

def build_text_lookup(v1_df: pd.DataFrame, v2_df: pd.DataFrame, v3_df: pd.DataFrame, v4_df: pd.DataFrame) -> Dict[str, PatentText]:
    """
    Build lens_id -> (abstract, claims) lookup.
    Priority: v4 -> v3 -> v2 -> v1.
    """
    lookup: Dict[str, PatentText] = {}

    for df in [v4_df, v3_df, v2_df, v1_df]:
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

def make_buckets(v1: Set[str], v2: Set[str], v3: Set[str], v4: Set[str]) -> Dict[str, Set[str]]:
    """
    a) patents in v1 not in v2, v3, or v4
    b) patents in v2 which are not in v1
    c) patents in v3 which are not in v1 or v2
    d) patents in v4 which are not in v1, v2, or v3
    """
    return {
        "a_v1_only": set(v1) - set(v2) - set(v3) - set(v4),
        "b_v2_minus_v1": set(v2) - set(v1),
        "c_v3_minus_v1_minus_v2": set(v3) - set(v1) - set(v2),
        "d_v4_minus_all": set(v4) - set(v1) - set(v2) - set(v3),
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

class GPUArchitectureLabel(BaseModel):
    is_gpu_architecture_design_patent: bool

def classify_gpu_architecture_design(client: OpenAI, patent: PatentText) -> bool:
    prompt = (
        "You are classifying patents for relevance.\n\n"
        "Task: Decide whether the patent is relevant to accelerator hardware architecture design.\n\n"
        "Target category:\n"
        "A patent is relevant if its claimed inventive contribution is the architecture or microarchitecture "
        "of a massively parallel compute accelerator, including GPUs, TPUs, NPUs, AI accelerators, vector "
        "processors, systolic-array accelerators, or other parallel processing devices.\n\n"
        "Output rule:\n"
        "Return only TRUE or FALSE.\n\n"
        "Decision standard:\n"
        "Return TRUE only if the patent's core claimed contribution is a hardware architectural mechanism "
        "inside the accelerator itself. Return FALSE otherwise.\n\n"
        "Count as TRUE when the patent claims one or more of the following inside an accelerator:\n"
        "- organization or structure of compute units, streaming multiprocessors, cores, tensor units, "
        "matrix units, vector units, processing elements, or execution pipelines\n"
        "- scheduling, dispatch, issue, control flow, warp/thread/block management, or synchronization "
        "implemented as hardware architectural support\n"
        "- memory hierarchy or data movement mechanisms, including caches, shared memory, scratchpads, "
        "register files, local memories, memory controllers, prefetching, tiling support, or bandwidth "
        "management inside the device\n"
        "- on-chip interconnect, network-on-chip, communication fabric, coherence, arbitration, or routing "
        "between accelerator components\n"
        "- architectural support for parallelism, sparsity, quantization, matrix operations, reduction, "
        "attention, convolution, or other workloads, but only when the novelty is in hardware mechanisms "
        "of the accelerator\n"
        "- partitioning, virtualization, multi-tenancy, isolation, or resource allocation, but only when "
        "implemented as accelerator hardware architecture rather than purely software management\n"
        "- interaction among accelerator submodules in a way that changes how the hardware executes, stores, "
        "moves, or coordinates computation\n\n"
        "Count as FALSE when the patent is primarily about:\n"
        "- machine learning models, neural network architectures, training methods, inference methods, or "
        "algorithmic improvements as such\n"
        "- software, compilers, drivers, kernels, runtimes, APIs, middleware, orchestration, or scheduling "
        "outside the hardware architecture itself\n"
        "- application-layer methods in graphics, vision, databases, robotics, networking, or other domains "
        "that merely run on an accelerator\n"
        "- general computing systems where an accelerator is only a target, environment, or example use case\n"
        "- semiconductor fabrication, process technology, lithography, packaging, chip stacking, materials, "
        "yield, or manufacturing methods\n"
        "- transistor-level, gate-level, analog, power-delivery, signal-integrity, clocking, or circuit-design "
        "details unless they clearly change accelerator architecture at the block or subsystem level\n"
        "- thermal management, cooling, board design, sockets, servers, racks, or mechanical integration\n"
        "- patents that mention GPUs, TPUs, NPUs, or accelerators only as implementation platforms without "
        "claiming a hardware architectural change to the accelerator itself\n\n"
        "Key distinction:\n"
        "The question is not whether the invention uses an accelerator. The question is whether the invention "
        "changes the internal hardware architecture of the accelerator.\n\n"
        "Priority rule:\n"
        "Focus on the claimed inventive contribution, not on background discussion, example embodiments, or "
        "application context. If the claims are primarily architectural, return TRUE. If the claims are "
        "primarily algorithmic, software-defined, manufacturing-related, or system-level without a concrete "
        "accelerator-internal hardware mechanism, return FALSE.\n\n"
        "Borderline cases:\n"
        "- If hardware and software are both discussed, return TRUE only if the novelty clearly resides in "
        "accelerator-internal hardware architecture.\n"
        "- If the patent concerns architectural support for a workload such as neural networks, graphics, or "
        "sparse computation, return TRUE only if the support is implemented through a concrete hardware "
        "mechanism inside the accelerator.\n"
        "- If unsure, prefer FALSE.\n"
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

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()

    print("Loading datasets from claims_added/ ...")
    v1_df = load_df(FILES["v1"])
    v2_df = load_df(FILES["v2"])
    v3_df = load_df(FILES["v3"])
    v4_df = load_df(FILES["v4"])

    v1 = ids_from_df(v1_df)
    v2 = ids_from_df(v2_df)
    v3 = ids_from_df(v3_df)
    v4 = ids_from_df(v4_df)

    print("\n==============================")
    print("Dataset Sizes")
    print("==============================")
    print(f"v1: {len(v1)}")
    print(f"v2: {len(v2)}")
    print(f"v3: {len(v3)}")
    print(f"v4: {len(v4)}")

    print("\n==============================")
    print("4x4 Overlap Table (Counts: |Vi ∩ Vj|)")
    print("==============================")
    counts, pct_row = overlap_tables({"v1": v1, "v2": v2, "v3": v3, "v4": v4})
    print(counts.to_string())

    print("\n==============================")
    print("4x4 Overlap Table (Row coverage: |Vi ∩ Vj| / |Vi|)")
    print("==============================")
    pct_disp = (pct_row * 100).round(2)
    print(pct_disp.to_string() + "  (%)")

    lookup = build_text_lookup(v1_df, v2_df, v3_df, v4_df)
    print(f"\nText lookup size (unique lens_id with text): {len(lookup)}")

    buckets = make_buckets(v1, v2, v3, v4)

    print("\n==============================")
    print("Bucket Sizes")
    print("==============================")
    for k, s in buckets.items():
        print(f"{k}: {len(s)}")

    N = 50
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
    out_df = out_df.loc[["a_v1_only", "b_v2_minus_v1", "c_v3_minus_v1_minus_v2", "d_v4_minus_all"]]
    print(out_df.to_string())

def analyze_corpus():
    """Analyze v5 corpus by sampling 50 patents and classifying them for GPU architecture design."""
    
    # Setup
    client = OpenAI()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load v5 data
    v5_path = DATA_DIR / "v5_processed.csv"
    df = load_df(v5_path)
    
    # Sample 50 patents
    if len(df) < 50:
        raise ValueError(f"v5 dataset has only {len(df)} patents, need at least 50")
    
    sampled_df = df.sample(n=50, random_state=42)
    
    results = []
    for _, row in sampled_df.iterrows():
        patent = PatentText(
            lens_id=row[ID_COL],
            abstract=row[ABSTRACT_COL],
            claims=row[CLAIMS_COL]
        )
        
        is_gpu_arch = classify_gpu_architecture_design(client, patent)
        
        results.append({
            "lens_id": patent.lens_id,
            "is_gpu_architecture_design_patent": is_gpu_arch
        })
    
    # Write to JSON
    output_path = ANALYSIS_DIR / "v5_sample_and_classification.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis complete. Sampled 50 patents from v5, classified {sum(r['is_gpu_architecture_design_patent'] for r in results)} as GPU architecture design patents.")
    print(f"Results saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    analyze_corpus()