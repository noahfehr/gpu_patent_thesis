import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

BASE_DIR = Path("data")
DATA_DIR = BASE_DIR / "claims_added"
ANALYSIS_DIR = BASE_DIR / "analysis"

ID_COL = "lens_id"
ABSTRACT_COL = "abstract"
CLAIMS_COL = "claims"
LABEL_COL = "is_gpu_architecture_design_patent"


@dataclass
class PatentText:
    lens_id: str
    abstract: str
    claims: str
    is_gpu_architecture_design_patent: bool = None


class GPUArchitectureLabel(BaseModel):
    is_gpu_architecture_design_patent: bool


PROMPT = (
    "You are classifying patents.\n\n"

    "Task:\n"
    "Decide whether a patent describes accelerator hardware architecture design.\n\n"

    "Output:\n"
    "Return TRUE if the invention concerns the design or modification of hardware architecture for a massively parallel compute accelerator (e.g., GPU, TPU, NPU, AI accelerator).\n"
    "Return FALSE otherwise.\n\n"

    "Core definition:\n"
    "Return TRUE when the invention introduces or modifies architectural mechanisms of a compute accelerator or of accelerator-specific hardware interaction with memory, interconnects, or peer controllers.\n\n"

    "Accelerator specificity requirement:\n"
    "Return TRUE if the invention concerns hardware architecture of a massively parallel compute device (e.g., GPU, TPU, NPU), including mechanisms adapted for high-throughput or parallel execution.\n"
    "Return TRUE if it is implemented as part of the accelerator’s hardware architecture or is adapted to its execution model (e.g., parallelism, memory access patterns, or coordination of many threads).\n"

   "This includes:\n"
    "- compute units, execution units, SIMD/SIMT lanes, tensor cores\n"
    "- on-chip or device-level interconnects specific to accelerator operation\n"
    "- accelerator-specific memory hierarchy mechanisms (e.g., shared memory, scratchpads, warp-level memory behavior)\n"
    "- hardware mechanisms for accelerator access to attached, shared, or controller-managed memory\n"
    "- MMU, arbitration, interface, or address-mapping mechanisms claimed as part of accelerator-specific hardware interaction\n"
    "- pipelines and parallel execution structures specific to accelerators\n"
    "- hardware scheduling or synchronization mechanisms inside the accelerator or tightly coupled to its operation\n\n"

    "Return FALSE when the invention primarily concerns:\n"
"- generic processor mechanisms described without adaptation to accelerator hardware or parallel execution context\n"
"- system-level memory management, paging, or allocation policies implemented primarily in software or OS/runtime logic\n"
"- runtime, compiler, or OS-level techniques (even if applied to GPUs)\n"
"- moving data between memories based on usage metrics, profiling, or application behavior where the novelty lies only in the policy or software logic and not in a new hardware mechanism\n"
"- application-level or software-driven optimization strategies\n"
"- machine learning algorithms or models\n"
"- semiconductor fabrication or circuit-level implementation\n\n"

    "Important rule:\n"
    "Mentioning a GPU or accelerator is not sufficient.\n"
    "If the invention manages how software uses memory on an accelerator, return FALSE.\n"
    "If the invention changes how the accelerator hardware itself operates internally, return TRUE.\n\n"

    "Decision rule:\n"
    "Focus on the claimed novelty.\n"
    "If the novelty is in hardware architecture inside the accelerator → TRUE.\n"
    "If the novelty is in system behavior, memory management policy, or software → FALSE."
)


def load_corpus(version: str) -> pd.DataFrame:
    path = DATA_DIR / f"{version}_processed.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    required = [ID_COL, ABSTRACT_COL, CLAIMS_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    df[ID_COL] = df[ID_COL].fillna("").astype(str).str.strip()
    df[ABSTRACT_COL] = df[ABSTRACT_COL].fillna("").astype(str)
    df[CLAIMS_COL] = df[CLAIMS_COL].fillna("").astype(str)

    df = df[df[ID_COL] != ""].copy()
    return df


def sample_patents(df: pd.DataFrame, count: int, random_state: int = 42) -> list[PatentText]:
    if len(df) < count:
        raise ValueError(f"Corpus has only {len(df)} rows, but count={count}")

    sampled_df = df.sample(n=count, random_state=random_state)

    return [
        PatentText(
            lens_id=row[ID_COL],
            abstract=row[ABSTRACT_COL],
            claims=row[CLAIMS_COL],
        )
        for _, row in sampled_df.iterrows()
    ]


def classify_gpu_architecture_design(client: OpenAI, patent: PatentText) -> bool:
    content = (
        f"lens_id: {patent.lens_id}\n\n"
        f"ABSTRACT:\n{patent.abstract}\n\n"
        f"CLAIMS:\n{patent.claims}\n"
    )

    resp = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Return only the structured boolean."},
            {"role": "user", "content": PROMPT + "\n\n" + content},
        ],
        text_format=GPUArchitectureLabel,
    )

    return resp.output_parsed.is_gpu_architecture_design_patent


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text())


def compare_classifications(version: str) -> list[dict]:
    sample_path = ANALYSIS_DIR / f"{version}_sample.json"

    # Support both naming conventions
    evaluated_candidates = [
        ANALYSIS_DIR / f"{version}_sample_evaluated.json",
        ANALYSIS_DIR / f"{version}_sample_evaluation.json",
    ]
    evaluated_path = next((p for p in evaluated_candidates if p.exists()), None)

    if evaluated_path is None:
        raise FileNotFoundError(
            f"Could not find evaluated file. Looked for: "
            f"{', '.join(str(p) for p in evaluated_candidates)}"
        )

    sample_rows = read_json(sample_path)
    evaluated_rows = read_json(evaluated_path)

    sample_by_id = {str(row[ID_COL]): row for row in sample_rows if ID_COL in row}
    evaluated_by_id = {str(row[ID_COL]): row for row in evaluated_rows if ID_COL in row}

    common_ids = sorted(set(sample_by_id) & set(evaluated_by_id))
    sample_only_ids = sorted(set(sample_by_id) - set(evaluated_by_id))
    evaluated_only_ids = sorted(set(evaluated_by_id) - set(sample_by_id))

    comparisons = []
    for lens_id in common_ids:
        sample_label = sample_by_id[lens_id].get(LABEL_COL)
        evaluated_label = evaluated_by_id[lens_id].get(LABEL_COL)

        comparisons.append(
            {
                "lens_id": lens_id,
                "sample_classification": sample_label,
                "evaluated_classification": evaluated_label,
                "match": sample_label == evaluated_label,
            }
        )

    comparable = [
        row for row in comparisons
        if row["sample_classification"] is not None
        and row["evaluated_classification"] is not None
    ]
    matches = sum(row["match"] for row in comparable)
    mismatches = [row for row in comparable if not row["match"]]

    print(f"Sample file: {sample_path}")
    print(f"Evaluated file: {evaluated_path}")
    print(f"Rows in sample: {len(sample_rows)}")
    print(f"Rows in evaluated: {len(evaluated_rows)}")
    print(f"Common lens_ids: {len(common_ids)}")
    print(f"Only in sample: {len(sample_only_ids)}")
    print(f"Only in evaluated: {len(evaluated_only_ids)}")
    print(f"Comparable classifications: {len(comparable)}")
    print(f"Matches: {matches}")
    print(f"Mismatches: {len(mismatches)}")

    if comparable:
        accuracy = matches / len(comparable)
        print(f"Agreement rate: {accuracy:.2%}")

    if sample_only_ids:
        print("\nLens IDs only in sample:")
        for lens_id in sample_only_ids[:10]:
            print(f"  - {lens_id}")
        if len(sample_only_ids) > 10:
            print(f"  ... and {len(sample_only_ids) - 10} more")

    if evaluated_only_ids:
        print("\nLens IDs only in evaluated:")
        for lens_id in evaluated_only_ids[:10]:
            print(f"  - {lens_id}")
        if len(evaluated_only_ids) > 10:
            print(f"  ... and {len(evaluated_only_ids) - 10} more")

    if mismatches:
        print("\nMismatches:")
        for row in mismatches[:20]:
            print(
                f"  - {row['lens_id']}: "
                f"sample={row['sample_classification']}, "
                f"evaluated={row['evaluated_classification']}"
            )
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")

    return comparisons


def evaluate_corpus(version: str, count: int) -> tuple[Path, Path]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()

    # df = load_corpus(version)
    # sampled = sample_patents(df, count)

    # sample_payload = [asdict(p) for p in sampled]
    sample_path = ANALYSIS_DIR / f"{version}_sample.json"
    # write_json(sample_path, sample_payload)
    sampled = [PatentText(**p) for p in read_json(sample_path)]    
    evaluations = []
    for i, patent in enumerate(sampled, start=1):
        try:
            label = classify_gpu_architecture_design(client, patent)
            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                LABEL_COL: bool(label),
                "error": None,
            })
        except Exception as e:
            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                LABEL_COL: None,
                "error": f"{type(e).__name__}: {e}",
            })

        if i % 10 == 0:
            print(f"Processed {i}/{len(sampled)}")

    eval_path = ANALYSIS_DIR / f"{version}_sample_evaluation.json"
    write_json(eval_path, evaluations)

    n_true = sum(
        1 for row in evaluations
        if row[LABEL_COL] is True
    )
    n_scored = sum(
        1 for row in evaluations
        if row[LABEL_COL] is not None
    )

    print(f"Sample written to: {sample_path}")
    print(f"Evaluation written to: {eval_path}")
    print(f"Scored {n_scored}/{len(evaluations)}")
    print(f"TRUE count: {n_true}")

    return sample_path, eval_path


if __name__ == "__main__":
    # evaluate_corpus("v5", 50)
    compare_classifications("v5")