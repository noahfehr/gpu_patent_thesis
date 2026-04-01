import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional

BASE_DIR = Path("data")
DATA_DIR = BASE_DIR / "claims_added"
ANALYSIS_DIR = BASE_DIR / "analysis"

ID_COL = "lens_id"
ABSTRACT_COL = "abstract"
CLAIMS_COL = "claims"
LABEL_COL = "is_gpu_architecture_design_patent"
IS_GPU_COL = "is_gpu"
IS_HARDWARE_COL = "is_hardware"

@dataclass
class PatentText:
    lens_id: str
    abstract: str
    claims: str
    is_gpu: Optional[bool] = None
    is_hardware: Optional[bool] = None


class GPURelatedLabel(BaseModel):
    is_gpu: bool


class HardwareLabel(BaseModel):
    is_hardware: bool

class TwoStageLabel(BaseModel):
    is_gpu: bool
    is_hardware: bool

ACCELERATOR_RELEVANT_PROMPT = (
    "You are classifying patents.\n\n"

    "Task:\n"
    "Decide whether a patent is relevant to the platform architecture or execution model of a massively parallel compute accelerator "
    "(e.g., GPU, TPU, NPU, AI/DNN accelerator).\n\n"

    "Output:\n"
    "Return TRUE if the claimed invention is specifically tied to how a massively parallel accelerator executes work, moves data, "
    "is programmed, or is architecturally supported.\n"
    "Return FALSE otherwise.\n\n"

    "Core rule:\n"
    "Return TRUE only if the claimed novelty is specific to the accelerator platform or its execution model. "
    "This includes mechanisms that directly support accelerator execution, programming, memory access, scheduling, synchronization, "
    "or data movement.\n\n"

    "Return TRUE for:\n"
    "- accelerator execution models and semantics (e.g., SIMT, warps, thread groups, work-items, shaders, ray tracing execution)\n"
    "- accelerator-specific compute organization (e.g., tensor units, PE/MAC arrays, streaming multiprocessors)\n"
    "- hardware or software mechanisms that directly support accelerator execution or programming\n"
    "- memory hierarchy, dataflow, buffering, tiling, address translation, or data movement mechanisms adapted to accelerator workloads\n"
    "- interconnect, host-device communication, chiplet, or memory-coupling mechanisms when the novelty is specifically in supporting accelerator execution or dataflow\n"
    "- compiler, runtime, driver, API, or programming model mechanisms when they are specifically designed for accelerator execution semantics\n"
    "- mechanisms for coordinating work between accelerator cores, memory, host processors, or peer accelerators when tied to the accelerator execution model\n\n"

    "Return FALSE for:\n"
    "- generic CPUs, generic SoCs, or generic system infrastructure with no accelerator-specific adaptation\n"
    "- generic memory, interconnect, DMA, storage, or networking inventions that could apply equally to any compute system\n"
    "- generic ML algorithms, model architectures, or training/inference methods without accelerator-specific implementation details\n"
    "- FPGA or programmable logic inventions unless the claimed novelty is specifically directed to accelerator execution or support for accelerator dataflow\n"
    "- fabrication, packaging, materials, process technology, or circuit implementation details unless the claimed novelty is specifically about supporting accelerator architecture or execution\n\n"

    "Important:\n"
    "Accelerator mention alone is not sufficient.\n"
    "Focus on claimed novelty, not just examples or possible use cases.\n"
    "Return TRUE only when the invention is specifically about the accelerator platform or execution model, including hardware or software that directly serves it."
)

HARDWARE_PROMPT = (
    "You are classifying patents.\n\n"

    "Task:\n"
    "Decide whether an accelerator-relevant patent describes hardware architecture that supports the execution model of a massively parallel compute accelerator "
    "(e.g., GPU, TPU, NPU, AI/DNN accelerator).\n\n"

    "Output:\n"
    "Return TRUE if the claimed novelty is in hardware architecture that directly supports accelerator execution, dataflow, memory access, or coordination.\n"
    "Return FALSE otherwise.\n\n"

    "Core rule:\n"
    "Return TRUE only if the invention introduces or modifies hardware mechanisms that serve the accelerator execution model. "
    "This includes hardware inside the accelerator as well as closely coupled hardware mechanisms outside the compute core when they are specifically designed to support accelerator execution.\n\n"

    "Return TRUE for:\n"
    "- compute units, SIMD/SIMT lanes, tensor units, MAC/PE arrays, shader or ray tracing hardware\n"
    "- execution control hardware such as scheduling, synchronization, arbitration, dependency handling, or work distribution\n"
    "- accelerator-specific memory hierarchy such as shared memory, scratchpad, local memory, cache behavior, tiling support, buffering, or register-file related mechanisms\n"
    "- MMU, TLB, address translation, page handling, prefetch, copy engines, or other memory-access hardware adapted to accelerator execution\n"
    "- on-device interconnect and data movement hardware between accelerator compute elements\n"
    "- host-device, device-memory, chiplet-memory, or peer-accelerator hardware mechanisms when the claimed novelty is specifically in serving accelerator execution or accelerator dataflow\n"
    "- dedicated boards, access engines, memory fabrics, or interconnect hardware specifically designed to feed or coordinate accelerator workloads\n\n"

    "Return FALSE for:\n"
    "- software only, including compiler, runtime, driver, OS, API, or scheduling logic not claimed as hardware\n"
    "- generic ML algorithms, training methods, inference methods, or model design\n"
    "- generic processors, generic DMA, generic memory controllers, generic interconnects, or generic system infrastructure not specifically adapted to accelerator execution\n"
    "- FPGA fabrics or programmable logic in the abstract, unless the claimed novelty is specifically a hardware architecture for serving accelerator execution\n"
    "- fabrication, packaging, process technology, or circuit-level implementation details unless the architectural novelty is specifically about support for accelerator execution\n\n"

    "Important:\n"
    "Accelerator mention alone is not sufficient.\n"
    "Return FALSE if the accelerator is only context.\n"
    "Return FALSE if the contribution is software or algorithmic rather than hardware architectural.\n\n"

    "Decision rule:\n"
    "Focus on claimed novelty.\n"
    "If the novelty is hardware architecture specifically designed to support the execution model of a massively parallel accelerator, return TRUE; otherwise return FALSE."
)

SINGLE_PROMPT = (
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

def build_patent_content(patent: PatentText) -> str:
    return (
        f"lens_id: {patent.lens_id}\n\n"
        f"ABSTRACT:\n{patent.abstract}\n\n"
        f"CLAIMS:\n{patent.claims}\n"
    )

def classify_is_gpu(client: OpenAI, patent: PatentText) -> bool:
    content = build_patent_content(patent)

    resp = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Return only the structured boolean."},
            {"role": "user", "content": ACCELERATOR_RELEVANT_PROMPT + "\n\n" + content},
        ],
        text_format=GPURelatedLabel,
    )

    return resp.output_parsed.is_gpu

def classify_is_hardware(client: OpenAI, patent: PatentText) -> bool:
    content = build_patent_content(patent)

    resp = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": "Return only the structured boolean."},
            {"role": "user", "content": HARDWARE_PROMPT + "\n\n" + content},
        ],
        text_format=HardwareLabel,
    )

    return resp.output_parsed.is_hardware

def classify_two_stage(client: OpenAI, patent: PatentText) -> dict[str, bool]:
    is_gpu = classify_is_gpu(client, patent)

    if not is_gpu:
        return {
            "is_gpu": False,
            "is_hardware": False,
        }

    is_hardware = classify_is_hardware(client, patent)

    return {
        "is_gpu": True,
        "is_hardware": is_hardware,
    }

# def classify_gpu_architecture_design(client: OpenAI, patent: PatentText) -> bool:
#     content = (
#         f"lens_id: {patent.lens_id}\n\n"
#         f"ABSTRACT:\n{patent.abstract}\n\n"
#         f"CLAIMS:\n{patent.claims}\n"
#     )

#     resp = client.responses.parse(
#         model="gpt-5-mini",
#         input=[
#             {"role": "system", "content": "Return only the structured boolean."},
#             {"role": "user", "content": PROMPT + "\n\n" + content},
#         ],
#         text_format=GPUArchitectureLabel,
#     )

#     return resp.output_parsed.is_gpu_architecture_design_patent


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

    df = load_corpus(version)
    sampled = sample_patents(df, count)

    sample_payload = [asdict(p) for p in sampled]
    sample_path = ANALYSIS_DIR / f"{version}_sample.json"
    write_json(sample_path, sample_payload)

    evaluations = []
    for i, patent in enumerate(sampled, start=1):
        try:
            result = classify_two_stage(client, patent)

            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                IS_GPU_COL: result["is_gpu"],
                IS_HARDWARE_COL: result["is_hardware"],
                LABEL_COL: result["is_gpu"] and result["is_hardware"],
                "error": None,
            })
        except Exception as e:
            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                IS_GPU_COL: None,
                IS_HARDWARE_COL: None,
                LABEL_COL: None,
                "error": f"{type(e).__name__}: {e}",
            })

        if i % 10 == 0:
            print(f"Processed {i}/{len(sampled)}")

    eval_path = ANALYSIS_DIR / f"{version}_sample_evaluation.json"
    write_json(eval_path, evaluations)

    n_gpu_scored = sum(1 for row in evaluations if row[IS_GPU_COL] is not None)
    n_gpu_true = sum(1 for row in evaluations if row[IS_GPU_COL] is True)
    n_hardware_true = sum(1 for row in evaluations if row[IS_HARDWARE_COL] is True)

    print(f"Sample written to: {sample_path}")
    print(f"Evaluation written to: {eval_path}")
    print(f"Scored {n_gpu_scored}/{len(evaluations)}")
    print(f"GPU TRUE count: {n_gpu_true}")
    print(f"Hardware TRUE count: {n_hardware_true}")

    return sample_path, eval_path

def analyze_sample_new_prompt(
    input_filename: str = "v6_even_sample_noah.json",
    output_filename: str = "v6_gpt_sample_update.json",
) -> tuple[Path, Path]:
    """
    Read an existing sampled JSON file from ANALYSIS_DIR, re-classify each row
    using the current two-stage prompts, and write the updated results to a new JSON file.

    Input file is expected to contain rows like:
    {
        "lens_id": "...",
        "abstract": "...",
        "claims": "...",
        ...
    }

    The existing classification in the input file is ignored. We only read:
    - lens_id
    - abstract
    - claims

    Output file contains:
    - lens_id
    - abstract
    - claims
    - is_gpu
    - is_hardware
    - error
    - eval_id
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    input_path = ANALYSIS_DIR / input_filename
    output_path = ANALYSIS_DIR / output_filename

    rows = read_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} must contain a JSON list of patent rows.")

    client = OpenAI()
    evaluations = []

    for i, row in enumerate(rows, start=1):
        lens_id = str(row.get(ID_COL, "")).strip()
        abstract = str(row.get(ABSTRACT_COL, "") or "")
        claims = str(row.get(CLAIMS_COL, "") or "")

        patent = PatentText(
            lens_id=lens_id,
            abstract=abstract,
            claims=claims,
        )

        try:
            result = classify_two_stage(client, patent)

            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                IS_GPU_COL: result["is_gpu"],
                IS_HARDWARE_COL: result["is_hardware"],
                LABEL_COL: result["is_gpu"] and result["is_hardware"],
                "error": None,
                "eval_id": i,
            })
        except Exception as e:
            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                IS_GPU_COL: None,
                IS_HARDWARE_COL: None,
                LABEL_COL: None,
                "error": f"{type(e).__name__}: {e}",
                "eval_id": i,
            })

        if i % 10 == 0:
            print(f"Processed {i}/{len(rows)}")

    write_json(output_path, evaluations)

    n_gpu_scored = sum(1 for row in evaluations if row[IS_GPU_COL] is not None)
    n_gpu_true = sum(1 for row in evaluations if row[IS_GPU_COL] is True)
    n_hardware_true = sum(1 for row in evaluations if row[IS_HARDWARE_COL] is True)
    n_errors = sum(1 for row in evaluations if row["error"] is not None)

    print(f"Input read from: {input_path}")
    print(f"Updated evaluations written to: {output_path}")
    print(f"Scored {n_gpu_scored}/{len(evaluations)}")
    print(f"GPU TRUE count: {n_gpu_true}")
    print(f"Hardware TRUE count: {n_hardware_true}")
    print(f"Errors: {n_errors}")

    return input_path, output_path

if __name__ == "__main__":
    # evaluate_corpus("v6", 1000)
    # compare_classifications("v5")
    # analyze_sample_new_prompt()
    analyze_sample_new_prompt(input_filename="v6_anurag.json", output_filename="v6_anurag_gpt_update.json")