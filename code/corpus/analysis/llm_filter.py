import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel

BASE_DIR = Path("data")
DATA_DIR = BASE_DIR / "claims_added"
ANALYSIS_DIR = BASE_DIR / "analysis"
RUNS_DIR = ANALYSIS_DIR / "runs"

ID_COL = "lens_id"
ABSTRACT_COL = "abstract"
CLAIMS_COL = "claims"

IS_ACCELERATOR_RELEVANT_COL = "is_accelerator_relevant"
IS_ACCELERATOR_HARDWARE_COL = "is_accelerator_hardware"
LABEL_COL = "is_accelerator_hardware_design_patent"


@dataclass
class PatentRecord:
    lens_id: str
    abstract: str
    claims: str


@dataclass
class RunConfig:
    dataset_version: str
    run_scope: str
    pipeline: str = "two_stage"
    prompt_version: str = "ts1"
    model: str = "gpt-5-mini"
    batch_size: int = 100
    max_retries: int = 3
    retry_sleep_seconds: float = 2.0


class AcceleratorRelevanceLabel(BaseModel):
    is_accelerator_relevant: bool


class AcceleratorHardwareLabel(BaseModel):
    is_accelerator_hardware: bool


PROMPT_ACCELERATOR_RELEVANCE = (
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

PROMPT_ACCELERATOR_HARDWARE = (
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


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_processed_ids(predictions_path: Path) -> set[str]:
    return {
        str(row.get(ID_COL, "")).strip()
        for row in read_jsonl(predictions_path)
        if str(row.get(ID_COL, "")).strip()
    }


def iter_dataframe_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start:start + batch_size]


def get_run_stem(config: RunConfig) -> str:
    return f"{config.dataset_version}__{config.run_scope}__{config.pipeline}__{config.prompt_version}"


def get_predictions_path(config: RunConfig) -> Path:
    return RUNS_DIR / f"{get_run_stem(config)}__predictions.jsonl"


def get_progress_path(config: RunConfig) -> Path:
    return RUNS_DIR / f"{get_run_stem(config)}__progress.json"


def get_manifest_path(config: RunConfig) -> Path:
    return RUNS_DIR / f"{get_run_stem(config)}__manifest.json"


def build_patent_content(patent: PatentRecord) -> str:
    return (
        f"lens_id: {patent.lens_id}\n\n"
        f"ABSTRACT:\n{patent.abstract}\n\n"
        f"CLAIMS:\n{patent.claims}\n"
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

    return df[df[ID_COL] != ""].copy()


def load_records_from_json(input_path: Path) -> list[PatentRecord]:
    rows = read_json(input_path)
    if not isinstance(rows, list):
        raise ValueError(f"{input_path} must contain a JSON list of patent rows.")

    records = []
    for row in rows:
        lens_id = str(row.get(ID_COL, "")).strip()
        if not lens_id:
            continue

        records.append(
            PatentRecord(
                lens_id=lens_id,
                abstract=str(row.get(ABSTRACT_COL, "") or ""),
                claims=str(row.get(CLAIMS_COL, "") or ""),
            )
        )
    return records


def classify_accelerator_relevance(client: OpenAI, patent: PatentRecord, model: str) -> bool:
    content = build_patent_content(patent)

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Return only the structured boolean."},
            {"role": "user", "content": PROMPT_ACCELERATOR_RELEVANCE + "\n\n" + content},
        ],
        text_format=AcceleratorRelevanceLabel,
    )

    return resp.output_parsed.is_accelerator_relevant


def classify_accelerator_hardware(client: OpenAI, patent: PatentRecord, model: str) -> bool:
    content = build_patent_content(patent)

    resp = client.responses.parse(
        model=model,
        input=[
            {"role": "system", "content": "Return only the structured boolean."},
            {"role": "user", "content": PROMPT_ACCELERATOR_HARDWARE + "\n\n" + content},
        ],
        text_format=AcceleratorHardwareLabel,
    )

    return resp.output_parsed.is_accelerator_hardware


def classify_two_stage_patent(client: OpenAI, patent: PatentRecord, model: str) -> dict[str, bool]:
    is_accelerator_relevant = classify_accelerator_relevance(client, patent, model=model)

    if not is_accelerator_relevant:
        return {
            IS_ACCELERATOR_RELEVANT_COL: False,
            IS_ACCELERATOR_HARDWARE_COL: False,
        }

    is_accelerator_hardware = classify_accelerator_hardware(client, patent, model=model)

    return {
        IS_ACCELERATOR_RELEVANT_COL: True,
        IS_ACCELERATOR_HARDWARE_COL: is_accelerator_hardware,
    }


def classify_with_retries(client: OpenAI, patent: PatentRecord, config: RunConfig) -> dict:
    for attempt in range(1, config.max_retries + 1):
        try:
            result = classify_two_stage_patent(client, patent, model=config.model)
            final_label = (
                result[IS_ACCELERATOR_RELEVANT_COL]
                and result[IS_ACCELERATOR_HARDWARE_COL]
            )

            return {
                ID_COL: patent.lens_id,
                IS_ACCELERATOR_RELEVANT_COL: result[IS_ACCELERATOR_RELEVANT_COL],
                IS_ACCELERATOR_HARDWARE_COL: result[IS_ACCELERATOR_HARDWARE_COL],
                LABEL_COL: final_label,
                "status": "success",
                "error": None,
                "error_type": None,
                "error_message": None,
                "attempts_used": attempt,
                "pipeline": config.pipeline,
                "prompt_version": config.prompt_version,
                "model": config.model,
            }
        except Exception as e:
            if attempt == config.max_retries:
                return {
                    ID_COL: patent.lens_id,
                    IS_ACCELERATOR_RELEVANT_COL: None,
                    IS_ACCELERATOR_HARDWARE_COL: None,
                    LABEL_COL: None,
                    "status": "error",
                    "error": f"{type(e).__name__}: {e}",
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "attempts_used": attempt,
                    "pipeline": config.pipeline,
                    "prompt_version": config.prompt_version,
                    "model": config.model,
                }

            time.sleep(config.retry_sleep_seconds * attempt)


def update_progress(
    progress_path: Path,
    *,
    run_id: str,
    total_rows: int,
    processed_rows: int,
    success_rows: int,
    error_rows: int,
    completed: bool,
) -> None:
    payload = {
        "run_id": run_id,
        "total_rows": total_rows,
        "processed_rows": processed_rows,
        "success_rows": success_rows,
        "error_rows": error_rows,
        "completed": completed,
        "updated_at_unix": time.time(),
    }
    write_json(progress_path, payload)


def run_patent_analysis_on_dataframe(df: pd.DataFrame, config: RunConfig) -> tuple[Path, Path, Path]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")

    client = OpenAI()

    predictions_path = get_predictions_path(config)
    progress_path = get_progress_path(config)
    manifest_path = get_manifest_path(config)

    all_records = [
        PatentRecord(
            lens_id=row[ID_COL],
            abstract=row[ABSTRACT_COL],
            claims=row[CLAIMS_COL],
        )
        for _, row in df.iterrows()
    ]

    processed_ids = load_processed_ids(predictions_path)
    remaining_records = [r for r in all_records if r.lens_id not in processed_ids]

    write_json(
        manifest_path,
        {
            "run_id": get_run_stem(config),
            "dataset_version": config.dataset_version,
            "run_scope": config.run_scope,
            "pipeline": config.pipeline,
            "prompt_version": config.prompt_version,
            "model": config.model,
            "total_rows": len(all_records),
            "already_processed_at_start": len(processed_ids),
            "remaining_at_start": len(remaining_records),
        },
    )

    existing_rows = read_jsonl(predictions_path)
    processed_rows = len(existing_rows)
    success_rows = sum(1 for row in existing_rows if row.get("status") == "success")
    error_rows = sum(1 for row in existing_rows if row.get("status") == "error")

    update_progress(
        progress_path,
        run_id=get_run_stem(config),
        total_rows=len(all_records),
        processed_rows=processed_rows,
        success_rows=success_rows,
        error_rows=error_rows,
        completed=False,
    )

    remaining_df = pd.DataFrame([{ID_COL: r.lens_id} for r in remaining_records])

    for batch_index, batch_df in enumerate(iter_dataframe_batches(remaining_df, config.batch_size), start=1):
        batch_ids = set(batch_df[ID_COL].tolist())
        batch_records = [r for r in remaining_records if r.lens_id in batch_ids]

        for patent in batch_records:
            row = classify_with_retries(client, patent, config)
            append_jsonl(predictions_path, row)

            processed_rows += 1
            if row["status"] == "success":
                success_rows += 1
            else:
                error_rows += 1

        update_progress(
            progress_path,
            run_id=get_run_stem(config),
            total_rows=len(all_records),
            processed_rows=processed_rows,
            success_rows=success_rows,
            error_rows=error_rows,
            completed=False,
        )

        print(
            f"Completed batch {batch_index}: "
            f"{processed_rows}/{len(all_records)} processed "
            f"(success={success_rows}, error={error_rows})"
        )

    update_progress(
        progress_path,
        run_id=get_run_stem(config),
        total_rows=len(all_records),
        processed_rows=processed_rows,
        success_rows=success_rows,
        error_rows=error_rows,
        completed=True,
    )

    print(f"Predictions written to: {predictions_path}")
    print(f"Progress written to: {progress_path}")
    print(f"Manifest written to: {manifest_path}")
    return predictions_path, progress_path, manifest_path


def run_patent_analysis(
    dataset_version: str,
    run_scope: str = "full",
    prompt_version: str = "ts1",
    batch_size: int = 100,
    model: str = "gpt-5-mini",
) -> tuple[Path, Path, Path]:
    df = load_corpus(dataset_version)
    config = RunConfig(
        dataset_version=dataset_version,
        run_scope=run_scope,
        prompt_version=prompt_version,
        model=model,
        batch_size=batch_size,
    )
    return run_patent_analysis_on_dataframe(df, config)


def run_patent_analysis_on_json(
    input_filename: str,
    dataset_version: str,
    run_scope: str,
    prompt_version: str = "ts1",
    batch_size: int = 100,
    model: str = "gpt-5-mini",
) -> tuple[Path, Path, Path]:
    input_path = ANALYSIS_DIR / input_filename
    records = load_records_from_json(input_path)

    df = pd.DataFrame(
        [
            {
                ID_COL: r.lens_id,
                ABSTRACT_COL: r.abstract,
                CLAIMS_COL: r.claims,
            }
            for r in records
        ]
    )

    config = RunConfig(
        dataset_version=dataset_version,
        run_scope=run_scope,
        prompt_version=prompt_version,
        model=model,
        batch_size=batch_size,
    )
    return run_patent_analysis_on_dataframe(df, config)


if __name__ == "__main__":
    # Full-corpus resumable run:
    # run_patent_analysis(
    #     dataset_version="v6",
    #     run_scope="full",
    #     prompt_version="ts1",
    #     batch_size=100,
    #     model="gpt-5-mini",
    # )

    # Re-run on an existing JSON sample:
    run_patent_analysis_on_json(
        input_filename="v6_anurag.json",
        dataset_version="v6",
        run_scope="anurag_sample",
        prompt_version="ts1",
        batch_size=50,
        model="gpt-5-mini",
    )
