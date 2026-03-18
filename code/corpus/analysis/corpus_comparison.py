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


@dataclass
class PatentText:
    lens_id: str
    abstract: str
    claims: str


class GPUArchitectureLabel(BaseModel):
    is_gpu_architecture_design_patent: bool


PROMPT = (
    "You are classifying patents.\n\n"
    "Task:\n"
    "Decide whether a patent describes accelerator hardware architecture design.\n\n"
    "Output:\n"
    "Return TRUE if the invention concerns the design or modification of hardware architecture for a massively parallel compute accelerator (e.g., GPU, TPU, NPU, AI accelerator).\n\n"
    "Return FALSE otherwise.\n\n"
    "Operational definition:\n\n"
    "Return TRUE when the invention introduces or modifies architectural mechanisms inside a compute accelerator, including hardware structures that determine how computation, memory access, or communication are performed within the device.\n\n"
    "This includes inventions concerning:\n"
    "- compute units, execution units, processing element arrays, tensor units\n"
    "- schedulers, interconnects, and communication mechanisms\n"
    "- memory subsystems, caching mechanisms, and register files\n"
    "- pipelines and parallel execution structures\n"
    "- hardware support for synchronization\n"
    "- other architectural mechanisms internal to the accelerator\n\n"
    "Return FALSE when the invention primarily concerns:\n"
    "- machine learning algorithms, model architectures, or training/inference techniques\n"
    "- software frameworks, compilers, kernels, or runtime systems\n"
    "- application-level methods that run on accelerators\n"
    "- general-purpose computing methods not tied to accelerator architecture\n"
    "- semiconductor fabrication, lithography, packaging, or materials\n"
    "- circuit-level implementation details without architectural implications\n"
    "- cooling, power delivery, or mechanical packaging\n\n"
    "Decision rule:\n\n"
    "Focus on what the patent claims as the inventive contribution.\n\n"
    "If the novelty lies in how the accelerator hardware itself is structured or operates internally, return TRUE.\n\n"
    "If the accelerator is only used as a platform, return FALSE."
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
            label = classify_gpu_architecture_design(client, patent)
            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                "is_gpu_architecture_design_patent": bool(label),
                "error": None,
            })
        except Exception as e:
            evaluations.append({
                "lens_id": patent.lens_id,
                "abstract": patent.abstract,
                "claims": patent.claims,
                "is_gpu_architecture_design_patent": None,
                "error": f"{type(e).__name__}: {e}",
            })

        if i % 10 == 0:
            print(f"Processed {i}/{len(sampled)}")

    eval_path = ANALYSIS_DIR / f"{version}_sample_evaluation.json"
    write_json(eval_path, evaluations)

    n_true = sum(
        1 for row in evaluations
        if row["is_gpu_architecture_design_patent"] is True
    )
    n_scored = sum(
        1 for row in evaluations
        if row["is_gpu_architecture_design_patent"] is not None
    )

    print(f"Sample written to: {sample_path}")
    print(f"Evaluation written to: {eval_path}")
    print(f"Scored {n_scored}/{len(evaluations)}")
    print(f"TRUE count: {n_true}")

    return sample_path, eval_path


if __name__ == "__main__":
    evaluate_corpus("v5", 50)