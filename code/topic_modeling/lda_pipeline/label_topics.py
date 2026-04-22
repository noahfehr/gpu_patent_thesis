from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

from model import build_all_topic_labeling_payloads, build_topic_labeling_payload


# -----------------------------------
# CLIENT
# -----------------------------------
client = OpenAI()


# -----------------------------------
# CONFIG
# -----------------------------------
@dataclass
class TopicLabelingConfig:
    """
    Configuration for OpenAI-based topic labeling.
    """

    model: str = "gpt-5-mini"
    top_n_docs: int = 3
    excerpt_chars: int = 1500
    n_passes: int = 3
    max_label_words: int = 6
    max_explanation_sentences: int = 3


# -----------------------------------
# PROMPTING
# -----------------------------------
def format_topic_labeling_prompt(
    payload: dict,
    max_label_words: int = 6,
    max_explanation_sentences: int = 3,
) -> str:
    """
    Construct a grounded prompt for topic labeling.
    """
    words = ", ".join(payload["top_words"])

    docs_str = ""
    for i, doc in enumerate(payload["top_docs"], start=1):
        docs_str += (
            f"\n--- Document {i} ---\n"
            f"lens_id: {doc['lens_id']}\n"
            f"topic_weight: {doc['topic_weight']:.4f}\n"
            f"text:\n{doc['text_excerpt']}\n"
        )

    return f"""
You are labeling one topic from an LDA topic model built on patent text related to AI accelerators and semiconductor hardware design.

Your task:
1. Assign a concise, descriptive topic label.
2. Explain briefly why that label fits.

Requirements:
- The label should be specific, not generic.
- The label should be {max_label_words} words or fewer.
- The explanation should be {max_explanation_sentences} sentences or fewer.
- Use both the top words and the representative documents.
- Prefer the underlying technical theme or design focus over surface-level wording.

Top words:
{words}

Top documents:
{docs_str}

Return STRICT JSON matching this schema:
{{
  "label": "...",
  "explanation": "..."
}}
""".strip()


# -----------------------------------
# OPENAI CALLS
# -----------------------------------
def _extract_response_text(response: Any) -> str:
    """
    Try the most common SDK response access patterns.
    """
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    try:
        for item in response.output:
            if getattr(item, "type", None) == "message":
                for content in item.content:
                    if hasattr(content, "text") and content.text:
                        return content.text
        pass
    except Exception:
        pass

    raise ValueError("Could not extract text from OpenAI response.")


def call_openai_labeling(
    prompt: str,
    config: TopicLabelingConfig,
) -> dict[str, str]:
    """
    Call OpenAI for one topic label with structured JSON output.
    Uses the Responses API.
    """
    response = client.responses.create(
        model=config.model,
        input=[
            {
                "role": "developer",
                "content": "Return only valid JSON that matches the provided schema.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "topic_label",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "explanation": {"type": "string"},
                    },
                    "required": ["label", "explanation"],
                    "additionalProperties": False,
                },
            }
        },
    )
    text = _extract_response_text(response)
    return normalize_label_response(json.loads(text))


# -----------------------------------
# NORMALIZATION
# -----------------------------------
def normalize_label_text(label: str) -> str:
    """
    Normalize a label for simple majority-vote grouping.
    """
    label = str(label).strip().lower()
    label = re.sub(r"[^a-z0-9\s\-/]", "", label)
    label = re.sub(r"\s+", " ", label)
    return label


def normalize_label_response(response: dict) -> dict[str, str]:
    """
    Enforce the expected output shape.
    """
    if not isinstance(response, dict):
        raise TypeError(f"Expected dict response, got {type(response)}")

    label = str(response.get("label", "")).strip()
    explanation = str(response.get("explanation", "")).strip()

    if not label:
        raise ValueError("Missing or empty 'label' in model response.")
    if not explanation:
        raise ValueError("Missing or empty 'explanation' in model response.")

    return {
        "label": label,
        "explanation": explanation,
    }


def select_majority_label(candidates: list[dict[str, str]]) -> dict[str, Any]:
    """
    Pick the majority label across passes using normalized exact-match grouping.
    If there is a tie, keep the earliest occurring candidate among tied labels.
    """
    if not candidates:
        raise ValueError("No candidates provided.")

    normalized_labels = [normalize_label_text(c["label"]) for c in candidates]
    counts = Counter(normalized_labels)
    max_count = max(counts.values())
    winning_norms = {label for label, count in counts.items() if count == max_count}

    selected_candidate = None
    selected_norm = None
    for candidate, norm_label in zip(candidates, normalized_labels):
        if norm_label in winning_norms:
            selected_candidate = candidate
            selected_norm = norm_label
            break

    if selected_candidate is None or selected_norm is None:
        raise RuntimeError("Failed to select a majority label.")

    return {
        "final_label": selected_candidate["label"],
        "final_explanation": selected_candidate["explanation"],
        "majority_count": int(counts[selected_norm]),
        "candidate_labels": [c["label"] for c in candidates],
        "candidate_explanations": [c["explanation"] for c in candidates],
        "candidate_counts": dict(counts),
    }


# -----------------------------------
# SINGLE TOPIC
# -----------------------------------
def label_topic_from_payload(
    payload: dict,
    config: TopicLabelingConfig | None = None,
) -> dict:
    """
    Label a single topic from a prebuilt payload using repeated generation.
    Returns the final chosen label plus all pass-level labels.
    """
    config = config or TopicLabelingConfig()
    prompt = format_topic_labeling_prompt(
        payload,
        max_label_words=config.max_label_words,
        max_explanation_sentences=config.max_explanation_sentences,
    )

    candidates = []
    errors = []
    for pass_idx in range(config.n_passes):
        try:
            result = call_openai_labeling(prompt, config=config)
            result["pass_index"] = pass_idx
            candidates.append(result)
        except Exception as e:
            errors.append({"pass_index": pass_idx, "error": str(e)})

    if not candidates:
        return {
            "topic_id": payload["topic_id"],
            "top_words": payload["top_words"],
            "top_docs": payload["top_docs"],
            "label": "ERROR",
            "explanation": "All labeling passes failed.",
            "n_successful_passes": 0,
            "n_requested_passes": config.n_passes,
            "majority_count": 0,
            "candidate_labels": [],
            "candidate_explanations": [],
            "candidate_counts": {},
            "errors": errors,
        }

    selected = select_majority_label(candidates)

    return {
        "topic_id": payload["topic_id"],
        "top_words": payload["top_words"],
        "top_docs": payload["top_docs"],
        "label": selected["final_label"],
        "explanation": selected["final_explanation"],
        "n_successful_passes": len(candidates),
        "n_requested_passes": config.n_passes,
        "majority_count": selected["majority_count"],
        "candidate_labels": selected["candidate_labels"],
        "candidate_explanations": selected["candidate_explanations"],
        "candidate_counts": selected["candidate_counts"],
        "errors": errors,
    }


def label_topic(
    df_docs: pd.DataFrame,
    doc_topic_df: pd.DataFrame,
    topic_words: dict[int, list[str]],
    topic_id: int,
    config: TopicLabelingConfig | None = None,
) -> dict:
    """
    Build a payload for one topic, then label it.
    """
    config = config or TopicLabelingConfig()

    payload = build_topic_labeling_payload(
        df_docs=df_docs,
        doc_topic_df=doc_topic_df,
        topic_words=topic_words,
        topic_id=topic_id,
        top_n_docs=config.top_n_docs,
        excerpt_chars=config.excerpt_chars,
    )
    return label_topic_from_payload(payload, config=config)


# -----------------------------------
# ALL TOPICS
# -----------------------------------
def label_all_topics(
    df_docs: pd.DataFrame,
    doc_topic_df: pd.DataFrame,
    topic_words: dict[int, list[str]],
    config: TopicLabelingConfig | None = None,
) -> list[dict]:
    """
    Label all topics sequentially.
    """
    config = config or TopicLabelingConfig()

    payloads = build_all_topic_labeling_payloads(
        df_docs=df_docs,
        doc_topic_df=doc_topic_df,
        topic_words=topic_words,
        top_n_docs=config.top_n_docs,
        excerpt_chars=config.excerpt_chars,
    )

    results = []
    for payload in payloads:
        topic_id = payload["topic_id"]
        print(f"Labeling topic {topic_id + 1}/{len(payloads)}")
        result = label_topic_from_payload(payload, config=config)
        results.append(result)

    return results


# -----------------------------------
# EVALUATION
# -----------------------------------
def topic_labels_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """
    Flatten topic labeling results into a dataframe.
    """
    rows = []
    for r in results:
        rows.append(
            {
                "topic_id": r["topic_id"],
                "label": r["label"],
                "explanation": r["explanation"],
                "top_words": ", ".join(r.get("top_words", [])),
                "n_successful_passes": r.get("n_successful_passes"),
                "n_requested_passes": r.get("n_requested_passes"),
                "majority_count": r.get("majority_count"),
                "candidate_labels": " | ".join(r.get("candidate_labels", [])),
            }
        )
    return pd.DataFrame(rows)


def evaluate_topic_labels(
    predicted_results: list[dict],
    human_labels_df: pd.DataFrame,
    topic_id_col: str = "topic_id",
    human_label_col: str = "human_label",
) -> tuple[pd.DataFrame, dict]:
    """
    Evaluate predicted labels against human labels using exact normalized match only.
    """
    pred_df = topic_labels_to_dataframe(predicted_results)[["topic_id", "label"]].copy()
    pred_df = pred_df.rename(columns={"label": "predicted_label"})

    merged = pred_df.merge(
        human_labels_df[[topic_id_col, human_label_col]].rename(
            columns={topic_id_col: "topic_id", human_label_col: "human_label"}
        ),
        on="topic_id",
        how="inner",
    )

    if merged.empty:
        raise ValueError("No overlapping topic_ids between predictions and human labels.")

    merged["predicted_label_norm"] = merged["predicted_label"].map(normalize_label_text)
    merged["human_label_norm"] = merged["human_label"].map(normalize_label_text)
    merged["exact_match"] = merged["predicted_label_norm"] == merged["human_label_norm"]

    summary = {
        "n_topics": int(len(merged)),
        "exact_match_rate": float(merged["exact_match"].mean()),
    }

    return merged.sort_values("topic_id").reset_index(drop=True), summary


# -----------------------------------
# SAVING
# -----------------------------------
def save_topic_labels(results: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def save_topic_labeling_outputs(
    results: list[dict],
    output_dir: str | Path,
    human_labels_df: pd.DataFrame | None = None,
) -> None:
    """
    Save a full topic labeling bundle.

    Outputs:
    - topic_labels.json
    - topic_labels.csv
    - topic_labels.parquet
    - topic_label_eval.csv (optional)
    - topic_label_eval_summary.json (optional)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_topic_labels(results, output_dir / "topic_labels.json")

    labels_df = topic_labels_to_dataframe(results)
    labels_df.to_csv(output_dir / "topic_labels.csv", index=False)
    labels_df.to_parquet(output_dir / "topic_labels.parquet", index=False)

    if human_labels_df is not None:
        eval_df, eval_summary = evaluate_topic_labels(
            predicted_results=results,
            human_labels_df=human_labels_df,
        )
        eval_df.to_csv(output_dir / "topic_label_eval.csv", index=False)
        eval_df.to_parquet(output_dir / "topic_label_eval.parquet", index=False)
        with open(output_dir / "topic_label_eval_summary.json", "w") as f:
            json.dump(eval_summary, f, indent=2)


# -----------------------------------
# RUN-DIR INTEGRATION
# -----------------------------------
def load_topic_words_json(path: str | Path) -> dict[int, list[str]]:
    """
    Load topic words from a saved JSON artifact and coerce keys back to int.
    """
    with open(path, "r") as f:
        obj = json.load(f)
    return {int(k): v for k, v in obj.items()}


def label_saved_run_outputs(
    df_docs: pd.DataFrame,
    doc_topic_path: str | Path,
    topic_words_path: str | Path,
    output_dir: str | Path,
    config: TopicLabelingConfig | None = None,
    human_labels_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Label a saved run from the artifacts produced by the LDA pipeline.

    This is intended to plug directly into folders like:
    - outputs/lda/runs/.../doc_topics/k20_seed0.parquet
    - outputs/lda/runs/.../topic_words/k20_seed0.json
    """
    config = config or TopicLabelingConfig()

    doc_topic_df = pd.read_parquet(doc_topic_path)
    topic_words = load_topic_words_json(topic_words_path)

    results = label_all_topics(
        df_docs=df_docs,
        doc_topic_df=doc_topic_df,
        topic_words=topic_words,
        config=config,
    )

    save_topic_labeling_outputs(
        results=results,
        output_dir=output_dir,
        human_labels_df=human_labels_df,
    )

    return results
