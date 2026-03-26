import json
import random
from pathlib import Path
from typing import Any
import copy


ANALYSIS_DIR = Path("data/analysis")


def load_json(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def even_sample(
    version: str,
    seed: int = 42,
    n_per_class: int = 50,
    source_suffix: str = "_sample_evaluation.json",
    output_suffix: str = "_even_sample.json",
) -> None:
    """
    Create a balanced, blinded sample from {version}_sample_evaluation.json.

    Example:
        sample("v6")
    """
    input_file = ANALYSIS_DIR / f"{version}{source_suffix}"
    output_file = ANALYSIS_DIR / f"{version}{output_suffix}"

    data = load_json(input_file)

    true_samples = [
        d for d in data if d.get("is_gpu_architecture_design_patent") is True
    ]
    false_samples = [
        d for d in data if d.get("is_gpu_architecture_design_patent") is False
    ]

    if len(true_samples) < n_per_class or len(false_samples) < n_per_class:
        raise ValueError(
            f"Not enough samples for {version}. "
            f"true={len(true_samples)}, false={len(false_samples)}, "
            f"needed={n_per_class} each."
        )

    rng = random.Random(seed)
    sampled_true = rng.sample(true_samples, n_per_class)
    sampled_false = rng.sample(false_samples, n_per_class)

    combined = sampled_true + sampled_false

    blinded = []
    for row in combined:
        row_copy = dict(row)
        row_copy["is_gpu_architecture_design_patent"] = None
        blinded.append(row_copy)

    rng.shuffle(blinded)
    write_json(output_file, blinded)

    print(f"Wrote {len(blinded)} rows to {output_file}")

def anurag_prep(
    eval_file=ANALYSIS_DIR / "v6_sample_evaluation.json",
    disagreement_file=ANALYSIS_DIR / "v6_disagreements.json",
    output_file=ANALYSIS_DIR / "v6_anurag.json",
    seed=42,
    target_pos=10,
    target_neg=10,
):
    data = load_json(eval_file)
    disagreements = load_json(disagreement_file)

    rng = random.Random(seed)

    disagreement_ids = {row["lens_id"] for row in disagreements}

    d_pos = [r for r in disagreements if r["model_label"] is True]
    d_neg = [r for r in disagreements if r["model_label"] is False]

    n_pos_needed = target_pos - len(d_pos)
    n_neg_needed = target_neg - len(d_neg)

    if n_pos_needed < 0 or n_neg_needed < 0:
        raise ValueError(
            f"Too many disagreement cases in one class: "
            f"{len(d_pos)} positive, {len(d_neg)} negative; "
            f"targets are {target_pos} positive and {target_neg} negative."
        )

    remaining = [r for r in data if r["lens_id"] not in disagreement_ids]

    remaining_pos = [
        r for r in remaining if r["is_gpu_architecture_design_patent"] is True
    ]
    remaining_neg = [
        r for r in remaining if r["is_gpu_architecture_design_patent"] is False
    ]

    if len(remaining_pos) < n_pos_needed or len(remaining_neg) < n_neg_needed:
        raise ValueError(
            f"Not enough remaining samples. Need {n_pos_needed} positive and "
            f"{n_neg_needed} negative, but only have "
            f"{len(remaining_pos)} positive and {len(remaining_neg)} negative."
        )

    sampled_pos = rng.sample(remaining_pos, n_pos_needed)
    sampled_neg = rng.sample(remaining_neg, n_neg_needed)

    final_rows = disagreements + sampled_pos + sampled_neg

    blinded = []
    for i, row in enumerate(final_rows):
        r = row.copy()
        r["is_gpu_architecture_design_patent"] = None
        if "model_label" in r:
            del r["model_label"]
        if "human_label" in r:
            del r["human_label"]
        r["eval_id"] = i
        blinded.append(r)

    rng.shuffle(blinded)
    write_json(output_file, blinded)

    print(
        f"Wrote {len(blinded)} rows to {output_file}. "
        f"Included {len(disagreements)} disagreements and filled to "
        f"{target_pos} positive / {target_neg} negative by model label."
    )

def evaluate(
    version: str,
    source_suffix: str = "_sample_evaluation.json",
    hand_suffix: str = "_even_sample.json",
    hand_label_key: str = "is_gpu_architecture_design_patent",
    hand_label_name: str = "human",
    disagreements_suffix: str | None = None,
) -> dict[str, float]:
    """
    Compare model predictions from {version}{source_suffix}
    against hand labels in {version}{hand_suffix}.

    Parameters
    ----------
    version : str
        Dataset version prefix, e.g. "v6".
    source_suffix : str
        Suffix for model prediction file.
    hand_suffix : str
        Suffix for hand-labeled file.
    hand_label_key : str
        Key in the hand-labeled file containing the ground-truth label.
    hand_label_name : str
        Name used in outputs, e.g. "human" or "anurag".
    disagreements_suffix : str | None
        Optional suffix for disagreement output file.
        If None, defaults to f"_{hand_label_name}_disagreements.json".

    Returns
    -------
    dict[str, float]
        Evaluation metrics and counts.
    """
    source_file = ANALYSIS_DIR / f"{version}{source_suffix}"
    hand_file = ANALYSIS_DIR / f"{version}{hand_suffix}"

    source_data = load_json(source_file)
    hand_data = load_json(hand_file)

    source_by_id = {row["lens_id"]: row for row in source_data}

    tp = tn = fp = fn = 0
    disagreements = []
    missing_ids = []
    unlabeled = []

    for row in hand_data:
        lens_id = row.get("lens_id")
        hand_label = row.get(hand_label_key)

        if lens_id not in source_by_id:
            missing_ids.append(lens_id)
            continue

        if hand_label is None:
            unlabeled.append(lens_id)
            continue

        model_label = source_by_id[lens_id].get("is_gpu_architecture_design_patent")

        if model_label is None:
            raise ValueError(
                f"Model label is None for lens_id={lens_id} in {source_file}"
            )

        if model_label is True and hand_label is True:
            tp += 1
        elif model_label is False and hand_label is False:
            tn += 1
        elif model_label is True and hand_label is False:
            fp += 1
            disagreements.append(
                {
                    "lens_id": lens_id,
                    "model_label": model_label,
                    f"{hand_label_name}_label": hand_label,
                    "abstract": row.get("abstract"),
                    "claims": row.get("claims"),
                }
            )
        elif model_label is False and hand_label is True:
            fn += 1
            disagreements.append(
                {
                    "lens_id": lens_id,
                    "model_label": model_label,
                    f"{hand_label_name}_label": hand_label,
                    "abstract": row.get("abstract"),
                    "claims": row.get("claims"),
                }
            )

    total = tp + tn + fp + fn
    if total == 0:
        raise ValueError("No labeled overlapping rows found to evaluate.")

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    results = {
        "n_evaluated": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_missing_ids": len(missing_ids),
        "n_unlabeled": len(unlabeled),
    }

    print(f"Evaluation against {hand_label_name} labels for {version}")
    print("-" * 40)
    print(f"Evaluated:  {total}")
    print(f"TP:         {tp}")
    print(f"TN:         {tn}")
    print(f"FP:         {fp}")
    print(f"FN:         {fn}")
    print(f"Accuracy:   {accuracy:.3f}")
    print(f"Precision:  {precision:.3f}")
    print(f"Recall:     {recall:.3f}")
    print(f"F1:         {f1:.3f}")
    print(f"Missing IDs:{len(missing_ids)}")
    print(f"Unlabeled:  {len(unlabeled)}")

    if disagreements_suffix is None:
        disagreements_suffix = f"_{hand_label_name}_disagreements.json"

    disagreements_file = ANALYSIS_DIR / f"{version}{disagreements_suffix}"
    write_json(disagreements_file, disagreements)
    print(f"Wrote disagreements to {disagreements_file}")

    return results


if __name__ == "__main__":
    # Example usage:
    # sample("v6")
    # evaluate("v6")
    evaluate(
        version="v6",
        hand_suffix="_anurag.json",
        hand_label_key="is_gpu_architecture_design_patent",
        hand_label_name="anurag",
    )
