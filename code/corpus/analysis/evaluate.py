import copy
import json
import random
from pathlib import Path
from typing import Any, Optional

ANALYSIS_DIR = Path("data/analysis")
RUNS_DIR = ANALYSIS_DIR / "runs"
METRICS_DIR = ANALYSIS_DIR / "metrics"

ID_COL = "lens_id"
ABSTRACT_COL = "abstract"
CLAIMS_COL = "claims"

IS_ACCELERATOR_RELEVANT_COL = "is_accelerator_relevant"
IS_ACCELERATOR_HARDWARE_COL = "is_accelerator_hardware"
LABEL_COL = "is_accelerator_hardware_design_patent"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def even_sample(
    predictions_path: Path,
    output_path: Path,
    seed: int = 42,
    n_per_class: int = 50,
    label_key: str = LABEL_COL,
) -> None:
    data = load_jsonl(predictions_path)

    true_samples = [d for d in data if d.get(label_key) is True]
    false_samples = [d for d in data if d.get(label_key) is False]

    if len(true_samples) < n_per_class or len(false_samples) < n_per_class:
        raise ValueError(
            f"Not enough samples. "
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
        row_copy[LABEL_COL] = None
        blinded.append(row_copy)

    rng.shuffle(blinded)
    write_json(output_path, blinded)
    print(f"Wrote {len(blinded)} rows to {output_path}")


def anurag_prep(
    predictions_path: Path,
    disagreement_file: Path,
    output_path: Path,
    seed: int = 42,
    target_pos: int = 10,
    target_neg: int = 10,
    label_key: str = LABEL_COL,
):
    data = load_jsonl(predictions_path)
    disagreements = load_json(disagreement_file)

    rng = random.Random(seed)
    disagreement_ids = {row[ID_COL] for row in disagreements}

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

    remaining = [r for r in data if r[ID_COL] not in disagreement_ids]
    remaining_pos = [r for r in remaining if r.get(label_key) is True]
    remaining_neg = [r for r in remaining if r.get(label_key) is False]

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
        r = copy.deepcopy(row)
        r[LABEL_COL] = None
        if "model_label" in r:
            del r["model_label"]
        if "human_label" in r:
            del r["human_label"]
        r["eval_id"] = i
        blinded.append(r)

    rng.shuffle(blinded)
    write_json(output_path, blinded)

    print(
        f"Wrote {len(blinded)} rows to {output_path}. "
        f"Included {len(disagreements)} disagreements and filled to "
        f"{target_pos} positive / {target_neg} negative by model label."
    )


def compare_predictions_to_labels(
    predictions_path: Path,
    labels_path: Path,
    prediction_label_key: str = LABEL_COL,
    hand_label_key: str = LABEL_COL,
    hand_label_name: str = "human",
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    prediction_rows = load_json(predictions_path)
    label_rows = load_json(labels_path)

    predictions_by_id = {row[ID_COL]: row for row in prediction_rows}
    labels_by_id = {row[ID_COL]: row for row in label_rows}

    tp = tn = fp = fn = 0
    disagreements = []
    missing_ids = []
    unlabeled = []

    for lens_id, label_row in labels_by_id.items():
        if lens_id not in predictions_by_id:
            missing_ids.append(lens_id)
            continue

        pred_row = predictions_by_id[lens_id]
        pred_label = pred_row.get(prediction_label_key)
        hand_label = label_row.get(hand_label_key)

        if hand_label is None:
            unlabeled.append(lens_id)
            continue

        if pred_label is None:
            raise ValueError(
                f"Prediction label is None for lens_id={lens_id} in {predictions_path}"
            )

        if pred_label is True and hand_label is True:
            tp += 1
        elif pred_label is False and hand_label is False:
            tn += 1
        elif pred_label is True and hand_label is False:
            fp += 1
            disagreements.append(
                {
                    ID_COL: lens_id,
                    "model_label": pred_label,
                    f"{hand_label_name}_label": hand_label,
                    ABSTRACT_COL: label_row.get(ABSTRACT_COL),
                    CLAIMS_COL: label_row.get(CLAIMS_COL),
                    IS_ACCELERATOR_RELEVANT_COL: pred_row.get(IS_ACCELERATOR_RELEVANT_COL),
                    IS_ACCELERATOR_HARDWARE_COL: pred_row.get(IS_ACCELERATOR_HARDWARE_COL),
                    LABEL_COL: pred_row.get(LABEL_COL),
                    "status": pred_row.get("status"),
                    "error": pred_row.get("error"),
                }
            )
        elif pred_label is False and hand_label is True:
            fn += 1
            disagreements.append(
                {
                    ID_COL: lens_id,
                    "model_label": pred_label,
                    f"{hand_label_name}_label": hand_label,
                    ABSTRACT_COL: label_row.get(ABSTRACT_COL),
                    CLAIMS_COL: label_row.get(CLAIMS_COL),
                    IS_ACCELERATOR_RELEVANT_COL: pred_row.get(IS_ACCELERATOR_RELEVANT_COL),
                    IS_ACCELERATOR_HARDWARE_COL: pred_row.get(IS_ACCELERATOR_HARDWARE_COL),
                    LABEL_COL: pred_row.get(LABEL_COL),
                    "status": pred_row.get("status"),
                    "error": pred_row.get("error"),
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
        "predictions_path": str(predictions_path),
        "labels_path": str(labels_path),
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
        "disagreements": disagreements,
    }

    print(f"Predictions: {predictions_path}")
    print(f"Labels:      {labels_path}")
    print("-" * 40)
    print(f"Evaluated:   {total}")
    print(f"TP:          {tp}")
    print(f"TN:          {tn}")
    print(f"FP:          {fp}")
    print(f"FN:          {fn}")
    print(f"Accuracy:    {accuracy:.3f}")
    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"F1:          {f1:.3f}")
    print(f"Missing IDs: {len(missing_ids)}")
    print(f"Unlabeled:   {len(unlabeled)}")

    if output_path is not None:
        write_json(output_path, results)
        print(f"Wrote comparison summary to {output_path}")

    return results


if __name__ == "__main__":
    # Example: balanced blinded sample from new JSONL predictions
    # even_sample(
    #     predictions_path=RUNS_DIR / "v6__full__two_stage__ts1__predictions.jsonl",
    #     output_path=ANALYSIS_DIR / "v6__full__two_stage__ts1__even_sample.json",
    #     seed=42,
    #     n_per_class=50,
    # )

    # Example: compare new JSONL predictions to hand labels
    # compare_predictions_to_labels(
    #     predictions_path=ANALYSIS_DIR / "legacy/v6_anuragsample_twostage_evaluation.json",
    #     labels_path=ANALYSIS_DIR / "v6_anurag_manual_classifications.json",
    #     prediction_label_key=LABEL_COL,
    #     hand_label_key=LABEL_COL,
    #     hand_label_name="anurag",
    #     output_path=METRICS_DIR / "v6__anurag_sample__two_stage__ts1__comparison.json",
    # )
    
    compare_predictions_to_labels(
        predictions_path=ANALYSIS_DIR / "legacy/v6_noahsample_twostage_evaluation.json",
        labels_path=ANALYSIS_DIR / "v6_noah_manual_classifications.json",
        prediction_label_key=LABEL_COL,
        hand_label_key=LABEL_COL,
        hand_label_name="noah",
        output_path=METRICS_DIR / "v6__noah_sample__two_stage__ts1__comparison.json",
    )
    pass
