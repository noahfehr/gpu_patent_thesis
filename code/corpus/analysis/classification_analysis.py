from pathlib import Path
import json

def load_predictions(path: Path):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # keep only successful runs
            if r.get("status") == "success":
                records.append(r)

    return records


def compute_metrics(records):
    total = len(records)

    # Stage 1
    stage1_true_records = [
        r for r in records if r.get("is_accelerator_relevant", False)
    ]
    n_stage1_true = len(stage1_true_records)

    # Stage 2 (only evaluated if Stage 1 = True)
    stage2_true_records = [
        r for r in stage1_true_records if r.get("is_accelerator_hardware", False)
    ]
    n_stage2_true = len(stage2_true_records)

    # Stage 3 (only evaluated if Stage 2 = True)
    stage3_true_records = [
        r for r in stage2_true_records if r.get("is_accelerator_hardware_design_patent", False)
    ]
    n_stage3_true = len(stage3_true_records)

    def pct(num, denom):
        return 100 * num / denom if denom > 0 else 0.0

    return {
        "total": total,

        # Stage 1 (global)
        "stage1_true": n_stage1_true,
        "stage1_pct": pct(n_stage1_true, total),

        # Stage 2 (conditional on Stage 1)
        "stage2_true": n_stage2_true,
        "stage2_pct_conditional": pct(n_stage2_true, n_stage1_true),

        # Stage 3 (conditional on Stage 2)
        "stage3_true": n_stage3_true,
        "stage3_pct_conditional": pct(n_stage3_true, n_stage2_true),

        # also useful: end-to-end yield
        "stage3_pct_overall": pct(n_stage3_true, total),
    }


def print_report(m):
    print("\n=== Two-Stage Pipeline Metrics ===")
    print(f"Total successful predictions: {m['total']}\n")

    print("Stage 1: Accelerator Relevant")
    print(f"  True count: {m['stage1_true']}")
    print(f"  % of total: {m['stage1_pct']:.2f}%\n")

    print("Stage 2: Accelerator Hardware (given Stage 1 = True)")
    print(f"  True count: {m['stage2_true']}")
    print(f"  % of Stage 1: {m['stage2_pct_conditional']:.2f}%\n")

    print("End-to-end yield:")
    print(f"  % of total reaching Stage 3 True: {m['stage3_pct_overall']:.2f}%\n")


if __name__ == "__main__":
    predictions_path = Path("data/analysis/runs/v6__full__two_stage__ts1__predictions.jsonl")

    if not predictions_path.exists():
        raise FileNotFoundError(f"File not found: {predictions_path}")

    records = load_predictions(predictions_path)
    metrics = compute_metrics(records)
    print_report(metrics)