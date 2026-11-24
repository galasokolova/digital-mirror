
import json
from pathlib import Path
from collections import defaultdict

from evaluate_diagnostic_v2 import (
    predict,
    evaluate,
    print_summary,
    save_csv,
)

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "ml/evidence_classifier/data/evaluation"

V3_DATASET = DATA_DIR / "diagnostic_claims_v3.json"


def main():
    with V3_DATASET.open(encoding="utf-8") as file:
        dataset = json.load(file)

    print("=== E06b DIAGNOSTIC V3 ===")
    print(f"Claims: {len(dataset)}")
    print(f"Pairs: {len(dataset) // 2}\n")

    results = evaluate(dataset)

    print_summary(results)

    # Save V3 results without overwriting V2.
    import evaluate_diagnostic_v2 as evaluator
    evaluator.OUTPUT = (
        DATA_DIR / "diagnostic_results_v3.csv"
    )
    evaluator.save_csv(results)

    print("\n=== CLASS ACCURACY ===")

    for label in ("SUPPORTS", "REFUTES"):
        items = [
            item for item in results
            if item["verifiedLabel"] == label
        ]

        correct = sum(item["correct"] for item in items)

        print(
            f"{label}: {correct}/{len(items)} "
            f"= {correct / len(items):.1%}"
        )

    print("\n=== V2 VS V3 ===")
    print("V2 accuracy: 46/60 = 76.7%")

    correct = sum(item["correct"] for item in results)

    print(
        f"V3 accuracy: {correct}/{len(results)} "
        f"= {correct / len(results):.1%}"
    )


if __name__ == "__main__":
    main()
