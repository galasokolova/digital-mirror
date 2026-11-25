
import json
import urllib.request
import urllib.error
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[4]

DATASET_PATH = (
    ROOT / "ml" / "evidence_classifier" / "data" / "evaluation"
    / "datasets" / "diagnostic_claims.json"
)

API_URL = "http://127.0.0.1:8000/predict"


def predict(item):
    payload = {
        "claim": item["claim"],
        "evidence": item["evidence"],
        "wikiTitle": item.get("wikiTitle"),
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    with DATASET_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    results = []
    pairs = defaultdict(list)

    print("\n=== E06b DIAGNOSTIC EVALUATION ===\n")

    print(
        f"{'ID':<12} {'Type':<12} {'Verified':<10} "
        f"{'Predicted':<10} {'P(SUP)':>8} {'Correct':>8}"
    )
    print("-" * 70)

    for item in dataset:
        try:
            prediction = predict(item)
        except (urllib.error.URLError, ValueError) as error:
            print(f"{item['id']}: ERROR: {error}")
            continue

        predicted = prediction["prediction"]
        verified = item["verifiedLabel"]
        probability = float(prediction["p_supports"])

        correct = predicted == verified

        result = {
            **item,
            "prediction": predicted,
            "pSupports": probability,
            "correct": correct,
        }

        results.append(result)
        pairs[item["pairId"]].append(result)

        print(
            f"{item['id']:<12} "
            f"{item['category']:<12} "
            f"{verified:<10} "
            f"{predicted:<10} "
            f"{probability:>8.4f} "
            f"{'YES' if correct else 'NO':>8}"
        )

    print("\n=== SUMMARY ===")

    if not results:
        print("No successful predictions.")
        return

    correct_count = sum(r["correct"] for r in results)
    accuracy = correct_count / len(results)

    print(f"Accuracy: {correct_count}/{len(results)} = {accuracy:.1%}")

    for label in ("SUPPORTS", "REFUTES"):
        subset = [r for r in results if r["verifiedLabel"] == label]
        if subset:
            count = sum(r["correct"] for r in subset)
            print(f"{label}: {count}/{len(subset)} = {count / len(subset):.1%}")

    print("\n=== CONTRADICTION SENSITIVITY ===")

    valid_pairs = 0
    correct_pairs = 0
    correctly_ordered = 0

    for pair_id, items in pairs.items():
        if len(items) != 2:
            continue

        positive = next(
            (r for r in items if r["verifiedLabel"] == "SUPPORTS"),
            None,
        )
        negative = next(
            (r for r in items if r["verifiedLabel"] == "REFUTES"),
            None,
        )

        if positive is None or negative is None:
            continue

        valid_pairs += 1

        delta = (
            positive["pSupports"]
            - negative["pSupports"]
        )

        both_correct = positive["correct"] and negative["correct"]

        correct_pairs += int(both_correct)
        correctly_ordered += int(delta > 0)

        print(
            f"{pair_id}: "
            f"P(true)={positive['pSupports']:.4f}, "
            f"P(false)={negative['pSupports']:.4f}, "
            f"delta={delta:+.4f}, "
            f"both_correct={both_correct}"
        )

    print("\n=== PAIR METRICS ===")
    print(f"Valid pairs: {valid_pairs}")
    print(f"Both correct: {correct_pairs}/{valid_pairs}")
    print(f"Correct probability ordering: {correctly_ordered}/{valid_pairs}")


if __name__ == "__main__":
    main()
