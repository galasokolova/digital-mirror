
import csv
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

DATASET = (
    ROOT / "ml/evidence_classifier/data/evaluation/diagnostic_claims_v2.json"
)

OUTPUT = (
    ROOT / "ml/evidence_classifier/data/evaluation/diagnostic_results_v2.csv"
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


def evaluate(dataset):
    results = []

    for index, item in enumerate(dataset, start=1):
        response = predict(item)

        predicted = response["prediction"]
        verified = item["verifiedLabel"]

        result = {
            **item,
            "prediction": predicted,
            "pSupports": float(response["p_supports"]),
            "pRefutes": float(response["p_refutes"]),
            "correct": predicted == verified,
        }

        results.append(result)

        print(
            f"[{index:02d}/{len(dataset)}] "
            f"{item['id']} "
            f"{item['category']:<12} "
            f"{verified:<8} -> {predicted:<8} "
            f"P(SUP)={result['pSupports']:.4f} "
            f"{'OK' if result['correct'] else 'WRONG'}"
        )

    return results


def print_summary(results):
    print("\n=== CATEGORY ACCURACY ===")

    categories = defaultdict(list)

    for result in results:
        categories[result["category"]].append(result)

    for category, items in categories.items():
        correct = sum(item["correct"] for item in items)

        print(
            f"{category:<12} "
            f"{correct:>2}/{len(items):<2} "
            f"{correct / len(items):.1%}"
        )

    total_correct = sum(item["correct"] for item in results)

    print("\n=== OVERALL ACCURACY ===")
    print(
        f"{total_correct}/{len(results)} "
        f"= {total_correct / len(results):.1%}"
    )

    print("\n=== PAIR ANALYSIS ===")

    pairs = defaultdict(list)

    for result in results:
        pairs[result["pairId"]].append(result)

    both_correct_count = 0
    correctly_ordered_count = 0
    contradiction_pairs = 0
    paraphrase_pairs = 0
    stable_paraphrases = 0

    for pair_id, items in pairs.items():
        if len(items) != 2:
            raise ValueError(f"Invalid pair: {pair_id}")

        first, second = items

        both_correct = first["correct"] and second["correct"]
        both_correct_count += int(both_correct)

        delta = first["pSupports"] - second["pSupports"]

        if first["category"] == "paraphrase":
            paraphrase_pairs += 1

            stable = (
                first["prediction"] == second["prediction"]
            )
            stable_paraphrases += int(stable)

            print(
                f"{pair_id} paraphrase "
                f"delta={delta:+.4f} "
                f"stable={stable} "
                f"both_correct={both_correct}"
            )
        else:
            contradiction_pairs += 1
            correctly_ordered_count += int(delta > 0)

            print(
                f"{pair_id} {first['category']:<12} "
                f"delta={delta:+.4f} "
                f"both_correct={both_correct}"
            )

    print("\n=== PAIR METRICS ===")
    print(
        f"Both correct: "
        f"{both_correct_count}/{len(pairs)}"
    )
    print(
        f"Correct contradiction ordering: "
        f"{correctly_ordered_count}/{contradiction_pairs}"
    )
    print(
        f"Stable paraphrases: "
        f"{stable_paraphrases}/{paraphrase_pairs}"
    )


def save_csv(results):
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "id",
        "pairId",
        "category",
        "claim",
        "evidence",
        "wikiTitle",
        "verifiedLabel",
        "prediction",
        "pSupports",
        "pRefutes",
        "correct",
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved: {OUTPUT}")


def main():
    with DATASET.open(encoding="utf-8") as file:
        dataset = json.load(file)

    print("=== E06b DIAGNOSTIC V2 ===")
    print(f"Claims: {len(dataset)}\n")

    results = evaluate(dataset)

    print_summary(results)
    save_csv(results)


if __name__ == "__main__":
    main()
