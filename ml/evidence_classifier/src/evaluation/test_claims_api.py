import json
import urllib.request
from pathlib import Path

BASE_URL = "http://127.0.0.1:8080"

PROJECT_ROOT = Path(__file__).resolve().parents[4]
CLAIMS_FILE = (
    PROJECT_ROOT
    / "orchestrator"
    / "src"
    / "main"
    / "resources"
    / "data"
    / "claims.json"
)


def main():
    with CLAIMS_FILE.open(encoding="utf-8") as file:
        claims = json.load(file)

    correct = 0

    print(
        f"{'ID':<12} {'Verified':<12} "
        f"{'Predicted':<12} {'P(SUPPORTS)':>12} {'Correct':>9}"
    )
    print("-" * 65)

    for claim in claims:
        claim_id = claim["id"]

        request = urllib.request.Request(
            f"{BASE_URL}/api/claims/{claim_id}/analyze",
            data=b"",
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                result = json.load(response)
        except Exception as error:
            print(f"{claim_id:<12} ERROR: {error}")
            continue

        verified = claim["verifiedLabel"]
        predicted = result["prediction"]
        p_supports = result["pSupports"]

        is_correct = verified == predicted
        correct += int(is_correct)

        print(
            f"{claim_id:<12} {verified:<12} "
            f"{predicted:<12} {p_supports:>12.4f} "
            f"{'YES' if is_correct else 'NO':>9}"
        )

    print("-" * 65)
    print(f"Correct: {correct}/{len(claims)}")
    print(f"Accuracy: {correct / len(claims):.1%}")


if __name__ == "__main__":
    main()
