from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[3]

INPUT = BASE / "data/experiments/fever/e07c_manual_review.csv"
OUTPUT = BASE / "data/experiments/fever/e07d_reviewed_pairs.csv"

DIFFERENCE_TYPES = {
    "1": "NEGATION",
    "2": "DIFFERENT_VALUE",
    "3": "DIFFERENT_SPECIFICITY",
    "4": "OTHER",
}


def ask_choice(prompt, choices):
    while True:
        answer = input(prompt).strip().lower()
        if answer in choices:
            return answer
        print(f"Choose one of: {', '.join(choices)}")


def main():
    if OUTPUT.exists():
        df = pd.read_csv(OUTPUT)
        print("Resuming previous review.")
    else:
        df = pd.read_csv(INPUT)
        print("Starting new E07d review.")

    # Initialize annotation columns with explicit nullable types
    for column in ["difference_type", "review_notes"]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype("string")

    for column in [
        "verified_contradiction",
        "evidence_valid_for_both_claims",
    ]:
        if column not in df.columns:
            df[column] = pd.NA
        df[column] = df[column].astype("boolean")

    for i, row in df.iterrows():
        if pd.notna(row["difference_type"]):
            continue

        print("\n" + "=" * 75)
        print(f"PAIR {i + 1} / {len(df)}")
        print("=" * 75)

        print("\nSUPPORTS CLAIM:")
        print(row["supports_claim"])

        print("\nSUPPORTS EVIDENCE:")
        print(row["supports_evidence"])

        print("\nREFUTES CLAIM:")
        print(row["refutes_claim"])

        print("\nREFUTES EVIDENCE:")
        print(row["refutes_evidence"])

        contradiction = ask_choice(
            "\nDo the claims contradict each other? "
            "(y/n/u=uncertain, s=skip, q=quit): ",
            ["y", "n", "u", "s", "q"],
        )

        if contradiction == "q":
            break
        if contradiction == "s":
            continue

        print("\nDIFFERENCE TYPE:")
        for key, value in DIFFERENCE_TYPES.items():
            print(f"  {key}. {value}")

        difference = ask_choice(
            "Type (1-4): ", list(DIFFERENCE_TYPES)
        )

        evidence = ask_choice(
            "Is the evidence sufficient to evaluate BOTH claims? "
            "(y/n/u=uncertain): ",
            ["y", "n", "u"],
        )

        notes = input("Review notes (optional): ").strip()

        boolean_answers = {
            "y": True,
            "n": False,
            "u": pd.NA,
        }

        df.at[i, "verified_contradiction"] = boolean_answers[contradiction]
        df.at[i, "difference_type"] = DIFFERENCE_TYPES[difference]
        df.at[i, "evidence_valid_for_both_claims"] = boolean_answers[evidence]
        df.at[i, "review_notes"] = notes

        df.to_csv(OUTPUT, index=False)
        print("Saved.")

    print(f"\nReview file: {OUTPUT}")


if __name__ == "__main__":
    main()
