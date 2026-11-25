from explore_switchboard import load_dataset
from preprocessing.damsl_normalizer import normalize_damsl_tag


def main():

    print("=== E04: UTTERANCE RECONSTRUCTION EXPLORATION ===")

    df = load_dataset()

    df["damsl_tag"] = df["act_tag"].apply(
        normalize_damsl_tag
    )

    df = df.sort_values(
        ["conversation_no", "transcript_index"]
    ).reset_index(drop=True)

    continuations = df[df["damsl_tag"] == "+"]

    print(f"\nTotal rows: {len(df):,}")
    print(f"Continuation rows: {len(continuations):,}")

    print("\n=== CALLER VALUES ===")

    print(df["caller"].value_counts().to_string())

    print("\n=== CONTINUATION CALLERS ===")

    print(
        continuations["caller"]
        .value_counts()
        .to_string()
    )

    print("\n=== SUBUTTERANCE INDEX DISTRIBUTION ===")

    print(
        continuations["subutterance_index"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\n=== CONTINUATION EXAMPLES ===")

    columns = [
        "conversation_no",
        "transcript_index",
        "caller",
        "utterance_index",
        "subutterance_index",
        "act_tag",
        "damsl_tag",
        "text"
    ]

    samples = continuations.sample(
        n=min(30, len(continuations)),
        random_state=42
    )

    for number, (index, row) in enumerate(
        samples.iterrows(),
        start=1
    ):

        print("\n" + "=" * 80)
        print(f"EXAMPLE {number}")
        print("=" * 80)

        context = df.iloc[
            max(0, index - 5):index + 4
        ]

        context = context[
            context["conversation_no"]
            == row["conversation_no"]
        ]

        for current_index, current_row in context.iterrows():

            marker = ">>>" if current_index == index else "   "

            print(f"\n{marker}")

            for column in columns:
                print(f"{column}: {current_row[column]}")

    print("\n=== EXPLORATION COMPLETE ===")


if __name__ == "__main__":
    main()