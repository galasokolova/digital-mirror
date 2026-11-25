from explore_switchboard import load_dataset


# ==================================================
# Configuration
# ==================================================

TARGET_TAGS = {
    "aa": "Agree / Accept",
    "ar": "Reject",
    "aap": "Accept-part",
    "arp": "Reject-part",
    "b": "Acknowledgment",
    "ba": "Appreciation",
    "h": "Hedge",
    "bh": "Backchannel in question form",
    "br": "Signal non-understanding",
    "bf": "Summarize / Rephrase",
    "ny": "Yes Answer",
    "nn": "No Answer",
    "ng": "Negative non-no answer",
    "na": "Affirmative non-yes answer",
    "qy": "Yes-No Question",
    "qw": "Wh-Question",
    "qy^d": "Declarative Yes-No Question",
    "qh": "Rhetorical Question",
    "fp": "Conventional Opening",
    "fc": "Conventional Closing",
    "ft": "Thanking",
    "sv": "Statement Opinion",
    "sd": "Statement Non-opinion",
}

SAMPLES_PER_TAG = 20
RANDOM_STATE = 42


# ==================================================
# Inspect original labels
# ==================================================

def explore_label_examples(df):

    for tag, description in TARGET_TAGS.items():

        subset = df[df["act_tag"] == tag]

        print("\n" + "=" * 70)
        print(f"TAG: {tag}")
        print(f"DESCRIPTION: {description}")
        print(f"TOTAL ROWS: {len(subset):,}")
        print("=" * 70)

        if subset.empty:
            continue

        samples = subset.sample(
            n=min(SAMPLES_PER_TAG, len(subset)),
            random_state=RANDOM_STATE
        )

        for index, (_, row) in enumerate(
            samples.iterrows(), start=1
        ):
            print(f"{index:02d}. {row['text']}")

            print(
                f"    Conversation: {row['conversation_no']}"
                f" | Caller: {row['caller']}"
                f" | Index: {row['transcript_index']}"
            )


# ==================================================
# Main
# ==================================================

def main():

    print("=== E02: SWITCHBOARD LABEL SEMANTICS ===")

    df = load_dataset()

    explore_label_examples(df)

    print("\n=== EXPLORATION COMPLETE ===")


if __name__ == "__main__":
    main()