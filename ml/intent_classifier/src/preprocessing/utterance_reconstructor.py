import pandas as pd


def reconstruct_utterances(df: pd.DataFrame):

    df = df.sort_values(
        ["conversation_no", "transcript_index"]
    ).reset_index(drop=True)

    reconstructed = []
    orphan_continuations = []

    last_utterance = {}

    merged_count = 0

    for row in df.to_dict("records"):

        key = (
            row["conversation_no"],
            row["caller"]
        )

        if row["damsl_tag"] == "+":

            if key not in last_utterance:

                orphan_continuations.append(row)
                continue

            target_index = last_utterance[key]

            target = reconstructed[target_index]

            target["text"] += " " + row["text"]

            target["segment_count"] += 1

            target["source_indices"].append(
                row["transcript_index"]
            )

            target["source_tags"].append(
                row["act_tag"]
            )

            target["end_transcript_index"] = (
                row["transcript_index"]
            )

            merged_count += 1

        else:

            new_utterance = row.copy()

            new_utterance["segment_count"] = 1

            new_utterance["source_indices"] = [
                row["transcript_index"]
            ]

            new_utterance["source_tags"] = [
                row["act_tag"]
            ]

            new_utterance["end_transcript_index"] = (
                row["transcript_index"]
            )

            reconstructed.append(new_utterance)

            last_utterance[key] = (
                len(reconstructed) - 1
            )

    return (
        pd.DataFrame(reconstructed),
        pd.DataFrame(orphan_continuations),
        merged_count
    )