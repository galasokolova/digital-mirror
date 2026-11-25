import re


def clean_text(text: str) -> str:

    if not isinstance(text, str):
        raise TypeError("Text must be a string")

    # Remove annotator comments
    text = re.sub(
        r"\*?\[\[.*?\]\]",
        " ",
        text
    )

    # Remove extended non-verbal annotations
    text = re.sub(
        r"<<.*?>>",
        " ",
        text
    )

    # Remove non-verbal annotations
    text = re.sub(
        r"<[^<>]*>",
        " ",
        text
    )

    # Remove annotation prefixes but preserve words
    text = re.sub(
        r"\{[A-Za-z]\s*",
        "",
        text
    )

    # Remove uncertain transcription markers
    # Preserve the transcribed words
    text = text.replace("((", " ")
    text = text.replace("))", " ")

    # Remove technical annotation symbols
    text = re.sub(
        r"[{}\[\]+/#*]",
        " ",
        text
    )

    # Remove interruption markers
    text = re.sub(
        r"-{2,}",
        " ",
        text
    )

    # Remove standalone hyphens
    text = re.sub(
        r"(?<!\S)-(?!\S)",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove spaces before punctuation
    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text
    )

    text = text.strip()

    # Remove punctuation-only texts
    if not any(char.isalnum() for char in text):
        return ""

    return text
