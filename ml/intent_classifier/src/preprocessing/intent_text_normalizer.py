import unicodedata


def normalize_text(text: str) -> str:

    return " ".join(text.lower().split())


def remove_punctuation(text: str) -> str:

    text = normalize_text(text)

    text = text.replace("’", "'")

    text = "".join(
        char
        if char == "'"
        or not unicodedata.category(char).startswith("P")
        else " "
        for char in text
    )

    return " ".join(text.split())