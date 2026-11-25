"""
Digital Mirror — Communicative Intent Taxonomy V1.

Maps selected Switchboard DAMSL dialogue acts
to application-specific communicative categories.
"""

TAXONOMY_VERSION = "1.0"

INTENT_MAPPING = {
    "fp": "GREETING",
    "fc": "GOODBYE",
    "ft": "GRATITUDE",

    "aa": "AGREEMENT",
    "ar": "DISAGREEMENT",

    "h": "UNCERTAINTY",
    "b": "ACKNOWLEDGMENT",

    "ny": "AFFIRMATIVE_ANSWER",
    "na": "AFFIRMATIVE_ANSWER",

    "nn": "NEGATIVE_ANSWER",
    "ng": "NEGATIVE_ANSWER",

    "qy": "QUESTION",
    "qw": "QUESTION",
    "qy^d": "QUESTION",
    "qw^d": "QUESTION",
    "qo": "QUESTION",
    "qrr": "QUESTION",

    "br": "CLARIFICATION",

    "sd": "STATEMENT",
    "sv": "STATEMENT",
}


INTENT_CLASSES = [
    "GREETING",
    "GOODBYE",
    "GRATITUDE",
    "AGREEMENT",
    "DISAGREEMENT",
    "UNCERTAINTY",
    "ACKNOWLEDGMENT",
    "AFFIRMATIVE_ANSWER",
    "NEGATIVE_ANSWER",
    "QUESTION",
    "CLARIFICATION",
    "STATEMENT",
]


def map_intent(damsl_tag: str) -> str | None:

    return INTENT_MAPPING.get(damsl_tag)