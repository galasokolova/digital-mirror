"""
Digital Mirror — Emotional Reaction Taxonomy V1.

Selects application-specific emotional categories
from the original GoEmotions taxonomy.
"""

TAXONOMY_VERSION = "1.0"

EMOTION_MAPPING = {
    "amusement": "AMUSEMENT",
    "annoyance": "ANNOYANCE",
    "approval": "APPROVAL",
    "confusion": "CONFUSION",
    "curiosity": "CURIOSITY",
    "disapproval": "DISAPPROVAL",
    "gratitude": "GRATITUDE",
    "realization": "REALIZATION",
    "surprise": "SURPRISE",
    "neutral": "NEUTRAL",
}

EMOTION_CLASSES = list(EMOTION_MAPPING.values())


def map_emotions(emotions: list[str]) -> list[str]:

    return sorted({
        EMOTION_MAPPING[emotion]
        for emotion in emotions
        if emotion in EMOTION_MAPPING
    })