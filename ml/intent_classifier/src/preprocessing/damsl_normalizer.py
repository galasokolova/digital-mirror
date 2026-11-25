"""
DAMSL tag normalization.

Adapted from Christopher Potts' swda.py (2011).

Original source:
https://compprag.christopherpotts.net/code-data/swda.py

License:
Creative Commons Attribution-NonCommercial-ShareAlike 3.0
https://creativecommons.org/licenses/by-nc-sa/3.0/

Changes:
- Extracted normalization into a standalone function.
- Removed dependency on the Utterance class.
- Added type hints and input validation.
- Reorganized conditional mappings.
"""

import re


PRESERVED_TAGS = {
    "qy^d",
    "qw^d",
    "b^m",
}

SPECIAL_TAGS = {
    "nn^e": "ng",
    "ny^e": "na",
}

TAG_MAPPING = {
    "qr": "qy",
    "qy": "qy",

    "fe": "ba",
    "ba": "ba",

    "oo": "oo_co_cc",
    "co": "oo_co_cc",
    "cc": "oo_co_cc",

    "fx": "sv",
    "sv": "sv",

    "aap": "aap_am",
    "am": "aap_am",

    "arp": "arp_nd",
    "nd": "arp_nd",

    "fo": 'fo_o_fw_"_by_bc',
    "o": 'fo_o_fw_"_by_bc',
    "fw": 'fo_o_fw_"_by_bc',
    '"': 'fo_o_fw_"_by_bc',
    "by": 'fo_o_fw_"_by_bc',
    "bc": 'fo_o_fw_"_by_bc',
}


def normalize_damsl_tag(raw_tag: str) -> str:

    if not isinstance(raw_tag, str):
        raise TypeError("DAMSL tag must be a string")

    tags = re.split(r"\s*[,;]\s*", raw_tag)

    normalized_tags = []

    for tag in tags:

        if tag in PRESERVED_TAGS:
            normalized_tags.append(tag)
            continue

        if tag in SPECIAL_TAGS:
            normalized_tags.append(SPECIAL_TAGS[tag])
            continue

        # Remove suffix modifiers
        tag = re.sub(r"(.)\^.*", r"\1", tag)

        # Remove additional annotation symbols
        tag = re.sub(r"[\(\)@*]", "", tag)

        # Map related categories
        tag = TAG_MAPPING.get(tag, tag)

        normalized_tags.append(tag)

    # Preserve the original author's first-tag policy
    return normalized_tags[0]