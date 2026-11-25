
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = ROOT / "ml" / "evidence_classifier" / "data" / "evaluation"

INPUT = DATA_DIR / "datasets" / "diagnostic_claims_v2.json"
OUTPUT = DATA_DIR / "datasets" / "diagnostic_claims_v3.json"

# Each tuple:
# (evidence, supported_claim, alternative_claim)
#
# The alternative is REFUTES except for paraphrase,
# where both claims are SUPPORTS.

NEW_PAIRS = {
    "entity": [
        (
            "Ottawa is the capital of Canada.",
            "Ottawa is Canada's capital.",
            "Toronto is Canada's capital.",
        ),
        (
            "Brasilia is the capital of Brazil.",
            "Brasilia is Brazil's capital.",
            "Rio de Janeiro is Brazil's capital.",
        ),
        (
            "The chemical symbol for gold is Au.",
            "Gold has the symbol Au.",
            "Gold has the symbol Ag.",
        ),
        (
            "The chemical symbol for sodium is Na.",
            "Sodium has the symbol Na.",
            "Sodium has the symbol K.",
        ),
        (
            "Alexander Fleming discovered penicillin.",
            "Alexander Fleming discovered penicillin.",
            "Isaac Newton discovered penicillin.",
        ),
        (
            "The pancreas produces insulin.",
            "The pancreas produces insulin.",
            "The liver produces insulin.",
        ),
        (
            "The heart pumps blood through the body.",
            "The heart pumps blood through the body.",
            "The lungs pump blood through the body.",
        ),
        (
            "Mount Everest is Earth's highest mountain above sea level.",
            "Mount Everest is the highest mountain above sea level.",
            "K2 is the highest mountain above sea level.",
        ),
        (
            "The Nile River flows through Egypt.",
            "The Nile River flows through Egypt.",
            "The Amazon River flows through Egypt.",
        ),
        (
            "Python was created by Guido van Rossum.",
            "Guido van Rossum created Python.",
            "James Gosling created Python.",
        ),
    ],

    "number": [
        (
            "A pentagon has five sides.",
            "A pentagon has five sides.",
            "A pentagon has seven sides.",
        ),
        (
            "A cube has six faces.",
            "A cube has six faces.",
            "A cube has eight faces.",
        ),
        (
            "An octagon has eight sides.",
            "An octagon has eight sides.",
            "An octagon has six sides.",
        ),
        (
            "A standard deck of playing cards contains 52 cards.",
            "A standard deck contains 52 cards.",
            "A standard deck contains 42 cards.",
        ),
        (
            "There are 24 hours in a standard day.",
            "A standard day contains 24 hours.",
            "A standard day contains 30 hours.",
        ),
        (
            "There are 60 seconds in one minute.",
            "One minute contains 60 seconds.",
            "One minute contains 100 seconds.",
        ),
        (
            "A byte contains eight bits.",
            "One byte contains eight bits.",
            "One byte contains sixteen bits.",
        ),
        (
            "Binary numbers use two digits: 0 and 1.",
            "Binary uses two digits.",
            "Binary uses ten digits.",
        ),
        (
            "The atomic number of carbon is 6.",
            "Carbon has atomic number 6.",
            "Carbon has atomic number 8.",
        ),
        (
            "The atomic number of oxygen is 8.",
            "Oxygen has atomic number 8.",
            "Oxygen has atomic number 6.",
        ),
    ],

    "date": [
        (
            "World War II ended in 1945.",
            "World War II ended in 1945.",
            "World War II ended in 1955.",
        ),
        (
            "The first artificial satellite, Sputnik 1, launched in 1957.",
            "Sputnik 1 launched in 1957.",
            "Sputnik 1 launched in 1967.",
        ),
        (
            "Yuri Gagarin became the first human in space in 1961.",
            "The first human spaceflight occurred in 1961.",
            "The first human spaceflight occurred in 1971.",
        ),
        (
            "The Titanic sank in 1912.",
            "The Titanic sank in 1912.",
            "The Titanic sank in 1922.",
        ),
        (
            "The European Union was established in 1993.",
            "The European Union was established in 1993.",
            "The European Union was established in 1983.",
        ),
        (
            "Wikipedia launched in 2001.",
            "Wikipedia launched in 2001.",
            "Wikipedia launched in 2011.",
        ),
        (
            "The first modern Olympic Games took place in 1896.",
            "The first modern Olympics took place in 1896.",
            "The first modern Olympics took place in 1906.",
        ),
        (
            "The Apollo 11 Moon landing occurred in 1969.",
            "Apollo 11 landed on the Moon in 1969.",
            "Apollo 11 landed on the Moon in 1979.",
        ),
        (
            "The Chernobyl nuclear disaster occurred in 1986.",
            "The Chernobyl disaster occurred in 1986.",
            "The Chernobyl disaster occurred in 1996.",
        ),
        (
            "The Hubble Space Telescope launched in 1990.",
            "Hubble launched in 1990.",
            "Hubble launched in 2000.",
        ),
    ],

    "negation": [
        (
            "The Earth has one natural Moon.",
            "Earth has a natural Moon.",
            "Earth has no natural Moon.",
        ),
        (
            "The Sun is a star.",
            "The Sun is a star.",
            "The Sun is not a star.",
        ),
        (
            "Iron is a chemical element.",
            "Iron is a chemical element.",
            "Iron is not a chemical element.",
        ),
        (
            "Sound requires a medium to propagate.",
            "Sound requires a medium to propagate.",
            "Sound does not require a medium to propagate.",
        ),
        (
            "Plants use sunlight during photosynthesis.",
            "Plants use sunlight during photosynthesis.",
            "Plants do not use sunlight during photosynthesis.",
        ),
        (
            "Java supports object-oriented programming.",
            "Java supports object-oriented programming.",
            "Java does not support object-oriented programming.",
        ),
        (
            "Python supports functions.",
            "Python supports functions.",
            "Python does not support functions.",
        ),
        (
            "DNA contains genetic information.",
            "DNA contains genetic information.",
            "DNA does not contain genetic information.",
        ),
        (
            "Electrons have negative electric charge.",
            "Electrons have negative electric charge.",
            "Electrons do not have negative electric charge.",
        ),
        (
            "The Pacific Ocean contains salt water.",
            "The Pacific Ocean contains salt water.",
            "The Pacific Ocean does not contain salt water.",
        ),
    ],

    "relation": [
        (
            "A dog is a mammal.",
            "A dog is a mammal.",
            "Every mammal is a dog.",
        ),
        (
            "A robin is a bird.",
            "A robin is a bird.",
            "Every bird is a robin.",
        ),
        (
            "A triangle is a polygon.",
            "A triangle is a polygon.",
            "Every polygon is a triangle.",
        ),
        (
            "A laptop is a computer.",
            "A laptop is a computer.",
            "Every computer is a laptop.",
        ),
        (
            "A whale is a mammal.",
            "A whale is a mammal.",
            "Every mammal is a whale.",
        ),
        (
            "London is located in England.",
            "London is located in England.",
            "England is located in London.",
        ),
        (
            "Rome is located in Italy.",
            "Rome is located in Italy.",
            "Italy is located in Rome.",
        ),
        (
            "A CPU executes machine instructions.",
            "A CPU executes machine instructions.",
            "Machine instructions execute a CPU.",
        ),
        (
            "The brain controls many bodily functions.",
            "The brain controls many bodily functions.",
            "Bodily functions control the brain.",
        ),
        (
            "The Earth is part of the Solar System.",
            "Earth is part of the Solar System.",
            "The Solar System is part of Earth.",
        ),
    ],

    "paraphrase": [
        (
            "The Sun is a star.",
            "The Sun is a star.",
            "Our Sun belongs to the category of stars.",
        ),
        (
            "Earth has one natural Moon.",
            "Earth has one natural Moon.",
            "There is one natural lunar satellite orbiting Earth.",
        ),
        (
            "A triangle has three sides.",
            "A triangle has three sides.",
            "Three sides form the boundary of a triangle.",
        ),
        (
            "The chemical symbol of oxygen is O.",
            "Oxygen has the chemical symbol O.",
            "The letter O represents oxygen in chemical notation.",
        ),
        (
            "Python was created by Guido van Rossum.",
            "Guido van Rossum created Python.",
            "The creator of Python is Guido van Rossum.",
        ),
        (
            "The United Nations was founded in 1945.",
            "The UN was founded in 1945.",
            "The organization known as the United Nations began in 1945.",
        ),
        (
            "Mars has two natural moons.",
            "Mars has two natural moons.",
            "Two moons orbit the planet Mars.",
        ),
        (
            "Jupiter is the largest planet in the Solar System.",
            "Jupiter is the largest planet in the Solar System.",
            "No other Solar System planet is larger than Jupiter.",
        ),
        (
            "A byte consists of eight bits.",
            "A byte consists of eight bits.",
            "Eight bits make up one byte.",
        ),
        (
            "The Earth revolves around the Sun.",
            "Earth revolves around the Sun.",
            "Our planet travels in an orbit around the Sun.",
        ),
    ],
}


def validate_v2(dataset):
    if len(dataset) != 60:
        raise ValueError(
            f"Expected 60 V2 records, found {len(dataset)}"
        )

    ids = [item["id"] for item in dataset]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate IDs in V2")

    required = {
        "id", "pairId", "category", "claim",
        "evidence", "wikiTitle", "verifiedLabel"
    }

    for item in dataset:
        missing = required - item.keys()
        if missing:
            raise ValueError(
                f"{item.get('id')}: missing {missing}"
            )


def validate_v3(dataset):
    if len(dataset) != 180:
        raise ValueError("Expected exactly 180 claims")

    ids = [item["id"] for item in dataset]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate claim IDs")

    groups = {}

    for item in dataset:
        groups.setdefault(item["pairId"], []).append(item)

    if len(groups) != 90:
        raise ValueError("Expected exactly 90 pairs")

    for pair_id, items in groups.items():
        if len(items) != 2:
            raise ValueError(f"{pair_id}: expected 2 claims")

        a, b = items

        if a["evidence"] != b["evidence"]:
            raise ValueError(f"{pair_id}: evidence mismatch")

        if a["category"] != b["category"]:
            raise ValueError(f"{pair_id}: category mismatch")

        labels = sorted(
            item["verifiedLabel"] for item in items
        )

        expected = (
            ["SUPPORTS", "SUPPORTS"]
            if a["category"] == "paraphrase"
            else ["REFUTES", "SUPPORTS"]
        )

        if labels != expected:
            raise ValueError(f"{pair_id}: invalid labels")

    counts = Counter(item["category"] for item in dataset)

    for category in NEW_PAIRS:
        if counts[category] != 30:
            raise ValueError(
                f"{category}: expected 30, got {counts[category]}"
            )

    return counts


def main():
    with INPUT.open(encoding="utf-8") as file:
        dataset = json.load(file)

    validate_v2(dataset)

    for category, pairs in NEW_PAIRS.items():
        if len(pairs) != 10:
            raise ValueError(
                f"{category}: expected 10 new pairs"
            )

    for category, pairs in NEW_PAIRS.items():
        for evidence, first, second in pairs:
            pair_number = (len(dataset) // 2) + 1
            pair_id = f"pair-{pair_number:03d}"

            second_label = (
                "SUPPORTS"
                if category == "paraphrase"
                else "REFUTES"
            )

            for variant, claim, label in (
                ("a", first, "SUPPORTS"),
                ("b", second, second_label),
            ):
                dataset.append({
                    "id": f"diag-{pair_number:03d}-{variant}",
                    "pairId": pair_id,
                    "category": category,
                    "claim": claim,
                    "evidence": evidence,
                    "wikiTitle": None,
                    "verifiedLabel": label,
                })

    counts = validate_v3(dataset)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(
            dataset,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("=== DIAGNOSTIC DATASET V3 ===")
    print(f"Output: {OUTPUT}")
    print(f"Claims: {len(dataset)}")
    print(f"Pairs: {len(dataset) // 2}")

    for category, count in sorted(counts.items()):
        print(f"{category}: {count}")

    print("Validation: PASSED")


if __name__ == "__main__":
    main()
