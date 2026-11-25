
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[4]

OUTPUT = (
    ROOT / "ml/evidence_classifier/data/evaluation/datasets/diagnostic_claims_v2.json"
)

# Format:
# category, evidence, supported claim, second claim, second label

PAIRS = [
    # ENTITY REPLACEMENT
    (
        "entity",
        "Venus is the hottest planet in the Solar System.",
        "Venus is the hottest planet in the Solar System.",
        "Mercury is the hottest planet in the Solar System.",
        "REFUTES",
    ),
    (
        "entity",
        "Jupiter is the largest planet in the Solar System.",
        "Jupiter is the largest planet in the Solar System.",
        "Saturn is the largest planet in the Solar System.",
        "REFUTES",
    ),
    (
        "entity",
        "The Pacific Ocean is Earth's largest ocean basin.",
        "The Pacific Ocean is Earth's largest ocean basin.",
        "The Atlantic Ocean is Earth's largest ocean basin.",
        "REFUTES",
    ),
    (
        "entity",
        "Neil Armstrong was the first person to walk on the Moon.",
        "Neil Armstrong was the first person to walk on the Moon.",
        "Buzz Aldrin was the first person to walk on the Moon.",
        "REFUTES",
    ),
    (
        "entity",
        "Canberra is the capital of Australia.",
        "Canberra is the capital of Australia.",
        "Sydney is the capital of Australia.",
        "REFUTES",
    ),

    # NUMERICAL CONTRADICTION
    (
        "number",
        "Mars has two natural moons: Phobos and Deimos.",
        "Mars has two natural moons.",
        "Mars has three natural moons.",
        "REFUTES",
    ),
    (
        "number",
        "A triangle has three sides.",
        "A triangle has three sides.",
        "A triangle has four sides.",
        "REFUTES",
    ),
    (
        "number",
        "A hexagon has six sides.",
        "A hexagon has six sides.",
        "A hexagon has eight sides.",
        "REFUTES",
    ),
    (
        "number",
        "The chemical formula for water is H2O, with two hydrogen atoms and one oxygen atom.",
        "A water molecule contains two hydrogen atoms.",
        "A water molecule contains three hydrogen atoms.",
        "REFUTES",
    ),
    (
        "number",
        "The adult human skeleton typically contains 206 bones.",
        "An adult human skeleton typically contains 206 bones.",
        "An adult human skeleton typically contains 306 bones.",
        "REFUTES",
    ),

    # DATE CONTRADICTION
    (
        "date",
        "The first human Moon landing occurred in 1969.",
        "Humans first landed on the Moon in 1969.",
        "Humans first landed on the Moon in 1979.",
        "REFUTES",
    ),
    (
        "date",
        "The Berlin Wall fell in November 1989.",
        "The Berlin Wall fell in 1989.",
        "The Berlin Wall fell in 1999.",
        "REFUTES",
    ),
    (
        "date",
        "The World Wide Web was invented by Tim Berners-Lee at CERN in 1989.",
        "Tim Berners-Lee invented the World Wide Web in 1989.",
        "Tim Berners-Lee invented the World Wide Web in 1999.",
        "REFUTES",
    ),
    (
        "date",
        "The United Nations was founded in 1945.",
        "The United Nations was founded in 1945.",
        "The United Nations was founded in 1955.",
        "REFUTES",
    ),
    (
        "date",
        "The first iPhone was introduced in 2007.",
        "The first iPhone was introduced in 2007.",
        "The first iPhone was introduced in 2017.",
        "REFUTES",
    ),

    # NEGATION
    (
        "negation",
        "Earth orbits the Sun.",
        "Earth orbits the Sun.",
        "Earth does not orbit the Sun.",
        "REFUTES",
    ),
    (
        "negation",
        "Light travels faster than sound in a vacuum-free comparison through air.",
        "Light travels faster than sound in air.",
        "Light does not travel faster than sound in air.",
        "REFUTES",
    ),
    (
        "negation",
        "The Moon is Earth's natural satellite.",
        "The Moon is a natural satellite of Earth.",
        "The Moon is not a natural satellite of Earth.",
        "REFUTES",
    ),
    (
        "negation",
        "Water contains oxygen atoms.",
        "Water contains oxygen.",
        "Water does not contain oxygen.",
        "REFUTES",
    ),
    (
        "negation",
        "Jupiter is a planet in the Solar System.",
        "Jupiter is a planet.",
        "Jupiter is not a planet.",
        "REFUTES",
    ),

    # RELATION REVERSAL
    (
        "relation",
        "The Earth orbits the Sun.",
        "The Earth orbits the Sun.",
        "The Sun orbits the Earth.",
        "REFUTES",
    ),
    (
        "relation",
        "The Moon orbits Earth.",
        "The Moon orbits Earth.",
        "Earth orbits the Moon.",
        "REFUTES",
    ),
    (
        "relation",
        "Paris is the capital of France.",
        "Paris is the capital of France.",
        "France is the capital of Paris.",
        "REFUTES",
    ),
    (
        "relation",
        "A square is a type of rectangle.",
        "A square is a rectangle.",
        "Every rectangle is a square.",
        "REFUTES",
    ),
    (
        "relation",
        "A dolphin is a mammal.",
        "A dolphin is a mammal.",
        "Every mammal is a dolphin.",
        "REFUTES",
    ),

    # SEMANTIC PARAPHRASE
    (
        "paraphrase",
        "Mars has two natural satellites named Phobos and Deimos.",
        "Mars has two natural moons.",
        "Two natural satellites orbit Mars.",
        "SUPPORTS",
    ),
    (
        "paraphrase",
        "Venus is the hottest planet in our Solar System.",
        "Venus is the hottest planet in the Solar System.",
        "No planet in our Solar System is hotter than Venus.",
        "SUPPORTS",
    ),
    (
        "paraphrase",
        "Jupiter is the largest planet in the Solar System.",
        "Jupiter is the Solar System's largest planet.",
        "The largest planet orbiting the Sun is Jupiter.",
        "SUPPORTS",
    ),
    (
        "paraphrase",
        "The United Nations was established in 1945.",
        "The United Nations was founded in 1945.",
        "The UN was established in the year 1945.",
        "SUPPORTS",
    ),
    (
        "paraphrase",
        "The chemical formula of water is H2O.",
        "Water has the chemical formula H2O.",
        "H2O is the chemical formula for water.",
        "SUPPORTS",
    ),
]


def main():
    dataset = []

    for index, pair in enumerate(PAIRS, start=1):
        category, evidence, first, second, second_label = pair
        pair_id = f"pair-{index:03d}"

        for variant, claim, label in (
            ("a", first, "SUPPORTS"),
            ("b", second, second_label),
        ):
            dataset.append({
                "id": f"diag-{index:03d}-{variant}",
                "pairId": pair_id,
                "category": category,
                "claim": claim,
                "evidence": evidence,
                "wikiTitle": None,
                "verifiedLabel": label,
            })

    assert len(PAIRS) == 30
    assert len(dataset) == 60
    assert len({item["id"] for item in dataset}) == 60

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(dataset, file, indent=2, ensure_ascii=False)

    counts = Counter(item["category"] for item in dataset)

    print("=== DIAGNOSTIC DATASET V2 ===")
    print(f"Output: {OUTPUT}")
    print(f"Pairs: {len(PAIRS)}")
    print(f"Claims: {len(dataset)}")

    for category, count in counts.items():
        print(f"{category}: {count}")

    print("Validation: PASSED")


if __name__ == "__main__":
    main()
