from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/raw/euvsdisinfo_all_texts_rename.csv"
OUTPUT = ROOT / "data/processed/euvsdisinfo_false.csv"

df = pd.read_csv(INPUT)

df["date"] = pd.to_datetime(
    df["Date"],
    format="%d.%m.%Y",
    errors="coerce"
)

df = df[df["date"] >= "2022-02-24"].copy()

# Initial candidate filter, not final topic validation
ukraine_pattern = r"\b(?:ukraine|ukrainian|zelensky|kyiv|kiev)\b"

topic_mask = (
    df["Title"].str.contains(
        ukraine_pattern, case=False, na=False
    )
    | df["Disinformation"].str.contains(
        ukraine_pattern, case=False, na=False
    )
    | df["Country"].fillna("").str.contains(
        "Ukraine", case=False
    )
)

df = df[topic_mask].copy()

df = df.drop_duplicates(subset=["Title"])

result = pd.DataFrame({
    "claim": df["Title"],
    "label": "FALSE",
    "date": df["date"],
    "original_text": df["Disinformation"],
    "factcheck_text": df["Information"],
    "factcheck_url": (
        "https://euvsdisinfo.eu"
        + df["Links"].astype(str)
    )
})

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT, index=False)

print("Selected records:", len(result))
print(result.head())

print("\n=== DATASET SUMMARY ===")
print("Total records:", len(result))
print("\nLabels:")
print(result["label"].value_counts())

print("\nMissing values:")
print(result.isnull().sum())

print("\nDuplicate claims:")
print(result["claim"].duplicated().sum())

print("\nDate range:")
print(result["date"].min(), "→", result["date"].max())
