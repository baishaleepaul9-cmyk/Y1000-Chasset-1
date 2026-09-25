from pathlib import Path
import pandas as pd

PHENOTYPE_FILE = Path(
    r"C:\Y1000_chassis_project\results\stage2A_multitrait_profile\species_multitrait_profile.csv"
)

OUT = Path(
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\genome_phenotype_reconciliation"
)

OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PHENOTYPE_FILE)

print("=" * 80)
print("PHENOTYPE DUPLICATE INVESTIGATION")
print("=" * 80)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Columns:")
for c in df.columns:
    print(" ", c)

# Normalize species
df["_species_key"] = (
    df["Species"]
    .astype(str)
    .str.strip()
    .str.replace("_", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.lower()
)

counts = (
    df["_species_key"]
    .value_counts()
    .rename_axis("Species_Key")
    .reset_index(name="Phenotype_Row_Count")
)

duplicates = counts[
    counts["Phenotype_Row_Count"] > 1
].copy()

print()
print("=" * 80)
print("DUPLICATE SUMMARY")
print("=" * 80)

print()
print("Unique species:", counts.shape[0])
print("Species with multiple rows:", len(duplicates))

print()
print("Total rows belonging to duplicated species:",
      int(
          df["_species_key"].isin(
              duplicates["Species_Key"]
          ).sum()
      )
)

duplicates.to_csv(
    OUT / "phenotype_duplicate_species_summary.csv",
    index=False
)

duplicate_rows = df[
    df["_species_key"].isin(
        duplicates["Species_Key"]
    )
].copy()

duplicate_rows.to_csv(
    OUT / "phenotype_duplicate_rows.csv",
    index=False
)

print()
print("Output:")
print(
    OUT / "phenotype_duplicate_species_summary.csv"
)
print(
    OUT / "phenotype_duplicate_rows.csv"
)

print()
print("=" * 80)
print("DUPLICATED SPECIES")
print("=" * 80)

for _, row in duplicates.iterrows():
    print(
        f"{row['Species_Key']:<60} "
        f"{row['Phenotype_Row_Count']} rows"
    )

print()
print("✓ Duplicate investigation complete.")