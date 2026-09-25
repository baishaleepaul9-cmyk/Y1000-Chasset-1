from pathlib import Path
import pandas as pd
import numpy as np

# ================================================================
# PATHS
# ================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

FEATURE_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_importance_analysis"
)

OUTPUT_DIR = FEATURE_DIR / "robust_feature_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ================================================================
# INPUT
# ================================================================

INPUT_FILE = (
    FEATURE_DIR
    / "full_dataset_busco_feature_importance_420.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("RANDOM vs PHYLOGENETIC FEATURE ROBUSTNESS ANALYSIS")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head().to_string(index=False))

# ================================================================
# IDENTIFY COLUMNS
# ================================================================

print("\n" + "=" * 80)
print("IDENTIFYING FEATURE / MODEL / PHENOTYPE COLUMNS")
print("=" * 80)

for col in df.columns:
    print(col)

# ================================================================
# NORMALIZE COLUMN NAMES
# ================================================================

df.columns = [
    str(c).strip()
    for c in df.columns
]

# ================================================================
# DETECT IMPORTANT COLUMNS
# ================================================================

feature_candidates = [
    c for c in df.columns
    if c.lower() in {
        "busco",
        "busco_id",
        "feature",
        "feature_name",
        "gene"
    }
]

model_candidates = [
    c for c in df.columns
    if c.lower() in {
        "model",
        "estimator"
    }
]

phenotype_candidates = [
    c for c in df.columns
    if c.lower() in {
        "phenotype",
        "target",
        "trait"
    }
]

importance_candidates = [
    c for c in df.columns
    if "importance" in c.lower()
]

print("\nFeature candidates:")
print(feature_candidates)

print("\nModel candidates:")
print(model_candidates)

print("\nPhenotype candidates:")
print(phenotype_candidates)

print("\nImportance candidates:")
print(importance_candidates)

# ================================================================
# STOP SAFELY IF STRUCTURE IS DIFFERENT
# ================================================================

if not feature_candidates:
    raise RuntimeError(
        "\nCould not identify BUSCO/feature column."
        "\nThe actual column names above must be inspected."
    )

if not importance_candidates:
    raise RuntimeError(
        "\nCould not identify importance column."
        "\nThe actual column names above must be inspected."
    )

FEATURE_COL = feature_candidates[0]
IMPORTANCE_COL = importance_candidates[0]

print("\nUsing:")
print("Feature column:", FEATURE_COL)
print("Importance column:", IMPORTANCE_COL)

# ================================================================
# BASIC CLEANING
# ================================================================

df[IMPORTANCE_COL] = pd.to_numeric(
    df[IMPORTANCE_COL],
    errors="coerce"
)

df = df.dropna(
    subset=[FEATURE_COL, IMPORTANCE_COL]
).copy()

df[FEATURE_COL] = (
    df[FEATURE_COL]
    .astype(str)
    .str.strip()
)

# ================================================================
# ABSOLUTE IMPORTANCE
# ================================================================

df["Absolute_Importance"] = (
    df[IMPORTANCE_COL].abs()
)

# ================================================================
# RANK FEATURES
# ================================================================

df["Importance_Rank"] = (
    df.groupby(
        phenotype_candidates[0]
        if phenotype_candidates
        else lambda x: True
    )[IMPORTANCE_COL]
    .rank(
        ascending=False,
        method="average"
    )
)

# ================================================================
# SAVE CLEANED FEATURE TABLE
# ================================================================

clean_file = (
    OUTPUT_DIR
    / "cleaned_full_dataset_feature_importance_420.csv"
)

df.to_csv(
    clean_file,
    index=False
)

print("\n✓ Cleaned feature table written:")
print(clean_file)

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE SUMMARY")
print("=" * 80)

print(
    df.sort_values(
        "Absolute_Importance",
        ascending=False
    ).head(30).to_string(index=False)
)

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)