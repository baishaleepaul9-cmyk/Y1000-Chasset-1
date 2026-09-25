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

INPUT_FILE = (
    FEATURE_DIR
    / "full_dataset_busco_feature_importance_420.csv"
)

# ================================================================
# SETTINGS
# ================================================================

TOP_N = 50

EXPECTED_PHENOTYPES = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
]

# ================================================================
# LOAD
# ================================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Could not find:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("=" * 80)
print("ROBUST BUSCO FEATURE PRIORITIZATION")
print("=" * 80)

print("\nInput:")
print(INPUT_FILE)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

# ================================================================
# QC
# ================================================================

required_columns = [
    "Phenotype",
    "Model",
    "BUSCO",
    "Importance"
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing required columns: {missing}"
    )

df["Phenotype"] = (
    df["Phenotype"]
    .astype(str)
    .str.strip()
)

df["Model"] = (
    df["Model"]
    .astype(str)
    .str.strip()
)

df["BUSCO"] = (
    df["BUSCO"]
    .astype(str)
    .str.strip()
)

df["Importance"] = pd.to_numeric(
    df["Importance"],
    errors="coerce"
)

df = df.dropna(
    subset=[
        "Phenotype",
        "Model",
        "BUSCO",
        "Importance"
    ]
).copy()

print("\nPhenotypes:")
print(sorted(df["Phenotype"].unique()))

print("\nModels:")
print(sorted(df["Model"].unique()))

print("\nUnique BUSCOs:")
print(df["BUSCO"].nunique())

# ================================================================
# CHECK EXPECTED PHENOTYPES
# ================================================================

missing_phenotypes = [
    p for p in EXPECTED_PHENOTYPES
    if p not in df["Phenotype"].unique()
]

if missing_phenotypes:
    raise RuntimeError(
        f"Expected phenotypes missing: {missing_phenotypes}"
    )

# ================================================================
# RANK WITHIN PHENOTYPE × MODEL
# ================================================================

print("\n" + "=" * 80)
print("CALCULATING WITHIN-MODEL FEATURE RANKS")
print("=" * 80)

df["Importance_Rank"] = (
    df.groupby(
        ["Phenotype", "Model"]
    )["Importance"]
    .rank(
        ascending=False,
        method="min"
    )
)

df["Top_N"] = (
    df["Importance_Rank"] <= TOP_N
)

# ================================================================
# TOP FEATURES PER PHENOTYPE × MODEL
# ================================================================

top_features = (
    df[df["Top_N"]]
    .sort_values(
        [
            "Phenotype",
            "Model",
            "Importance_Rank"
        ]
    )
    .copy()
)

top_file = (
    OUTPUT_DIR
    / f"top_{TOP_N}_buscos_by_phenotype_model_420.csv"
)

top_features.to_csv(
    top_file,
    index=False
)

print(
    f"\n✓ Top {TOP_N} feature table written:"
)
print(top_file)

# ================================================================
# CROSS-MODEL CONSISTENCY
# ================================================================

print("\n" + "=" * 80)
print("CALCULATING CROSS-MODEL CONSISTENCY")
print("=" * 80)

# Number of distinct models in each phenotype
model_counts = (
    df.groupby("Phenotype")["Model"]
    .nunique()
    .to_dict()
)

# Count how many models place each BUSCO in their top N
consistency = (
    top_features
    .groupby(
        ["Phenotype", "BUSCO"]
    )
    .agg(
        Models_TopN=(
            "Model",
            "nunique"
        ),
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Max_Importance=(
            "Importance",
            "max"
        ),
        Mean_Rank=(
            "Importance_Rank",
            "mean"
        ),
        Best_Rank=(
            "Importance_Rank",
            "min"
        )
    )
    .reset_index()
)

consistency["Total_Models"] = (
    consistency["Phenotype"]
    .map(model_counts)
)

consistency["Model_Consistency"] = (
    consistency["Models_TopN"]
    / consistency["Total_Models"]
)

# ================================================================
# SORT
# ================================================================

consistency = consistency.sort_values(
    [
        "Phenotype",
        "Models_TopN",
        "Mean_Importance"
    ],
    ascending=[
        True,
        False,
        False
    ]
)

consistency_file = (
    OUTPUT_DIR
    / "busco_cross_model_consistency_420.csv"
)

consistency.to_csv(
    consistency_file,
    index=False
)

print("\n✓ Cross-model consistency table written:")
print(consistency_file)

# ================================================================
# MULTI-TRAIT BUSCO ANALYSIS
# ================================================================

print("\n" + "=" * 80)
print("IDENTIFYING MULTI-TRAIT BUSCOs")
print("=" * 80)

# Aggregate across phenotype/model combinations
multi = (
    consistency
    .groupby("BUSCO")
    .agg(
        Phenotypes_Associated=(
            "Phenotype",
            "nunique"
        ),
        Total_TopN_Appearances=(
            "Models_TopN",
            "sum"
        ),
        Mean_Importance_All_Traits=(
            "Mean_Importance",
            "mean"
        ),
        Mean_Rank_All_Traits=(
            "Mean_Rank",
            "mean"
        )
    )
    .reset_index()
)

multi["Multi_Trait"] = (
    multi["Phenotypes_Associated"] >= 2
)

multi = multi.sort_values(
    [
        "Phenotypes_Associated",
        "Total_TopN_Appearances",
        "Mean_Importance_All_Traits"
    ],
    ascending=False
)

multi_file = (
    OUTPUT_DIR
    / "multi_trait_busco_candidates_420.csv"
)

multi.to_csv(
    multi_file,
    index=False
)

print("\n✓ Multi-trait BUSCO table written:")
print(multi_file)

# ================================================================
# STRONG CANDIDATES
# ================================================================

strong_candidates = multi[
    multi["Multi_Trait"]
].copy()

strong_file = (
    OUTPUT_DIR
    / "strong_multi_trait_busco_candidates_420.csv"
)

strong_candidates.to_csv(
    strong_file,
    index=False
)

print(
    "\n✓ Strong multi-trait candidates written:"
)
print(strong_file)

# ================================================================
# SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    f"\nTotal BUSCOs analyzed: "
    f"{df['BUSCO'].nunique()}"
)

print(
    f"Total phenotypes: "
    f"{df['Phenotype'].nunique()}"
)

print(
    f"Total models: "
    f"{df['Model'].nunique()}"
)

print(
    f"Multi-trait BUSCO candidates: "
    f"{len(strong_candidates)}"
)

print("\nTop multi-trait candidates:")

print(
    strong_candidates
    .head(30)
    .to_string(index=False)
)

# ================================================================
# FINAL
# ================================================================

print("\n" + "=" * 80)
print("ROBUST BUSCO PRIORITIZATION COMPLETE")
print("=" * 80)

print("\nOutputs:")
print(top_file)
print(consistency_file)
print(multi_file)
print(strong_file)