from pathlib import Path
import pandas as pd
import numpy as np

# ================================================================
# FINAL PARETO–ROBUST BUSCO PRIORITIZATION
# Y1000+ Yeast Chassis Project
# ================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

BASE_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_importance_analysis"
    / "robust_feature_analysis"
    / "integrated_robust_candidates"
)

PARETO_DIR = BASE_DIR / "pareto_busco_analysis"

OUTPUT_DIR = BASE_DIR / "final_candidate_prioritization"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ================================================================
# INPUT FILES
# ================================================================

ROBUST_FILE = (
    BASE_DIR
    / "robust_all_three_buscos_420.csv"
)

PARETO_OCCUPANCY_FILE = (
    PARETO_DIR
    / "pareto_robust_busco_occupancy_420.csv"
)

RANKED_PARETO_FILE = (
    PARETO_DIR
    / "ranked_pareto_robust_buscos_420.csv"
)

PARETO_MATRIX_FILE = (
    PARETO_DIR
    / "pareto_robust_busco_matrix.csv"
)

# ================================================================
# CHECK INPUTS
# ================================================================

print("=" * 80)
print("FINAL PARETO–ROBUST BUSCO PRIORITIZATION")
print("=" * 80)

input_files = {
    "Robust BUSCO candidates": ROBUST_FILE,
    "Pareto occupancy table": PARETO_OCCUPANCY_FILE,
    "Ranked Pareto table": RANKED_PARETO_FILE,
    "Pareto BUSCO matrix": PARETO_MATRIX_FILE,
}

for name, path in input_files.items():

    print("\nChecking:")
    print(path)

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print("✓ Found")

# ================================================================
# LOAD ROBUST BUSCO TABLE
# ================================================================

print("\n" + "=" * 80)
print("LOADING ROBUST BUSCO CANDIDATES")
print("=" * 80)

robust = pd.read_csv(ROBUST_FILE)

print("\nShape:")
print(robust.shape)

print("\nColumns:")
print(robust.columns.tolist())

# ================================================================
# LOAD PARETO OCCUPANCY
# ================================================================

print("\n" + "=" * 80)
print("LOADING PARETO OCCUPANCY")
print("=" * 80)

pareto = pd.read_csv(PARETO_OCCUPANCY_FILE)

print("\nShape:")
print(pareto.shape)

print("\nColumns:")
print(pareto.columns.tolist())

# ================================================================
# LOAD RANKED PARETO TABLE
# ================================================================

print("\n" + "=" * 80)
print("LOADING RANKED PARETO TABLE")
print("=" * 80)

ranked = pd.read_csv(RANKED_PARETO_FILE)

print("\nShape:")
print(ranked.shape)

print("\nColumns:")
print(ranked.columns.tolist())

# ================================================================
# STANDARDIZE COLUMN NAMES
# ================================================================

for table in [robust, pareto, ranked]:

    table.columns = [
        str(c).strip()
        for c in table.columns
    ]

# ================================================================
# CHECK BUSCO COLUMN
# ================================================================

for name, table in [
    ("robust", robust),
    ("pareto", pareto),
    ("ranked", ranked)
]:

    if "BUSCO" not in table.columns:

        raise RuntimeError(
            f"\n{name} table does not contain BUSCO column."
        )

# ================================================================
# REMOVE DUPLICATES
# ================================================================

robust = robust.drop_duplicates(
    subset=["BUSCO"]
).copy()

pareto = pareto.drop_duplicates(
    subset=["BUSCO"]
).copy()

ranked = ranked.drop_duplicates(
    subset=["BUSCO"]
).copy()

print("\nUnique BUSCOs:")

print(
    "Robust:",
    robust["BUSCO"].nunique()
)

print(
    "Pareto:",
    pareto["BUSCO"].nunique()
)

print(
    "Ranked:",
    ranked["BUSCO"].nunique()
)

# ================================================================
# MERGE ROBUST + PARETO INFORMATION
# ================================================================

print("\n" + "=" * 80)
print("INTEGRATING ROBUST + PARETO INFORMATION")
print("=" * 80)

# ------------------------------------------------
# Select useful columns from robust table
# ------------------------------------------------

robust_cols = [
    "BUSCO",
    "Phenotypes_Associated",
    "Total_TopN_Appearances",
    "Mean_Importance_All_Traits",
    "Mean_Rank_All_Traits",
    "Multi_Trait",
    "N_Model_Phenotypes",
    "Mean_Importance",
    "Max_Importance",
    "N_Present_Taxa",
    "Prevalence",
    "Phylogenetic_Concentration_Index",
    "MultiTrait_Candidate",
    "Repeatedly_Important",
    "Phylogenetic_Analyzed",
    "Robust_All_Three",
]

robust_cols = [
    c for c in robust_cols
    if c in robust.columns
]

robust_sub = robust[robust_cols].copy()

# ------------------------------------------------
# Select Pareto columns
# ------------------------------------------------

pareto_cols = [
    "BUSCO",
    "Pareto_Present",
    "Pareto_Taxa",
    "Pareto_Occupancy",
    "Pareto_Occupancy_Percent",
]

pareto_cols = [
    c for c in pareto_cols
    if c in pareto.columns
]

pareto_sub = pareto[pareto_cols].copy()

# ================================================================
# MERGE
# ================================================================

integrated = pd.merge(
    robust_sub,
    pareto_sub,
    on="BUSCO",
    how="inner"
)

print("\nIntegrated shape:")
print(integrated.shape)

print("\nIntegrated columns:")
print(integrated.columns.tolist())

# ================================================================
# NUMERIC CLEANING
# ================================================================

numeric_cols = [
    "Phenotypes_Associated",
    "Total_TopN_Appearances",
    "Mean_Importance_All_Traits",
    "Mean_Rank_All_Traits",
    "N_Model_Phenotypes",
    "Mean_Importance",
    "Max_Importance",
    "N_Present_Taxa",
    "Prevalence",
    "Phylogenetic_Concentration_Index",
    "Pareto_Present",
    "Pareto_Taxa",
    "Pareto_Occupancy",
    "Pareto_Occupancy_Percent",
]

for col in numeric_cols:

    if col in integrated.columns:

        integrated[col] = pd.to_numeric(
            integrated[col],
            errors="coerce"
        )

# ================================================================
# CREATE COMPONENT SCORES
# ================================================================

print("\n" + "=" * 80)
print("CALCULATING PRIORITIZATION COMPONENTS")
print("=" * 80)

# ------------------------------------------------
# 1. Pareto occupancy score
# ------------------------------------------------

integrated["Pareto_Occupancy_Score"] = (
    integrated["Pareto_Occupancy"]
)

# ------------------------------------------------
# 2. Multi-trait score
# ------------------------------------------------

integrated["MultiTrait_Score"] = (
    integrated["Phenotypes_Associated"]
    / 3.0
)

# ------------------------------------------------
# 3. Model consistency score
# ------------------------------------------------

# Maximum possible model-phenotype combinations:
# 3 models × 3 phenotypes = 9

integrated["Model_Consistency_Score"] = (
    integrated["N_Model_Phenotypes"]
    / 9.0
)

# ------------------------------------------------
# 4. Repeated top-N appearance score
# ------------------------------------------------

# Robust candidates were derived from repeated
# top-N feature appearances.

max_appearances = (
    integrated["Total_TopN_Appearances"].max()
)

if max_appearances > 0:

    integrated["Repeated_Importance_Score"] = (
        integrated["Total_TopN_Appearances"]
        / max_appearances
    )

else:

    integrated["Repeated_Importance_Score"] = 0.0

# ------------------------------------------------
# 5. ML importance score
# ------------------------------------------------

max_importance = (
    integrated["Mean_Importance"].max()
)

if max_importance > 0:

    integrated["ML_Importance_Score"] = (
        integrated["Mean_Importance"]
        / max_importance
    )

else:

    integrated["ML_Importance_Score"] = 0.0

# ------------------------------------------------
# 6. Phylogenetic concentration
# ------------------------------------------------

# Lower Normalized Mean Distance corresponds to
# stronger phylogenetic concentration.
#
# Therefore use inverse concentration where possible.

if "Phylogenetic_Concentration_Index" in integrated.columns:

    concentration = (
        integrated["Phylogenetic_Concentration_Index"]
    )

    # The index is treated as an existing
    # concentration measure rather than recalculated.

    integrated["Phylogenetic_Score"] = (
        concentration.clip(lower=0)
    )

else:

    integrated["Phylogenetic_Score"] = np.nan

# ================================================================
# FINAL PRIORITIZATION SCORE
# ================================================================

print("\n" + "=" * 80)
print("CALCULATING FINAL PRIORITIZATION SCORE")
print("=" * 80)

# IMPORTANT:
# This is a prioritization index for downstream analysis.
# It is NOT a biological causal score.

weights = {
    "Pareto_Occupancy_Score": 0.30,
    "MultiTrait_Score": 0.20,
    "Model_Consistency_Score": 0.15,
    "Repeated_Importance_Score": 0.15,
    "ML_Importance_Score": 0.15,
    "Phylogenetic_Score": 0.05,
}

# ------------------------------------------------
# Normalize phylogenetic score safely
# ------------------------------------------------

if (
    integrated["Phylogenetic_Score"]
    .notna()
    .any()
):

    pmin = integrated["Phylogenetic_Score"].min()
    pmax = integrated["Phylogenetic_Score"].max()

    if pmax > pmin:

        integrated["Phylogenetic_Score_Normalized"] = (
            (
                integrated["Phylogenetic_Score"]
                - pmin
            )
            /
            (
                pmax
                - pmin
            )
        )

    else:

        integrated["Phylogenetic_Score_Normalized"] = 0.0

else:

    integrated["Phylogenetic_Score_Normalized"] = 0.0

# ------------------------------------------------
# Rebuild score using normalized phylogenetic score
# ------------------------------------------------

integrated["Final_Prioritization_Score"] = (

    weights["Pareto_Occupancy_Score"]
    * integrated["Pareto_Occupancy_Score"]

    +

    weights["MultiTrait_Score"]
    * integrated["MultiTrait_Score"]

    +

    weights["Model_Consistency_Score"]
    * integrated["Model_Consistency_Score"]

    +

    weights["Repeated_Importance_Score"]
    * integrated["Repeated_Importance_Score"]

    +

    weights["ML_Importance_Score"]
    * integrated["ML_Importance_Score"]

    +

    weights["Phylogenetic_Score"]
    * integrated["Phylogenetic_Score_Normalized"]
)

# ================================================================
# FINAL RANK
# ================================================================

integrated = integrated.sort_values(
    "Final_Prioritization_Score",
    ascending=False
).reset_index(drop=True)

integrated["Final_Priority_Rank"] = (
    np.arange(1, len(integrated) + 1)
)

# ================================================================
# CANDIDATE CLASSIFICATION
# ================================================================

def classify_candidate(row):

    occupancy = row["Pareto_Occupancy"]

    phenotypes = row["Phenotypes_Associated"]

    models = row["N_Model_Phenotypes"]

    if (
        occupancy >= 1.0
        and phenotypes >= 3
        and models >= 6
    ):

        return "Core_MultiTrait_Pareto"

    elif (
        occupancy >= 0.8
        and phenotypes >= 3
    ):

        return "Strong_MultiTrait_Pareto"

    elif (
        occupancy >= 0.8
        and phenotypes >= 2
    ):

        return "MultiTrait_Pareto"

    elif occupancy >= 0.8:

        return "High_Occupancy_Pareto"

    else:

        return "Pareto_Associated"

integrated["Candidate_Class"] = (
    integrated.apply(
        classify_candidate,
        axis=1
    )
)

# ================================================================
# SAVE COMPLETE TABLE
# ================================================================

complete_output = (
    OUTPUT_DIR
    / "final_pareto_robust_busco_prioritization_420.csv"
)

integrated.to_csv(
    complete_output,
    index=False
)

print("\n✓ Complete prioritization table written:")
print(complete_output)

# ================================================================
# TOP 30
# ================================================================

print("\n" + "=" * 80)
print("TOP 30 FINAL PARETO–ROBUST BUSCO CANDIDATES")
print("=" * 80)

display_cols = [
    "Final_Priority_Rank",
    "BUSCO",
    "Candidate_Class",
    "Pareto_Occupancy_Percent",
    "Phenotypes_Associated",
    "N_Model_Phenotypes",
    "Mean_Importance",
    "Max_Importance",
    "Phylogenetic_Concentration_Index",
    "Final_Prioritization_Score",
]

display_cols = [
    c for c in display_cols
    if c in integrated.columns
]

print(
    integrated[
        display_cols
    ]
    .head(30)
    .to_string(index=False)
)

# ================================================================
# CORE CANDIDATES
# ================================================================

core = integrated[
    integrated["Candidate_Class"]
    == "Core_MultiTrait_Pareto"
].copy()

core_output = (
    OUTPUT_DIR
    / "core_multitrait_pareto_buscos_420.csv"
)

core.to_csv(
    core_output,
    index=False
)

print("\n" + "=" * 80)
print("CORE MULTI-TRAIT PARETO CANDIDATES")
print("=" * 80)

print(
    "Number of candidates:",
    len(core)
)

print(
    "\nWritten:"
)

print(core_output)

# ================================================================
# HIGH-OCCUPANCY CANDIDATES
# ================================================================

high = integrated[
    integrated["Pareto_Occupancy"] >= 0.80
].copy()

high_output = (
    OUTPUT_DIR
    / "high_occupancy_pareto_robust_buscos_420.csv"
)

high.to_csv(
    high_output,
    index=False
)

print("\n" + "=" * 80)
print("HIGH-OCCUPANCY PARETO CANDIDATES")
print("=" * 80)

print(
    "Number of candidates:",
    len(high)
)

print(
    "\nWritten:"
)

print(high_output)

# ================================================================
# TOP 10
# ================================================================

top10 = integrated.head(10).copy()

top10_output = (
    OUTPUT_DIR
    / "top10_final_pareto_robust_buscos_420.csv"
)

top10.to_csv(
    top10_output,
    index=False
)

print("\n" + "=" * 80)
print("TOP 10")
print("=" * 80)

print(
    top10[
        display_cols
    ].to_string(index=False)
)

print("\nWritten:")
print(top10_output)

# ================================================================
# SUMMARY
# ================================================================

summary = pd.DataFrame({
    "Metric": [
        "Robust BUSCO candidates",
        "Integrated Pareto BUSCO candidates",
        "Core multi-trait Pareto candidates",
        "High-occupancy Pareto candidates",
        "100% Pareto occupancy candidates",
        ">=80% Pareto occupancy candidates",
    ],

    "Value": [
        len(robust),
        len(integrated),
        len(core),
        len(high),
        (
            integrated["Pareto_Occupancy"]
            >= 1.0
        ).sum(),
        (
            integrated["Pareto_Occupancy"]
            >= 0.8
        ).sum(),
    ]
})

summary_output = (
    OUTPUT_DIR
    / "final_pareto_busco_prioritization_summary_420.csv"
)

summary.to_csv(
    summary_output,
    index=False
)

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    summary.to_string(index=False)
)

print("\n✓ Summary written:")
print(summary_output)

# ================================================================
# OUTPUT DIRECTORY
# ================================================================

print("\n" + "=" * 80)
print("FINAL OUTPUT FILES")
print("=" * 80)

for f in [
    complete_output,
    core_output,
    high_output,
    top10_output,
    summary_output,
]:

    print(f)

print("\n" + "=" * 80)
print("FINAL PARETO–ROBUST BUSCO PRIORITIZATION COMPLETE")
print("=" * 80)