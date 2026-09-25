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

ROBUST_DIR = FEATURE_DIR / "robust_feature_analysis"

OUTPUT_DIR = ROBUST_DIR / "integrated_robust_candidates"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ================================================================
# INPUT FILES
# ================================================================

MULTI_TRAIT_FILE = (
    ROBUST_DIR
    / "strong_multi_trait_busco_candidates_420.csv"
)

REPEATED_FILE = (
    FEATURE_DIR
    / "repeatedly_important_buscos_420.csv"
)

PHYLO_FILE = (
    FEATURE_DIR
    / "important_busco_phylogenetic_concentration_420.csv"
)

print("=" * 80)
print("INTEGRATED ROBUST BUSCO CANDIDATE ANALYSIS")
print("=" * 80)

# ================================================================
# CHECK FILES
# ================================================================

for f in [MULTI_TRAIT_FILE, REPEATED_FILE, PHYLO_FILE]:
    print("\nChecking:")
    print(f)

    if not f.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{f}"
        )

    print("✓ Found")

# ================================================================
# LOAD
# ================================================================

multi = pd.read_csv(MULTI_TRAIT_FILE)
repeated = pd.read_csv(REPEATED_FILE)
phylo = pd.read_csv(PHYLO_FILE)

print("\nInput shapes:")
print("Multi-trait:", multi.shape)
print("Repeated:", repeated.shape)
print("Phylogenetic:", phylo.shape)

# ================================================================
# NORMALIZE BUSCO IDs
# ================================================================

for df in [multi, repeated, phylo]:
    df["BUSCO"] = (
        df["BUSCO"]
        .astype(str)
        .str.strip()
    )

# ================================================================
# SETS
# ================================================================

multi_set = set(multi["BUSCO"])
repeated_set = set(repeated["BUSCO"])
phylo_set = set(phylo["BUSCO"])

print("\nBUSCO counts:")
print("Multi-trait candidates:", len(multi_set))
print("Repeatedly important:", len(repeated_set))
print("Phylogenetic analysis:", len(phylo_set))

# ================================================================
# INTERSECTIONS
# ================================================================

multi_repeated = multi_set & repeated_set

multi_phylo = multi_set & phylo_set

all_three = (
    multi_set
    & repeated_set
    & phylo_set
)

print("\n" + "=" * 80)
print("INTERSECTION RESULTS")
print("=" * 80)

print(
    "\nMulti-trait ∩ Repeated importance:",
    len(multi_repeated)
)

print(
    "Multi-trait ∩ Phylogenetic:",
    len(multi_phylo)
)

print(
    "Multi-trait ∩ Repeated ∩ Phylogenetic:",
    len(all_three)
)

# ================================================================
# MERGE ALL AVAILABLE INFORMATION
# ================================================================

integrated = multi.merge(
    repeated,
    on="BUSCO",
    how="left",
    suffixes=(
        "_MultiTrait",
        "_Repeated"
    )
)

integrated = integrated.merge(
    phylo,
    on="BUSCO",
    how="left"
)

# ================================================================
# FLAGS
# ================================================================

integrated["MultiTrait_Candidate"] = (
    integrated["BUSCO"].isin(multi_set)
)

integrated["Repeatedly_Important"] = (
    integrated["BUSCO"].isin(repeated_set)
)

integrated["Phylogenetic_Analyzed"] = (
    integrated["BUSCO"].isin(phylo_set)
)

integrated["Robust_All_Three"] = (
    integrated["BUSCO"].isin(all_three)
)

# ================================================================
# SAVE FULL INTEGRATED TABLE
# ================================================================

full_output = (
    OUTPUT_DIR
    / "integrated_robust_busco_candidates_420.csv"
)

integrated.to_csv(
    full_output,
    index=False
)

print("\n✓ Full integrated table:")
print(full_output)

# ================================================================
# SAVE MULTI-TRAIT + REPEATED
# ================================================================

multi_repeated_df = (
    integrated[
        integrated["BUSCO"].isin(multi_repeated)
    ]
    .copy()
)

multi_repeated_file = (
    OUTPUT_DIR
    / "multi_trait_repeated_buscos_420.csv"
)

multi_repeated_df.to_csv(
    multi_repeated_file,
    index=False
)

print("\n✓ Multi-trait + repeated candidates:")
print(multi_repeated_file)

# ================================================================
# SAVE ALL THREE
# ================================================================

all_three_df = (
    integrated[
        integrated["BUSCO"].isin(all_three)
    ]
    .copy()
)

all_three_file = (
    OUTPUT_DIR
    / "robust_all_three_buscos_420.csv"
)

all_three_df.to_csv(
    all_three_file,
    index=False
)

print("\n✓ Robust all-three candidates:")
print(all_three_file)

# ================================================================
# SORT ROBUST CANDIDATES
# ================================================================

if not all_three_df.empty:

    sort_columns = []

    if "N_Model_Phenotypes" in all_three_df.columns:
        sort_columns.append(
            "N_Model_Phenotypes"
        )

    if "Mean_Importance" in all_three_df.columns:
        sort_columns.append(
            "Mean_Importance"
        )

    if "Phylogenetic_Concentration_Index" in all_three_df.columns:
        sort_columns.append(
            "Phylogenetic_Concentration_Index"
        )

    if sort_columns:

        ascending = [
            False
            for _ in sort_columns
        ]

        ranked = (
            all_three_df
            .sort_values(
                sort_columns,
                ascending=ascending
            )
        )

    else:
        ranked = all_three_df.copy()

    ranked_file = (
        OUTPUT_DIR
        / "ranked_robust_all_three_buscos_420.csv"
    )

    ranked.to_csv(
        ranked_file,
        index=False
    )

    print("\n✓ Ranked robust candidates:")
    print(ranked_file)

    print("\n" + "=" * 80)
    print("TOP ROBUST BUSCO CANDIDATES")
    print("=" * 80)

    print(
        ranked.head(30).to_string(
            index=False
        )
    )

else:

    print(
        "\n⚠ No BUSCOs were found in all three sets."
    )

# ================================================================
# SUMMARY
# ================================================================

summary = pd.DataFrame({
    "Category": [
        "Multi-trait",
        "Repeatedly important",
        "Phylogenetic analysis",
        "Multi-trait + repeated",
        "Multi-trait + phylogenetic",
        "All three"
    ],
    "N_BUSCOs": [
        len(multi_set),
        len(repeated_set),
        len(phylo_set),
        len(multi_repeated),
        len(multi_phylo),
        len(all_three)
    ]
})

summary_file = (
    OUTPUT_DIR
    / "robust_busco_intersection_summary_420.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print("\n✓ Summary:")
print(summary_file)

print("\n")
print(summary.to_string(index=False))

print("\n" + "=" * 80)
print("INTEGRATED ROBUST BUSCO ANALYSIS COMPLETE")
print("=" * 80)