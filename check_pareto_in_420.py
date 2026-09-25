# -*- coding: utf-8 -*-

"""
================================================================================
PARETO CANDIDATES ↔ FINAL 420-TAXON ML DATASET
================================================================================

Purpose
-------
Check whether all previously identified Pareto candidates are represented
in the FINAL 420-taxon ML dataset.

This script:
    1. Loads the final 420-taxon variable-BUSCO ML matrix.
    2. Checks all 11 previously identified Pareto candidates.
    3. Matches primarily by Assembly Accession.
    4. Reports phenotype values for matched candidates.
    5. Separates candidates present vs absent.
    6. Does NOT modify the ML dataset.

Output
------
results\stage5_phylogeny_ml_dataset\pareto_qc\
    pareto_membership_420_ml_dataset.csv
    pareto_summary_420.csv
"""

from pathlib import Path
import pandas as pd
import re


# ==============================================================================
# PATHS
# ==============================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

ML_MATRIX = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "feature_qc"
    / "y1000_420_variable_busco_ml_matrix.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "pareto_qc"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# PREVIOUSLY IDENTIFIED PARETO CANDIDATES
# ==============================================================================

PARETO_CANDIDATES = [
    {
        "Assembly_Accession": "GCA_003705225.1",
        "Species": "Ambrosiozyma vanderkliftii",
    },
    {
        "Assembly_Accession": "GCA_003123585.1",
        "Species": "Barnettozyma californica",
    },
    {
        "Assembly_Accession": "GCA_003709245.3",
        "Species": "Cyberlindnera saturnus",
    },
    {
        "Assembly_Accession": "GCA_030558845.1",
        "Species": "Kodamaea laetipori",
    },
    {
        "Assembly_Accession": "GCA_030563145.1",
        "Species": "Schwanniomyces polymorphus var. africanus",
    },
    {
        "Assembly_Accession": "GCA_030463025.1",
        "Species": "Schwanniomyces polymorphus",
    },
    {
        "Assembly_Accession": "GCA_030583345.1",
        "Species": "Schwanniomyces pseudopolymorphus",
    },
    {
        "Assembly_Accession": "GCA_030583405.1",
        "Species": "Sugiyamaella americana",
    },
    {
        "Assembly_Accession": "GCA_030579815.1",
        "Species": "Sugiyamaella smithiae",
    },
    {
        "Assembly_Accession": "GCA_030558095.1",
        "Species": "Teunomyces funiuensis",
    },
    {
        "Assembly_Accession": "GCA_030564625.1",
        "Species": "Zygoascus hellenicus",
    },
]


# ==============================================================================
# HELPERS
# ==============================================================================

def clean_accession(value):
    """
    Normalize assembly accession strings.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip()

    # Keep standard accession/version format.
    match = re.search(r"(GC[AF]_\d+\.\d+)", value)

    if match:
        return match.group(1)

    return value


def normalize_species(value):
    """
    Normalize species names for secondary matching.
    """
    if pd.isna(value):
        return ""

    value = str(value).strip().lower()

    value = re.sub(r"\s+", " ", value)

    return value


# ==============================================================================
# HEADER
# ==============================================================================

print("=" * 80)
print("PARETO CANDIDATES ↔ FINAL 420-TAXON ML DATASET")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("ML dataset:")
print(ML_MATRIX)

print()
print("Output directory:")
print(OUTPUT_DIR)


# ==============================================================================
# CHECK INPUT
# ==============================================================================

if not ML_MATRIX.exists():
    raise FileNotFoundError(
        f"\nFinal 420-taxon ML matrix not found:\n{ML_MATRIX}"
    )


# ==============================================================================
# LOAD ML DATASET
# ==============================================================================

print()
print("=" * 80)
print("LOADING FINAL 420-TAXON ML DATASET")
print("=" * 80)

df = pd.read_csv(ML_MATRIX)

print()
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# ==============================================================================
# BASIC VALIDATION
# ==============================================================================

required_columns = [
    "Species",
    "Assembly_Accession",
]

missing_columns = [
    c for c in required_columns
    if c not in df.columns
]

if missing_columns:
    raise RuntimeError(
        f"Required columns missing from ML dataset: {missing_columns}"
    )


if len(df) != 420:
    raise RuntimeError(
        f"Expected 420 taxa but found {len(df)}."
    )


print()
print("✓ Final ML dataset contains exactly 420 taxa.")


# ==============================================================================
# NORMALIZE IDENTIFIERS
# ==============================================================================

df["_accession_key"] = df["Assembly_Accession"].apply(clean_accession)
df["_species_key"] = df["Species"].apply(normalize_species)


# ==============================================================================
# CHECK DUPLICATES
# ==============================================================================

duplicate_accessions = (
    df["_accession_key"]
    .value_counts()
)

duplicate_accessions = duplicate_accessions[
    duplicate_accessions > 1
]

if len(duplicate_accessions) > 0:
    print()
    print("WARNING: duplicate assembly accessions detected:")
    print(duplicate_accessions.to_string())
else:
    print("✓ Assembly accessions are unique.")


# ==============================================================================
# PARETO MEMBERSHIP
# ==============================================================================

print()
print("=" * 80)
print("PARETO MEMBERSHIP")
print("=" * 80)

results = []

for candidate in PARETO_CANDIDATES:

    accession = candidate["Assembly_Accession"]
    pareto_species = candidate["Species"]

    accession_key = clean_accession(accession)
    species_key = normalize_species(pareto_species)

    # --------------------------------------------------------------------------
    # Primary match: Assembly accession
    # --------------------------------------------------------------------------

    accession_match = df[
        df["_accession_key"] == accession_key
    ]

    # --------------------------------------------------------------------------
    # Secondary match: species name
    # --------------------------------------------------------------------------

    species_match = df[
        df["_species_key"] == species_key
    ]

    # --------------------------------------------------------------------------
    # Determine match
    # --------------------------------------------------------------------------

    if len(accession_match) == 1:

        row = accession_match.iloc[0]

        result = {
            "Pareto_Assembly_Accession": accession,
            "Pareto_Species": pareto_species,
            "Present_in_420": True,
            "Matched_Assembly_Accession": row["Assembly_Accession"],
            "Matched_Species": row["Species"],
            "Match_Type": "Assembly_Accession",
            "N_Strains": row.get("N_Strains", None),
            "Carbon_Breadth": row.get("Carbon_Breadth", None),
            "Nitrogen_Breadth": row.get("Nitrogen_Breadth", None),
            "Utilized_Median_Growth": row.get(
                "Utilized_Median_Growth",
                None
            ),
        }

        print(
            f"✓ {accession}  {pareto_species}"
        )

        print(
            f"    Present in final 420 ML dataset."
        )

        if "Carbon_Breadth" in df.columns:
            print(
                f"    Phenotype: "
                f"Carbon={row['Carbon_Breadth']}, "
                f"Nitrogen={row['Nitrogen_Breadth']}, "
                f"Growth={row['Utilized_Median_Growth']}"
            )

    elif len(accession_match) > 1:

        raise RuntimeError(
            f"Multiple ML rows found for accession {accession}"
        )

    elif len(species_match) == 1:

        row = species_match.iloc[0]

        result = {
            "Pareto_Assembly_Accession": accession,
            "Pareto_Species": pareto_species,
            "Present_in_420": True,
            "Matched_Assembly_Accession": row["Assembly_Accession"],
            "Matched_Species": row["Species"],
            "Match_Type": "Species",
            "N_Strains": row.get("N_Strains", None),
            "Carbon_Breadth": row.get("Carbon_Breadth", None),
            "Nitrogen_Breadth": row.get("Nitrogen_Breadth", None),
            "Utilized_Median_Growth": row.get(
                "Utilized_Median_Growth",
                None
            ),
        }

        print(
            f"✓ {accession}  {pareto_species}"
        )

        print(
            "    Present in final 420 ML dataset "
            "(matched by species)."
        )

    elif len(species_match) > 1:

        raise RuntimeError(
            f"Multiple ML rows found for species {pareto_species}"
        )

    else:

        result = {
            "Pareto_Assembly_Accession": accession,
            "Pareto_Species": pareto_species,
            "Present_in_420": False,
            "Matched_Assembly_Accession": "",
            "Matched_Species": "",
            "Match_Type": "Not found",
            "N_Strains": None,
            "Carbon_Breadth": None,
            "Nitrogen_Breadth": None,
            "Utilized_Median_Growth": None,
        }

        print(
            f"✗ {accession}  {pareto_species}"
        )

        print(
            "    NOT PRESENT in final 420 ML dataset."
        )

    results.append(result)


# ==============================================================================
# RESULTS DATAFRAME
# ==============================================================================

results_df = pd.DataFrame(results)


# ==============================================================================
# SUMMARY
# ==============================================================================

present = results_df[
    results_df["Present_in_420"] == True
]

absent = results_df[
    results_df["Present_in_420"] == False
]


print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print()
print(
    f"Total Pareto candidates:       {len(results_df)}"
)

print(
    f"Present in 420 ML dataset:     {len(present)}"
)

print(
    f"Absent from 420 ML dataset:    {len(absent)}"
)


# ==============================================================================
# ABSENT CANDIDATES
# ==============================================================================

if len(absent) > 0:

    print()
    print("Pareto candidates NOT in 420:")

    for _, row in absent.iterrows():

        print(
            f"  {row['Pareto_Assembly_Accession']} "
            f"{row['Pareto_Species']}"
        )

else:

    print()
    print("✓ ALL Pareto candidates are represented in the 420-taxon dataset.")


# ==============================================================================
# SAVE MEMBERSHIP TABLE
# ==============================================================================

membership_file = (
    OUTPUT_DIR
    / "pareto_membership_420_ml_dataset.csv"
)

results_df.to_csv(
    membership_file,
    index=False
)


# ==============================================================================
# SAVE SUMMARY
# ==============================================================================

summary = pd.DataFrame(
    [
        {
            "Total_Pareto_Candidates": len(results_df),
            "Present_in_420": len(present),
            "Absent_from_420": len(absent),
            "Final_ML_Taxa": len(df),
        }
    ]
)

summary_file = (
    OUTPUT_DIR
    / "pareto_summary_420.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ==============================================================================
# CLEAN TEMPORARY COLUMNS
# ==============================================================================

# These columns only existed in memory and were never written to the
# original ML matrix.

del df["_accession_key"]
del df["_species_key"]


# ==============================================================================
# FINAL REPORT
# ==============================================================================

print()
print("=" * 80)
print("PARETO ↔ 420 QC COMPLETE")
print("=" * 80)

print()
print("Final ML taxa:")
print(f"  {len(df)}")

print()
print("Pareto candidates:")
print(f"  {len(results_df)}")

print()
print("Present:")
print(f"  {len(present)}")

print()
print("Absent:")
print(f"  {len(absent)}")

print()
print("Membership table:")
print(membership_file)

print()
print("Summary:")
print(summary_file)

print()
print("✓ Final 420 ML dataset was NOT modified.")
print("✓ Pareto candidate list was NOT modified.")
print("✓ Pareto membership QC completed.")