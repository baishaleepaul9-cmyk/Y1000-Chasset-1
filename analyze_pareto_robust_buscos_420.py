from pathlib import Path
import pandas as pd
import numpy as np


# ================================================================
# PARETO-ASSOCIATED ROBUST BUSCO ANALYSIS
# ================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")


# ================================================================
# PATHS
# ================================================================

ROBUST_DIR = (
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


BUSCO_MATRIX = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "genomic_features"
    / "y1000_busco_complete_matrix.csv"
)


PARETO_FILE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "pareto_qc"
    / "pareto_420_species_level_reconciliation.csv"
)


OUTPUT_DIR = (
    ROBUST_DIR
    / "pareto_busco_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ROBUST_FILE = (
    ROBUST_DIR
    / "robust_all_three_buscos_420.csv"
)


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("PARETO-ASSOCIATED ROBUST BUSCO ANALYSIS")
print("=" * 80)


# ================================================================
# CHECK FILES
# ================================================================

print("\n" + "=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)


INPUT_FILES = {
    "Robust BUSCO candidates": ROBUST_FILE,
    "Complete BUSCO matrix": BUSCO_MATRIX,
    "Pareto reconciliation": PARETO_FILE,
}


for label, filepath in INPUT_FILES.items():

    print("\nChecking:")
    print(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{filepath}"
        )

    print("✓ Found")


# ================================================================
# LOAD ROBUST BUSCOs
# ================================================================

print("\n" + "=" * 80)
print("LOADING ROBUST BUSCO CANDIDATES")
print("=" * 80)


robust = pd.read_csv(ROBUST_FILE)

print("\nShape:")
print(robust.shape)

print("\nColumns:")
print(robust.columns.tolist())


if "BUSCO" not in robust.columns:
    raise RuntimeError(
        "Robust candidate file does not contain a BUSCO column."
    )


robust["BUSCO"] = (
    robust["BUSCO"]
    .astype(str)
    .str.strip()
)


robust_buscos = (
    robust["BUSCO"]
    .dropna()
    .unique()
    .tolist()
)


print("\nNumber of robust BUSCOs:")
print(len(robust_buscos))


# ================================================================
# LOAD COMPLETE BUSCO MATRIX
# ================================================================

print("\n" + "=" * 80)
print("LOADING COMPLETE BUSCO MATRIX")
print("=" * 80)


busco = pd.read_csv(BUSCO_MATRIX)

print("\nShape:")
print(busco.shape)

print("\nFirst columns:")
print(
    busco.columns[:10].tolist()
)


# ================================================================
# CHECK BUSCO MATRIX
# ================================================================

required_metadata = [
    "Species",
    "Assembly_Accession"
]


for col in required_metadata:

    if col not in busco.columns:
        raise RuntimeError(
            f"BUSCO matrix does not contain required column: {col}"
        )


metadata_columns = [
    "Species",
    "Assembly_Accession"
]


busco_columns = [
    c for c in busco.columns
    if c not in metadata_columns
]


print("\nTotal BUSCO columns:")
print(len(busco_columns))


# ================================================================
# CHECK ROBUST BUSCO AVAILABILITY
# ================================================================

print("\n" + "=" * 80)
print("CHECKING ROBUST BUSCO AVAILABILITY")
print("=" * 80)


available_buscos = [
    b for b in robust_buscos
    if b in busco_columns
]


missing_buscos = [
    b for b in robust_buscos
    if b not in busco_columns
]


print("\nRobust BUSCOs:")
print(len(robust_buscos))

print("Available in complete matrix:")
print(len(available_buscos))

print("Missing from complete matrix:")
print(len(missing_buscos))


if missing_buscos:

    print("\nMissing BUSCOs:")

    for b in missing_buscos:
        print(b)


if len(available_buscos) == 0:
    raise RuntimeError(
        "None of the robust BUSCOs were found in the complete BUSCO matrix."
    )


# ================================================================
# LOAD PARETO RECONCILIATION
# ================================================================

print("\n" + "=" * 80)
print("LOADING PARETO RECONCILIATION")
print("=" * 80)


pareto = pd.read_csv(PARETO_FILE)

print("\nShape:")
print(pareto.shape)

print("\nColumns:")
print(pareto.columns.tolist())


# ================================================================
# ACTUAL PARETO FILE STRUCTURE
# ================================================================
#
# The actual file uses:
#
#   Status
#   Matched_Assembly_Accession
#   Matched_Species
#
# rather than Present_in_420.
#
# Therefore:
#
#   Status == matched
#
# is used to identify Pareto taxa represented
# in the 420-taxon ML dataset.
# ================================================================


required_pareto_columns = [
    "Pareto_Assembly_Accession",
    "Pareto_Species",
    "Status",
    "Matched_Assembly_Accession",
    "Matched_Species"
]


for col in required_pareto_columns:

    if col not in pareto.columns:

        raise RuntimeError(
            f"Pareto file does not contain required column: {col}"
        )


# ================================================================
# INSPECT STATUS VALUES
# ================================================================

print("\n" + "=" * 80)
print("PARETO MATCH STATUS")
print("=" * 80)


print(
    pareto["Status"]
    .value_counts(dropna=False)
    .to_string()
)


# ================================================================
# SELECT MATCHED PARETO TAXA
# ================================================================

#
# We use the presence of Matched_Assembly_Accession rather than
# assuming a particular spelling of Status.
#
# This is safer because the actual file already contains the
# resolved accession for successfully matched taxa.
#


pareto_420 = pareto[
    pareto["Matched_Assembly_Accession"].notna()
].copy()


print("\nTotal Pareto candidates:")
print(len(pareto))


print("\nPareto candidates matched to 420:")
print(len(pareto_420))


print("\nMatched Pareto taxa:")


print(
    pareto_420[
        [
            "Pareto_Species",
            "Matched_Assembly_Accession",
            "Matched_Species"
        ]
    ].to_string(index=False)
)


# ================================================================
# EXTRACT MATCHED ACCESSIONS
# ================================================================

pareto_accessions = (
    pareto_420[
        "Matched_Assembly_Accession"
    ]
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


print("\nUnique matched Pareto accessions:")
print(len(pareto_accessions))


# ================================================================
# MATCH AGAINST BUSCO MATRIX
# ================================================================

print("\n" + "=" * 80)
print("MATCHING PARETO TAXA TO BUSCO MATRIX")
print("=" * 80)


busco["Assembly_Accession"] = (
    busco["Assembly_Accession"]
    .astype(str)
    .str.strip()
)


pareto_matrix = busco[
    busco["Assembly_Accession"]
    .isin(pareto_accessions)
].copy()


print("\nMatched Pareto taxa in BUSCO matrix:")
print(len(pareto_matrix))


# ================================================================
# FIND MISSING PARETO TAXA
# ================================================================

found_accessions = set(
    pareto_matrix[
        "Assembly_Accession"
    ].astype(str)
)


missing_pareto = [
    accession
    for accession in pareto_accessions
    if accession not in found_accessions
]


print("\nMissing Pareto accessions from BUSCO matrix:")
print(len(missing_pareto))


if missing_pareto:

    print("\nMissing accessions:")

    for accession in missing_pareto:
        print(accession)


# ================================================================
# SAVE PARETO MATCHING
# ================================================================

pareto_matching = pareto.copy()


pareto_matching["BUSCO_Matrix_Matched"] = (
    pareto_matching[
        "Matched_Assembly_Accession"
    ]
    .astype(str)
    .isin(found_accessions)
)


matching_file = (
    OUTPUT_DIR
    / "pareto_busco_matrix_matching_420.csv"
)


pareto_matching.to_csv(
    matching_file,
    index=False
)


print("\n✓ Pareto matching table written:")
print(matching_file)


# ================================================================
# EXTRACT ROBUST BUSCO MATRIX
# ================================================================

print("\n" + "=" * 80)
print("EXTRACTING ROBUST BUSCOs FOR PARETO TAXA")
print("=" * 80)


robust_pareto = pareto_matrix[
    [
        "Species",
        "Assembly_Accession"
    ]
    + available_buscos
].copy()


print("\nRobust Pareto matrix shape:")
print(robust_pareto.shape)


# ================================================================
# SAVE ROBUST PARETO MATRIX
# ================================================================

matrix_file = (
    OUTPUT_DIR
    / "pareto_robust_busco_matrix.csv"
)


robust_pareto.to_csv(
    matrix_file,
    index=False
)


print("\n✓ Pareto robust BUSCO matrix written:")
print(matrix_file)


# ================================================================
# CALCULATE PARETO BUSCO OCCUPANCY
# ================================================================

print("\n" + "=" * 80)
print("CALCULATING PARETO BUSCO OCCUPANCY")
print("=" * 80)


occupancy_records = []


for busco_id in available_buscos:

    values = pd.to_numeric(
        robust_pareto[busco_id],
        errors="coerce"
    )


    n_present = int(
        (values > 0).sum()
    )


    n_total = len(values)


    occupancy = (
        n_present / n_total
        if n_total > 0
        else np.nan
    )


    occupancy_records.append(
        {
            "BUSCO": busco_id,
            "Pareto_Taxa": n_total,
            "Pareto_Present": n_present,
            "Pareto_Absent": n_total - n_present,
            "Pareto_Occupancy": occupancy,
            "Pareto_Occupancy_Percent": (
                occupancy * 100
                if not np.isnan(occupancy)
                else np.nan
            )
        }
    )


occupancy_df = pd.DataFrame(
    occupancy_records
)


# ================================================================
# MERGE ROBUST FEATURE INFORMATION
# ================================================================

print("\n" + "=" * 80)
print("INTEGRATING ROBUST FEATURE INFORMATION")
print("=" * 80)


occupancy_df = occupancy_df.merge(
    robust,
    on="BUSCO",
    how="left"
)


print("\nIntegrated table shape:")
print(occupancy_df.shape)


# ================================================================
# SAVE OCCUPANCY
# ================================================================

occupancy_file = (
    OUTPUT_DIR
    / "pareto_robust_busco_occupancy_420.csv"
)


occupancy_df.to_csv(
    occupancy_file,
    index=False
)


print("\n✓ Pareto BUSCO occupancy table written:")
print(occupancy_file)


# ================================================================
# RANK BY OBSERVED EVIDENCE
# ================================================================

ranked = occupancy_df.sort_values(
    [
        "Pareto_Occupancy",
        "N_Model_Phenotypes",
        "Mean_Importance"
    ],
    ascending=[
        False,
        False,
        False
    ]
).copy()


ranked_file = (
    OUTPUT_DIR
    / "ranked_pareto_robust_buscos_420.csv"
)


ranked.to_csv(
    ranked_file,
    index=False
)


print("\n✓ Ranked Pareto BUSCO table written:")
print(ranked_file)


# ================================================================
# 100% PARETO OCCUPANCY
# ================================================================

pareto_complete = ranked[
    ranked["Pareto_Occupancy"] == 1.0
].copy()


complete_file = (
    OUTPUT_DIR
    / "pareto_complete_occupancy_robust_buscos_420.csv"
)


pareto_complete.to_csv(
    complete_file,
    index=False
)


# ================================================================
# >=80% PARETO OCCUPANCY
# ================================================================

pareto_high = ranked[
    ranked["Pareto_Occupancy"] >= 0.80
].copy()


high_file = (
    OUTPUT_DIR
    / "pareto_high_occupancy_robust_buscos_420.csv"
)


pareto_high.to_csv(
    high_file,
    index=False
)


print("\n✓ Complete-occupancy BUSCOs:")
print(complete_file)


print("\n✓ High-occupancy BUSCOs:")
print(high_file)


# ================================================================
# TOP 30
# ================================================================

print("\n" + "=" * 80)
print("TOP ROBUST BUSCOs IN PARETO TAXA")
print("=" * 80)


display_columns = [
    "BUSCO",
    "Pareto_Present",
    "Pareto_Taxa",
    "Pareto_Occupancy",
    "Pareto_Occupancy_Percent",
    "Phenotypes_Associated",
    "N_Model_Phenotypes",
    "Mean_Importance",
    "Max_Importance",
    "Phylogenetic_Concentration_Index"
]


available_display_columns = [
    c
    for c in display_columns
    if c in ranked.columns
]


print(
    ranked[
        available_display_columns
    ]
    .head(30)
    .to_string(index=False)
)


# ================================================================
# MATCHED PARETO TAXA TABLE
# ================================================================

taxon_summary = pareto_matrix[
    [
        "Species",
        "Assembly_Accession"
    ]
].copy()


taxon_summary_file = (
    OUTPUT_DIR
    / "pareto_busco_matched_taxa_420.csv"
)


taxon_summary.to_csv(
    taxon_summary_file,
    index=False
)


print("\n" + "=" * 80)
print("MATCHED PARETO TAXA")
print("=" * 80)


print(
    taxon_summary.to_string(index=False)
)


# ================================================================
# SUMMARY
# ================================================================

summary = pd.DataFrame(
    {
        "Metric": [
            "Robust BUSCO candidates",
            "Robust BUSCOs available in complete matrix",
            "Total Pareto candidates",
            "Pareto candidates matched to 420",
            "Pareto candidates matched in BUSCO matrix",
            "Pareto candidates missing from BUSCO matrix",
            "Robust BUSCOs with 100% Pareto occupancy",
            "Robust BUSCOs with >=80% Pareto occupancy"
        ],
        "Value": [
            len(robust_buscos),
            len(available_buscos),
            len(pareto),
            len(pareto_accessions),
            len(pareto_matrix),
            len(missing_pareto),
            len(pareto_complete),
            len(pareto_high)
        ]
    }
)


summary_file = (
    OUTPUT_DIR
    / "pareto_robust_busco_analysis_summary_420.csv"
)


summary.to_csv(
    summary_file,
    index=False
)


print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)


print(
    summary.to_string(index=False)
)


# ================================================================
# OUTPUT FILES
# ================================================================

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)


for filepath in [
    matching_file,
    matrix_file,
    occupancy_file,
    ranked_file,
    complete_file,
    high_file,
    taxon_summary_file,
    summary_file
]:

    print(filepath)


# ================================================================
# COMPLETE
# ================================================================

print("\n" + "=" * 80)
print("PARETO ROBUST BUSCO ANALYSIS COMPLETE")
print("=" * 80)