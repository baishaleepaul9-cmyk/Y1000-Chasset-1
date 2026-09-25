from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# Y1000+ GENOMIC FEATURE MATRIX — QC
# =============================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

MATRIX_FILE = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "genomic_features"
    / "y1000_busco_presence_absence_matrix.csv"
)

OUTPUT_DIR = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
    / "postbusco_phylogeny_v2"
    / "genomic_features"
    / "qc"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_GENOMES = 436
EXPECTED_FEATURES = 2137

ID_COLUMNS = [
    "Species",
    "Assembly_Accession"
]


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("Y1000+ GENOMIC FEATURE MATRIX QC")
print("=" * 80)

print()
print("Input matrix:")
print(MATRIX_FILE)

print()
print("QC output directory:")
print(OUTPUT_DIR)


# =============================================================================
# CHECK INPUT
# =============================================================================

if not MATRIX_FILE.exists():

    raise FileNotFoundError(
        f"\nMatrix not found:\n{MATRIX_FILE}"
    )


# =============================================================================
# LOAD MATRIX
# =============================================================================

print()
print("=" * 80)
print("LOADING MATRIX")
print("=" * 80)

df = pd.read_csv(
    MATRIX_FILE
)

print()
print(
    f"Rows: {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)


# =============================================================================
# IDENTIFIER CHECK
# =============================================================================

print()
print("=" * 80)
print("IDENTIFIER QC")
print("=" * 80)

missing_id_columns = [
    column
    for column in ID_COLUMNS
    if column not in df.columns
]

if missing_id_columns:

    raise RuntimeError(
        "Missing identifier columns:\n"
        + "\n".join(
            missing_id_columns
        )
    )


print()
print("✓ Species column present")
print("✓ Assembly_Accession column present")


# =============================================================================
# EXPECTED DIMENSIONS
# =============================================================================

print()
print("=" * 80)
print("DIMENSION QC")
print("=" * 80)

n_genomes = len(df)

feature_columns = [
    column
    for column in df.columns
    if column not in ID_COLUMNS
]

n_features = len(
    feature_columns
)

print()
print(
    f"Genomes:          {n_genomes}"
)

print(
    f"BUSCO features:   {n_features}"
)

print(
    f"Expected genomes: {EXPECTED_GENOMES}"
)

print(
    f"Expected features:{EXPECTED_FEATURES}"
)


if n_genomes != EXPECTED_GENOMES:

    print()
    print(
        f"! WARNING: expected {EXPECTED_GENOMES} genomes "
        f"but found {n_genomes}"
    )

else:

    print()
    print(
        "✓ Correct number of genomes"
    )


if n_features != EXPECTED_FEATURES:

    print()
    print(
        f"! WARNING: expected {EXPECTED_FEATURES} features "
        f"but found {n_features}"
    )

else:

    print(
        "✓ Correct number of BUSCO features"
    )


# =============================================================================
# DUPLICATE TAXA
# =============================================================================

print()
print("=" * 80)
print("TAXON QC")
print("=" * 80)

duplicate_species = (
    df["Species"]
    .duplicated()
)

duplicate_accessions = (
    df["Assembly_Accession"]
    .duplicated()
)

print()
print(
    "Duplicate species:",
    int(duplicate_species.sum())
)

print(
    "Duplicate accessions:",
    int(duplicate_accessions.sum())
)


if duplicate_accessions.any():

    print()
    print(
        "Duplicated accessions:"
    )

    print(
        df.loc[
            duplicate_accessions,
            [
                "Species",
                "Assembly_Accession"
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "✓ All assembly accessions are unique"
    )


# =============================================================================
# CHECK FOR MISSING VALUES
# =============================================================================

print()
print("=" * 80)
print("MISSING VALUE QC")
print("=" * 80)

feature_df = df[
    feature_columns
]

total_missing = int(
    feature_df
    .isna()
    .sum()
    .sum()
)

print()
print(
    f"Total missing feature values: "
    f"{total_missing}"
)

if total_missing == 0:

    print(
        "✓ No missing values in genomic features"
    )

else:

    print(
        "! Missing values detected"
    )


# =============================================================================
# CHECK BINARY VALUES
# =============================================================================

print()
print("=" * 80)
print("BINARY VALUE QC")
print("=" * 80)

unique_values = np.unique(
    feature_df.values
)

print()
print(
    "Unique feature values:"
)

print(
    unique_values
)

non_binary = [
    value
    for value in unique_values
    if value not in {0, 1}
]

if not non_binary:

    print()
    print(
        "✓ Matrix contains only 0/1 values"
    )

else:

    print()
    print(
        "! Non-binary values detected:"
    )

    print(
        non_binary
    )


# =============================================================================
# FEATURE PREVALENCE
# =============================================================================

print()
print("=" * 80)
print("FEATURE PREVALENCE")
print("=" * 80)

feature_counts = (
    feature_df
    .sum(axis=0)
)

feature_prevalence = (
    feature_counts
    / n_genomes
    * 100
)

prevalence_df = pd.DataFrame(
    {
        "BUSCO_ID":
            feature_columns,

        "Present_Count":
            feature_counts.values,

        "Present_Percent":
            feature_prevalence.values,

        "Absent_Count":
            n_genomes
            -
            feature_counts.values,

        "Absent_Percent":
            100
            -
            feature_prevalence.values
    }
)

prevalence_df = (
    prevalence_df
    .sort_values(
        "Present_Percent"
    )
)


# =============================================================================
# INVARIANT FEATURES
# =============================================================================

print()
print("=" * 80)
print("INVARIANT FEATURE CHECK")
print("=" * 80)

absent_everywhere = (
    prevalence_df[
        "Present_Count"
    ]
    == 0
)

present_everywhere = (
    prevalence_df[
        "Present_Count"
    ]
    == n_genomes
)

n_absent_everywhere = int(
    absent_everywhere.sum()
)

n_present_everywhere = int(
    present_everywhere.sum()
)

n_invariant = (
    n_absent_everywhere
    +
    n_present_everywhere
)

n_variable = (
    n_features
    -
    n_invariant
)

print()
print(
    f"Features absent in all genomes: "
    f"{n_absent_everywhere}"
)

print(
    f"Features present in all genomes: "
    f"{n_present_everywhere}"
)

print(
    f"Total invariant features: "
    f"{n_invariant}"
)

print(
    f"Variable features: "
    f"{n_variable}"
)


# =============================================================================
# PREVALENCE BINS
# =============================================================================

print()
print("=" * 80)
print("PREVALENCE DISTRIBUTION")
print("=" * 80)


def prevalence_category(percent):

    if percent == 0:
        return "0%"

    elif percent <= 5:
        return "0–5%"

    elif percent <= 25:
        return "5–25%"

    elif percent <= 50:
        return "25–50%"

    elif percent <= 75:
        return "50–75%"

    elif percent < 95:
        return "75–95%"

    elif percent < 100:
        return "95–<100%"

    else:
        return "100%"


prevalence_df[
    "Prevalence_Bin"
] = (
    prevalence_df[
        "Present_Percent"
    ]
    .apply(
        prevalence_category
    )
)


bin_order = [
    "0%",
    "0–5%",
    "5–25%",
    "25–50%",
    "50–75%",
    "75–95%",
    "95–<100%",
    "100%"
]


prevalence_summary = (
    prevalence_df[
        "Prevalence_Bin"
    ]
    .value_counts()
    .reindex(
        bin_order,
        fill_value=0
    )
    .reset_index()
)

prevalence_summary.columns = [
    "Prevalence_Bin",
    "Number_of_Features"
]


for _, row in prevalence_summary.iterrows():

    print(
        f"{row['Prevalence_Bin']:>10} : "
        f"{int(row['Number_of_Features'])}"
    )


# =============================================================================
# RARE FEATURES
# =============================================================================

print()
print("=" * 80)
print("RARE FEATURE CHECK")
print("=" * 80)

rare_5 = prevalence_df[
    prevalence_df[
        "Present_Percent"
    ] <= 5
]

rare_10 = prevalence_df[
    prevalence_df[
        "Present_Percent"
    ] <= 10
]

print()
print(
    f"Features present in ≤5% genomes: "
    f"{len(rare_5)}"
)

print(
    f"Features present in ≤10% genomes: "
    f"{len(rare_10)}"
)


# =============================================================================
# HIGH-PREVALENCE FEATURES
# =============================================================================

print()
print("=" * 80)
print("HIGH-PREVALENCE FEATURE CHECK")
print("=" * 80)

high_95 = prevalence_df[
    prevalence_df[
        "Present_Percent"
    ] > 95
]

high_99 = prevalence_df[
    prevalence_df[
        "Present_Percent"
    ] > 99
]

print()
print(
    f"Features present in >95% genomes: "
    f"{len(high_95)}"
)

print(
    f"Features present in >99% genomes: "
    f"{len(high_99)}"
)


# =============================================================================
# VARIABLE FEATURE SET
# =============================================================================

variable_features = prevalence_df[
    (
        prevalence_df[
            "Present_Count"
        ] > 0
    )
    &
    (
        prevalence_df[
            "Present_Count"
        ] < n_genomes
    )
]

print()
print(
    f"Variable features available: "
    f"{len(variable_features)}"
)


# =============================================================================
# SAVE PREVALENCE TABLE
# =============================================================================

prevalence_file = (
    OUTPUT_DIR
    / "busco_feature_prevalence_qc.csv"
)

prevalence_df.to_csv(
    prevalence_file,
    index=False
)

print()
print(
    "Feature prevalence table:"
)

print(
    prevalence_file
)


# =============================================================================
# SAVE PREVALENCE SUMMARY
# =============================================================================

prevalence_summary_file = (
    OUTPUT_DIR
    / "busco_prevalence_distribution.csv"
)

prevalence_summary.to_csv(
    prevalence_summary_file,
    index=False
)


# =============================================================================
# SAVE INVARIANT FEATURES
# =============================================================================

invariant_df = prevalence_df[
    (
        prevalence_df[
            "Present_Count"
        ]
        == 0
    )
    |
    (
        prevalence_df[
            "Present_Count"
        ]
        == n_genomes
    )
].copy()

invariant_file = (
    OUTPUT_DIR
    / "busco_invariant_features.csv"
)

invariant_df.to_csv(
    invariant_file,
    index=False
)


# =============================================================================
# SAVE VARIABLE FEATURES
# =============================================================================

variable_file = (
    OUTPUT_DIR
    / "busco_variable_features.csv"
)

variable_features.to_csv(
    variable_file,
    index=False
)


# =============================================================================
# SAVE QC SUMMARY
# =============================================================================

qc_summary = pd.DataFrame(
    [
        {
            "Metric":
                "Genomes",

            "Value":
                n_genomes
        },

        {
            "Metric":
                "BUSCO_Features",

            "Value":
                n_features
        },

        {
            "Metric":
                "Missing_Cells",

            "Value":
                total_missing
        },

        {
            "Metric":
                "Duplicate_Species",

            "Value":
                int(
                    duplicate_species.sum()
                )
        },

        {
            "Metric":
                "Duplicate_Accessions",

            "Value":
                int(
                    duplicate_accessions.sum()
                )
        },

        {
            "Metric":
                "Features_Absent_Everywhere",

            "Value":
                n_absent_everywhere
        },

        {
            "Metric":
                "Features_Present_Everywhere",

            "Value":
                n_present_everywhere
        },

        {
            "Metric":
                "Invariant_Features",

            "Value":
                n_invariant
        },

        {
            "Metric":
                "Variable_Features",

            "Value":
                n_variable
        },

        {
            "Metric":
                "Features_Present_<=5pct",

            "Value":
                len(rare_5)
        },

        {
            "Metric":
                "Features_Present_<=10pct",

            "Value":
                len(rare_10)
        },

        {
            "Metric":
                "Features_Present_>95pct",

            "Value":
                len(high_95)
        },

        {
            "Metric":
                "Features_Present_>99pct",

            "Value":
                len(high_99)
        }
    ]
)

summary_file = (
    OUTPUT_DIR
    / "genomic_matrix_qc_summary.csv"
)

qc_summary.to_csv(
    summary_file,
    index=False
)


# =============================================================================
# FINAL REPORT
# =============================================================================

print()
print("=" * 80)
print("QC COMPLETE")
print("=" * 80)

print()
print(
    f"Genomes:                 {n_genomes}"
)

print(
    f"BUSCO features:          {n_features}"
)

print(
    f"Missing cells:            {total_missing}"
)

print(
    f"Invariant features:       {n_invariant}"
)

print(
    f"Variable features:        {n_variable}"
)

print(
    f"Rare (≤5%):               {len(rare_5)}"
)

print(
    f"High (>95%):              {len(high_95)}"
)

print()

if (
    n_genomes == EXPECTED_GENOMES
    and
    n_features == EXPECTED_FEATURES
    and
    total_missing == 0
    and
    not non_binary
    and
    not duplicate_accessions.any()
):

    print(
        "✓ BASIC MATRIX QC PASSED"
    )

else:

    print(
        "! REVIEW QC WARNINGS ABOVE"
    )


print()
print(
    "QC outputs:"
)

print(
    f"  {summary_file}"
)

print(
    f"  {prevalence_file}"
)

print(
    f"  {prevalence_summary_file}"
)

print(
    f"  {invariant_file}"
)

print(
    f"  {variable_file}"
)

print()
print(
    "No input files were modified."
)