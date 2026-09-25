import os
import re
import sys
import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT = r"C:\Y1000_chassis_project"

ML_DATASET = os.path.join(
    PROJECT,
    "results",
    "stage5_phylogeny_ml_dataset",
    "genome_phenotype_reconciliation",
    "genome_phenotype_ml_dataset.csv",
)

OUTPUT_DIR = os.path.join(
    PROJECT,
    "results",
    "stage5_phylogeny_ml_dataset",
    "feature_qc",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Expected final structure
EXPECTED_TAXA = 420
EXPECTED_BUSCO = 2137

IDENTIFIER_COLUMNS = [
    "Species",
    "Assembly_Accession",
]

PHENOTYPE_COLUMNS = [
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
]

SOURCE_COLUMN = "Phenotype_Source_Species"


# =============================================================================
# HELPERS
# =============================================================================

def norm(value):
    if pd.isna(value):
        return ""
    return re.sub(
        r"\s+",
        " ",
        str(value).strip().lower()
    )


def is_busco_id(column):
    """
    BUSCO IDs in this project look like:

        0at4891
        9724at4891
        9995at4891

    Therefore:
        digits + 'at' + digits
    """

    return bool(
        re.fullmatch(
            r"\d+at\d+",
            str(column).strip()
        )
    )


def fail(message):
    print()
    print("ERROR:")
    print(message)
    print()
    sys.exit(1)


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("FINAL 420-TAXON ML DATASET QC")
print("=" * 80)

print()
print("Project:")
print(PROJECT)

print()
print("Input:")
print(ML_DATASET)

print()
print("Output:")
print(OUTPUT_DIR)


# =============================================================================
# CHECK INPUT
# =============================================================================

if not os.path.exists(ML_DATASET):
    fail(
        f"ML dataset does not exist:\n{ML_DATASET}"
    )


# =============================================================================
# LOAD DATASET
# =============================================================================

print()
print("=" * 80)
print("LOADING FINAL ML DATASET")
print("=" * 80)

df = pd.read_csv(ML_DATASET)

print()
print(f"Rows:    {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


# =============================================================================
# IDENTIFY COLUMN TYPES
# =============================================================================

print()
print("=" * 80)
print("COLUMN STRUCTURE")
print("=" * 80)

print()
print("All columns:")
print()

for i, col in enumerate(df.columns, start=1):
    print(f"{i:4d}. {col}")


# =============================================================================
# IDENTIFIER QC
# =============================================================================

print()
print("=" * 80)
print("IDENTIFIER QC")
print("=" * 80)

missing_identifiers = [
    c for c in IDENTIFIER_COLUMNS
    if c not in df.columns
]

if missing_identifiers:
    fail(
        "Missing identifier columns:\n"
        + "\n".join(missing_identifiers)
    )

print("✓ Species column present")
print("✓ Assembly_Accession column present")


# =============================================================================
# PHENOTYPE COLUMN QC
# =============================================================================

print()
print("=" * 80)
print("PHENOTYPE COLUMN QC")
print("=" * 80)

missing_phenotype = [
    c for c in PHENOTYPE_COLUMNS
    if c not in df.columns
]

if missing_phenotype:
    fail(
        "Missing phenotype columns:\n"
        + "\n".join(missing_phenotype)
    )

print("✓ All six phenotype variables present")

if SOURCE_COLUMN in df.columns:
    print("✓ Phenotype_Source_Species present")
else:
    print("⚠ Phenotype_Source_Species not present")


# =============================================================================
# TAXON QC
# =============================================================================

print()
print("=" * 80)
print("TAXON QC")
print("=" * 80)

species = df["Species"].astype(str).str.strip()
accessions = df["Assembly_Accession"].astype(str).str.strip()

duplicate_species = species.duplicated().sum()
duplicate_accessions = accessions.duplicated().sum()

print()
print(f"Taxa:                  {len(df)}")
print(f"Expected taxa:         {EXPECTED_TAXA}")
print(f"Duplicate species:     {duplicate_species}")
print(f"Duplicate accessions:  {duplicate_accessions}")

if len(df) != EXPECTED_TAXA:
    fail(
        f"Expected {EXPECTED_TAXA} taxa but found {len(df)}."
    )

if duplicate_species != 0:
    fail(
        "Duplicate species detected."
    )

if duplicate_accessions != 0:
    fail(
        "Duplicate assembly accessions detected."
    )

print()
print("✓ Exactly 420 unique taxa")
print("✓ Assembly accessions are unique")


# =============================================================================
# BUSCO FEATURE DETECTION
# =============================================================================

print()
print("=" * 80)
print("IDENTIFYING BUSCO FEATURES")
print("=" * 80)

busco_columns = [
    c for c in df.columns
    if is_busco_id(c)
]

non_busco_columns = [
    c for c in df.columns
    if c not in busco_columns
]

print()
print(f"BUSCO-like columns detected: {len(busco_columns)}")

print()
print("Non-BUSCO columns:")

for c in non_busco_columns:
    print(f"  {c}")


# =============================================================================
# BUSCO ID QC
# =============================================================================

print()
print("=" * 80)
print("BUSCO ID QC")
print("=" * 80)

busco_ids = [str(c).strip() for c in busco_columns]

unique_busco_ids = set(busco_ids)

print()
print(f"BUSCO count:       {len(busco_columns)}")
print(f"Unique BUSCO IDs:  {len(unique_busco_ids)}")

duplicate_busco_ids = [
    x for x in sorted(busco_ids)
    if busco_ids.count(x) > 1
]

if duplicate_busco_ids:
    print()
    print("Duplicate BUSCO IDs:")
    for x in sorted(set(duplicate_busco_ids)):
        print(f"  {x}")

    fail(
        "Duplicate BUSCO IDs detected."
    )

print("Duplicate BUSCO IDs: 0")
print("✓ BUSCO IDs are unique")


# =============================================================================
# BUSCO COUNT CHECK
# =============================================================================

if len(busco_columns) != EXPECTED_BUSCO:

    extra = sorted(
        set(busco_columns) -
        set(
            c for c in busco_columns
        )
    )

    print()
    print("BUSCO COUNT MISMATCH")
    print()
    print(f"Expected BUSCO features: {EXPECTED_BUSCO}")
    print(f"Detected BUSCO features: {len(busco_columns)}")

    fail(
        f"Expected {EXPECTED_BUSCO} BUSCO features "
        f"but detected {len(busco_columns)}."
    )

print()
print(f"✓ Correct number of BUSCO features: {EXPECTED_BUSCO}")


# =============================================================================
# EXPECTED TOTAL COLUMN COUNT
# =============================================================================

print()
print("=" * 80)
print("COLUMN COUNT VALIDATION")
print("=" * 80)

# The source/species column is metadata, not a phenotype variable.
expected_total_columns = (
    len(IDENTIFIER_COLUMNS)
    + len(PHENOTYPE_COLUMNS)
    + EXPECTED_BUSCO
    + (1 if SOURCE_COLUMN in df.columns else 0)
)

print()
print("Column calculation:")

print(
    f"Identifiers:                 {len(IDENTIFIER_COLUMNS)}"
)

print(
    f"Phenotype variables:         {len(PHENOTYPE_COLUMNS)}"
)

print(
    f"Phenotype source column:     "
    f"{1 if SOURCE_COLUMN in df.columns else 0}"
)

print(
    f"BUSCO features:              {EXPECTED_BUSCO}"
)

print(
    f"Expected total columns:      {expected_total_columns}"
)

print(
    f"Actual total columns:        {df.shape[1]}"
)

if df.shape[1] != expected_total_columns:
    fail(
        f"Expected {expected_total_columns} total columns "
        f"but found {df.shape[1]}."
    )

print()
print("✓ Total column count is correct")


# =============================================================================
# CHECK UNEXPECTED NON-BUSCO COLUMNS
# =============================================================================

print()
print("=" * 80)
print("NON-BUSCO COLUMN VALIDATION")
print("=" * 80)

expected_non_busco = (
    IDENTIFIER_COLUMNS
    + PHENOTYPE_COLUMNS
)

if SOURCE_COLUMN in df.columns:
    expected_non_busco.append(SOURCE_COLUMN)

unexpected_non_busco = [
    c for c in non_busco_columns
    if c not in expected_non_busco
]

missing_expected_non_busco = [
    c for c in expected_non_busco
    if c not in df.columns
]

if unexpected_non_busco:

    print()
    print("Unexpected non-BUSCO columns:")

    for c in unexpected_non_busco:
        print(f"  {c}")

    fail(
        "Unexpected non-BUSCO columns detected."
    )

if missing_expected_non_busco:

    print()
    print("Missing expected non-BUSCO columns:")

    for c in missing_expected_non_busco:
        print(f"  {c}")

    fail(
        "Expected non-BUSCO columns are missing."
    )

print()
print("✓ Non-BUSCO column structure is correct")


# =============================================================================
# BUSCO MATRIX
# =============================================================================

busco = df[busco_columns].copy()


# =============================================================================
# CONVERT BUSCO VALUES SAFELY
# =============================================================================

print()
print("=" * 80)
print("BUSCO VALUE QC")
print("=" * 80)

# Convert strings such as "0", "1", " 0 ", "1.0" safely.
for col in busco.columns:
    busco[col] = pd.to_numeric(
        busco[col],
        errors="coerce"
    )

# Check conversion-created missing values.
conversion_missing = int(
    busco.isna().sum().sum()
)

if conversion_missing > 0:

    print()
    print(
        f"Values that could not be converted to numeric: "
        f"{conversion_missing}"
    )

    fail(
        "BUSCO matrix contains non-numeric values."
    )

# Check missing values.
missing_busco = int(
    busco.isna().sum().sum()
)

print()
print(f"Missing BUSCO cells: {missing_busco}")

if missing_busco != 0:
    fail(
        "BUSCO matrix contains missing values."
    )

# Get unique values without sorting mixed types.
unique_values = sorted(
    pd.unique(
        busco.to_numpy().ravel()
    ).tolist()
)

print()
print("Unique BUSCO values:")
print(unique_values)

unexpected_values = [
    x for x in unique_values
    if x not in (0, 1)
]

if unexpected_values:

    print()
    print("Unexpected BUSCO values:")
    print(unexpected_values)

    fail(
        "BUSCO matrix contains values other than 0/1."
    )

print()
print("✓ BUSCO matrix is binary")
print("✓ No missing BUSCO values")


# =============================================================================
# TAXON × BUSCO DIMENSIONS
# =============================================================================

print()
print("=" * 80)
print("BUSCO MATRIX DIMENSIONS")
print("=" * 80)

print()
print(f"Rows:             {busco.shape[0]}")
print(f"BUSCO columns:    {busco.shape[1]}")

if busco.shape[0] != EXPECTED_TAXA:
    fail(
        f"Expected {EXPECTED_TAXA} BUSCO rows "
        f"but found {busco.shape[0]}."
    )

if busco.shape[1] != EXPECTED_BUSCO:
    fail(
        f"Expected {EXPECTED_BUSCO} BUSCO columns "
        f"but found {busco.shape[1]}."
    )

print()
print("✓ BUSCO matrix dimensions are correct")


# =============================================================================
# BUSCO PREVALENCE
# =============================================================================

print()
print("=" * 80)
print("BUSCO PREVALENCE ANALYSIS")
print("=" * 80)

prevalence_count = busco.sum(axis=0)

prevalence_percent = (
    prevalence_count /
    EXPECTED_TAXA *
    100
)

prevalence_df = pd.DataFrame({
    "BUSCO_ID": busco.columns,
    "Present_Count": prevalence_count.values,
    "Prevalence_Percent": prevalence_percent.values,
})

prevalence_df["Absent_Count"] = (
    EXPECTED_TAXA -
    prevalence_df["Present_Count"]
)

prevalence_df["Invariant"] = (
    (prevalence_df["Present_Count"] == 0)
    |
    (prevalence_df["Present_Count"] == EXPECTED_TAXA)
)

prevalence_df["Rare_le_5pct"] = (
    prevalence_df["Prevalence_Percent"] <= 5
)

prevalence_df["Rare_le_10pct"] = (
    prevalence_df["Prevalence_Percent"] <= 10
)

prevalence_df["High_gt_95pct"] = (
    prevalence_df["Prevalence_Percent"] > 95
)

prevalence_df["High_gt_99pct"] = (
    prevalence_df["Prevalence_Percent"] > 99
)

absent_all = int(
    (prevalence_df["Present_Count"] == 0).sum()
)

present_all = int(
    (prevalence_df["Present_Count"] == EXPECTED_TAXA).sum()
)

invariant = int(
    prevalence_df["Invariant"].sum()
)

variable = int(
    EXPECTED_BUSCO - invariant
)

rare_5 = int(
    prevalence_df["Rare_le_5pct"].sum()
)

rare_10 = int(
    prevalence_df["Rare_le_10pct"].sum()
)

high_95 = int(
    prevalence_df["High_gt_95pct"].sum()
)

high_99 = int(
    prevalence_df["High_gt_99pct"].sum()
)

print()
print(
    f"Total BUSCO features:          {EXPECTED_BUSCO}"
)

print(
    f"Absent in all taxa:            {absent_all}"
)

print(
    f"Present in all taxa:            {present_all}"
)

print(
    f"Total invariant:                {invariant}"
)

print(
    f"Variable:                        {variable}"
)

print(
    f"Rare ≤5%:                        {rare_5}"
)

print(
    f"Rare ≤10%:                       {rare_10}"
)

print(
    f"High >95%:                       {high_95}"
)

print(
    f"High >99%:                       {high_99}"
)


# =============================================================================
# PHENOTYPE QC
# =============================================================================

print()
print("=" * 80)
print("PHENOTYPE QC")
print("=" * 80)

phenotype = df[PHENOTYPE_COLUMNS].copy()

for col in PHENOTYPE_COLUMNS:
    phenotype[col] = pd.to_numeric(
        phenotype[col],
        errors="coerce"
    )

missing_phenotype = (
    phenotype.isna().sum()
)

print()
print("Missing phenotype values:")

print(
    missing_phenotype.to_string()
)

total_missing_phenotype = int(
    missing_phenotype.sum()
)

if total_missing_phenotype != 0:
    fail(
        "Phenotype matrix contains missing values."
    )

print()
print("✓ No missing phenotype values")


# =============================================================================
# PHENOTYPE VARIANCE
# =============================================================================

print()
print("Phenotype variance:")

phenotype_variance = phenotype.var(
    axis=0,
    ddof=0
)

print(
    phenotype_variance.to_string()
)

zero_variance = [
    c for c in PHENOTYPE_COLUMNS
    if phenotype_variance[c] == 0
]

print()
print("Zero-variance phenotype variables:")

if zero_variance:
    for c in zero_variance:
        print(f"  {c}")
else:
    print("  None")

    print()
    print("✓ All phenotype variables have variation")


# =============================================================================
# SPECIES / ACCESSION MISSINGNESS
# =============================================================================

print()
print("=" * 80)
print("IDENTIFIER MISSINGNESS")
print("=" * 80)

missing_species = int(
    df["Species"].isna().sum()
)

missing_accessions = int(
    df["Assembly_Accession"].isna().sum()
)

print()
print(
    f"Missing Species:              {missing_species}"
)

print(
    f"Missing Assembly_Accession:   {missing_accessions}"
)

if missing_species or missing_accessions:
    fail(
        "Identifier columns contain missing values."
    )

print()
print("✓ No missing identifiers")


# =============================================================================
# PHENOTYPE SOURCE QC
# =============================================================================

if SOURCE_COLUMN in df.columns:

    print()
    print("=" * 80)
    print("PHENOTYPE SOURCE QC")
    print("=" * 80)

    missing_source = int(
        df[SOURCE_COLUMN].isna().sum()
    )

    print()
    print(
        f"Missing phenotype source values: "
        f"{missing_source}"
    )

    if missing_source != 0:
        fail(
            "Phenotype_Source_Species contains missing values."
        )

    print()
    print("✓ Phenotype source information complete")


# =============================================================================
# RECOMMENDED ML FEATURE SET
# =============================================================================

print()
print("=" * 80)
print("RECOMMENDED ML FEATURE SET")
print("=" * 80)

variable_buscos = prevalence_df.loc[
    ~prevalence_df["Invariant"],
    "BUSCO_ID"
].tolist()

print()
print(
    f"Original BUSCO features:       {EXPECTED_BUSCO}"
)

print(
    f"Invariant features removed:    {invariant}"
)

print(
    f"Recommended variable features: {len(variable_buscos)}"
)

print()
print(
    "✓ Recommended filtering removes only invariant BUSCOs."
)

print(
    "✓ Variable high-prevalence BUSCOs are retained."
)

print(
    "✓ Rare features are not automatically discarded."
)


# =============================================================================
# SAVE PREVALENCE TABLE
# =============================================================================

prevalence_path = os.path.join(
    OUTPUT_DIR,
    "busco_prevalence_420.csv"
)

prevalence_df.to_csv(
    prevalence_path,
    index=False
)


# =============================================================================
# SAVE VARIABLE BUSCO LIST
# =============================================================================

variable_path = os.path.join(
    OUTPUT_DIR,
    "busco_variable_features_420.csv"
)

pd.DataFrame({
    "BUSCO_ID": variable_buscos
}).to_csv(
    variable_path,
    index=False
)


# =============================================================================
# SAVE INVARIANT BUSCO LIST
# =============================================================================

invariant_path = os.path.join(
    OUTPUT_DIR,
    "busco_invariant_features_420.csv"
)

pd.DataFrame({
    "BUSCO_ID": prevalence_df.loc[
        prevalence_df["Invariant"],
        "BUSCO_ID"
    ].tolist()
}).to_csv(
    invariant_path,
    index=False
)


# =============================================================================
# CREATE FILTERED ML MATRIX
# =============================================================================

filtered_columns = (
    IDENTIFIER_COLUMNS
    +
    (
        [SOURCE_COLUMN]
        if SOURCE_COLUMN in df.columns
        else []
    )
    +
    PHENOTYPE_COLUMNS
    +
    variable_buscos
)

filtered_df = df[filtered_columns].copy()

filtered_matrix_path = os.path.join(
    OUTPUT_DIR,
    "y1000_420_variable_busco_ml_matrix.csv"
)

filtered_df.to_csv(
    filtered_matrix_path,
    index=False
)


# =============================================================================
# PREVALENCE DISTRIBUTION
# =============================================================================

def prevalence_category(x):

    if x == 0:
        return "0%"

    if x <= 5:
        return "0–5%"

    if x <= 25:
        return "5–25%"

    if x <= 50:
        return "25–50%"

    if x <= 75:
        return "50–75%"

    if x <= 95:
        return "75–95%"

    if x < 100:
        return "95–<100%"

    return "100%"


prevalence_distribution = (
    prevalence_df["Prevalence_Percent"]
    .apply(prevalence_category)
    .value_counts()
)

distribution_order = [
    "0%",
    "0–5%",
    "5–25%",
    "25–50%",
    "50–75%",
    "75–95%",
    "95–<100%",
    "100%",
]

prevalence_distribution = (
    prevalence_distribution
    .reindex(
        distribution_order,
        fill_value=0
    )
)

distribution_df = pd.DataFrame({
    "Prevalence_Category":
        prevalence_distribution.index,
    "Feature_Count":
        prevalence_distribution.values
})

distribution_path = os.path.join(
    OUTPUT_DIR,
    "busco_prevalence_distribution_420.csv"
)

distribution_df.to_csv(
    distribution_path,
    index=False
)


# =============================================================================
# PHENOTYPE SUMMARY
# =============================================================================

phenotype_summary = pd.DataFrame({
    "Variable": PHENOTYPE_COLUMNS,
    "N": [
        phenotype[c].count()
        for c in PHENOTYPE_COLUMNS
    ],
    "Mean": [
        phenotype[c].mean()
        for c in PHENOTYPE_COLUMNS
    ],
    "SD": [
        phenotype[c].std()
        for c in PHENOTYPE_COLUMNS
    ],
    "Min": [
        phenotype[c].min()
        for c in PHENOTYPE_COLUMNS
    ],
    "Max": [
        phenotype[c].max()
        for c in PHENOTYPE_COLUMNS
    ],
    "Variance": [
        phenotype_variance[c]
        for c in PHENOTYPE_COLUMNS
    ],
})

phenotype_summary_path = os.path.join(
    OUTPUT_DIR,
    "phenotype_summary_420.csv"
)

phenotype_summary.to_csv(
    phenotype_summary_path,
    index=False
)


# =============================================================================
# QC SUMMARY
# =============================================================================

qc_summary = pd.DataFrame({
    "Metric": [
        "Taxa",
        "Expected_Taxa",
        "BUSCO_features",
        "Expected_BUSCO_features",
        "Missing_BUSCO_cells",
        "Absent_all_taxa",
        "Present_all_taxa",
        "Invariant_BUSCO_features",
        "Variable_BUSCO_features",
        "Rare_le_5_percent",
        "Rare_le_10_percent",
        "High_gt_95_percent",
        "High_gt_99_percent",
        "Missing_phenotype_cells",
        "Duplicate_species",
        "Duplicate_accessions",
        "Total_columns",
        "Expected_total_columns",
        "QC_status",
    ],
    "Value": [
        len(df),
        EXPECTED_TAXA,
        len(busco_columns),
        EXPECTED_BUSCO,
        missing_busco,
        absent_all,
        present_all,
        invariant,
        variable,
        rare_5,
        rare_10,
        high_95,
        high_99,
        total_missing_phenotype,
        duplicate_species,
        duplicate_accessions,
        df.shape[1],
        expected_total_columns,
        "PASSED",
    ],
})

qc_summary_path = os.path.join(
    OUTPUT_DIR,
    "ml_dataset_qc_summary_420.csv"
)

qc_summary.to_csv(
    qc_summary_path,
    index=False
)


# =============================================================================
# FINAL VALIDATION
# =============================================================================

print()
print("=" * 80)
print("FINAL VALIDATION")
print("=" * 80)

print()
print(f"Final taxa:                 {len(df)}")
print(f"Final BUSCO features:       {len(busco_columns)}")
print(f"Variable BUSCO features:    {variable}")
print(f"Final total columns:        {df.shape[1]}")
print(f"Expected total columns:     {expected_total_columns}")

print()
print(f"Missing BUSCO cells:        {missing_busco}")
print(
    f"Missing phenotype cells:    "
    f"{total_missing_phenotype}"
)

print(
    f"Duplicate species:          "
    f"{duplicate_species}"
)

print(
    f"Duplicate accessions:       "
    f"{duplicate_accessions}"
)

print()
print("✓ 420 taxa validated")
print("✓ 2137 BUSCO features validated")
print("✓ BUSCO matrix is binary")
print("✓ No missing BUSCO values")
print("✓ No missing phenotype values")
print("✓ Species are unique")
print("✓ Assembly accessions are unique")
print("✓ Column structure is correct")
print("✓ QC PASSED")


# =============================================================================
# OUTPUTS
# =============================================================================

print()
print("=" * 80)
print("420-TAXON ML DATASET QC COMPLETE")
print("=" * 80)

print()
print("Filtered ML matrix:")
print(filtered_matrix_path)

print()
print("QC summary:")
print(qc_summary_path)

print()
print("BUSCO prevalence:")
print(prevalence_path)

print()
print("BUSCO prevalence distribution:")
print(distribution_path)

print()
print("Invariant BUSCOs:")
print(invariant_path)

print()
print("Variable BUSCOs:")
print(variable_path)

print()
print("Phenotype summary:")
print(phenotype_summary_path)

print()
print("✓ Original ML dataset was NOT modified.")
print("✓ QC completed successfully.")