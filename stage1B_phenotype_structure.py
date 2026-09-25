# ============================================================
# Y1000+ PROJECT
# Stage 1B: Phenotype Structure & Multi-Constraint Analysis
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# 1. PATHS
# ============================================================

PROJECT_DIR = r"C:\Y1000_chassis_project"

INPUT_FILE = os.path.join(
    PROJECT_DIR,
    "data",
    "phenotype",
    "media-3.xlsx"
)

RESULTS_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "stage1B_phenotype_structure"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 75)
print("Y1000+ STAGE 1B — PHENOTYPE STRUCTURE ANALYSIS")
print("=" * 75)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/9] Loading phenotype dataset...")

df = pd.read_excel(
    INPUT_FILE,
    header=1
)

df = df.dropna(
    how="all"
)

df = df.dropna(
    axis=1,
    how="all"
)

# Standardize column names
df.columns = [
    str(col).strip()
    for col in df.columns
]

# Rename first column
df = df.rename(
    columns={
        df.columns[0]: "Strain_ID"
    }
)

print("Dataset shape:", df.shape)

print(
    "Unique species:",
    df["Species"].nunique()
)

print(
    "Unique strain IDs:",
    df["Strain_ID"].nunique()
)


# ============================================================
# 3. DEFINE PHENOTYPE GROUPS
# ============================================================

print("\n[2/9] Defining carbon and nitrogen phenotypes...")


# ------------------------------------------------------------
# Carbon-growth conditions
# ------------------------------------------------------------

carbon_conditions = [
    "Cellobiose",
    "Citrate",
    "D-Glucosamine",
    "Fructose",
    "Galactose",
    "Glucose",
    "Glycerol",
    "L-Arabinose",
    "L-Sorbose",
    "Lactose",
    "Maltose",
    "Mannose",
    "myo-Inositol",
    "Rhamnose",
    "Xylose",
    "Raffinose",
    "Sucrose",
    "DL-Lactate"
]


# ------------------------------------------------------------
# Nitrogen-growth conditions
# ------------------------------------------------------------

nitrogen_conditions = [
    "Allantoin",
    "Creatinine",
    "L-Lysine",
    "Nitrate",
    "Nitrite",
    "Urea"
]


# Check that every expected column exists
missing_carbon = [
    x for x in carbon_conditions
    if x not in df.columns
]

missing_nitrogen = [
    x for x in nitrogen_conditions
    if x not in df.columns
]

if missing_carbon:
    raise ValueError(
        f"Missing carbon columns: {missing_carbon}"
    )

if missing_nitrogen:
    raise ValueError(
        f"Missing nitrogen columns: {missing_nitrogen}"
    )


print(
    "\nCarbon conditions:",
    len(carbon_conditions)
)

print(
    "Nitrogen conditions:",
    len(nitrogen_conditions)
)


# ============================================================
# 4. CONVERT PHENOTYPES TO NUMERIC
# ============================================================

print("\n[3/9] Converting growth measurements to numeric...")

phenotype_columns = (
    carbon_conditions
    + nitrogen_conditions
)

for col in phenotype_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# ============================================================
# 5. DUPLICATE SPECIES / STRAIN INSPECTION
# ============================================================

print("\n[4/9] Inspecting duplicate species and strain IDs...")


# ------------------------------------------------------------
# Duplicate species
# ------------------------------------------------------------

species_counts = (
    df["Species"]
    .value_counts()
)

duplicate_species_names = (
    species_counts[
        species_counts > 1
    ]
    .index
    .tolist()
)


duplicate_species = df[
    df["Species"].isin(
        duplicate_species_names
    )
].copy()


print(
    "\nSpecies represented by multiple rows:",
    len(duplicate_species_names)
)

print(
    "Rows belonging to those species:",
    len(duplicate_species)
)


duplicate_species.to_csv(
    os.path.join(
        RESULTS_DIR,
        "duplicate_species_inspection.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# Duplicate strain IDs
# ------------------------------------------------------------

duplicate_ids = df[
    df["Strain_ID"].duplicated(
        keep=False
    )
].copy()


print(
    "\nDuplicate strain-ID rows:",
    len(duplicate_ids)
)


duplicate_ids.to_csv(
    os.path.join(
        RESULTS_DIR,
        "duplicate_strain_ID_inspection.csv"
    ),
    index=False
)


# ============================================================
# 6. PHENOTYPE COVERAGE
# ============================================================

print("\n[5/9] Calculating phenotype coverage...")


coverage = pd.DataFrame({

    "Phenotype": phenotype_columns,

    "Non_Missing": [
        df[col].notna().sum()
        for col in phenotype_columns
    ],

    "Missing": [
        df[col].isna().sum()
        for col in phenotype_columns
    ]

})

coverage["Coverage_Percent"] = (
    coverage["Non_Missing"]
    / len(df)
    * 100
)


print("\nPhenotype coverage:")

print(
    coverage.to_string(
        index=False
    )
)


coverage.to_csv(
    os.path.join(
        RESULTS_DIR,
        "phenotype_coverage.csv"
    ),
    index=False
)


# ============================================================
# 7. CARBON / NITROGEN PERFORMANCE SCORES
# ============================================================

print("\n[6/9] Constructing standardized phenotype scores...")


# ------------------------------------------------------------
# Why percentile/rank scoring?
#
# Different substrates have very different growth-rate scales.
# For example, glucose has a much larger numerical range than
# some other substrates.
#
# We therefore calculate a within-condition percentile.
# A value near 1 means relatively strong growth on that
# particular substrate among the Y1000+ dataset.
# ------------------------------------------------------------


carbon_percentiles = pd.DataFrame(
    index=df.index
)

for col in carbon_conditions:

    carbon_percentiles[col] = (
        df[col]
        .rank(
            pct=True,
            method="average"
        )
    )


nitrogen_percentiles = pd.DataFrame(
    index=df.index
)

for col in nitrogen_conditions:

    nitrogen_percentiles[col] = (
        df[col]
        .rank(
            pct=True,
            method="average"
        )
    )


# ------------------------------------------------------------
# Require sufficient measurements
#
# Carbon:
# at least 70% of carbon conditions
#
# Nitrogen:
# at least 70% of nitrogen conditions
# ------------------------------------------------------------

minimum_carbon_conditions = int(
    np.ceil(
        0.70 * len(carbon_conditions)
    )
)

minimum_nitrogen_conditions = int(
    np.ceil(
        0.70 * len(nitrogen_conditions)
    )
)


df["Carbon_Measured_Count"] = (
    df[carbon_conditions]
    .notna()
    .sum(axis=1)
)

df["Nitrogen_Measured_Count"] = (
    df[nitrogen_conditions]
    .notna()
    .sum(axis=1)
)


df["Carbon_Performance"] = np.nan

df.loc[
    df["Carbon_Measured_Count"]
    >= minimum_carbon_conditions,
    "Carbon_Performance"
] = (
    carbon_percentiles
    .mean(axis=1)
)


df["Nitrogen_Performance"] = np.nan

df.loc[
    df["Nitrogen_Measured_Count"]
    >= minimum_nitrogen_conditions,
    "Nitrogen_Performance"
] = (
    nitrogen_percentiles
    .mean(axis=1)
)


print(
    "\nMinimum carbon measurements required:",
    minimum_carbon_conditions
)

print(
    "Minimum nitrogen measurements required:",
    minimum_nitrogen_conditions
)


print(
    "\nCarbon Performance available for:",
    df["Carbon_Performance"].notna().sum(),
    "rows"
)

print(
    "Nitrogen Performance available for:",
    df["Nitrogen_Performance"].notna().sum(),
    "rows"
)


# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

print("\n[7/9] Calculating phenotype correlations...")


correlation_columns = [
    "Carbon Breadth",
    "Nitrogen Breadth",
    "Carbon_Performance",
    "Nitrogen_Performance"
]


correlation_matrix = (
    df[correlation_columns]
    .corr(
        method="spearman"
    )
)


print("\nSpearman correlation matrix:")

print(
    correlation_matrix.round(3)
)


correlation_matrix.to_csv(
    os.path.join(
        RESULTS_DIR,
        "phenotype_spearman_correlations.csv"
    )
)


# ------------------------------------------------------------
# Correlation heatmap
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 7)
)

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar(
    label="Spearman correlation"
)

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)

plt.title(
    "Y1000+ Phenotype Correlations"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "phenotype_correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. PCA OF CARBON PHENOTYPES
# ============================================================

print("\n[8/9] Performing PCA on carbon-growth profiles...")


carbon_pca_data = df[
    carbon_conditions
].copy()


# Keep rows with enough carbon measurements
valid_carbon = (
    carbon_pca_data
    .notna()
    .sum(axis=1)
    >= minimum_carbon_conditions
)


carbon_pca_data = carbon_pca_data[
    valid_carbon
].copy()


# Median imputation for PCA only
carbon_pca_data = carbon_pca_data.fillna(
    carbon_pca_data.median()
)


# Standardize each carbon condition
scaler = StandardScaler()

carbon_scaled = scaler.fit_transform(
    carbon_pca_data
)


# PCA
pca = PCA()

carbon_pca = pca.fit_transform(
    carbon_scaled
)


explained = pca.explained_variance_ratio_


print(
    "\nVariance explained:"
)

for i, value in enumerate(
    explained[:10],
    start=1
):

    print(
        f"PC{i}: {value * 100:.2f}%"
    )


# Save PCA results
pca_results = pd.DataFrame(
    carbon_pca[:, :5],
    columns=[
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "PC5"
    ]
)

pca_results["Strain_ID"] = (
    df.loc[
        carbon_pca_data.index,
        "Strain_ID"
    ].values
)

pca_results["Species"] = (
    df.loc[
        carbon_pca_data.index,
        "Species"
    ].values
)


pca_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "carbon_PCA_results.csv"
    ),
    index=False
)


# PCA plot
plt.figure(
    figsize=(9, 7)
)

plt.scatter(
    carbon_pca[:, 0],
    carbon_pca[:, 1],
    alpha=0.6
)

plt.xlabel(
    f"PC1 ({explained[0] * 100:.1f}% variance)"
)

plt.ylabel(
    f"PC2 ({explained[1] * 100:.1f}% variance)"
)

plt.title(
    "PCA of Y1000+ Carbon-Growth Profiles"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "carbon_PCA.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 10. MULTI-CONSTRAINT PROFILE
# ============================================================

print("\n[9/9] Creating preliminary multi-constraint profile...")


# We are NOT selecting final candidates yet.
#
# This table simply gives us the three main quantitative
# dimensions we are currently investigating:
#
# 1. Carbon breadth
# 2. Nitrogen breadth
# 3. Carbon performance
#
# Nitrogen performance is retained as an additional dimension.


profile_columns = [
    "Strain_ID",
    "Species",
    "Carbon Breadth",
    "Nitrogen Breadth",
    "Carbon_Performance",
    "Nitrogen_Performance",
    "Carbon Class",
    "Nitrogen Class"
]


profile = df[
    profile_columns
].copy()


profile.to_csv(
    os.path.join(
        RESULTS_DIR,
        "preliminary_multi_constraint_profile.csv"
    ),
    index=False
)


# ============================================================
# 11. SUMMARY STATISTICS
# ============================================================

summary_columns = [
    "Carbon Breadth",
    "Nitrogen Breadth",
    "Carbon_Performance",
    "Nitrogen_Performance"
]


summary = df[
    summary_columns
].describe()


summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "multi_constraint_summary_statistics.csv"
    )
)


# ============================================================
# 12. SAVE ENHANCED DATASET
# ============================================================

enhanced_file = os.path.join(
    RESULTS_DIR,
    "y1000_stage1B_enhanced_phenotype.csv"
)


df.to_csv(
    enhanced_file,
    index=False
)


# ============================================================
# 13. FINAL REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "stage1B_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Y1000+ STAGE 1B — PHENOTYPE STRUCTURE ANALYSIS\n"
    )

    f.write(
        "=" * 65 + "\n\n"
    )

    f.write(
        f"Total rows: {len(df)}\n"
    )

    f.write(
        f"Unique species: {df['Species'].nunique()}\n"
    )

    f.write(
        f"Unique strain IDs: {df['Strain_ID'].nunique()}\n\n"
    )

    f.write(
        f"Carbon conditions: "
        f"{len(carbon_conditions)}\n"
    )

    f.write(
        f"Nitrogen conditions: "
        f"{len(nitrogen_conditions)}\n\n"
    )

    f.write(
        "Duplicate species:\n"
    )

    f.write(
        f"{len(duplicate_species_names)} species "
        f"have multiple rows.\n\n"
    )

    f.write(
        "Duplicate strain IDs:\n"
    )

    f.write(
        f"{len(duplicate_ids)} rows "
        f"have duplicated strain IDs.\n\n"
    )

    f.write(
        "Minimum carbon measurements required: "
        f"{minimum_carbon_conditions}\n"
    )

    f.write(
        "Minimum nitrogen measurements required: "
        f"{minimum_nitrogen_conditions}\n\n"
    )

    f.write(
        "Spearman correlations:\n\n"
    )

    f.write(
        correlation_matrix.round(3).to_string()
    )

    f.write(
        "\n\nPCA variance explained:\n"
    )

    for i, value in enumerate(
        explained[:10],
        start=1
    ):

        f.write(
            f"PC{i}: {value * 100:.2f}%\n"
        )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)

print(
    "STAGE 1B ANALYSIS COMPLETE"
)

print("=" * 75)


print("\nResults saved to:")

print(RESULTS_DIR)


print("\nKey output files:")

print(
    "1. duplicate_species_inspection.csv"
)

print(
    "2. duplicate_strain_ID_inspection.csv"
)

print(
    "3. phenotype_coverage.csv"
)

print(
    "4. phenotype_spearman_correlations.csv"
)

print(
    "5. phenotype_correlation_heatmap.png"
)

print(
    "6. carbon_PCA_results.csv"
)

print(
    "7. carbon_PCA.png"
)

print(
    "8. preliminary_multi_constraint_profile.csv"
)

print(
    "9. y1000_stage1B_enhanced_phenotype.csv"
)

print(
    "10. stage1B_report.txt"
)

print(
    "\nIMPORTANT:"
)

print(
    "The multi-constraint phenotype has NOT been finalized."
)

print(
    "Use the Stage 1B results to determine the final"
)

print(
    "industrial constraints before proceeding to genomics."
)
