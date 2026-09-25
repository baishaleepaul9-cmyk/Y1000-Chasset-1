# ============================================================
# Y1000+ PROJECT
# Stage 1: Phenotype Data Audit
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

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
    "stage1_phenotype_audit"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 70)
print("Y1000+ STAGE 1 — PHENOTYPE DATA AUDIT")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)


# ------------------------------------------------------------
# 2. LOAD EXCEL FILE
# ------------------------------------------------------------

print("\n[1/8] Loading Excel file...")

# The first row of the Excel file is a description/title.
# The SECOND row contains the actual column names.
raw = pd.read_excel(
    INPUT_FILE,
    header=1
)

print("Raw shape:", raw.shape)

# Remove completely empty rows and columns
raw = raw.dropna(how="all")
raw = raw.dropna(axis=1, how="all")

df = raw.copy()

print("Cleaned shape:", df.shape)


# ------------------------------------------------------------
# 3. STANDARDIZE COLUMN NAMES
# ------------------------------------------------------------

print("\n[2/8] Standardizing column names...")

df.columns = [
    str(col).strip()
    for col in df.columns
]

# The first column is the Y1000+ strain identifier
first_column = df.columns[0]

df = df.rename(
    columns={
        first_column: "Strain_ID"
    }
)

# Check that Species exists
if "Species" not in df.columns:
    print("\nERROR: Species column was not found.")
    print("\nColumns detected:")
    for col in df.columns:
        print(repr(col))

    raise ValueError(
        "Species column was not found. "
        "Please check the Excel file."
    )


print("\nColumns found:")

for i, col in enumerate(df.columns, start=1):
    print(f"{i:2d}. {col}")


# ------------------------------------------------------------
# 4. BASIC DATA INFORMATION
# ------------------------------------------------------------

print("\n[3/8] Basic dataset information...")

print("\nNumber of rows:", len(df))

print(
    "Number of columns:",
    len(df.columns)
)

print(
    "Number of unique species:",
    df["Species"].nunique()
)

print(
    "Number of unique strain IDs:",
    df["Strain_ID"].nunique()
)


# ------------------------------------------------------------
# 5. DATA TYPES
# ------------------------------------------------------------

print("\n[4/8] Checking data types...")

# Columns expected to contain categorical/text information
categorical_columns = [
    "Strain_ID",
    "Species",
    "Carbon Class",
    "Nitrogen Class"
]

# Convert all remaining columns to numeric
# Non-numeric entries become NaN.
for col in df.columns:

    if col not in categorical_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )


print("\nData types:")

print(df.dtypes)


# ------------------------------------------------------------
# 6. MISSING VALUE ANALYSIS
# ------------------------------------------------------------

print("\n[5/8] Calculating missing values...")

missing = pd.DataFrame({

    "Column": df.columns,

    "Missing_Count": [
        df[col].isna().sum()
        for col in df.columns
    ],

    "Total": [
        len(df)
        for col in df.columns
    ]

})

missing["Missing_Percent"] = (
    missing["Missing_Count"]
    / missing["Total"]
    * 100
)

missing = missing.sort_values(
    "Missing_Percent",
    ascending=False
)


print("\nMissing-value summary:")

print(
    missing.to_string(index=False)
)


# Save missingness report
missing.to_csv(
    os.path.join(
        RESULTS_DIR,
        "missingness_summary.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 7. DUPLICATE ANALYSIS
# ------------------------------------------------------------

print("\n[6/8] Checking duplicates...")

# Duplicate species
duplicate_species = df[
    df.duplicated(
        subset=["Species"],
        keep=False
    )
].copy()

# Duplicate strain IDs
duplicate_strains = df[
    df.duplicated(
        subset=["Strain_ID"],
        keep=False
    )
].copy()


print(
    "\nDuplicate species rows:",
    len(duplicate_species)
)

print(
    "Duplicate strain ID rows:",
    len(duplicate_strains)
)


# Save duplicate reports
duplicate_species.to_csv(
    os.path.join(
        RESULTS_DIR,
        "duplicate_species.csv"
    ),
    index=False
)

duplicate_strains.to_csv(
    os.path.join(
        RESULTS_DIR,
        "duplicate_strain_ids.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 8. PHENOTYPE SUMMARY
# ------------------------------------------------------------

print("\n[7/8] Creating phenotype summary...")

# Identify numeric columns
numeric_columns = df.select_dtypes(
    include=np.number
).columns.tolist()


phenotype_summary = pd.DataFrame({

    "Phenotype": numeric_columns,

    "Non_Missing": [
        df[col].notna().sum()
        for col in numeric_columns
    ],

    "Missing": [
        df[col].isna().sum()
        for col in numeric_columns
    ],

    "Missing_Percent": [
        df[col].isna().mean() * 100
        for col in numeric_columns
    ],

    "Mean": [
        df[col].mean()
        for col in numeric_columns
    ],

    "Median": [
        df[col].median()
        for col in numeric_columns
    ],

    "Std": [
        df[col].std()
        for col in numeric_columns
    ],

    "Minimum": [
        df[col].min()
        for col in numeric_columns
    ],

    "Maximum": [
        df[col].max()
        for col in numeric_columns
    ]

})


print("\nPhenotype summary:")

print(
    phenotype_summary.to_string(
        index=False
    )
)


phenotype_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "phenotype_summary.csv"
    ),
    index=False
)


# ------------------------------------------------------------
# 9. CARBON / NITROGEN CLASS DISTRIBUTIONS
# ------------------------------------------------------------

print("\n[8/8] Checking phenotype classes...")


class_columns = [
    "Carbon Class",
    "Nitrogen Class"
]


for col in class_columns:

    if col in df.columns:

        print("\n" + col)

        print(
            df[col]
            .value_counts(dropna=False)
        )

        class_counts = (
            df[col]
            .value_counts(dropna=False)
            .rename_axis(col)
            .reset_index(name="Count")
        )

        output_name = (
            col.lower()
            .replace(" ", "_")
            + "_distribution.csv"
        )

        class_counts.to_csv(
            os.path.join(
                RESULTS_DIR,
                output_name
            ),
            index=False
        )


# ------------------------------------------------------------
# 10. SAVE CLEANED DATASET
# ------------------------------------------------------------

print("\nSaving cleaned phenotype dataset...")

cleaned_file = os.path.join(
    RESULTS_DIR,
    "y1000_cleaned_phenotype.csv"
)

df.to_csv(
    cleaned_file,
    index=False
)


# ------------------------------------------------------------
# 11. MISSINGNESS PLOT
# ------------------------------------------------------------

print("\nCreating missingness plot...")

plot_data = missing[
    missing["Missing_Count"] > 0
].copy()


if len(plot_data) > 0:

    plt.figure(figsize=(12, 7))

    plt.bar(
        plot_data["Column"],
        plot_data["Missing_Percent"]
    )

    plt.xticks(
        rotation=90,
        ha="right"
    )

    plt.ylabel(
        "Missing values (%)"
    )

    plt.xlabel(
        "Phenotype / Variable"
    )

    plt.title(
        "Y1000+ Phenotype Dataset — Missingness"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "missingness_plot.png"
        ),
        dpi=300
    )

    plt.close()

else:

    print(
        "No missing values detected."
    )


# ------------------------------------------------------------
# 12. CARBON BREADTH DISTRIBUTION
# ------------------------------------------------------------

if "Carbon Breadth" in df.columns:

    plt.figure(figsize=(8, 6))

    plt.hist(
        df["Carbon Breadth"].dropna(),
        bins=20
    )

    plt.xlabel(
        "Carbon Breadth"
    )

    plt.ylabel(
        "Number of species"
    )

    plt.title(
        "Distribution of Carbon Breadth"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "carbon_breadth_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 13. NITROGEN BREADTH DISTRIBUTION
# ------------------------------------------------------------

if "Nitrogen Breadth" in df.columns:

    plt.figure(figsize=(8, 6))

    plt.hist(
        df["Nitrogen Breadth"].dropna(),
        bins=20
    )

    plt.xlabel(
        "Nitrogen Breadth"
    )

    plt.ylabel(
        "Number of species"
    )

    plt.title(
        "Distribution of Nitrogen Breadth"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            "nitrogen_breadth_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 14. CREATE FINAL TEXT REPORT
# ------------------------------------------------------------

print("\nCreating Stage 1 report...")

report_file = os.path.join(
    RESULTS_DIR,
    "stage1_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Y1000+ STAGE 1 — PHENOTYPE DATA AUDIT\n"
    )

    f.write(
        "=" * 60 + "\n\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Columns: {len(df.columns)}\n"
    )

    f.write(
        f"Unique species: "
        f"{df['Species'].nunique()}\n"
    )

    f.write(
        f"Unique strain IDs: "
        f"{df['Strain_ID'].nunique()}\n\n"
    )

    f.write(
        "Missingness:\n\n"
    )

    f.write(
        missing.to_string(index=False)
    )

    f.write(
        "\n\nPhenotype summary:\n\n"
    )

    f.write(
        phenotype_summary.to_string(index=False)
    )


# ------------------------------------------------------------
# 15. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 70)

print(
    "STAGE 1 AUDIT COMPLETE"
)

print("=" * 70)


print("\nResults saved to:")

print(RESULTS_DIR)


print("\nImportant files:")

print(
    "1. y1000_cleaned_phenotype.csv"
)

print(
    "2. missingness_summary.csv"
)

print(
    "3. phenotype_summary.csv"
)

print(
    "4. duplicate_species.csv"
)

print(
    "5. duplicate_strain_ids.csv"
)

print(
    "6. stage1_report.txt"
)

print(
    "7. missingness_plot.png"
)

print(
    "8. carbon_breadth_distribution.png"
)

print(
    "9. nitrogen_breadth_distribution.png"
)


print("\nNext step:")

print(
    "Review the phenotype coverage before selecting "
    "the final industrial constraints."
)