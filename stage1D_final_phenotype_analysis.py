# ============================================================
# Y1000+ PROJECT
# Stage 1D: Final Phenotype Dimension Analysis
# ============================================================

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


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
    "stage1D_final_phenotype"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 75)
print("Y1000+ STAGE 1D — FINAL PHENOTYPE DIMENSION ANALYSIS")
print("=" * 75)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/7] Loading phenotype dataset...")

df = pd.read_excel(
    INPUT_FILE,
    header=1
)

df = df.dropna(how="all")
df = df.dropna(axis=1, how="all")

df.columns = [
    str(col).strip()
    for col in df.columns
]

df = df.rename(
    columns={
        df.columns[0]: "Strain_ID"
    }
)


# ============================================================
# 3. CARBON CONDITIONS
# ============================================================

print("\n[2/7] Defining carbon conditions...")

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


for col in carbon_conditions:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


carbon = df[
    carbon_conditions
].copy()


# ============================================================
# 4. DEFINE UTILIZATION
# ============================================================

print("\n[3/7] Defining utilized carbon conditions...")


# IMPORTANT:
# We use growth > 0 as the operational definition of
# a measured positive-growth condition.
#
# This is NOT yet claiming that every tiny positive value
# represents biologically meaningful growth.
#
# We will inspect the resulting distributions before
# finalizing the threshold.


positive_growth = carbon > 0


df["Positive_Carbon_Count"] = (
    positive_growth
    .sum(axis=1)
)


df["Measured_Carbon_Count"] = (
    carbon.notna()
    .sum(axis=1)
)


# ============================================================
# 5. PERFORMANCE CONDITIONAL ON UTILIZATION
# ============================================================

print(
    "\n[4/7] Calculating performance among utilized "
    "carbon sources..."
)


# Replace non-positive values with NaN.
#
# This means that performance metrics below describe
# how strongly the organism grows on substrates where
# it exhibits positive growth, rather than treating
# non-growth as poor growth.


positive_only = carbon.where(
    carbon > 0
)


# ------------------------------------------------------------
# Mean growth on utilized substrates
# ------------------------------------------------------------

df["Utilized_Mean_Growth"] = (
    positive_only.mean(
        axis=1
    )
)


# ------------------------------------------------------------
# Median growth on utilized substrates
# ------------------------------------------------------------

df["Utilized_Median_Growth"] = (
    positive_only.median(
        axis=1
    )
)


# ------------------------------------------------------------
# Lower quartile among utilized substrates
# ------------------------------------------------------------

df["Utilized_Q25_Growth"] = (
    positive_only.quantile(
        0.25,
        axis=1
    )
)


# ------------------------------------------------------------
# 10th percentile among utilized substrates
# ------------------------------------------------------------

df["Utilized_Q10_Growth"] = (
    positive_only.quantile(
        0.10,
        axis=1
    )
)


# ------------------------------------------------------------
# Standard deviation
# ------------------------------------------------------------

df["Utilized_SD_Growth"] = (
    positive_only.std(
        axis=1
    )
)


# ------------------------------------------------------------
# Coefficient of variation
# ------------------------------------------------------------

df["Utilized_CV"] = (
    df["Utilized_SD_Growth"]
    / df["Utilized_Mean_Growth"]
)


# ============================================================
# 6. CORRELATION WITH CARBON BREADTH
# ============================================================

print(
    "\n[5/7] Comparing performance dimensions "
    "against Carbon Breadth..."
)


candidate_columns = [

    "Utilized_Mean_Growth",

    "Utilized_Median_Growth",

    "Utilized_Q25_Growth",

    "Utilized_Q10_Growth",

    "Utilized_SD_Growth",

    "Utilized_CV"

]


results = []


for col in candidate_columns:

    valid = df[
        [
            "Carbon Breadth",
            col
        ]
    ].dropna()

    rho, p = spearmanr(
        valid["Carbon Breadth"],
        valid[col]
    )

    results.append({

        "Metric": col,

        "Spearman_Rho_with_Carbon_Breadth": rho,

        "P_value": p,

        "Non_Missing": len(valid)

    })


comparison = pd.DataFrame(
    results
)


print("\nCandidate performance dimensions:")

print(
    comparison.to_string(
        index=False
    )
)


comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "performance_vs_breadth.csv"
    ),
    index=False
)


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

print(
    "\n[6/7] Creating correlation matrix..."
)


correlation_columns = [

    "Carbon Breadth",

    "Utilized_Mean_Growth",

    "Utilized_Median_Growth",

    "Utilized_Q25_Growth",

    "Utilized_Q10_Growth",

    "Utilized_SD_Growth",

    "Utilized_CV"

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
        "final_phenotype_correlation_matrix.csv"
    )
)


# ============================================================
# 8. DISTRIBUTION OF POSITIVE-GROWTH COUNTS
# ============================================================

print(
    "\n[7/7] Creating final diagnostic plots..."
)


plt.figure(
    figsize=(8, 6)
)

plt.hist(
    df["Positive_Carbon_Count"],
    bins=np.arange(
        -0.5,
        len(carbon_conditions) + 1.5,
        1
    )
)

plt.xlabel(
    "Number of carbon sources with positive growth"
)

plt.ylabel(
    "Number of strains"
)

plt.title(
    "Distribution of Positive Carbon Utilization"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "positive_carbon_count_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Breadth vs conditional median growth
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Carbon Breadth"],
    df["Utilized_Median_Growth"],
    alpha=0.5
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Median Growth Among Utilized Substrates"
)

plt.title(
    "Carbon Breadth vs Conditional Growth Performance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "breadth_vs_conditional_median.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Breadth vs conditional Q25
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Carbon Breadth"],
    df["Utilized_Q25_Growth"],
    alpha=0.5
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Q25 Growth Among Utilized Substrates"
)

plt.title(
    "Carbon Breadth vs Lower-Tail Growth Performance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "breadth_vs_conditional_q25.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. SAVE FINAL ANALYTICAL DATASET
# ============================================================

final_columns = [

    "Strain_ID",
    "Species",

    "Carbon Breadth",
    "Nitrogen Breadth",

    "Carbon Class",
    "Nitrogen Class",

    "Measured_Carbon_Count",
    "Positive_Carbon_Count",

    "Utilized_Mean_Growth",
    "Utilized_Median_Growth",

    "Utilized_Q25_Growth",
    "Utilized_Q10_Growth",

    "Utilized_SD_Growth",
    "Utilized_CV"

]


final_df = df[
    final_columns
].copy()


final_df.to_csv(
    os.path.join(
        RESULTS_DIR,
        "y1000_final_phenotype_candidates.csv"
    ),
    index=False
)


# ============================================================
# 10. REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "stage1D_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Y1000+ STAGE 1D — FINAL PHENOTYPE DIMENSION ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Candidate performance dimensions:\n\n"
    )

    f.write(
        comparison.to_string(
            index=False
        )
    )

    f.write(
        "\n\nSpearman correlation matrix:\n\n"
    )

    f.write(
        correlation_matrix.round(3).to_string()
    )

    f.write(
        "\n\nPositive carbon utilization summary:\n\n"
    )

    f.write(
        df["Positive_Carbon_Count"]
        .describe()
        .to_string()
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)

print(
    "STAGE 1D COMPLETE"
)

print("=" * 75)

print("\nResults saved to:")

print(RESULTS_DIR)

print("\nMost important outputs:")

print(
    "1. performance_vs_breadth.csv"
)

print(
    "2. final_phenotype_correlation_matrix.csv"
)

print(
    "3. y1000_final_phenotype_candidates.csv"
)

print(
    "4. stage1D_report.txt"
)

print(
    "\nNo final phenotype has been selected automatically."
)

print(
    "The final choice will be based on redundancy, "
    "biological interpretation, and data structure."
)