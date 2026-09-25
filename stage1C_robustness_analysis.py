# ============================================================
# Y1000+ PROJECT
# Stage 1C: Growth Robustness Analysis
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import spearmanr


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
    "stage1C_robustness"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 75)
print("Y1000+ STAGE 1C — GROWTH ROBUSTNESS ANALYSIS")
print("=" * 75)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/8] Loading phenotype dataset...")

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

print("Dataset shape:", df.shape)


# ============================================================
# 3. DEFINE CARBON CONDITIONS
# ============================================================

print("\n[2/8] Defining carbon conditions...")

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


# ============================================================
# 4. BASIC CARBON STATISTICS
# ============================================================

print("\n[3/8] Calculating carbon-growth statistics...")


carbon = df[
    carbon_conditions
].copy()


# Number of observed conditions
df["Carbon_Measured_Count"] = (
    carbon.notna()
    .sum(axis=1)
)


# ------------------------------------------------------------
# Mean growth
# ------------------------------------------------------------

df["Carbon_Mean_Growth"] = (
    carbon.mean(axis=1)
)


# ------------------------------------------------------------
# Median growth
# ------------------------------------------------------------

df["Carbon_Median_Growth"] = (
    carbon.median(axis=1)
)


# ------------------------------------------------------------
# Standard deviation
# ------------------------------------------------------------

df["Carbon_SD_Growth"] = (
    carbon.std(axis=1)
)


# ------------------------------------------------------------
# Coefficient of variation
#
# CV = SD / Mean
#
# Lower CV = more consistent relative growth.
# ------------------------------------------------------------

df["Carbon_CV"] = (
    df["Carbon_SD_Growth"]
    / df["Carbon_Mean_Growth"].replace(
        0,
        np.nan
    )
)


# ------------------------------------------------------------
# Minimum growth
# ------------------------------------------------------------

df["Carbon_Min_Growth"] = (
    carbon.min(axis=1)
)


# ------------------------------------------------------------
# Maximum growth
# ------------------------------------------------------------

df["Carbon_Max_Growth"] = (
    carbon.max(axis=1)
)


# ------------------------------------------------------------
# Lower-tail performance
#
# 25th percentile represents performance under
# relatively unfavorable carbon conditions.
# ------------------------------------------------------------

df["Carbon_Q25_Growth"] = (
    carbon.quantile(
        0.25,
        axis=1
    )
)


# ------------------------------------------------------------
# 10th percentile
# ------------------------------------------------------------

df["Carbon_Q10_Growth"] = (
    carbon.quantile(
        0.10,
        axis=1
    )
)


# ============================================================
# 5. PERCENTILE-BASED ROBUSTNESS
# ============================================================

print("\n[4/8] Calculating percentile-based robustness...")


# Each substrate has a different growth-rate scale.
#
# Therefore, convert each substrate to a percentile
# across the Y1000+ population.


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


# ------------------------------------------------------------
# Average relative performance
# ------------------------------------------------------------

df["Relative_Carbon_Performance"] = (
    carbon_percentiles.mean(
        axis=1
    )
)


# ------------------------------------------------------------
# Median relative performance
# ------------------------------------------------------------

df["Relative_Carbon_Median"] = (
    carbon_percentiles.median(
        axis=1
    )
)


# ------------------------------------------------------------
# Lower-tail relative performance
# ------------------------------------------------------------

df["Relative_Carbon_Q25"] = (
    carbon_percentiles.quantile(
        0.25,
        axis=1
    )
)


df["Relative_Carbon_Q10"] = (
    carbon_percentiles.quantile(
        0.10,
        axis=1
    )
)


# ============================================================
# 6. ROBUSTNESS CANDIDATE SUMMARY
# ============================================================

print("\n[5/8] Comparing candidate robustness measures...")


# Load Carbon Breadth
if "Carbon Breadth" not in df.columns:

    raise ValueError(
        "Carbon Breadth column not found."
    )


candidate_columns = [

    "Carbon Breadth",

    "Carbon_Mean_Growth",

    "Carbon_Median_Growth",

    "Carbon_SD_Growth",

    "Carbon_CV",

    "Carbon_Min_Growth",

    "Carbon_Q25_Growth",

    "Carbon_Q10_Growth",

    "Relative_Carbon_Performance",

    "Relative_Carbon_Median",

    "Relative_Carbon_Q25",

    "Relative_Carbon_Q10"

]


candidate_summary = pd.DataFrame(
    index=candidate_columns,
    columns=[
        "Spearman_rho_with_Carbon_Breadth",
        "P_value",
        "Non_Missing"
    ]
)


for col in candidate_columns:

    valid = df[
        [
            "Carbon Breadth",
            col
        ]
    ].dropna()

    if len(valid) > 2:

        rho, p = spearmanr(
            valid["Carbon Breadth"],
            valid[col]
        )

        candidate_summary.loc[
            col,
            "Spearman_rho_with_Carbon_Breadth"
        ] = rho

        candidate_summary.loc[
            col,
            "P_value"
        ] = p

        candidate_summary.loc[
            col,
            "Non_Missing"
        ] = len(valid)


print("\nCandidate robustness measures:")

print(
    candidate_summary.to_string()
)


candidate_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_candidate_comparison.csv"
    )
)


# ============================================================
# 7. CORRELATION MATRIX
# ============================================================

print("\n[6/8] Creating robustness correlation matrix...")


correlation_columns = [

    "Carbon Breadth",

    "Carbon_Mean_Growth",

    "Carbon_Median_Growth",

    "Carbon_SD_Growth",

    "Carbon_CV",

    "Carbon_Min_Growth",

    "Carbon_Q25_Growth",

    "Carbon_Q10_Growth",

    "Relative_Carbon_Performance",

    "Relative_Carbon_Median",

    "Relative_Carbon_Q25",

    "Relative_Carbon_Q10"

]


robustness_corr = (
    df[correlation_columns]
    .corr(
        method="spearman"
    )
)


robustness_corr.to_csv(
    os.path.join(
        RESULTS_DIR,
        "robustness_correlation_matrix.csv"
    )
)


# ============================================================
# 8. PLOTS
# ============================================================

print("\n[7/8] Creating comparison plots...")


# ------------------------------------------------------------
# Plot 1 — Carbon Breadth vs Mean Growth
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Carbon Breadth"],
    df["Carbon_Mean_Growth"],
    alpha=0.5
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Mean Growth Rate"
)

plt.title(
    "Carbon Breadth vs Mean Carbon Growth"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "carbon_breadth_vs_mean_growth.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 2 — Carbon Breadth vs Lower-Tail Growth
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Carbon Breadth"],
    df["Relative_Carbon_Q25"],
    alpha=0.5
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Relative Carbon Q25"
)

plt.title(
    "Carbon Breadth vs Lower-Tail Carbon Performance"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "carbon_breadth_vs_lower_tail.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 3 — Breadth vs CV
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    df["Carbon Breadth"],
    df["Carbon_CV"],
    alpha=0.5
)

plt.xlabel(
    "Carbon Breadth"
)

plt.ylabel(
    "Carbon Growth CV"
)

plt.title(
    "Carbon Breadth vs Growth Variability"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "carbon_breadth_vs_cv.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Plot 4 — Robustness candidate distributions
# ------------------------------------------------------------

plot_candidates = [

    "Carbon_Mean_Growth",

    "Carbon_Median_Growth",

    "Carbon_Q25_Growth",

    "Relative_Carbon_Performance",

    "Relative_Carbon_Q25"

]


for col in plot_candidates:

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        df[col].dropna(),
        bins=30
    )

    plt.xlabel(col)

    plt.ylabel(
        "Number of strains"
    )

    plt.title(
        f"Distribution of {col}"
    )

    plt.tight_layout()

    safe_name = (
        col
        .lower()
        .replace(" ", "_")
    )

    plt.savefig(
        os.path.join(
            RESULTS_DIR,
            f"{safe_name}_distribution.png"
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# 9. SAVE COMPLETE ROBUSTNESS DATASET
# ============================================================

print("\n[8/8] Saving enhanced dataset...")


enhanced_columns = [

    "Strain_ID",
    "Species",

    "Carbon Breadth",
    "Nitrogen Breadth",

    "Carbon Class",
    "Nitrogen Class",

    "Carbon_Measured_Count",

    "Carbon_Mean_Growth",
    "Carbon_Median_Growth",

    "Carbon_SD_Growth",
    "Carbon_CV",

    "Carbon_Min_Growth",
    "Carbon_Max_Growth",

    "Carbon_Q25_Growth",
    "Carbon_Q10_Growth",

    "Relative_Carbon_Performance",
    "Relative_Carbon_Median",

    "Relative_Carbon_Q25",
    "Relative_Carbon_Q10"

]


enhanced = df[
    [
        col for col in enhanced_columns
        if col in df.columns
    ]
].copy()


enhanced.to_csv(
    os.path.join(
        RESULTS_DIR,
        "y1000_stage1C_robustness_dataset.csv"
    ),
    index=False
)


# ============================================================
# 10. REPORT
# ============================================================

report_file = os.path.join(
    RESULTS_DIR,
    "stage1C_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Y1000+ STAGE 1C — GROWTH ROBUSTNESS ANALYSIS\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"Rows: {len(df)}\n"
    )

    f.write(
        f"Carbon conditions: "
        f"{len(carbon_conditions)}\n\n"
    )

    f.write(
        "Candidate robustness measures:\n\n"
    )

    f.write(
        candidate_summary.to_string()
    )

    f.write(
        "\n\nInterpretation should consider:\n"
    )

    f.write(
        "1. Redundancy with Carbon Breadth\n"
    )

    f.write(
        "2. Biological interpretability\n"
    )

    f.write(
        "3. Missingness\n"
    )

    f.write(
        "4. Whether the metric captures robustness "
        "rather than simply breadth\n"
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)

print(
    "STAGE 1C ANALYSIS COMPLETE"
)

print("=" * 75)


print("\nResults saved to:")

print(RESULTS_DIR)


print("\nMost important file:")

print(
    "robustness_candidate_comparison.csv"
)

print(
    "\nAlso inspect:"
)

print(
    "robustness_correlation_matrix.csv"
)

print(
    "carbon_breadth_vs_mean_growth.png"
)

print(
    "carbon_breadth_vs_lower_tail.png"
)

print(
    "carbon_breadth_vs_cv.png"
)

print(
    "\nIMPORTANT:"
)

print(
    "No final robustness metric has been selected."
)

print(
    "Use the results to select the biologically "
    "appropriate third industrial constraint."
)