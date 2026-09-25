# ============================================================
# Y1000+ PROJECT
# Stage 1E: Growth Threshold & Phenotype Definition Validation
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
    "stage1E_threshold_validation"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


print("=" * 75)
print("Y1000+ STAGE 1E — GROWTH THRESHOLD VALIDATION")
print("=" * 75)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("\n[1/8] Loading Y1000+ phenotype data...")

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
# 3. CARBON CONDITIONS
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


carbon = df[
    carbon_conditions
].copy()


# ============================================================
# 4. INSPECT ZERO / NEAR-ZERO VALUES
# ============================================================

print("\n[3/8] Inspecting zero and near-zero growth values...")


all_growth_values = carbon.values.flatten()

all_growth_values = all_growth_values[
    ~np.isnan(all_growth_values)
]


thresholds = [
    0,
    0.0001,
    0.0005,
    0.001,
    0.002,
    0.005,
    0.01,
    0.02
]


threshold_summary = []

for threshold in thresholds:

    threshold_summary.append({

        "Threshold": threshold,

        "Values_At_or_Below": (
            all_growth_values <= threshold
        ).sum(),

        "Percent_At_or_Below": (
            all_growth_values <= threshold
        ).mean() * 100

    })


threshold_summary = pd.DataFrame(
    threshold_summary
)


print("\nGrowth-value threshold summary:")

print(
    threshold_summary.to_string(
        index=False
    )
)


threshold_summary.to_csv(
    os.path.join(
        RESULTS_DIR,
        "growth_threshold_summary.csv"
    ),
    index=False
)


# ============================================================
# 5. PER-CONDITION DISTRIBUTIONS
# ============================================================

print(
    "\n[4/8] Calculating per-condition growth statistics..."
)


condition_stats = []

for col in carbon_conditions:

    values = carbon[col].dropna()

    condition_stats.append({

        "Carbon_Source": col,

        "N": len(values),

        "Zero_Count": (
            values == 0
        ).sum(),

        "Zero_Percent": (
            values == 0
        ).mean() * 100,

        "Below_0.001_Percent": (
            values <= 0.001
        ).mean() * 100,

        "Below_0.005_Percent": (
            values <= 0.005
        ).mean() * 100,

        "Median": values.median(),

        "Q25": values.quantile(0.25),

        "Q75": values.quantile(0.75),

        "Maximum": values.max()

    })


condition_stats = pd.DataFrame(
    condition_stats
)


print(
    "\nPer-condition statistics:"
)

print(
    condition_stats.to_string(
        index=False
    )
)


condition_stats.to_csv(
    os.path.join(
        RESULTS_DIR,
        "per_condition_growth_statistics.csv"
    ),
    index=False
)


# ============================================================
# 6. COMPARE THREE UTILIZATION DEFINITIONS
# ============================================================

print(
    "\n[5/8] Comparing candidate utilization definitions..."
)


# ------------------------------------------------------------
# Definition A:
# strictly positive growth
# ------------------------------------------------------------

positive_count = (
    carbon > 0
).sum(axis=1)


# ------------------------------------------------------------
# Definition B:
# growth > 0.001
# ------------------------------------------------------------

threshold_001_count = (
    carbon > 0.001
).sum(axis=1)


# ------------------------------------------------------------
# Definition C:
# growth > 0.005
# ------------------------------------------------------------

threshold_005_count = (
    carbon > 0.005
).sum(axis=1
)


utilization_comparison = pd.DataFrame({

    "Strain_ID": df["Strain_ID"],

    "Species": df["Species"],

    "Published_Carbon_Breadth": df[
        "Carbon Breadth"
    ],

    "Positive_Growth_Count": positive_count,

    "Growth_Above_0.001_Count": threshold_001_count,

    "Growth_Above_0.005_Count": threshold_005_count

})


# Correlations with published Carbon Breadth

comparison_results = []

for col in [

    "Positive_Growth_Count",

    "Growth_Above_0.001_Count",

    "Growth_Above_0.005_Count"

]:

    valid = utilization_comparison[
        [
            "Published_Carbon_Breadth",
            col
        ]
    ].dropna()

    rho, p = spearmanr(
        valid["Published_Carbon_Breadth"],
        valid[col]
    )

    comparison_results.append({

        "Operational_Definition": col,

        "Spearman_Rho_with_Published_Breadth": rho,

        "P_value": p

    })


comparison_results = pd.DataFrame(
    comparison_results
)


print(
    "\nComparison with published Carbon Breadth:"
)

print(
    comparison_results.to_string(
        index=False
    )
)


comparison_results.to_csv(
    os.path.join(
        RESULTS_DIR,
        "utilization_definition_comparison.csv"
    ),
    index=False
)


utilization_comparison.to_csv(
    os.path.join(
        RESULTS_DIR,
        "utilization_definition_per_strain.csv"
    ),
    index=False
)


# ============================================================
# 7. TEST WHETHER OUR POSITIVE-GROWTH DEFINITION
# APPROXIMATES THE PUBLISHED BREADTH
# ============================================================

print(
    "\n[6/8] Comparing operational counts with "
    "published Carbon Breadth..."
)


comparison_table = pd.DataFrame({

    "Published_Carbon_Breadth":
        df["Carbon Breadth"],

    "Positive_Growth_Count":
        positive_count,

    "Above_0.001_Count":
        threshold_001_count,

    "Above_0.005_Count":
        threshold_005_count

})


comparison_table["Difference_Positive"] = (
    comparison_table[
        "Positive_Growth_Count"
    ]
    -
    comparison_table[
        "Published_Carbon_Breadth"
    ]
)


comparison_table["Difference_0.001"] = (
    comparison_table[
        "Above_0.001_Count"
    ]
    -
    comparison_table[
        "Published_Carbon_Breadth"
    ]
)


comparison_table["Difference_0.005"] = (
    comparison_table[
        "Above_0.005_Count"
    ]
    -
    comparison_table[
        "Published_Carbon_Breadth"
    ]
)


comparison_table.to_csv(
    os.path.join(
        RESULTS_DIR,
        "breadth_definition_differences.csv"
    ),
    index=False
)


# ============================================================
# 8. DISTRIBUTION PLOTS
# ============================================================

print(
    "\n[7/8] Creating diagnostic plots..."
)


# ------------------------------------------------------------
# Growth value distribution
# ------------------------------------------------------------

plt.figure(
    figsize=(9, 6)
)

plt.hist(
    all_growth_values,
    bins=60
)

plt.xlabel(
    "Growth rate"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Distribution of Y1000+ Carbon Growth Rates"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "growth_rate_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Low-growth distribution
# ------------------------------------------------------------

low_values = all_growth_values[
    all_growth_values <= 0.02
]


plt.figure(
    figsize=(9, 6)
)

plt.hist(
    low_values,
    bins=50
)

plt.xlabel(
    "Growth rate (0–0.02)"
)

plt.ylabel(
    "Frequency"
)

plt.title(
    "Near-Zero Y1000+ Growth Values"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "near_zero_growth_distribution.png"
    ),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# Published breadth vs operational breadth
# ------------------------------------------------------------

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    comparison_table[
        "Published_Carbon_Breadth"
    ],
    comparison_table[
        "Positive_Growth_Count"
    ],
    alpha=0.5
)

plt.xlabel(
    "Published Carbon Breadth"
)

plt.ylabel(
    "Count of Growth > 0"
)

plt.title(
    "Published Carbon Breadth vs Positive Growth Count"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "published_vs_positive_breadth.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 9. FINAL REPORT
# ============================================================

print(
    "\n[8/8] Creating Stage 1E report..."
)


report_file = os.path.join(
    RESULTS_DIR,
    "stage1E_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "Y1000+ STAGE 1E — GROWTH THRESHOLD VALIDATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        "Y1000+ source interpretation:\n"
    )

    f.write(
        "The source dataset contains growth rates calculated "
        "using grofit. The dataset does not provide a universal "
        "numeric utilization threshold in media-3.xlsx.\n\n"
    )

    f.write(
        "Growth threshold summary:\n\n"
    )

    f.write(
        threshold_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nOperational definition comparison:\n\n"
    )

    f.write(
        comparison_results.to_string(
            index=False
        )
    )

    f.write(
        "\n\nPer-condition statistics:\n\n"
    )

    f.write(
        condition_stats.to_string(
            index=False
        )
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 75)

print(
    "STAGE 1E COMPLETE"
)

print("=" * 75)


print(
    "\nResults saved to:"
)

print(
    RESULTS_DIR
)

print(
    "\nMost important outputs:"
)

print(
    "1. growth_threshold_summary.csv"
)

print(
    "2. per_condition_growth_statistics.csv"
)

print(
    "3. utilization_definition_comparison.csv"
)

print(
    "4. breadth_definition_differences.csv"
)

print(
    "5. growth_rate_distribution.png"
)

print(
    "6. near_zero_growth_distribution.png"
)

print(
    "7. stage1E_report.txt"
)

print(
    "\nDo NOT select a threshold automatically."
)

print(
    "We will use the results to finalize the phenotype "
    "definition based on the original Y1000+ data."
)