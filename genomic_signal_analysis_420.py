# =============================================================================
# GENOMIC SIGNAL ANALYSIS — Y1000+ YEAST — 420 TAXA
# =============================================================================
#
# Purpose
# -------
# Quantify whether genomic/BUSCO features provide predictive information
# beyond the phylogenetic baseline.
#
# Existing results used:
#   1. Publication-grade validation results
#   2. Diagnostic genomic-vs-phylogeny results
#
# This script DOES NOT overwrite:
#   - existing models
#   - existing OOF predictions
#   - existing validation results
#
# Outputs:
#   genomic_signal_analysis_420/
#       tables/
#       figures/
#       genomic_signal_summary_420.csv
#
# =============================================================================

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

BASE_DIR = (
    r"C:\Y1000_chassis_project\results"
    r"\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml"
    r"\phylogeny_cv"
    r"\model_training"
)

OPT_DIR = os.path.join(
    BASE_DIR,
    "performance_optimization_420"
)

PUB_DIR = os.path.join(
    OPT_DIR,
    "publication_validation_420"
)

DIAG_DIR = os.path.join(
    BASE_DIR,
    "diagnostic_model_420"
)

OUT_DIR = os.path.join(
    BASE_DIR,
    "genomic_signal_analysis_420"
)

TABLE_DIR = os.path.join(
    OUT_DIR,
    "tables"
)

FIG_DIR = os.path.join(
    OUT_DIR,
    "figures"
)

for directory in [OUT_DIR, TABLE_DIR, FIG_DIR]:
    os.makedirs(directory, exist_ok=True)


# =============================================================================
# PHENOTYPE CONFIGURATION
# =============================================================================

CONFIG = {
    "Carbon_Breadth": {
        "feature_set": "Robust_79",
        "model": "SVR_RBF",
    },

    "Nitrogen_Breadth": {
        "feature_set": "Filtered_Variable",
        "model": "SVR_RBF",
    },

    "Utilized_Median_Growth": {
        "feature_set": "Phenotype_Selected_Top50",
        "model": "GradientBoosting",
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def find_column(df, candidates):
    """
    Return the first matching column from candidates.
    Matching is case-insensitive.
    """
    lower_map = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def load_csv(path, description):
    print("\n" + "=" * 80)
    print(description)
    print("=" * 80)
    print("File:")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    df = pd.read_csv(path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    return df


# =============================================================================
# LOAD PUBLICATION VALIDATION SUMMARY
# =============================================================================

publication_summary_path = os.path.join(
    PUB_DIR,
    "tables",
    "publication_grade_validation_summary_420.csv"
)

pub_df = load_csv(
    publication_summary_path,
    "LOADING PUBLICATION VALIDATION SUMMARY"
)


# =============================================================================
# LOAD DIAGNOSTIC SUMMARY
# =============================================================================

diagnostic_summary_path = os.path.join(
    DIAG_DIR,
    "tables",
    "diagnostic_model_summary_420.csv"
)

diag_df = load_csv(
    diagnostic_summary_path,
    "LOADING DIAGNOSTIC MODEL SUMMARY"
)


# =============================================================================
# NORMALIZE COLUMN NAMES
# =============================================================================

pub_df.columns = [str(c).strip() for c in pub_df.columns]
diag_df.columns = [str(c).strip() for c in diag_df.columns]


# =============================================================================
# DISPLAY AVAILABLE INFORMATION
# =============================================================================

print("\n" + "=" * 80)
print("PUBLICATION SUMMARY COLUMNS")
print("=" * 80)

print(list(pub_df.columns))

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY COLUMNS")
print("=" * 80)

print(list(diag_df.columns))


# =============================================================================
# LOCATE IMPORTANT COLUMNS
# =============================================================================

pub_pheno_col = find_column(
    pub_df,
    ["Phenotype"]
)

diag_pheno_col = find_column(
    diag_df,
    ["Phenotype"]
)

if pub_pheno_col is None:
    raise ValueError(
        "Publication summary does not contain a Phenotype column."
    )

if diag_pheno_col is None:
    raise ValueError(
        "Diagnostic summary does not contain a Phenotype column."
    )


# =============================================================================
# BUILD RESULTS
# =============================================================================

results = []


for phenotype, config in CONFIG.items():

    print("\n\n" + "=" * 80)
    print(f"GENOMIC SIGNAL ANALYSIS — {phenotype}")
    print("=" * 80)

    feature_set = config["feature_set"]
    model = config["model"]

    # -------------------------------------------------------------------------
    # PUBLICATION VALIDATION
    # -------------------------------------------------------------------------

    pub_rows = pub_df[
        pub_df[pub_pheno_col].astype(str).str.strip()
        == phenotype
    ]

    if pub_rows.empty:

        print(
            f"\nWARNING: {phenotype} not found in publication summary."
        )

        continue

    pub_row = pub_rows.iloc[0]

    # -------------------------------------------------------------------------
    # DIAGNOSTIC RESULTS
    # -------------------------------------------------------------------------

    diag_rows = diag_df[
        diag_df[diag_pheno_col].astype(str).str.strip()
        == phenotype
    ]

    if diag_rows.empty:

        print(
            f"\nWARNING: {phenotype} not found in diagnostic summary."
        )

        continue

    diag_row = diag_rows.iloc[0]

    # -------------------------------------------------------------------------
    # EXTRACT VALUES
    # -------------------------------------------------------------------------

    existing_r2 = safe_float(
        pub_row.get("R2", np.nan)
    )

    diagnostic_r2 = safe_float(
        diag_row.get(
            "Diagnostic_Genomic_R2",
            np.nan
        )
    )

    phylogeny_r2 = safe_float(
        diag_row.get(
            "Phylogeny_R2",
            np.nan
        )
    )

    existing_genomic_r2 = safe_float(
        diag_row.get(
            "Existing_Genomic_R2",
            existing_r2
        )
    )

    existing_delta = safe_float(
        diag_row.get(
            "Delta_R2_Existing_vs_Phylogeny",
            np.nan
        )
    )

    diagnostic_delta = safe_float(
        diag_row.get(
            "Delta_R2_Diagnostic_vs_Phylogeny",
            np.nan
        )
    )

    diagnostic_vs_existing = safe_float(
        diag_row.get(
            "Delta_R2_Diagnostic_vs_Existing",
            np.nan
        )
    )

    permutation_p = safe_float(
        diag_row.get(
            "Permutation_P",
            np.nan
        )
    )

    permutation_n = safe_float(
        diag_row.get(
            "Permutation_N",
            np.nan
        )
    )

    null_mean = safe_float(
        diag_row.get(
            "Permutation_Null_Mean_R2",
            np.nan
        )
    )

    null_95 = safe_float(
        diag_row.get(
            "Permutation_Null_95th_R2",
            np.nan
        )
    )

    existing_rmse = safe_float(
        diag_row.get(
            "Existing_Genomic_RMSE",
            np.nan
        )
    )

    diagnostic_rmse = safe_float(
        diag_row.get(
            "Diagnostic_Genomic_RMSE",
            np.nan
        )
    )

    existing_mae = safe_float(
        diag_row.get(
            "Existing_Genomic_MAE",
            np.nan
        )
    )

    diagnostic_mae = safe_float(
        diag_row.get(
            "Diagnostic_Genomic_MAE",
            np.nan
        )
    )

    n_taxa = safe_float(
        diag_row.get(
            "N_Taxa",
            420
        )
    )

    # -------------------------------------------------------------------------
    # CALCULATE ADDITIONAL EFFECTS
    # -------------------------------------------------------------------------

    if np.isfinite(existing_r2) and np.isfinite(phylogeny_r2):
        genomic_gain_existing = (
            existing_r2 - phylogeny_r2
        )
    else:
        genomic_gain_existing = np.nan

    if np.isfinite(diagnostic_r2) and np.isfinite(phylogeny_r2):
        genomic_gain_diagnostic = (
            diagnostic_r2 - phylogeny_r2
        )
    else:
        genomic_gain_diagnostic = np.nan

    # -------------------------------------------------------------------------
    # INTERPRETATION
    # -------------------------------------------------------------------------

    if np.isfinite(genomic_gain_existing):

        if genomic_gain_existing > 0:
            existing_signal = "Positive genomic contribution"
        elif genomic_gain_existing < 0:
            existing_signal = "No positive genomic contribution"
        else:
            existing_signal = "No detectable genomic contribution"

    else:
        existing_signal = "Not available"

    if np.isfinite(diagnostic_vs_existing):

        if diagnostic_vs_existing > 0:
            diagnostic_effect = "Diagnostic improvement"
        elif diagnostic_vs_existing < 0:
            diagnostic_effect = "Diagnostic model lower than existing"
        else:
            diagnostic_effect = "No change"

    else:
        diagnostic_effect = "Not available"

    print("\nPhenotype:")
    print(phenotype)

    print("\nFeature set:")
    print(feature_set)

    print("\nModel:")
    print(model)

    print("\nN taxa:")
    print(n_taxa)

    print("\nPhylogeny R²:")
    print(phylogeny_r2)

    print("\nExisting genomic R²:")
    print(existing_r2)

    print("\nDiagnostic genomic R²:")
    print(diagnostic_r2)

    print("\nExisting genomic ΔR² vs phylogeny:")
    print(existing_delta)

    print("\nDiagnostic genomic ΔR² vs phylogeny:")
    print(diagnostic_delta)

    print("\nDiagnostic ΔR² vs existing:")
    print(diagnostic_vs_existing)

    print("\nPermutation p-value:")
    print(permutation_p)

    print("\nInterpretation:")
    print(existing_signal)

    print("\nDiagnostic comparison:")
    print(diagnostic_effect)

    # -------------------------------------------------------------------------
    # STORE
    # -------------------------------------------------------------------------

    results.append({

        "Phenotype": phenotype,

        "Feature_Set": feature_set,

        "Model": model,

        "N_Taxa": int(n_taxa)
        if np.isfinite(n_taxa)
        else np.nan,

        "Phylogeny_R2": phylogeny_r2,

        "Existing_Genomic_R2": existing_r2,

        "Diagnostic_Genomic_R2": diagnostic_r2,

        "Existing_Genomic_Gain_vs_Phylogeny":
            genomic_gain_existing,

        "Diagnostic_Genomic_Gain_vs_Phylogeny":
            genomic_gain_diagnostic,

        "Delta_R2_Existing_vs_Phylogeny":
            existing_delta,

        "Delta_R2_Diagnostic_vs_Phylogeny":
            diagnostic_delta,

        "Delta_R2_Diagnostic_vs_Existing":
            diagnostic_vs_existing,

        "Existing_Genomic_RMSE":
            existing_rmse,

        "Diagnostic_Genomic_RMSE":
            diagnostic_rmse,

        "Existing_Genomic_MAE":
            existing_mae,

        "Diagnostic_Genomic_MAE":
            diagnostic_mae,

        "Permutation_N":
            permutation_n,

        "Permutation_P":
            permutation_p,

        "Permutation_Null_Mean_R2":
            null_mean,

        "Permutation_Null_95th_R2":
            null_95,

        "Existing_Signal_Interpretation":
            existing_signal,

        "Diagnostic_Interpretation":
            diagnostic_effect,
    })


# =============================================================================
# FINAL TABLE
# =============================================================================

results_df = pd.DataFrame(results)


print("\n\n" + "=" * 80)
print("GENOMIC SIGNAL SUMMARY")
print("=" * 80)

if results_df.empty:

    raise RuntimeError(
        "No phenotype results were successfully extracted."
    )

print(
    results_df[
        [
            "Phenotype",
            "Phylogeny_R2",
            "Existing_Genomic_R2",
            "Diagnostic_Genomic_R2",
            "Existing_Genomic_Gain_vs_Phylogeny",
            "Diagnostic_Genomic_Gain_vs_Phylogeny",
            "Delta_R2_Diagnostic_vs_Existing",
            "Permutation_P",
        ]
    ].to_string(index=False)
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

summary_path = os.path.join(
    TABLE_DIR,
    "genomic_signal_summary_420.csv"
)

results_df.to_csv(
    summary_path,
    index=False
)

print("\n✓ Summary saved:")
print(summary_path)


# =============================================================================
# FIGURE 1 — R² COMPARISON
# =============================================================================

plot_df = results_df.copy()

phenotypes = plot_df["Phenotype"].tolist()

x = np.arange(len(phenotypes))
width = 0.25

plt.figure(figsize=(11, 7))

plt.bar(
    x - width,
    plot_df["Phylogeny_R2"],
    width,
    label="Phylogeny"
)

plt.bar(
    x,
    plot_df["Existing_Genomic_R2"],
    width,
    label="Existing genomic"
)

plt.bar(
    x + width,
    plot_df["Diagnostic_Genomic_R2"],
    width,
    label="Diagnostic genomic"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    x,
    phenotypes,
    rotation=20,
    ha="right"
)

plt.ylabel("R²")
plt.title(
    "Phylogeny versus Genomic Predictive Signal — Y1000+"
)

plt.legend()

plt.tight_layout()

r2_fig = os.path.join(
    FIG_DIR,
    "phylogeny_vs_genomic_R2_420.png"
)

plt.savefig(
    r2_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n✓ R² comparison figure saved:")
print(r2_fig)


# =============================================================================
# FIGURE 2 — ΔR² RELATIVE TO PHYLOGENY
# =============================================================================

plt.figure(figsize=(11, 7))

plt.bar(
    x - width / 2,
    plot_df["Existing_Genomic_Gain_vs_Phylogeny"],
    width,
    label="Existing genomic"
)

plt.bar(
    x + width / 2,
    plot_df["Diagnostic_Genomic_Gain_vs_Phylogeny"],
    width,
    label="Diagnostic genomic"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    x,
    phenotypes,
    rotation=20,
    ha="right"
)

plt.ylabel("ΔR² relative to phylogeny")
plt.title(
    "Independent Genomic Contribution Beyond Phylogeny"
)

plt.legend()

plt.tight_layout()

delta_fig = os.path.join(
    FIG_DIR,
    "genomic_delta_R2_vs_phylogeny_420.png"
)

plt.savefig(
    delta_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n✓ ΔR² figure saved:")
print(delta_fig)


# =============================================================================
# FIGURE 3 — EXISTING VS DIAGNOSTIC
# =============================================================================

plt.figure(figsize=(11, 7))

plt.bar(
    x - width / 2,
    plot_df["Existing_Genomic_R2"],
    width,
    label="Existing genomic"
)

plt.bar(
    x + width / 2,
    plot_df["Diagnostic_Genomic_R2"],
    width,
    label="Diagnostic genomic"
)

plt.axhline(
    0,
    linewidth=1
)

plt.xticks(
    x,
    phenotypes,
    rotation=20,
    ha="right"
)

plt.ylabel("Genomic R²")
plt.title(
    "Existing versus Diagnostic Genomic Models"
)

plt.legend()

plt.tight_layout()

comparison_fig = os.path.join(
    FIG_DIR,
    "existing_vs_diagnostic_genomic_R2_420.png"
)

plt.savefig(
    comparison_fig,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\n✓ Existing-vs-diagnostic figure saved:")
print(comparison_fig)


# =============================================================================
# WRITE TEXT INTERPRETATION
# =============================================================================

interpretation_path = os.path.join(
    TABLE_DIR,
    "genomic_signal_interpretation_420.txt"
)

with open(
    interpretation_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "GENOMIC SIGNAL ANALYSIS — Y1000+ YEAST\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Purpose:\n"
    )

    f.write(
        "Quantify whether genomic/BUSCO-based models provide "
        "predictive information beyond the phylogenetic baseline.\n\n"
    )

    for _, row in results_df.iterrows():

        f.write(
            f"PHENOTYPE: {row['Phenotype']}\n"
        )

        f.write(
            f"Feature set: {row['Feature_Set']}\n"
        )

        f.write(
            f"Model: {row['Model']}\n"
        )

        f.write(
            f"Phylogeny R2: {row['Phylogeny_R2']}\n"
        )

        f.write(
            f"Existing genomic R2: "
            f"{row['Existing_Genomic_R2']}\n"
        )

        f.write(
            f"Diagnostic genomic R2: "
            f"{row['Diagnostic_Genomic_R2']}\n"
        )

        f.write(
            f"Existing genomic gain vs phylogeny: "
            f"{row['Existing_Genomic_Gain_vs_Phylogeny']}\n"
        )

        f.write(
            f"Diagnostic genomic gain vs phylogeny: "
            f"{row['Diagnostic_Genomic_Gain_vs_Phylogeny']}\n"
        )

        f.write(
            f"Diagnostic change vs existing: "
            f"{row['Delta_R2_Diagnostic_vs_Existing']}\n"
        )

        f.write(
            f"Permutation p-value: "
            f"{row['Permutation_P']}\n"
        )

        f.write(
            f"Interpretation: "
            f"{row['Existing_Signal_Interpretation']}\n"
        )

        f.write("\n" + "-" * 80 + "\n\n")


print("\n✓ Interpretation file saved:")
print(interpretation_path)


# =============================================================================
# FINAL MESSAGE
# =============================================================================

print("\n\n" + "=" * 80)
print("GENOMIC SIGNAL ANALYSIS COMPLETE")
print("=" * 80)

print("\nMain output directory:")
print(OUT_DIR)

print("\nTables:")
print(TABLE_DIR)

print("\nFigures:")
print(FIG_DIR)

print("\nSummary:")
print(summary_path)

print("\nExisting models and OOF predictions were NOT overwritten.")

print("\nNext scientific stage:")
print(
    "If genomic signal is supported, proceed to feature-level "
    "interpretation / SHAP / pathway mapping."
)

print(
    "If genomic signal is weak, report the phylogenetic dependence "
    "and avoid overclaiming genomic prediction."
)