# ============================================================
# AUDIT VALIDATED PHYLOGENY-AWARE ML MODELS — Y1000+ / 420
# ============================================================
#
# Purpose:
#   Compare EXISTING model-performance outputs from the
#   phylogeny-aware ML pipeline.
#
#   This script DOES NOT retrain models.
#   It DOES NOT create new predictions.
#   It DOES NOT select models from raw training performance.
#
#   It audits the already-generated validation results and
#   identifies the validated configuration for each phenotype.
#
# Phenotypes:
#   1. Carbon_Breadth
#   2. Nitrogen_Breadth
#   3. Utilized_Median_Growth
#
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

RESULTS = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

# ============================================================
# POSSIBLE PERFORMANCE OUTPUT DIRECTORIES
# ============================================================

SEARCH_DIRS = [
    RESULTS / "performance_optimization_420",
    RESULTS / "model_performance_420",
    RESULTS / "feature_importance_analysis",
    RESULTS / "final_models_420",
    RESULTS,
]

# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT = (
    RESULTS
    / "validated_model_audit_420"
)

TABLES = OUTPUT / "tables"
REPORTS = OUTPUT / "reports"

TABLES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_files(patterns):

    found = []

    for directory in SEARCH_DIRS:

        if not directory.exists():
            continue

        for pattern in patterns:

            found.extend(directory.rglob(pattern))

    # remove duplicates
    unique = []

    for f in found:

        if f not in unique:
            unique.append(f)

    return unique


def normalize_columns(df):

    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


def detect_column(df, candidates):

    lookup = {
        str(c).lower().replace(" ", "_"): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().replace(" ", "_")

        if key in lookup:
            return lookup[key]

    return None


def numeric(df, column):

    if column is None:
        return pd.Series(np.nan, index=df.index)

    return pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("VALIDATED PHYLOGENY-AWARE ML MODEL AUDIT")
print("Y1000+ / 420 TAXA")
print("=" * 80)

print()
print("This script audits EXISTING validation outputs.")
print("No models are retrained.")
print("No new predictions are generated.")
print()

# ============================================================
# SEARCH FOR PERFORMANCE TABLES
# ============================================================

print("=" * 80)
print("SEARCHING EXISTING ML PERFORMANCE OUTPUTS")
print("=" * 80)

patterns = [
    "*performance*.csv",
    "*optimization*.csv",
    "*model*.csv",
    "*fold*.csv",
]

files = find_files(patterns)

if not files:

    raise FileNotFoundError(
        "\nNo existing model-performance CSV files were found.\n"
        f"Searched under:\n{RESULTS}"
    )

print()
print("Candidate performance files found:")

for f in files:

    print(f"  {f}")

# ============================================================
# LOAD RELEVANT PERFORMANCE TABLES
# ============================================================

records = []

print()
print("=" * 80)
print("READING PERFORMANCE TABLES")
print("=" * 80)

for file in files:

    try:

        df = pd.read_csv(file)

    except Exception:

        continue

    df = normalize_columns(df)

    if len(df) == 0:
        continue

    # --------------------------------------------------------
    # Detect phenotype
    # --------------------------------------------------------

    phenotype_col = detect_column(
        df,
        [
            "Phenotype",
            "Target",
            "Response",
            "Trait"
        ]
    )

    # --------------------------------------------------------
    # Detect model
    # --------------------------------------------------------

    model_col = detect_column(
        df,
        [
            "Model",
            "Best_Model",
            "Algorithm"
        ]
    )

    # --------------------------------------------------------
    # Detect feature set
    # --------------------------------------------------------

    feature_col = detect_column(
        df,
        [
            "Feature_Set",
            "FeatureSet",
            "Features",
            "Feature_Configuration"
        ]
    )

    # --------------------------------------------------------
    # Detect R2
    # --------------------------------------------------------

    r2_candidates = [
        "OOF_R2",
        "Overall_OOF_R2",
        "Mean_CV_R2",
        "Mean_R2",
        "CV_R2",
        "R2",
        "R_squared",
    ]

    r2_col = detect_column(
        df,
        r2_candidates
    )

    # --------------------------------------------------------
    # Detect RMSE
    # --------------------------------------------------------

    rmse_candidates = [
        "OOF_RMSE",
        "Overall_OOF_RMSE",
        "Mean_CV_RMSE",
        "Mean_RMSE",
        "CV_RMSE",
        "RMSE",
    ]

    rmse_col = detect_column(
        df,
        rmse_candidates
    )

    # --------------------------------------------------------
    # Detect MAE
    # --------------------------------------------------------

    mae_col = detect_column(
        df,
        [
            "OOF_MAE",
            "Overall_OOF_MAE",
            "Mean_CV_MAE",
            "Mean_MAE",
            "CV_MAE",
            "MAE",
        ]
    )

    # --------------------------------------------------------
    # Need at least phenotype/model/performance
    # --------------------------------------------------------

    if phenotype_col is None:
        continue

    if model_col is None:
        continue

    if r2_col is None and rmse_col is None:
        continue

    temp = pd.DataFrame()

    temp["Phenotype"] = df[phenotype_col].astype(str)

    temp["Model"] = df[model_col].astype(str)

    if feature_col is not None:

        temp["Feature_Set"] = (
            df[feature_col]
            .astype(str)
        )

    else:

        temp["Feature_Set"] = "Not reported"

    temp["R2"] = numeric(
        df,
        r2_col
    )

    temp["RMSE"] = numeric(
        df,
        rmse_col
    )

    temp["MAE"] = numeric(
        df,
        mae_col
    )

    temp["Source_File"] = str(file)

    records.append(temp)

# ============================================================
# COMBINE
# ============================================================

if not records:

    raise RuntimeError(
        "\nPerformance files were found, but none contained "
        "recognizable phenotype/model/performance columns."
    )

performance = pd.concat(
    records,
    ignore_index=True
)

# Remove meaningless rows
performance = performance[
    performance["Phenotype"].notna()
].copy()

performance = performance[
    ~performance["Phenotype"].str.lower().isin(
        ["nan", "none", ""]
    )
]

# ============================================================
# STANDARDIZE PHENOTYPE NAMES
# ============================================================

phenotype_map = {

    "carbon breadth":
        "Carbon_Breadth",

    "carbon_breadth":
        "Carbon_Breadth",

    "nitrogen breadth":
        "Nitrogen_Breadth",

    "nitrogen_breadth":
        "Nitrogen_Breadth",

    "utilized median growth":
        "Utilized_Median_Growth",

    "utilized_median_growth":
        "Utilized_Median_Growth",

}

performance["Phenotype"] = (
    performance["Phenotype"]
    .str.strip()
    .map(
        lambda x:
        phenotype_map.get(
            x.lower(),
            x
        )
    )
)

# ============================================================
# SAVE RAW AUDIT TABLE
# ============================================================

raw_output = (
    TABLES
    / "all_existing_ml_performance_records_420.csv"
)

performance.to_csv(
    raw_output,
    index=False
)

print()
print(f"Performance records collected: {len(performance)}")
print()
print(f"Saved:")
print(raw_output)

# ============================================================
# KEEP PROJECT PHENOTYPES
# ============================================================

target_phenotypes = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
]

audit = performance[
    performance["Phenotype"].isin(
        target_phenotypes
    )
].copy()

# ============================================================
# REMOVE DUPLICATES
# ============================================================

audit = audit.drop_duplicates(
    subset=[
        "Phenotype",
        "Feature_Set",
        "Model",
        "R2",
        "RMSE",
        "MAE",
        "Source_File",
    ]
)

# ============================================================
# PERFORMANCE SUMMARY
# ============================================================

summary_rows = []

for phenotype in target_phenotypes:

    subset = audit[
        audit["Phenotype"] == phenotype
    ].copy()

    if subset.empty:
        continue

    summary_rows.append({

        "Phenotype":
            phenotype,

        "Number_of_Configurations":
            len(subset),

        "Best_R2":
            subset["R2"].max()
            if subset["R2"].notna().any()
            else np.nan,

        "Best_R2_Model":
            subset.loc[
                subset["R2"].idxmax(),
                "Model"
            ]
            if subset["R2"].notna().any()
            else "Unavailable",

        "Best_R2_Feature_Set":
            subset.loc[
                subset["R2"].idxmax(),
                "Feature_Set"
            ]
            if subset["R2"].notna().any()
            else "Unavailable",

        "Lowest_RMSE":
            subset["RMSE"].min()
            if subset["RMSE"].notna().any()
            else np.nan,

        "Lowest_RMSE_Model":
            subset.loc[
                subset["RMSE"].idxmin(),
                "Model"
            ]
            if subset["RMSE"].notna().any()
            else "Unavailable",

        "Lowest_RMSE_Feature_Set":
            subset.loc[
                subset["RMSE"].idxmin(),
                "Feature_Set"
            ]
            if subset["RMSE"].notna().any()
            else "Unavailable",
    })

summary = pd.DataFrame(
    summary_rows
)

summary_file = (
    TABLES
    / "validated_model_performance_summary_420.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 80)
print("VALIDATED MODEL PERFORMANCE SUMMARY")
print("=" * 80)

for _, row in summary.iterrows():

    print()
    print(row["Phenotype"])

    print(
        f"  Configurations evaluated : "
        f"{row['Number_of_Configurations']}"
    )

    print(
        f"  Best available R²        : "
        f"{row['Best_R2']}"
    )

    print(
        f"  Model                    : "
        f"{row['Best_R2_Model']}"
    )

    print(
        f"  Feature set              : "
        f"{row['Best_R2_Feature_Set']}"
    )

    print(
        f"  Lowest RMSE              : "
        f"{row['Lowest_RMSE']}"
    )

    print(
        f"  RMSE model               : "
        f"{row['Lowest_RMSE_Model']}"
    )

# ============================================================
# RANK EXISTING CONFIGURATIONS
# ============================================================

ranked = audit.copy()

# R² is the primary criterion.
# RMSE is retained as a secondary diagnostic.

ranked["R2_Rank"] = (
    ranked
    .groupby("Phenotype")["R2"]
    .rank(
        ascending=False,
        method="min"
    )
)

ranked["RMSE_Rank"] = (
    ranked
    .groupby("Phenotype")["RMSE"]
    .rank(
        ascending=True,
        method="min"
    )
)

ranked_file = (
    TABLES
    / "existing_model_configuration_ranking_420.csv"
)

ranked.to_csv(
    ranked_file,
    index=False
)

# ============================================================
# IDENTIFY THE EXISTING VALIDATED CONFIGURATION
# ============================================================

selected_rows = []

for phenotype in target_phenotypes:

    subset = ranked[
        ranked["Phenotype"] == phenotype
    ].copy()

    subset = subset[
        subset["R2"].notna()
    ]

    if subset.empty:
        continue

    # Highest already-reported validation R².
    best = subset.sort_values(
        by=[
            "R2",
            "RMSE"
        ],
        ascending=[
            False,
            True
        ]
    ).iloc[0]

    selected_rows.append({

        "Phenotype":
            phenotype,

        "Selected_Model":
            best["Model"],

        "Selected_Feature_Set":
            best["Feature_Set"],

        "Validated_R2":
            best["R2"],

        "Validated_RMSE":
            best["RMSE"],

        "Validated_MAE":
            best["MAE"],

        "Selection_Basis":
            "Best already-reported validation R2",

        "Source_File":
            best["Source_File"],
    })

selected = pd.DataFrame(
    selected_rows
)

selected_file = (
    TABLES
    / "validated_selected_model_configurations_420.csv"
)

selected.to_csv(
    selected_file,
    index=False
)

# ============================================================
# IMPORTANT: FLAG NEGATIVE R2
# ============================================================

selected["Predictive_Status"] = np.where(
    selected["Validated_R2"] > 0,
    "Positive_validation_R2",
    "Non_positive_validation_R2"
)

status_file = (
    TABLES
    / "validated_model_predictive_status_420.csv"
)

selected.to_csv(
    status_file,
    index=False
)

# ============================================================
# REPORT
# ============================================================

report_file = (
    REPORTS
    / "validated_phylogeny_aware_ml_model_audit_420.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "VALIDATED PHYLOGENY-AWARE ML MODEL AUDIT\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        "Purpose\n"
    )

    f.write(
        "This audit compares existing phylogeny-aware ML "
        "validation outputs without retraining models.\n\n"
    )

    f.write(
        "The audit is restricted to the three project phenotypes:\n"
    )

    for p in target_phenotypes:
        f.write(f"  - {p}\n")

    f.write("\n")

    f.write(
        "MODEL SELECTION PRINCIPLE\n"
    )

    f.write("-" * 80 + "\n")

    f.write(
        "The selected configuration is the best already-reported "
        "validation configuration based primarily on validation R2. "
        "No new model fitting is performed by this script.\n\n"
    )

    f.write(
        "SELECTED CONFIGURATIONS\n"
    )

    f.write("-" * 80 + "\n")

    for _, row in selected.iterrows():

        f.write(
            f"{row['Phenotype']}\n"
        )

        f.write(
            f"  Model       : "
            f"{row['Selected_Model']}\n"
        )

        f.write(
            f"  Feature set : "
            f"{row['Selected_Feature_Set']}\n"
        )

        f.write(
            f"  Validation R2 : "
            f"{row['Validated_R2']}\n"
        )

        f.write(
            f"  Validation RMSE : "
            f"{row['Validated_RMSE']}\n"
        )

        f.write(
            f"  Validation MAE : "
            f"{row['Validated_MAE']}\n"
        )

        f.write(
            f"  Status : "
            f"{row['Predictive_Status']}\n"
        )

        f.write("\n")

    f.write(
        "INTERPRETATION\n"
    )

    f.write("-" * 80 + "\n")

    if (
        len(selected) > 0
        and
        (selected["Validated_R2"] <= 0).any()
    ):

        f.write(
            "At least one selected phenotype has a non-positive "
            "validation R2. This indicates that the existing "
            "phylogeny-aware model does not demonstrate predictive "
            "performance above the corresponding baseline for that "
            "phenotype under the evaluated validation scheme.\n\n"
        )

        f.write(
            "These results should not be presented as strong "
            "predictive genome-to-phenotype performance. Further "
            "methodological investigation is warranted before "
            "claiming robust prediction.\n\n"
        )

    else:

        f.write(
            "The selected configurations have positive validation "
            "R2 values. These configurations can be carried forward "
            "for downstream interpretation, subject to independent "
            "validation.\n\n"
        )

    f.write(
        "IMPORTANT\n"
    )

    f.write("-" * 80 + "\n")

    f.write(
        "This audit does not establish independent generalization. "
        "Independent validation remains necessary before claiming "
        "external predictive performance.\n"
    )

# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 80)
print("MODEL AUDIT COMPLETE")
print("=" * 80)

print()
print("TABLES:")

print(
    raw_output
)

print(
    summary_file
)

print(
    ranked_file
)

print(
    selected_file
)

print(
    status_file
)

print()
print("REPORT:")

print(
    report_file
)

print()
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

if len(selected) > 0:

    for _, row in selected.iterrows():

        print(
            f"{row['Phenotype']} -> "
            f"{row['Selected_Model']} + "
            f"{row['Selected_Feature_Set']} | "
            f"R2={row['Validated_R2']}"
        )

print()
print(
    "No models were retrained."
)

print(
    "No new predictions were generated."
)

print(
    "Use the selected validated configurations only after "
    "reviewing the validation metrics."
)

print("=" * 80)