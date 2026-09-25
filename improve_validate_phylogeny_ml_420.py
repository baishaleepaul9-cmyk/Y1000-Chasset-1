# -*- coding: utf-8 -*-

"""
improve_validate_phylogeny_ml_420.py

FINAL VALIDATION / MODEL-IMPROVEMENT ANALYSIS
Y1000+ / 420 TAXA

This version:

1. Loads the existing 420-taxon genomic ML matrix.
2. Loads the existing 79 robust BUSCO feature set.
3. Loads the existing phylogenetic CV assignments.
4. Reuses previously completed phylogenetic CV results when available.
5. Reuses previously completed repeated-random CV results when available.
6. Reuses previously completed permutation/null results when available.
7. Performs empirical permutation p-value calculation.
8. Compares phylogenetic and random validation.
9. Identifies validated model candidates.
10. Produces a final validation report.

IMPORTANT
---------
- Existing final models are NOT overwritten.
- Existing candidate-chassis results are NOT modified.
- Existing feature sets are NOT modified.
- Existing completed CV/permutation calculations are reused.
- Missing calculations are generated only when necessary.
- This script does NOT claim that a Brownian covariance matrix is directly
  incorporated into the ML estimator.
- The framework is genome-based ML with phylogeny-aware validation.

FIX APPLIED
-----------
The original script created Overall_OOF_R2 correctly, but then merged the same
column a second time. Pandas therefore generated suffixed columns and the
script subsequently failed with:

    KeyError: 'Overall_OOF_R2'

This version avoids that duplicate merge and explicitly detects the correct
observed-R2 column.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

from sklearn.model_selection import KFold

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PROJECT PATHS
# =============================================================================

ROOT = Path(r"C:\Y1000_chassis_project")

RESULTS = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
)

ML_MATRIX = (
    RESULTS
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)

PHYLO_CV = (
    RESULTS
    / "phylogeny_cv"
    / "phylogenetic_cv_assignments_420.csv"
)

ROBUST_BUSCOS = (
    RESULTS
    / "phylogeny_cv"
    / "model_training"
    / "feature_importance_analysis"
    / "robust_feature_analysis"
    / "integrated_robust_candidates"
    / "robust_all_three_buscos_420.csv"
)

OUTPUT_DIR = (
    RESULTS
    / "model_improvement_validation_420"
)

TABLES = OUTPUT_DIR / "tables"
FIGURES = OUTPUT_DIR / "figures"
REPORTS = OUTPUT_DIR / "reports"

for directory in [
    TABLES,
    FIGURES,
    REPORTS,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# SETTINGS
# =============================================================================

PHENOTYPES = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
]

EXPECTED_TAXA = 420
EXPECTED_ROBUST_FEATURES = 79

N_PHYLO_FOLDS = 5

N_RANDOM_REPEATS = 10
N_RANDOM_FOLDS = 5

N_PERMUTATIONS = 100

RANDOM_STATE = 42


# =============================================================================
# EXISTING RESULT FILES
# =============================================================================

PHYLO_RESULTS_FILE = (
    TABLES
    / "phylogenetic_cv_revalidation_results_420.csv"
)

PHYLO_SUMMARY_FILE = (
    TABLES
    / "phylogenetic_cv_model_summary_420.csv"
)

RANDOM_RESULTS_FILE = (
    TABLES
    / "repeated_random_cv_results_420.csv"
)

RANDOM_SUMMARY_FILE = (
    TABLES
    / "repeated_random_cv_model_summary_420.csv"
)

COMPARISON_FILE = (
    TABLES
    / "random_vs_phylogenetic_model_comparison_420.csv"
)

PERMUTATION_RESULTS_FILE = (
    TABLES
    / "phylogenetic_permutation_null_results_420.csv"
)

PERMUTATION_SUMMARY_FILE = (
    TABLES
    / "phylogenetic_permutation_null_summary_420.csv"
)

FINAL_COMPARISON_FILE = (
    TABLES
    / "FINAL_MODEL_VALIDATION_COMPARISON_420.csv"
)

CANDIDATE_FILE = (
    TABLES
    / "final_validated_model_candidates_420.csv"
)

STATUS_FILE = (
    TABLES
    / "validated_predictive_status_420.csv"
)


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def build_models():

    models = {

        "Ridge": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                Ridge(
                    alpha=10.0
                )
            ),
        ]),

        "ElasticNet": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.5,
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                )
            ),
        ]),

        "SVR_RBF": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=10.0,
                    epsilon=0.1,
                    gamma="scale",
                )
            ),
        ]),

        "SVR_RBF_RobustScaler": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                RobustScaler()
            ),
            (
                "model",
                SVR(
                    kernel="rbf",
                    C=10.0,
                    epsilon=0.1,
                    gamma="scale",
                )
            ),
        ]),

        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]),

        "ExtraTrees": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]),

        "GradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                )
            ),
        ]),

        "HistGradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.03,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                )
            ),
        ]),
    }

    return models


# =============================================================================
# METRIC FUNCTIONS
# =============================================================================

def calculate_metrics(
    y_true,
    y_pred
):

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    try:

        r2 = r2_score(
            y_true,
            y_pred
        )

    except Exception:

        r2 = np.nan

    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
    }


# =============================================================================
# EMPIRICAL P-VALUE
# =============================================================================

def empirical_p_value(
    observed,
    null_values
):

    if pd.isna(observed):
        return np.nan

    null_values = np.asarray(
        null_values,
        dtype=float
    )

    null_values = null_values[
        np.isfinite(
            null_values
        )
    ]

    if len(null_values) == 0:
        return np.nan

    return (
        1
        + np.sum(
            null_values >= observed
        )
    ) / (
        len(null_values)
        + 1
    )


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("VALIDATING / IMPROVING PHYLOGENY-AWARE ML")
print("Y1000+ / 420 TAXA")
print("=" * 80)

print()
print("RESUME-AWARE VERSION")
print()
print("This script:")
print("  - does NOT overwrite existing final models")
print("  - does NOT modify candidate chassis")
print("  - does NOT modify the existing feature set")
print("  - reuses completed CV results")
print("  - reuses completed permutation results")
print("  - only calculates missing results")
print()


# =============================================================================
# CHECK INPUTS
# =============================================================================

print("=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)

for path, label in [
    (
        ML_MATRIX,
        "420-taxon ML matrix"
    ),
    (
        PHYLO_CV,
        "Phylogenetic CV assignments"
    ),
    (
        ROBUST_BUSCOS,
        "Robust BUSCO feature set"
    ),
]:

    print()
    print(label)
    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print("FOUND")


# =============================================================================
# LOAD ML MATRIX
# =============================================================================

print()
print("=" * 80)
print("LOADING 420-TAXON GENOMIC ML MATRIX")
print("=" * 80)

df = pd.read_csv(
    ML_MATRIX
)

print()
print("Rows:", len(df))
print("Columns:", len(df.columns))

if len(df) != EXPECTED_TAXA:

    print(
        "WARNING:"
        f" Expected {EXPECTED_TAXA} taxa but found {len(df)}."
    )


# =============================================================================
# LOAD PHYLOGENETIC CV ASSIGNMENTS
# =============================================================================

print()
print("=" * 80)
print("LOADING PHYLOGENETIC CV ASSIGNMENTS")
print("=" * 80)

cv = pd.read_csv(
    PHYLO_CV
)

print()
print("Rows:", len(cv))
print("Columns:", len(cv.columns))

required_cv_columns = [
    "Assembly_Accession",
    "Phylogenetic_Fold",
]

for column in required_cv_columns:

    if column not in cv.columns:

        raise RuntimeError(
            f"Missing required CV column: {column}"
        )


# =============================================================================
# NORMALIZE ACCESSIONS
# =============================================================================

df["Assembly_Accession"] = (
    df["Assembly_Accession"]
    .astype(str)
    .str.strip()
)

cv["Assembly_Accession"] = (
    cv["Assembly_Accession"]
    .astype(str)
    .str.strip()
)


# =============================================================================
# MERGE PHYLOGENETIC FOLDS
# =============================================================================

cv_lookup = (
    cv[
        [
            "Assembly_Accession",
            "Phylogenetic_Fold",
        ]
    ]
    .drop_duplicates(
        subset="Assembly_Accession"
    )
)

df = df.merge(
    cv_lookup,
    on="Assembly_Accession",
    how="left"
)

if df["Phylogenetic_Fold"].isna().any():

    missing = df.loc[
        df["Phylogenetic_Fold"].isna(),
        [
            "Species",
            "Assembly_Accession",
        ]
    ]

    print()
    print("Missing phylogenetic assignments:")
    print(
        missing.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Some taxa do not have phylogenetic CV assignments."
    )

df["Phylogenetic_Fold"] = (
    df["Phylogenetic_Fold"]
    .astype(int)
)

phylo_folds = sorted(
    df["Phylogenetic_Fold"]
    .unique()
)

print()
print("Phylogenetic folds:")
print(phylo_folds)

print()
print("Phylogenetic fold sizes:")
print(
    df["Phylogenetic_Fold"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# LOAD ROBUST BUSCO FEATURES
# =============================================================================

print()
print("=" * 80)
print("LOADING ROBUST BUSCO FEATURE SET")
print("=" * 80)

robust = pd.read_csv(
    ROBUST_BUSCOS
)

print()
print("Rows:", len(robust))
print("Columns:", len(robust.columns))

possible_busco_columns = [
    c
    for c in robust.columns
    if str(c).lower()
    in [
        "busco",
        "busco_id",
        "busco_family",
    ]
]

if not possible_busco_columns:

    raise RuntimeError(
        "Could not identify BUSCO column in robust feature file."
    )

BUSCO_COLUMN = possible_busco_columns[0]

robust_buscos = (
    robust[BUSCO_COLUMN]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print()
print("Robust BUSCOs:", len(robust_buscos))

if len(robust_buscos) != EXPECTED_ROBUST_FEATURES:

    print(
        "WARNING:"
        f" expected {EXPECTED_ROBUST_FEATURES},"
        f" found {len(robust_buscos)}."
    )

available_buscos = [
    b
    for b in robust_buscos
    if b in df.columns
]

print(
    "BUSCOs available in ML matrix:",
    len(available_buscos)
)

if len(available_buscos) == 0:

    raise RuntimeError(
        "No robust BUSCO features were found in the ML matrix."
    )


# =============================================================================
# SAVE FEATURE AUDIT
# =============================================================================

feature_audit = pd.DataFrame({
    "BUSCO": robust_buscos,
    "Available_In_420_Matrix": [
        b in df.columns
        for b in robust_buscos
    ]
})

feature_audit.to_csv(
    TABLES
    / "robust_BUSCO_feature_audit_420.csv",
    index=False
)


# =============================================================================
# SAVE VALIDATION MATRIX
# =============================================================================

validation_matrix = df[
    [
        "Species",
        "Assembly_Accession",
        "Phylogenetic_Fold",
    ]
    + available_buscos
    + PHENOTYPES
].copy()

validation_matrix.to_csv(
    TABLES
    / "validation_training_matrix_420.csv",
    index=False
)


# =============================================================================
# MODELS
# =============================================================================

models = build_models()

print()
print("=" * 80)
print("MODELS TO BE EVALUATED")
print("=" * 80)

for model_name in models:

    print(
        "  ",
        model_name
    )


# =============================================================================
# CV RESULTS
# =============================================================================

print()
print("=" * 80)
print("CHECKING EXISTING CROSS-VALIDATION RESULTS")
print("=" * 80)


# -----------------------------------------------------------------------------
# PHYLOGENETIC CV
# -----------------------------------------------------------------------------

if (
    PHYLO_RESULTS_FILE.exists()
    and PHYLO_SUMMARY_FILE.exists()
):

    print()
    print("Existing phylogenetic CV results detected.")
    print("REUSING existing results.")
    print(
        PHYLO_RESULTS_FILE
    )

    phylo_df = pd.read_csv(
        PHYLO_RESULTS_FILE
    )

    phylo_summary = pd.read_csv(
        PHYLO_SUMMARY_FILE
    )

else:

    print()
    print("Existing phylogenetic CV results not found.")
    print("Running phylogenetic CV...")

    phylo_results = []

    for phenotype in PHENOTYPES:

        print()
        print("=" * 80)
        print(
            "PHENOTYPE:",
            phenotype
        )
        print("=" * 80)

        y = pd.to_numeric(
            df[phenotype],
            errors="coerce"
        )

        valid = y.notna()

        X = df.loc[
            valid,
            available_buscos
        ].copy()

        y = y.loc[
            valid
        ].copy()

        metadata = df.loc[
            valid,
            [
                "Species",
                "Assembly_Accession",
                "Phylogenetic_Fold",
            ]
        ].copy()

        for model_name, model in models.items():

            print(
                "  Model:",
                model_name
            )

            all_pred = np.full(
                len(y),
                np.nan
            )

            for fold in phylo_folds:

                train_idx = (
                    metadata[
                        "Phylogenetic_Fold"
                    ]
                    != fold
                )

                test_idx = (
                    metadata[
                        "Phylogenetic_Fold"
                    ]
                    == fold
                )

                estimator = clone(
                    model
                )

                estimator.fit(
                    X.loc[train_idx],
                    y.loc[train_idx]
                )

                pred = estimator.predict(
                    X.loc[test_idx]
                )

                positions = np.where(
                    test_idx
                )[0]

                all_pred[
                    positions
                ] = pred

                metrics = calculate_metrics(
                    y.loc[test_idx],
                    pred
                )

                phylo_results.append({
                    "Phenotype":
                        phenotype,

                    "Model":
                        model_name,

                    "Fold":
                        int(fold),

                    "N_Train":
                        int(train_idx.sum()),

                    "N_Test":
                        int(test_idx.sum()),

                    "RMSE":
                        metrics["RMSE"],

                    "MAE":
                        metrics["MAE"],

                    "R2":
                        metrics["R2"],
                })

            overall = calculate_metrics(
                y,
                all_pred
            )

            phylo_results.append({
                "Phenotype":
                    phenotype,

                "Model":
                    model_name,

                "Fold":
                    "OOF_OVERALL",

                "N_Train":
                    len(y),

                "N_Test":
                    len(y),

                "RMSE":
                    overall["RMSE"],

                "MAE":
                    overall["MAE"],

                "R2":
                    overall["R2"],
            })

    phylo_df = pd.DataFrame(
        phylo_results
    )

    phylo_df.to_csv(
        PHYLO_RESULTS_FILE,
        index=False
    )

    phylo_fold_only = phylo_df[
        phylo_df["Fold"]
        != "OOF_OVERALL"
    ].copy()

    phylo_summary = (
        phylo_fold_only
        .groupby(
            [
                "Phenotype",
                "Model"
            ]
        )
        .agg(
            Mean_RMSE=("RMSE", "mean"),
            SD_RMSE=("RMSE", "std"),
            Mean_MAE=("MAE", "mean"),
            SD_MAE=("MAE", "std"),
            Mean_R2=("R2", "mean"),
            SD_R2=("R2", "std"),
            Min_R2=("R2", "min"),
            Max_R2=("R2", "max"),
        )
        .reset_index()
    )

    phylo_overall = phylo_df[
        phylo_df["Fold"]
        == "OOF_OVERALL"
    ].copy()

    phylo_overall = phylo_overall[
        [
            "Phenotype",
            "Model",
            "RMSE",
            "MAE",
            "R2",
        ]
    ].rename(
        columns={
            "RMSE":
                "Overall_OOF_RMSE",

            "MAE":
                "Overall_OOF_MAE",

            "R2":
                "Overall_OOF_R2",
        }
    )

    phylo_summary = phylo_summary.merge(
        phylo_overall,
        on=[
            "Phenotype",
            "Model",
        ],
        how="left"
    )

    phylo_summary.to_csv(
        PHYLO_SUMMARY_FILE,
        index=False
    )


# =============================================================================
# REPEATED RANDOM CV
# =============================================================================

print()
print("=" * 80)
print("CHECKING REPEATED RANDOM CV RESULTS")
print("=" * 80)

if (
    RANDOM_RESULTS_FILE.exists()
    and RANDOM_SUMMARY_FILE.exists()
):

    print()
    print("Existing repeated random CV results detected.")
    print("REUSING existing results.")

    random_df = pd.read_csv(
        RANDOM_RESULTS_FILE
    )

    random_summary = pd.read_csv(
        RANDOM_SUMMARY_FILE
    )

else:

    print()
    print("Existing random CV results not found.")
    print("Running repeated random CV...")

    random_results = []

    for phenotype in PHENOTYPES:

        print()
        print(
            "PHENOTYPE:",
            phenotype
        )

        y = pd.to_numeric(
            df[phenotype],
            errors="coerce"
        )

        valid = y.notna()

        X = df.loc[
            valid,
            available_buscos
        ].copy()

        y = y.loc[
            valid
        ].copy()

        for model_name, model in models.items():

            print(
                "  Model:",
                model_name
            )

            for repeat in range(
                N_RANDOM_REPEATS
            ):

                rkf = KFold(
                    n_splits=N_RANDOM_FOLDS,
                    shuffle=True,
                    random_state=(
                        RANDOM_STATE
                        + repeat
                    ),
                )

                for fold_number, (
                    train_idx,
                    test_idx
                ) in enumerate(
                    rkf.split(X),
                    start=1
                ):

                    estimator = clone(
                        model
                    )

                    estimator.fit(
                        X.iloc[train_idx],
                        y.iloc[train_idx]
                    )

                    pred = estimator.predict(
                        X.iloc[test_idx]
                    )

                    metrics = calculate_metrics(
                        y.iloc[test_idx],
                        pred
                    )

                    random_results.append({
                        "Phenotype":
                            phenotype,

                        "Model":
                            model_name,

                        "Repeat":
                            int(repeat + 1),

                        "Fold":
                            int(fold_number),

                        "RMSE":
                            metrics["RMSE"],

                        "MAE":
                            metrics["MAE"],

                        "R2":
                            metrics["R2"],
                    })

    random_df = pd.DataFrame(
        random_results
    )

    random_df.to_csv(
        RANDOM_RESULTS_FILE,
        index=False
    )

    random_summary = (
        random_df
        .groupby(
            [
                "Phenotype",
                "Model"
            ]
        )
        .agg(
            Mean_RMSE=("RMSE", "mean"),
            SD_RMSE=("RMSE", "std"),
            Mean_MAE=("MAE", "mean"),
            SD_MAE=("MAE", "std"),
            Mean_R2=("R2", "mean"),
            SD_R2=("R2", "std"),
            Min_R2=("R2", "min"),
            Max_R2=("R2", "max"),
        )
        .reset_index()
    )

    random_summary.to_csv(
        RANDOM_SUMMARY_FILE,
        index=False
    )


# =============================================================================
# ENSURE OVERALL PHYLOGENETIC R2 EXISTS
# =============================================================================

if "Overall_OOF_R2" not in phylo_summary.columns:

    print()
    print(
        "Overall_OOF_R2 not found in existing phylogenetic summary."
    )

    phylo_overall = phylo_df[
        phylo_df["Fold"].astype(str)
        == "OOF_OVERALL"
    ].copy()

    if len(phylo_overall) == 0:

        raise RuntimeError(
            "Could not reconstruct Overall_OOF_R2."
        )

    phylo_overall = phylo_overall[
        [
            "Phenotype",
            "Model",
            "RMSE",
            "MAE",
            "R2",
        ]
    ].rename(
        columns={
            "RMSE":
                "Overall_OOF_RMSE",

            "MAE":
                "Overall_OOF_MAE",

            "R2":
                "Overall_OOF_R2",
        }
    )

    phylo_summary = phylo_summary.merge(
        phylo_overall,
        on=[
            "Phenotype",
            "Model",
        ],
        how="left"
    )


# =============================================================================
# RANDOM VS PHYLOGENETIC COMPARISON
# =============================================================================

print()
print("=" * 80)
print("BUILDING RANDOM VS PHYLOGENETIC COMPARISON")
print("=" * 80)

comparison = phylo_summary.merge(
    random_summary,
    on=[
        "Phenotype",
        "Model",
    ],
    suffixes=(
        "_Phylogenetic",
        "_Random",
    )
)

comparison[
    "R2_Difference_Random_Minus_Phylogenetic"
] = (
    comparison[
        "Mean_R2_Random"
    ]
    -
    comparison[
        "Mean_R2_Phylogenetic"
    ]
)

comparison[
    "RMSE_Difference_Random_Minus_Phylogenetic"
] = (
    comparison[
        "Mean_RMSE_Random"
    ]
    -
    comparison[
        "Mean_RMSE_Phylogenetic"
    ]
)


# =============================================================================
# PERMUTATION / NULL ANALYSIS
# =============================================================================

print()
print("=" * 80)
print("PERMUTATION / NULL ANALYSIS")
print("=" * 80)

print()
print(
    f"Number of permutations: {N_PERMUTATIONS}"
)

if PERMUTATION_RESULTS_FILE.exists():

    print()
    print(
        "Existing permutation results detected."
    )

    print(
        "REUSING existing permutation results."
    )

    print(
        PERMUTATION_RESULTS_FILE
    )

    permutation_df = pd.read_csv(
        PERMUTATION_RESULTS_FILE
    )

else:

    print()
    print(
        "No existing permutation results found."
    )

    print(
        "Running permutation/null analysis..."
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    permutation_results = []

    for phenotype in PHENOTYPES:

        print()
        print(
            "Permutation phenotype:",
            phenotype
        )

        y_original = pd.to_numeric(
            df[phenotype],
            errors="coerce"
        )

        valid = y_original.notna()

        X = df.loc[
            valid,
            available_buscos
        ].copy()

        y_original = y_original.loc[
            valid
        ].copy()

        metadata = df.loc[
            valid,
            [
                "Phylogenetic_Fold"
            ]
        ].copy()

        for model_name, model in models.items():

            print(
                "  Model:",
                model_name
            )

            for permutation in range(
                N_PERMUTATIONS
            ):

                y_perm = pd.Series(
                    rng.permutation(
                        y_original.values
                    ),
                    index=y_original.index
                )

                pred = np.full(
                    len(y_perm),
                    np.nan
                )

                for fold in phylo_folds:

                    train_idx = (
                        metadata[
                            "Phylogenetic_Fold"
                        ]
                        != fold
                    )

                    test_idx = (
                        metadata[
                            "Phylogenetic_Fold"
                        ]
                        == fold
                    )

                    estimator = clone(
                        model
                    )

                    estimator.fit(
                        X.loc[train_idx],
                        y_perm.loc[train_idx]
                    )

                    fold_pred = estimator.predict(
                        X.loc[test_idx]
                    )

                    pred[
                        np.where(test_idx)[0]
                    ] = fold_pred

                metrics = calculate_metrics(
                    y_perm,
                    pred
                )

                permutation_results.append({
                    "Phenotype":
                        phenotype,

                    "Model":
                        model_name,

                    "Permutation":
                        int(permutation + 1),

                    "RMSE":
                        metrics["RMSE"],

                    "MAE":
                        metrics["MAE"],

                    "R2":
                        metrics["R2"],
                })

                # Periodic checkpoint
                if (
                    len(permutation_results)
                    % 50
                    == 0
                ):

                    checkpoint_df = pd.DataFrame(
                        permutation_results
                    )

                    checkpoint_df.to_csv(
                        PERMUTATION_RESULTS_FILE,
                        index=False
                    )

                    print(
                        "    Checkpoint saved:"
                        f" {len(permutation_results)} records"
                    )

    permutation_df = pd.DataFrame(
        permutation_results
    )

    permutation_df.to_csv(
        PERMUTATION_RESULTS_FILE,
        index=False
    )


# =============================================================================
# PERMUTATION SUMMARY
# =============================================================================

print()
print("=" * 80)
print("PERMUTATION SUMMARY")
print("=" * 80)

permutation_summary = (
    permutation_df
    .groupby(
        [
            "Phenotype",
            "Model"
        ]
    )
    .agg(
        Null_Mean_R2=("R2", "mean"),
        Null_SD_R2=("R2", "std"),
        Null_Max_R2=("R2", "max"),
        Null_Mean_RMSE=("RMSE", "mean"),
        Null_SD_RMSE=("RMSE", "std"),
    )
    .reset_index()
)

permutation_summary.to_csv(
    PERMUTATION_SUMMARY_FILE,
    index=False
)


# =============================================================================
# MERGE NULL SUMMARY
# =============================================================================

comparison = comparison.merge(
    permutation_summary,
    on=[
        "Phenotype",
        "Model",
    ],
    how="left"
)


# =============================================================================
# FIXED EMPIRICAL NULL TEST
# =============================================================================

print()
print("=" * 80)
print("CALCULATING EMPIRICAL PERMUTATION P-VALUES")
print("=" * 80)


# IMPORTANT:
#
# comparison already contains Overall_OOF_R2 because it comes from
# phylo_summary.
#
# DO NOT merge phylo_overall into comparison again.
#
# This is the bug that caused:
#
# KeyError: 'Overall_OOF_R2'
#
# in the original script.


if "Overall_OOF_R2" in comparison.columns:

    observed_r2_column = (
        "Overall_OOF_R2"
    )

elif "Overall_OOF_R2_x" in comparison.columns:

    observed_r2_column = (
        "Overall_OOF_R2_x"
    )

elif "Overall_OOF_R2_y" in comparison.columns:

    observed_r2_column = (
        "Overall_OOF_R2_y"
    )

else:

    raise KeyError(
        "\nCould not find observed phylogenetic OOF R2.\n"
        "Available comparison columns:\n"
        + "\n".join(
            [
                str(c)
                for c in comparison.columns
            ]
        )
    )

print()
print(
    "Observed R2 column:",
    observed_r2_column
)


pvalues = []

for _, row in comparison.iterrows():

    subset = permutation_df[
        (
            permutation_df[
                "Phenotype"
            ]
            ==
            row[
                "Phenotype"
            ]
        )
        &
        (
            permutation_df[
                "Model"
            ]
            ==
            row[
                "Model"
            ]
        )
    ]

    observed_r2 = row[
        observed_r2_column
    ]

    p = empirical_p_value(
        observed_r2,
        subset[
            "R2"
        ].values
    )

    pvalues.append(
        p
    )


comparison[
    "Permutation_Empirical_P_R2"
] = pvalues


# =============================================================================
# STABILITY FLAG
# =============================================================================

print()
print("=" * 80)
print("CALCULATING PHYLOGENETIC FOLD STABILITY")
print("=" * 80)

phylo_fold_only = phylo_df[
    phylo_df["Fold"].astype(str)
    != "OOF_OVERALL"
].copy()

stability = (
    phylo_fold_only
    .assign(
        Positive=lambda x:
        x["R2"] > 0
    )
    .groupby(
        [
            "Phenotype",
            "Model"
        ]
    )["Positive"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "Positive":
                "Phylogenetic_R2_Positive_Fold_Fraction"
        }
    )
)

comparison = comparison.merge(
    stability,
    on=[
        "Phenotype",
        "Model",
    ],
    how="left"
)


# =============================================================================
# POTENTIALLY USEFUL FLAG
# =============================================================================

comparison[
    "Potentially_Useful"
] = (
    (
        comparison[
            observed_r2_column
        ]
        > 0
    )
    &
    (
        comparison[
            "Permutation_Empirical_P_R2"
        ]
        < 0.05
    )
)


# =============================================================================
# SAVE FINAL COMPARISON
# =============================================================================

comparison.to_csv(
    FINAL_COMPARISON_FILE,
    index=False
)


# =============================================================================
# IDENTIFY BEST VALIDATED CANDIDATES
# =============================================================================

print()
print("=" * 80)
print("BEST VALIDATED MODEL CANDIDATES")
print("=" * 80)

candidate_rows = []

for phenotype in PHENOTYPES:

    subset = comparison[
        comparison[
            "Phenotype"
        ]
        ==
        phenotype
    ].copy()

    supported = subset[
        subset[
            "Potentially_Useful"
        ]
        ==
        True
    ].copy()

    if len(supported) > 0:

        best = supported.sort_values(
            [
                observed_r2_column,
                "Mean_R2_Phylogenetic",
                "Mean_RMSE_Phylogenetic",
            ],
            ascending=[
                False,
                False,
                True,
            ]
        ).iloc[0]

    else:

        best = subset.sort_values(
            [
                observed_r2_column,
                "Mean_R2_Phylogenetic",
                "Mean_RMSE_Phylogenetic",
            ],
            ascending=[
                False,
                False,
                True,
            ]
        ).iloc[0]

    candidate_rows.append(
        best.to_dict()
    )

    print()
    print(
        phenotype
    )

    print(
        "Candidate model:",
        best["Model"]
    )

    print(
        "Phylogenetic CV mean R2:",
        round(
            best[
                "Mean_R2_Phylogenetic"
            ],
            6
        )
    )

    print(
        "Overall phylogenetic OOF R2:",
        round(
            best[
                observed_r2_column
            ],
            6
        )
    )

    print(
        "Permutation empirical p:",
        round(
            best[
                "Permutation_Empirical_P_R2"
            ],
            6
        )
    )

    print(
        "Positive phylogenetic fold fraction:",
        round(
            best[
                "Phylogenetic_R2_Positive_Fold_Fraction"
            ],
            6
        )
    )

    print(
        "Potentially useful:",
        best[
            "Potentially_Useful"
        ]
    )


candidate_df = pd.DataFrame(
    candidate_rows
)


# Normalize column name in final candidate table
if (
    observed_r2_column
    != "Overall_OOF_R2"
):

    candidate_df[
        "Overall_OOF_R2"
    ] = candidate_df[
        observed_r2_column
    ]


candidate_df.to_csv(
    CANDIDATE_FILE,
    index=False
)


# =============================================================================
# MODEL STATUS
# =============================================================================

status_rows = []

for _, row in candidate_df.iterrows():

    useful = bool(
        row[
            "Potentially_Useful"
        ]
    )

    if useful:

        status = (
            "SUPPORTED_FOR_FURTHER_VALIDATION"
        )

    else:

        status = (
            "NO_CLEAR_PREDICTIVE_SUPPORT"
        )

    status_rows.append({

        "Phenotype":
            row[
                "Phenotype"
            ],

        "Candidate_Model":
            row[
                "Model"
            ],

        "Overall_Phylogenetic_OOF_R2":
            row[
                "Overall_OOF_R2"
            ],

        "Permutation_Empirical_P":
            row[
                "Permutation_Empirical_P_R2"
            ],

        "Phylogenetic_R2_Positive_Fold_Fraction":
            row[
                "Phylogenetic_R2_Positive_Fold_Fraction"
            ],

        "Status":
            status,
    })


status_df = pd.DataFrame(
    status_rows
)

status_df.to_csv(
    STATUS_FILE,
    index=False
)


# =============================================================================
# REPORT
# =============================================================================

report_path = (
    REPORTS
    / "phylogeny_aware_ml_validation_report_420.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 80
        + "\n"
    )

    report.write(
        "PHYLOGENY-AWARE ML VALIDATION REPORT\n"
    )

    report.write(
        "Y1000+ / 420 TAXA\n"
    )

    report.write(
        "=" * 80
        + "\n\n"
    )

    report.write(
        "DATA\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    report.write(
        f"Taxa evaluated: {len(df)}\n"
    )

    report.write(
        "Robust BUSCO features: "
        f"{len(available_buscos)}\n"
    )

    report.write(
        f"Phylogenetic folds: {len(phylo_folds)}\n"
    )

    report.write(
        f"Random CV repeats: {N_RANDOM_REPEATS}\n"
    )

    report.write(
        f"Random CV folds per repeat: "
        f"{N_RANDOM_FOLDS}\n"
    )

    report.write(
        f"Permutation tests: "
        f"{N_PERMUTATIONS}\n\n"
    )

    report.write(
        "VALIDATION DESIGN\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    report.write(
        "Genomic predictors consist of the established "
        "robust BUSCO feature set.\n"
    )

    report.write(
        "Phylogeny is incorporated through phylogenetic "
        "cross-validation folds.\n"
    )

    report.write(
        "Repeated random cross-validation is used as a "
        "complementary comparison.\n"
    )

    report.write(
        "Permutation testing evaluates whether observed "
        "OOF R2 exceeds the tested label-permutation null.\n\n"
    )

    report.write(
        "IMPORTANT INTERPRETATION\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    report.write(
        "The current framework uses genomic BUSCO features "
        "and phylogeny-aware validation.\n"
    )

    report.write(
        "The phylogenetic covariance matrix is NOT directly "
        "incorporated into the ML estimator by this script.\n"
    )

    report.write(
        "Therefore results should be described as "
        "'phylogeny-aware validation of genomic ML models' "
        "rather than as a phylogenetic regression model.\n\n"
    )

    report.write(
        "FINAL VALIDATED CANDIDATES\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    for _, row in candidate_df.iterrows():

        report.write(
            f"\nPhenotype: "
            f"{row['Phenotype']}\n"
        )

        report.write(
            f"Model: "
            f"{row['Model']}\n"
        )

        report.write(
            f"Mean phylogenetic CV R2: "
            f"{row['Mean_R2_Phylogenetic']:.6f}\n"
        )

        report.write(
            f"Overall phylogenetic OOF R2: "
            f"{row['Overall_OOF_R2']:.6f}\n"
        )

        report.write(
            f"Mean phylogenetic CV RMSE: "
            f"{row['Mean_RMSE_Phylogenetic']:.6f}\n"
        )

        report.write(
            f"Permutation empirical p-value: "
            f"{row['Permutation_Empirical_P_R2']:.6f}\n"
        )

        report.write(
            f"Positive phylogenetic fold fraction: "
            f"{row['Phylogenetic_R2_Positive_Fold_Fraction']:.6f}\n"
        )

        report.write(
            f"Potentially useful: "
            f"{row['Potentially_Useful']}\n"
        )

    report.write(
        "\n\n"
    )

    report.write(
        "INTERPRETATION GUIDANCE\n"
    )

    report.write(
        "-" * 80
        + "\n"
    )

    report.write(
        "A substantially higher random-CV R2 than "
        "phylogenetic-CV R2 may indicate that ordinary "
        "random splitting benefits from phylogenetic "
        "relatedness.\n"
    )

    report.write(
        "A positive phylogenetic OOF R2 combined with a "
        "low permutation empirical p-value indicates that "
        "the observed signal exceeds the tested label-"
        "permutation null.\n"
    )

    report.write(
        "A low R2 should not be described as strong "
        "predictive power.\n"
    )

    report.write(
        "Permutation significance does not imply high "
        "predictive accuracy; it indicates evidence against "
        "the tested null distribution.\n"
    )

    report.write(
        "No existing final model was overwritten.\n"
    )


# =============================================================================
# FINAL CONSOLE SUMMARY
# =============================================================================

print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)

print()
print(
    "Taxa:",
    len(df)
)

print(
    "Robust BUSCO features:",
    len(available_buscos)
)

print(
    "Phylogenetic folds:",
    len(phylo_folds)
)

print(
    "Permutation tests:",
    N_PERMUTATIONS
)

print()
print("=" * 80)
print("FINAL VALIDATION CANDIDATES")
print("=" * 80)

print(
    candidate_df[
        [
            "Phenotype",
            "Model",
            "Mean_R2_Phylogenetic",
            "Overall_OOF_R2",
            "Permutation_Empirical_P_R2",
            "Phylogenetic_R2_Positive_Fold_Fraction",
            "Potentially_Useful",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# OUTPUT FILES
# =============================================================================

print()
print("=" * 80)
print("OUTPUT FILES")
print("=" * 80)

output_files = [
    TABLES / "robust_BUSCO_feature_audit_420.csv",
    TABLES / "validation_training_matrix_420.csv",
    PHYLO_RESULTS_FILE,
    PHYLO_SUMMARY_FILE,
    RANDOM_RESULTS_FILE,
    RANDOM_SUMMARY_FILE,
    COMPARISON_FILE,
    PERMUTATION_RESULTS_FILE,
    PERMUTATION_SUMMARY_FILE,
    FINAL_COMPARISON_FILE,
    CANDIDATE_FILE,
    STATUS_FILE,
    report_path,
]

for path in output_files:

    print()
    print(path)


print()
print("=" * 80)
print("IMPORTANT")
print("=" * 80)

print()
print(
    "Existing final models were NOT overwritten."
)

print(
    "Candidate-chassis results were NOT modified."
)

print(
    "Existing permutation results were reused when available."
)

print(
    "Existing CV results were reused when available."
)

print(
    "This validation does NOT automatically replace the "
    "existing final model."
)

print()
print("=" * 80)