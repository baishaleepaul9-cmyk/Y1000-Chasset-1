#!/usr/bin/env python3
"""
FINAL PHYLOGENY-AWARE ML MODELS — 420 TAXA
==========================================

Purpose
-------
Build the final genome -> phenotype models for:

    1. Carbon_Breadth
    2. Nitrogen_Breadth
    3. Utilized_Median_Growth

The script uses the already-established robust BUSCO candidate set
and phylogenetic CV assignments.

Feature selection
-----------------
The 79 robust BUSCOs are treated as a pre-defined candidate feature pool.

For each phenotype:

    - load 420-taxon ML matrix
    - load robust BUSCO candidate list
    - load phylogenetic CV assignments
    - evaluate several regression models using phylogenetic folds
    - calculate RMSE, MAE and R2
    - select the model based on mean phylogenetic CV RMSE
    - refit selected model on all 420 taxa
    - save final model
    - save final predictions
    - save out-of-fold predictions
    - save model comparison summary
    - save model metadata
    - save final BUSCO feature list

IMPORTANT FIX
-------------
NumPy scalar types such as np.int64 and np.float64 are not directly
JSON serializable. The metadata therefore uses a recursive converter
before json.dump().
"""

from pathlib import Path
import warnings
import json
import pickle

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"C:\Y1000_chassis_project")


ML_MATRIX = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)


ROBUST_BUSCOS = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_importance_analysis"
    / "robust_feature_analysis"
    / "integrated_robust_candidates"
    / "robust_all_three_buscos_420.csv"
)


PHYLO_CV = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "phylogenetic_cv_assignments_420.csv"
)


OUTPUT_DIR = (
    ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "final_models_420"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CONFIGURATION
# =============================================================================

PHENOTYPES = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
]

N_FOLDS = 5

RANDOM_STATE = 42

EXPECTED_ROBUST_N = 79


# =============================================================================
# JSON SERIALIZATION FIX
# =============================================================================

def make_json_serializable(obj):
    """
    Recursively convert NumPy/Pandas objects into native Python objects
    that can be serialized by json.dump().
    """

    # NumPy integer
    if isinstance(obj, np.integer):
        return int(obj)

    # NumPy floating point
    if isinstance(obj, np.floating):
        return float(obj)

    # NumPy boolean
    if isinstance(obj, np.bool_):
        return bool(obj)

    # NumPy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Pandas Series
    if isinstance(obj, pd.Series):
        return obj.tolist()

    # Pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")

    # Dictionary
    if isinstance(obj, dict):
        return {
            str(key): make_json_serializable(value)
            for key, value in obj.items()
        }

    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    # Native Python object
    return obj


def save_json(data, path):
    """
    Save data to JSON after recursively converting NumPy/Pandas objects.
    """

    clean_data = make_json_serializable(data)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            clean_data,
            f,
            indent=4,
            allow_nan=False,
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rmse(y_true, y_pred):

    return np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )


def safe_r2(y_true, y_pred):
    """
    R2 can fail for a fold containing effectively constant y values.
    """

    try:

        return r2_score(
            y_true,
            y_pred,
        )

    except Exception:

        return np.nan


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):

    model = clone(model)

    model.fit(
        X_train,
        y_train,
    )

    pred = model.predict(
        X_test
    )

    return {
        "RMSE": rmse(
            y_test,
            pred,
        ),

        "MAE": mean_absolute_error(
            y_test,
            pred,
        ),

        "R2": safe_r2(
            y_test,
            pred,
        ),

        "Predictions": pred,
    }


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
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                Ridge(
                    alpha=10.0
                ),
            ),
        ]),


        "ElasticNet": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.5,
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),


        "SVR_RBF": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "scaler",
                StandardScaler(),
            ),

            (
                "model",
                SVR(
                    kernel="rbf",
                    C=10.0,
                    epsilon=0.1,
                    gamma="scale",
                ),
            ),
        ]),


        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),


        "ExtraTrees": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),


        "GradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),

            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=2,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }

    return models


# =============================================================================
# START
# =============================================================================

print("=" * 80)
print("FINAL PHYLOGENY-AWARE ML MODELING — 420 TAXA")
print("=" * 80)


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print("\nCHECKING INPUT FILES")
print("=" * 80)


for f in [
    ML_MATRIX,
    ROBUST_BUSCOS,
    PHYLO_CV,
]:

    print("\nChecking:")
    print(f)

    if not f.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{f}"
        )

    print("✓ Found")


# =============================================================================
# LOAD ML MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("LOADING 420-TAXON ML MATRIX")
print("=" * 80)


df = pd.read_csv(
    ML_MATRIX
)


print("\nShape:")
print(df.shape)


print("\nColumns:")
print(
    df.columns.tolist()[:20]
)


required_metadata = [
    "Species",
    "Assembly_Accession",
]


for col in required_metadata:

    if col not in df.columns:

        raise RuntimeError(
            f"Required column missing from ML matrix: {col}"
        )


# =============================================================================
# LOAD ROBUST BUSCO SET
# =============================================================================

print("\n" + "=" * 80)
print("LOADING ROBUST BUSCO CANDIDATES")
print("=" * 80)


robust = pd.read_csv(
    ROBUST_BUSCOS
)


print("\nShape:")
print(robust.shape)


print("\nColumns:")
print(
    robust.columns.tolist()
)


# =============================================================================
# DETECT BUSCO COLUMN
# =============================================================================

if "BUSCO" in robust.columns:

    busco_col = "BUSCO"

else:

    possible_busco_cols = [
        c
        for c in robust.columns
        if c.lower()
        in {
            "busco",
            "busco_family",
            "busco_id",
        }
    ]

    if not possible_busco_cols:

        raise RuntimeError(
            "Could not identify BUSCO column in robust candidate file."
        )

    busco_col = possible_busco_cols[0]


robust_buscos = (
    robust[busco_col]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)


print("\nRobust BUSCO count:")
print(
    len(robust_buscos)
)


if len(robust_buscos) != EXPECTED_ROBUST_N:

    print(
        f"\nWARNING: Expected approximately "
        f"{EXPECTED_ROBUST_N} robust BUSCOs, "
        f"but found {len(robust_buscos)}."
    )


# =============================================================================
# INTERSECT WITH ML MATRIX
# =============================================================================

available_buscos = [
    b
    for b in robust_buscos
    if b in df.columns
]


missing_buscos = [
    b
    for b in robust_buscos
    if b not in df.columns
]


print("\nBUSCO FEATURE AVAILABILITY")
print("-" * 80)

print(
    "Robust BUSCOs:",
    len(robust_buscos)
)

print(
    "Available in ML matrix:",
    len(available_buscos)
)

print(
    "Missing from ML matrix:",
    len(missing_buscos)
)


if missing_buscos:

    print("\nMissing BUSCOs:")

    for b in missing_buscos:

        print(
            " ",
            b
        )


if len(available_buscos) == 0:

    raise RuntimeError(
        "No robust BUSCOs were found in the 420-taxon ML matrix."
    )


# =============================================================================
# LOAD PHYLOGENETIC CV
# =============================================================================

print("\n" + "=" * 80)
print("LOADING PHYLOGENETIC CV ASSIGNMENTS")
print("=" * 80)


cv = pd.read_csv(
    PHYLO_CV
)


print("\nShape:")
print(
    cv.shape
)


print("\nColumns:")
print(
    cv.columns.tolist()
)


required_cv_cols = [
    "Species",
    "Assembly_Accession",
    "Phylogenetic_Fold",
]


for col in required_cv_cols:

    if col not in cv.columns:

        raise RuntimeError(
            f"Required phylogenetic CV column missing: {col}"
        )


# =============================================================================
# RECONCILE ML MATRIX WITH PHYLOGENETIC CV
# =============================================================================

print("\n" + "=" * 80)
print("RECONCILING ML MATRIX WITH PHYLOGENETIC CV")
print("=" * 80)


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


# Check for duplicate accessions in CV
duplicate_cv_accessions = (
    cv["Assembly_Accession"]
    .duplicated()
    .sum()
)


if duplicate_cv_accessions > 0:

    print(
        "\nWARNING:"
    )

    print(
        "Duplicate Assembly_Accession entries in CV:",
        duplicate_cv_accessions
    )


cv_lookup = (
    cv[
        [
            "Assembly_Accession",
            "Phylogenetic_Fold",
        ]
    ]
    .drop_duplicates(
        subset=[
            "Assembly_Accession"
        ]
    )
)


df = df.merge(
    cv_lookup,
    on="Assembly_Accession",
    how="left",
)


print(
    "\nML rows:",
    len(df)
)


print(
    "Rows with phylogenetic fold:",
    df["Phylogenetic_Fold"].notna().sum()
)


print(
    "Rows without phylogenetic fold:",
    df["Phylogenetic_Fold"].isna().sum()
)


if df["Phylogenetic_Fold"].isna().any():

    missing = df.loc[
        df["Phylogenetic_Fold"].isna(),
        [
            "Species",
            "Assembly_Accession",
        ],
    ]

    print(
        "\nMissing taxa:"
    )

    print(
        missing.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Some 420-taxon ML rows do not have phylogenetic CV assignments."
    )


# =============================================================================
# CLEAN PHYLOGENETIC FOLDS
# =============================================================================

df["Phylogenetic_Fold"] = (
    df["Phylogenetic_Fold"]
    .astype(int)
)


# IMPORTANT:
# Explicitly convert NumPy integers to native Python ints.
# This prevents the original JSON serialization error.

folds = sorted(
    int(x)
    for x in
    df["Phylogenetic_Fold"]
    .dropna()
    .unique()
)


print(
    "\nPhylogenetic folds:"
)

print(
    folds
)


if len(folds) != N_FOLDS:

    raise RuntimeError(
        f"Expected {N_FOLDS} phylogenetic folds, "
        f"found {len(folds)}."
    )


# =============================================================================
# CHECK FOLD SIZES
# =============================================================================

print("\nPhylogenetic fold sizes:")
print(
    df["Phylogenetic_Fold"]
    .value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# SAVE FINAL FEATURE LIST
# =============================================================================

feature_table = pd.DataFrame({
    "BUSCO": available_buscos
})


feature_table.to_csv(
    OUTPUT_DIR
    / "final_model_busco_features_420.csv",
    index=False,
)


print(
    "\n✓ Final candidate feature list written:"
)

print(
    OUTPUT_DIR
    / "final_model_busco_features_420.csv"
)


# =============================================================================
# MODELING
# =============================================================================

models = build_models()


all_model_results = []

all_oof_predictions = []

final_model_summary = []


# =============================================================================
# LOOP THROUGH PHENOTYPES
# =============================================================================

for phenotype in PHENOTYPES:

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        f"PHENOTYPE: {phenotype}"
    )

    print(
        "=" * 80
    )


    # -------------------------------------------------------------------------
    # CHECK PHENOTYPE
    # -------------------------------------------------------------------------

    if phenotype not in df.columns:

        raise RuntimeError(
            f"Phenotype column not found: {phenotype}"
        )


    # -------------------------------------------------------------------------
    # FEATURE MATRIX
    # -------------------------------------------------------------------------

    X = df[
        available_buscos
    ].copy()


    y = pd.to_numeric(
        df[phenotype],
        errors="coerce",
    )


    valid = y.notna()


    X = X.loc[
        valid
    ].copy()


    y = y.loc[
        valid
    ].copy()


    meta = df.loc[
        valid,
        [
            "Species",
            "Assembly_Accession",
            "Phylogenetic_Fold",
        ],
    ].copy()


    print(
        "\nSamples:"
    )

    print(
        len(y)
    )


    print(
        "\nFeature count:"
    )

    print(
        X.shape[1]
    )


    print(
        "\nPhenotype statistics:"
    )

    print(
        y.describe()
    )


    # =========================================================================
    # PHYLOGENETIC CV MODEL COMPARISON
    # =========================================================================

    phenotype_results = []


    for model_name, model in models.items():

        print(
            "\n"
            + "-" * 80
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "-" * 80
        )


        fold_results = []


        oof_pred = np.full(
            len(y),
            np.nan,
            dtype=float,
        )


        # =====================================================================
        # FIVE PHYLOGENETIC FOLDS
        # =====================================================================

        for fold in folds:

            train_idx = (
                meta[
                    "Phylogenetic_Fold"
                ]
                != fold
            )


            test_idx = (
                meta[
                    "Phylogenetic_Fold"
                ]
                == fold
            )


            X_train = X.loc[
                train_idx
            ]


            X_test = X.loc[
                test_idx
            ]


            y_train = y.loc[
                train_idx
            ]


            y_test = y.loc[
                test_idx
            ]


            result = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )


            pred = result[
                "Predictions"
            ]


            oof_pred[
                np.where(test_idx)[0]
            ] = pred


            fold_result = {

                "Phenotype":
                    phenotype,

                "Model":
                    model_name,

                "Fold":
                    int(fold),

                "N_Train":
                    int(len(y_train)),

                "N_Test":
                    int(len(y_test)),

                "RMSE":
                    float(result["RMSE"]),

                "MAE":
                    float(result["MAE"]),

                "R2":
                    float(result["R2"])
                    if not np.isnan(result["R2"])
                    else np.nan,
            }


            fold_results.append(
                fold_result
            )


            all_model_results.append(
                fold_result
            )


            print(
                f"Fold {fold}: "
                f"RMSE={result['RMSE']:.5f}, "
                f"MAE={result['MAE']:.5f}, "
                f"R2={result['R2']:.5f}"
            )


        # =====================================================================
        # OVERALL OOF PERFORMANCE
        # =====================================================================

        overall_rmse = rmse(
            y,
            oof_pred,
        )


        overall_mae = mean_absolute_error(
            y,
            oof_pred,
        )


        overall_r2 = safe_r2(
            y,
            oof_pred,
        )


        mean_rmse = np.mean([
            x["RMSE"]
            for x in fold_results
        ])


        sd_rmse = np.std(
            [
                x["RMSE"]
                for x in fold_results
            ],
            ddof=1,
        )


        mean_mae = np.mean([
            x["MAE"]
            for x in fold_results
        ])


        mean_r2 = np.nanmean([
            x["R2"]
            for x in fold_results
        ])


        summary_row = {

            "Phenotype":
                phenotype,

            "Model":
                model_name,

            "Mean_CV_RMSE":
                float(mean_rmse),

            "SD_CV_RMSE":
                float(sd_rmse),

            "Mean_CV_MAE":
                float(mean_mae),

            "Mean_CV_R2":
                float(mean_r2),

            "Overall_OOF_RMSE":
                float(overall_rmse),

            "Overall_OOF_MAE":
                float(overall_mae),

            "Overall_OOF_R2":
                float(overall_r2),

            "N_Samples":
                int(len(y)),

            "N_Features":
                int(len(available_buscos)),
        }


        phenotype_results.append(
            summary_row
        )


        # =====================================================================
        # SAVE OOF PREDICTIONS
        # =====================================================================

        oof_df = meta.copy()


        oof_df["Phenotype"] = phenotype


        oof_df["Observed"] = (
            y.values
        )


        oof_df["Predicted"] = (
            oof_pred
        )


        oof_df["Model"] = (
            model_name
        )


        all_oof_predictions.append(
            oof_df
        )


    # =========================================================================
    # MODEL COMPARISON TABLE
    # =========================================================================

    phenotype_results_df = pd.DataFrame(
        phenotype_results
    )


    phenotype_results_df = (
        phenotype_results_df
        .sort_values(
            by=[
                "Mean_CV_RMSE",
                "Overall_OOF_RMSE",
            ],
            ascending=True,
        )
    )


    phenotype_results_df.to_csv(
        OUTPUT_DIR
        / f"{phenotype}_model_comparison_420.csv",
        index=False,
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"MODEL COMPARISON — {phenotype}"
    )

    print(
        "=" * 80
    )


    print(
        phenotype_results_df[
            [
                "Model",
                "Mean_CV_RMSE",
                "SD_CV_RMSE",
                "Mean_CV_MAE",
                "Mean_CV_R2",
                "Overall_OOF_RMSE",
                "Overall_OOF_R2",
            ]
        ].to_string(
            index=False
        )
    )


    # =========================================================================
    # SELECT FINAL MODEL
    # =========================================================================

    best_row = (
        phenotype_results_df
        .iloc[0]
    )


    best_model_name = (
        str(best_row["Model"])
    )


    best_model = models[
        best_model_name
    ]


    print(
        "\nFINAL MODEL SELECTED:"
    )

    print(
        best_model_name
    )


    print(
        f"Mean phylogenetic CV RMSE: "
        f"{best_row['Mean_CV_RMSE']:.6f}"
    )


    print(
        f"Mean phylogenetic CV R²: "
        f"{best_row['Mean_CV_R2']:.6f}"
    )


    print(
        f"Overall OOF RMSE: "
        f"{best_row['Overall_OOF_RMSE']:.6f}"
    )


    print(
        f"Overall OOF R²: "
        f"{best_row['Overall_OOF_R2']:.6f}"
    )


    # =========================================================================
    # REFIT FINAL MODEL ON ALL 420 TAXA
    # =========================================================================

    print(
        "\nRefitting selected model on all available taxa..."
    )


    final_model = clone(
        best_model
    )


    final_model.fit(
        X,
        y,
    )


    final_pred = (
        final_model.predict(X)
    )


    # =========================================================================
    # SAVE FINAL MODEL
    # =========================================================================

    model_path = (
        OUTPUT_DIR
        / f"final_{phenotype}_model_420.pkl"
    )


    with open(
        model_path,
        "wb",
    ) as f:

        pickle.dump(
            final_model,
            f,
        )


    # =========================================================================
    # SAVE FINAL PREDICTIONS
    # =========================================================================

    prediction_df = (
        meta.copy()
    )


    prediction_df["Phenotype"] = (
        phenotype
    )


    prediction_df["Observed"] = (
        y.values
    )


    prediction_df[
        "Final_Model_Predicted"
    ] = final_pred


    prediction_df["Residual"] = (
        y.values
        - final_pred
    )


    prediction_df.to_csv(
        OUTPUT_DIR
        / f"final_{phenotype}_predictions_420.csv",
        index=False,
    )


    # =========================================================================
    # SAVE MODEL METADATA
    # =========================================================================

    metadata = {

        "phenotype":
            phenotype,

        "model":
            best_model_name,

        "n_samples":
            int(len(y)),

        "n_features":
            int(len(available_buscos)),

        "features":
            list(available_buscos),

        "cv_strategy":
            "5-fold phylogenetic CV",

        # Explicitly convert folds to native Python int
        "phylogenetic_folds":
            [int(fold) for fold in folds],

        "mean_cv_rmse":
            float(
                best_row[
                    "Mean_CV_RMSE"
                ]
            ),

        "sd_cv_rmse":
            float(
                best_row[
                    "SD_CV_RMSE"
                ]
            ),

        "mean_cv_mae":
            float(
                best_row[
                    "Mean_CV_MAE"
                ]
            ),

        "mean_cv_r2":
            float(
                best_row[
                    "Mean_CV_R2"
                ]
            ),

        "overall_oof_rmse":
            float(
                best_row[
                    "Overall_OOF_RMSE"
                ]
            ),

        "overall_oof_mae":
            float(
                best_row[
                    "Overall_OOF_MAE"
                ]
            ),

        "overall_oof_r2":
            float(
                best_row[
                    "Overall_OOF_R2"
                ]
            ),

        "random_state":
            int(RANDOM_STATE),
    }


    metadata_path = (
        OUTPUT_DIR
        / f"final_{phenotype}_metadata_420.json"
    )


    # -------------------------------------------------------------------------
    # FIX:
    # Use the safe JSON writer instead of json.dump(metadata, ...)
    # -------------------------------------------------------------------------

    save_json(
        metadata,
        metadata_path,
    )


    final_model_summary.append(
        metadata
    )


    print(
        "\n✓ Final model saved:"
    )

    print(
        model_path
    )


    print(
        "✓ Final predictions saved:"
    )

    print(
        OUTPUT_DIR
        / f"final_{phenotype}_predictions_420.csv"
    )


    print(
        "✓ Metadata saved:"
    )

    print(
        metadata_path
    )


# =============================================================================
# SAVE COMBINED RESULTS
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "SAVING COMBINED RESULTS"
)

print(
    "=" * 80
)


# =============================================================================
# ALL MODEL RESULTS
# =============================================================================

all_results_df = pd.DataFrame(
    all_model_results
)


all_results_df.to_csv(
    OUTPUT_DIR
    / "all_phylogenetic_cv_model_results_420.csv",
    index=False,
)


print(
    "\n✓ All CV model results written:"
)

print(
    OUTPUT_DIR
    / "all_phylogenetic_cv_model_results_420.csv"
)


# =============================================================================
# ALL OOF PREDICTIONS
# =============================================================================

all_oof_df = pd.concat(
    all_oof_predictions,
    ignore_index=True,
)


all_oof_df.to_csv(
    OUTPUT_DIR
    / "all_phylogenetic_oof_predictions_420.csv",
    index=False,
)


print(
    "\n✓ All OOF predictions written:"
)

print(
    OUTPUT_DIR
    / "all_phylogenetic_oof_predictions_420.csv"
)


# =============================================================================
# FINAL MODEL SUMMARY
# =============================================================================

final_summary_df = pd.DataFrame(
    final_model_summary
)


final_summary_df.to_csv(
    OUTPUT_DIR
    / "FINAL_MODEL_SUMMARY_420.csv",
    index=False,
)


print(
    "\n✓ Final model summary written:"
)

print(
    OUTPUT_DIR
    / "FINAL_MODEL_SUMMARY_420.csv"
)


# =============================================================================
# FINAL FEATURE MATRIX
# =============================================================================

final_feature_columns = [
    "Species",
    "Assembly_Accession",
]

final_feature_columns += (
    available_buscos
)

final_feature_columns += (
    PHENOTYPES
)

final_feature_columns += [
    "Phylogenetic_Fold"
]


final_feature_matrix = (
    df[
        final_feature_columns
    ].copy()
)


final_feature_matrix.to_csv(
    OUTPUT_DIR
    / "final_model_training_matrix_420.csv",
    index=False,
)


print(
    "\n✓ Final model training matrix written:"
)

print(
    OUTPUT_DIR
    / "final_model_training_matrix_420.csv"
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL MODEL SUMMARY"
)

print(
    "=" * 80
)


print(
    final_summary_df[
        [
            "phenotype",
            "model",
            "n_samples",
            "n_features",
            "mean_cv_rmse",
            "mean_cv_r2",
            "overall_oof_rmse",
            "overall_oof_r2",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# FINAL MODEL PATHS
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL PHYLOGENY-AWARE MODELING COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nFinal models:"
)


for phenotype in PHENOTYPES:

    print(
        OUTPUT_DIR
        / f"final_{phenotype}_model_420.pkl"
    )


print(
    "\nFinal model metadata:"
)


for phenotype in PHENOTYPES:

    print(
        OUTPUT_DIR
        / f"final_{phenotype}_metadata_420.json"
    )


print(
    "\nFinal feature list:"
)

print(
    OUTPUT_DIR
    / "final_model_busco_features_420.csv"
)


print(
    "\nFinal model summary:"
)

print(
    OUTPUT_DIR
    / "FINAL_MODEL_SUMMARY_420.csv"
)


print(
    "\nFinal OOF predictions:"
)

print(
    OUTPUT_DIR
    / "all_phylogenetic_oof_predictions_420.csv"
)


print(
    "\nFinal training matrix:"
)

print(
    OUTPUT_DIR
    / "final_model_training_matrix_420.csv"
)


# =============================================================================
# NEXT STAGE
# =============================================================================

print(
    "\n"
    + "=" * 80
)

print(
    "NEXT STAGE: SHAP / XAI ON THE FINAL MODELS"
)

print(
    "=" * 80
)