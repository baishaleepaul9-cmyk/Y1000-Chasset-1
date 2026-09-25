# =============================================================================
# Y1000+ STAGE 6A
# PHYLOGENY-AWARE FEATURE-SET OPTIMIZATION
# =============================================================================
#
# Purpose:
#   Determine whether predictive performance improves when moving beyond the
#   current 79 robust BUSCO feature panel.
#
# Feature-set comparisons:
#   1. Robust 79 BUSCOs
#   2. All variable BUSCOs
#   3. Filtered variable BUSCOs
#   4. Phenotype-selected Top 50 BUSCOs
#   5. Phenotype-selected Top 100 BUSCOs
#   6. Phenotype-selected Top 200 BUSCOs
#
# Models:
#   Ridge
#   ElasticNet
#   SVR-RBF
#   Random Forest
#   Extra Trees
#   Gradient Boosting
#
# Validation:
#   Outer: 5-fold PHYLOGENETIC CV
#   Inner: 3-fold CV for hyperparameter tuning
#
# IMPORTANT:
#   - Feature filtering occurs inside outer training folds only.
#   - Feature selection occurs inside outer training folds only.
#   - Hyperparameter tuning occurs inside outer training folds only.
#   - Outer phylogenetic test folds remain untouched until prediction.
#   - Existing final models are NOT overwritten.
#
# FIXES:
#   - Removed problematic .astype("Int64") on NumPy OOF fold arrays.
#   - Fold values are stored as normal numeric values.
#   - Robust JSON serialization.
#   - Safe NaN/Inf handling.
#   - Avoids nested joblib parallelism during GridSearchCV.
#
# =============================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import f_regression

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Y1000_chassis_project"
)

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
)


# =============================================================================
# INPUT FILES
# =============================================================================

ML_MATRIX = (
    RESULTS_ROOT
    / "phylogeny_aware_ml"
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)


VARIABLE_MATRIX = (
    RESULTS_ROOT
    / "feature_qc"
    / "y1000_420_variable_busco_ml_matrix.csv"
)


ROBUST_BUSCO_FILE = (
    RESULTS_ROOT
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_importance_analysis"
    / "robust_feature_analysis"
    / "integrated_robust_candidates"
    / "robust_all_three_buscos_420.csv"
)


PHYLO_CV_FILE = (
    RESULTS_ROOT
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "phylogenetic_cv_assignments_420.csv"
)


# =============================================================================
# OUTPUT DIRECTORY
# =============================================================================

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "performance_optimization_420"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


FEATURE_OUTPUT = (
    OUTPUT_ROOT
    / "selected_features"
)

FEATURE_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


PREDICTION_OUTPUT = (
    OUTPUT_ROOT
    / "oof_predictions"
)

PREDICTION_OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# PHENOTYPES
# =============================================================================

PHENOTYPES = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
]


# =============================================================================
# METADATA / TARGET COLUMNS
# =============================================================================

META_COLUMNS = [
    "Species",
    "Assembly_Accession",
    "Phenotype_Source_Species",
    "N_Strains"
]

TARGET_COLUMNS = PHENOTYPES.copy()


# =============================================================================
# RANDOM STATE / CV
# =============================================================================

RANDOM_STATE = 42

EXPECTED_TAXA = 420

N_OUTER_FOLDS = 5

N_INNER_FOLDS = 3


# =============================================================================
# FEATURE FILTERING
# =============================================================================

MIN_PREVALENCE = 0.05

MAX_PREVALENCE = 0.95


# =============================================================================
# PHENOTYPE FEATURE SELECTION
# =============================================================================

TOP_K_VALUES = [
    50,
    100,
    200
]


# =============================================================================
# TREE MODEL SIZE
# =============================================================================

N_ESTIMATORS = 300


# =============================================================================
# PARALLELISM
# =============================================================================
#
# IMPORTANT:
# GridSearchCV + tree models both using all cores can create nested
# parallelism and cause joblib worker problems.
#
# Therefore:
#
#   GridSearchCV = 1 worker
#   Tree models  = all available cores
#
# =============================================================================

GRID_N_JOBS = 1

TREE_N_JOBS = -1


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("Y1000+ PHYLOGENY-AWARE FEATURE OPTIMIZATION")
print("=" * 80)

print("\nProject:")
print(PROJECT_ROOT)

print("\nOutput:")
print(OUTPUT_ROOT)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_id(value):
    """
    Normalize identifiers for safe matching.
    """
    return str(value).strip()


def json_safe(obj):
    """
    Recursively convert NumPy/Pandas objects into JSON-safe objects.
    """

    if isinstance(obj, dict):
        return {
            str(key): json_safe(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            json_safe(value)
            for value in obj
        ]

    if isinstance(obj, np.ndarray):
        return [
            json_safe(value)
            for value in obj.tolist()
        ]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if obj is None:
        return None

    try:
        missing = pd.isna(obj)

        if isinstance(missing, (bool, np.bool_)):

            if bool(missing):
                return None

    except Exception:
        pass

    return obj


def safe_float(value):
    """
    Convert value to finite float.
    """

    try:

        value = float(value)

        if np.isfinite(value):
            return value

        return np.nan

    except Exception:

        return np.nan


def rmse(y_true, y_pred):
    """
    Root mean squared error.
    """

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def safe_r2(y_true, y_pred):
    """
    Safe R2 calculation.
    """

    try:

        value = r2_score(
            y_true,
            y_pred
        )

        value = float(value)

        if not np.isfinite(value):
            return np.nan

        return value

    except Exception:

        return np.nan


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def get_models():

    models = {

        # ---------------------------------------------------------------------
        # RIDGE
        # ---------------------------------------------------------------------

        "Ridge":

        Pipeline([
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                Ridge()
            )
        ]),


        # ---------------------------------------------------------------------
        # ELASTIC NET
        # ---------------------------------------------------------------------

        "ElasticNet":

        Pipeline([
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                ElasticNet(
                    max_iter=20000,
                    random_state=RANDOM_STATE
                )
            )
        ]),


        # ---------------------------------------------------------------------
        # SVR RBF
        # ---------------------------------------------------------------------

        "SVR_RBF":

        Pipeline([
            (
                "scale",
                StandardScaler()
            ),

            (
                "model",
                SVR(
                    kernel="rbf"
                )
            )
        ]),


        # ---------------------------------------------------------------------
        # RANDOM FOREST
        # ---------------------------------------------------------------------

        "RandomForest":

        RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=TREE_N_JOBS
        ),


        # ---------------------------------------------------------------------
        # EXTRA TREES
        # ---------------------------------------------------------------------

        "ExtraTrees":

        ExtraTreesRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=TREE_N_JOBS
        ),


        # ---------------------------------------------------------------------
        # GRADIENT BOOSTING
        # ---------------------------------------------------------------------

        "GradientBoosting":

        GradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    }

    return models


# =============================================================================
# HYPERPARAMETER GRIDS
# =============================================================================

def get_parameter_grids():

    return {

        # ---------------------------------------------------------------------
        # RIDGE
        # ---------------------------------------------------------------------

        "Ridge": {

            "model__alpha": [
                0.1,
                1.0,
                10.0,
                100.0
            ]
        },


        # ---------------------------------------------------------------------
        # ELASTIC NET
        # ---------------------------------------------------------------------

        "ElasticNet": {

            "model__alpha": [
                0.001,
                0.01,
                0.1,
                1.0
            ],

            "model__l1_ratio": [
                0.1,
                0.5,
                0.9
            ]
        },


        # ---------------------------------------------------------------------
        # SVR
        # ---------------------------------------------------------------------

        "SVR_RBF": {

            "model__C": [
                0.1,
                1.0,
                10.0
            ],

            "model__gamma": [
                "scale",
                0.01,
                0.1
            ],

            "model__epsilon": [
                0.05,
                0.1,
                0.2
            ]
        },


        # ---------------------------------------------------------------------
        # RANDOM FOREST
        # ---------------------------------------------------------------------

        "RandomForest": {

            "n_estimators": [
                300
            ],

            "max_depth": [
                None,
                10,
                20
            ],

            "min_samples_leaf": [
                1,
                2,
                5
            ],

            "max_features": [
                "sqrt",
                0.5
            ]
        },


        # ---------------------------------------------------------------------
        # EXTRA TREES
        # ---------------------------------------------------------------------

        "ExtraTrees": {

            "n_estimators": [
                300
            ],

            "max_depth": [
                None,
                10,
                20
            ],

            "min_samples_leaf": [
                1,
                2,
                5
            ],

            "max_features": [
                "sqrt",
                0.5
            ]
        },


        # ---------------------------------------------------------------------
        # GRADIENT BOOSTING
        # ---------------------------------------------------------------------

        "GradientBoosting": {

            "n_estimators": [
                100,
                200
            ],

            "learning_rate": [
                0.03,
                0.05,
                0.1
            ],

            "max_depth": [
                2,
                3
            ],

            "min_samples_leaf": [
                2,
                5
            ]
        }
    }


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print("\n" + "=" * 80)
print("CHECKING INPUT FILES")
print("=" * 80)


required_files = {

    "ML matrix":
        ML_MATRIX,

    "Variable BUSCO matrix":
        VARIABLE_MATRIX,

    "Robust BUSCO file":
        ROBUST_BUSCO_FILE,

    "Phylogenetic CV file":
        PHYLO_CV_FILE
}


for description, path in required_files.items():

    print(
        f"\nChecking {description}:"
    )

    print(path)

    if not path.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}"
        )

    print("✓ Found")


# =============================================================================
# LOAD MAIN ML MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("LOADING PHYLOGENY-AWARE ML MATRIX")
print("=" * 80)


ml = pd.read_csv(
    ML_MATRIX
)


print("\nShape:")
print(ml.shape)


print("\nFirst columns:")
print(
    list(
        ml.columns[:15]
    )
)


if len(ml) != EXPECTED_TAXA:

    print(
        f"\nWARNING: Expected {EXPECTED_TAXA} taxa "
        f"but found {len(ml)}."
    )


# =============================================================================
# LOAD VARIABLE BUSCO MATRIX
# =============================================================================

print("\n" + "=" * 80)
print("LOADING VARIABLE BUSCO MATRIX")
print("=" * 80)


variable = pd.read_csv(
    VARIABLE_MATRIX
)


print("\nShape:")
print(variable.shape)


print("\nFirst columns:")
print(
    list(
        variable.columns[:10]
    )
)


# =============================================================================
# LOAD ROBUST BUSCO LIST
# =============================================================================

print("\n" + "=" * 80)
print("LOADING ROBUST BUSCO CANDIDATES")
print("=" * 80)


robust_df = pd.read_csv(
    ROBUST_BUSCO_FILE
)


print("\nShape:")
print(robust_df.shape)


print("\nColumns:")
print(
    list(
        robust_df.columns
    )
)


if "BUSCO" not in robust_df.columns:

    raise RuntimeError(
        "Robust BUSCO file does not contain a BUSCO column."
    )


robust_buscos = (

    robust_df["BUSCO"]

    .astype(str)

    .str.strip()

    .tolist()
)


# Remove duplicates while preserving order

robust_buscos = list(
    dict.fromkeys(
        robust_buscos
    )
)


print("\nRobust BUSCO count:")
print(
    len(robust_buscos)
)


# =============================================================================
# LOAD PHYLOGENETIC CV
# =============================================================================

print("\n" + "=" * 80)
print("LOADING PHYLOGENETIC CV ASSIGNMENTS")
print("=" * 80)


cv = pd.read_csv(
    PHYLO_CV_FILE
)


print("\nShape:")
print(cv.shape)


print("\nColumns:")
print(
    list(
        cv.columns
    )
)


required_cv_columns = [
    "Assembly_Accession",
    "Phylogenetic_Fold"
]


for column in required_cv_columns:

    if column not in cv.columns:

        raise RuntimeError(
            f"Phylogenetic CV file missing required column: {column}"
        )


# =============================================================================
# RECONCILE TAXA
# =============================================================================

print("\n" + "=" * 80)
print("RECONCILING TAXA")
print("=" * 80)


ml["Assembly_Accession"] = (

    ml["Assembly_Accession"]

    .astype(str)

    .str.strip()
)


cv["Assembly_Accession"] = (

    cv["Assembly_Accession"]

    .astype(str)

    .str.strip()
)


cv_small = (

    cv[
        [
            "Assembly_Accession",
            "Phylogenetic_Fold"
        ]
    ]

    .drop_duplicates(
        subset=[
            "Assembly_Accession"
        ]
    )
)


if len(cv_small) != len(cv):

    print(
        "\nWARNING: Duplicate Assembly_Accession values "
        "found in phylogenetic CV file."
    )


ml = ml.merge(

    cv_small,

    on="Assembly_Accession",

    how="left"
)


print("\nML rows:")
print(
    len(ml)
)


print("\nRows with phylogenetic fold:")
print(
    ml["Phylogenetic_Fold"]
    .notna()
    .sum()
)


print("\nRows without phylogenetic fold:")
print(
    ml["Phylogenetic_Fold"]
    .isna()
    .sum()
)


if ml["Phylogenetic_Fold"].isna().any():

    missing = ml.loc[
        ml["Phylogenetic_Fold"].isna(),
        [
            "Species",
            "Assembly_Accession"
        ]
    ]

    print("\nMissing taxa:")
    print(
        missing.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Some ML taxa do not have phylogenetic fold assignments."
    )


# =============================================================================
# SAFE FOLD CONVERSION
# =============================================================================

ml["Phylogenetic_Fold"] = pd.to_numeric(
    ml["Phylogenetic_Fold"],
    errors="coerce"
)


if ml["Phylogenetic_Fold"].isna().any():

    raise RuntimeError(
        "Invalid phylogenetic fold values detected."
    )


ml["Phylogenetic_Fold"] = (

    ml["Phylogenetic_Fold"]

    .round()

    .astype(int)
)


folds = sorted(
    ml["Phylogenetic_Fold"]
    .unique()
    .tolist()
)


print("\nPhylogenetic folds:")
print(folds)


print("\nFold sizes:")
print(
    ml["Phylogenetic_Fold"]
    .value_counts()
    .sort_index()
)


if len(folds) != N_OUTER_FOLDS:

    raise RuntimeError(
        f"Expected {N_OUTER_FOLDS} phylogenetic folds "
        f"but found {len(folds)}."
    )


# =============================================================================
# IDENTIFY BUSCO FEATURES
# =============================================================================

print("\n" + "=" * 80)
print("IDENTIFYING BUSCO FEATURES")
print("=" * 80)


non_busco_columns = set(
    META_COLUMNS
    +
    TARGET_COLUMNS
    +
    [
        "Phylogenetic_Fold"
    ]
)


busco_columns = [
    column
    for column in ml.columns
    if column not in non_busco_columns
]


print(
    "\nBUSCO feature count in phylogeny-aware matrix:"
)

print(
    len(busco_columns)
)


# =============================================================================
# IDENTIFY VARIABLE BUSCOs
# =============================================================================

variable_busco_candidates = [

    column

    for column in variable.columns

    if column not in (
        META_COLUMNS
        +
        TARGET_COLUMNS
    )
]


variable_buscos = [

    column

    for column in variable_busco_candidates

    if column in busco_columns
]


print(
    "\nVariable BUSCO features available:"
)

print(
    len(variable_buscos)
)


# =============================================================================
# ROBUST BUSCO AVAILABILITY
# =============================================================================

robust_available = [

    busco

    for busco in robust_buscos

    if busco in busco_columns
]


robust_missing = [

    busco

    for busco in robust_buscos

    if busco not in busco_columns
]


print(
    "\nRobust BUSCOs available:"
)

print(
    len(robust_available)
)


print(
    "\nRobust BUSCOs missing:"
)

print(
    len(robust_missing)
)


if robust_missing:

    print(
        "\nFirst missing robust BUSCOs:"
    )

    print(
        robust_missing[:20]
    )


if len(robust_available) == 0:

    raise RuntimeError(
        "No robust BUSCOs are available in the ML matrix."
    )


# =============================================================================
# BASE FEATURE SETS
# =============================================================================

feature_set_definitions = {

    "Robust_79":
        robust_available,

    "All_Variable":
        variable_buscos
}


print("\n" + "=" * 80)
print("BASE FEATURE SETS")
print("=" * 80)


for name, features in feature_set_definitions.items():

    print(
        f"{name}: {len(features)} features"
    )


# =============================================================================
# RESULT CONTAINERS
# =============================================================================

all_results = []

all_predictions = []

selected_feature_records = []

best_model_records = []


# =============================================================================
# OUTER PHYLOGENETIC CV
# =============================================================================

for phenotype in PHENOTYPES:

    print("\n\n")
    print("=" * 80)
    print(
        f"PHENOTYPE: {phenotype}"
    )
    print("=" * 80)


    # =========================================================================
    # CHECK TARGET
    # =========================================================================

    if phenotype not in ml.columns:

        raise RuntimeError(
            f"Phenotype not found in ML matrix: {phenotype}"
        )


    y_all = pd.to_numeric(
        ml[phenotype],
        errors="coerce"
    )


    valid_target = y_all.notna()


    if not valid_target.all():

        print(
            f"\nRemoving "
            f"{np.sum(~valid_target)} "
            f"rows with missing phenotype."
        )


    data = ml.loc[
        valid_target
    ].copy()


    y = (

        y_all.loc[
            valid_target
        ]

        .astype(float)
    )


    print("\nSamples:")
    print(
        len(data)
    )


    print("\nPhenotype statistics:")
    print(
        y.describe()
    )


    # =========================================================================
    # FEATURE SET NAMES
    # =========================================================================

    phenotype_selected_names = [

        f"Phenotype_Selected_Top{k}"

        for k in TOP_K_VALUES
    ]


    feature_set_names = [

        "Robust_79",

        "All_Variable",

        "Filtered_Variable"

    ] + phenotype_selected_names


    # =========================================================================
    # OOF STORAGE
    # =========================================================================

    prediction_store = {}


    for feature_set_name in feature_set_names:

        for model_name in get_models().keys():

            prediction_store[
                (
                    feature_set_name,
                    model_name
                )
            ] = {

                "y_true":
                    np.full(
                        len(data),
                        np.nan,
                        dtype=float
                    ),

                "y_pred":
                    np.full(
                        len(data),
                        np.nan,
                        dtype=float
                    ),

                # IMPORTANT:
                # Keep fold as normal float NumPy array.
                # DO NOT use pandas nullable Int64 here.
                "fold":
                    np.full(
                        len(data),
                        np.nan,
                        dtype=float
                    )
            }


    # =========================================================================
    # OUTER FOLDS
    # =========================================================================

    for fold in folds:

        print("\n" + "-" * 80)

        print(
            f"{phenotype} — "
            f"OUTER PHYLOGENETIC FOLD {fold}"
        )

        print("-" * 80)


        train_mask = (

            data["Phylogenetic_Fold"]
            != fold
        )


        test_mask = (

            data["Phylogenetic_Fold"]
            == fold
        )


        train_idx = np.where(
            train_mask
        )[0]


        test_idx = np.where(
            test_mask
        )[0]


        print(
            f"Training taxa: {len(train_idx)}"
        )

        print(
            f"Test taxa: {len(test_idx)}"
        )


        # =====================================================================
        # BASE MATRIX
        # =====================================================================

        X_base = (

            data[
                busco_columns
            ]

            .apply(
                pd.to_numeric,
                errors="coerce"
            )

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )


        X_train_all = X_base.iloc[
            train_idx
        ]


        X_test_all = X_base.iloc[
            test_idx
        ]


        y_train = y.iloc[
            train_idx
        ].values


        y_test = y.iloc[
            test_idx
        ].values


        # =====================================================================
        # FEATURE SET CONSTRUCTION
        # =====================================================================

        fold_feature_sets = {}


        # ---------------------------------------------------------------------
        # FEATURE SET 1 — ROBUST 79
        # ---------------------------------------------------------------------

        robust_features = [

            feature

            for feature in robust_available

            if feature in X_train_all.columns
        ]


        fold_feature_sets[
            "Robust_79"
        ] = robust_features


        # ---------------------------------------------------------------------
        # FEATURE SET 2 — ALL VARIABLE BUSCOs
        # ---------------------------------------------------------------------

        all_variable_features = [

            feature

            for feature in variable_buscos

            if feature in X_train_all.columns
        ]


        fold_feature_sets[
            "All_Variable"
        ] = all_variable_features


        # ---------------------------------------------------------------------
        # FEATURE SET 3 — FILTERED VARIABLE BUSCOs
        # ---------------------------------------------------------------------
        #
        # IMPORTANT:
        # Filtering is performed using TRAINING DATA ONLY.
        # ---------------------------------------------------------------------

        filtered_features = []


        for feature in all_variable_features:

            values = X_train_all[
                feature
            ].values


            prevalence = np.mean(
                values > 0
            )


            if (
                MIN_PREVALENCE
                <= prevalence
                <= MAX_PREVALENCE
            ):

                filtered_features.append(
                    feature
                )


        fold_feature_sets[
            "Filtered_Variable"
        ] = filtered_features


        # ---------------------------------------------------------------------
        # PHENOTYPE-ASSOCIATED FEATURE SELECTION
        # ---------------------------------------------------------------------

        candidate_features = (
            filtered_features.copy()
        )


        if len(candidate_features) == 0:

            raise RuntimeError(
                f"No filtered BUSCOs available "
                f"in fold {fold} "
                f"for phenotype {phenotype}."
            )


        X_train_candidates = (

            X_train_all[
                candidate_features
            ]

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )


        # ---------------------------------------------------------------------
        # F-REGRESSION
        # ---------------------------------------------------------------------

        scores, pvalues = f_regression(

            X_train_candidates,

            y_train
        )


        score_df = pd.DataFrame({

            "BUSCO":
                candidate_features,

            "F_score":
                scores,

            "P_value":
                pvalues
        })


        score_df["F_score"] = (

            score_df["F_score"]

            .replace(
                [np.inf, -np.inf],
                np.nan
            )

            .fillna(0)
        )


        score_df = (

            score_df

            .sort_values(
                "F_score",
                ascending=False
            )
        )


        # ---------------------------------------------------------------------
        # TOP 50 / 100 / 200
        # ---------------------------------------------------------------------

        for k in TOP_K_VALUES:

            k_actual = min(
                k,
                len(score_df)
            )


            selected = (

                score_df

                .head(k_actual)

                ["BUSCO"]

                .tolist()
            )


            fold_feature_sets[
                f"Phenotype_Selected_Top{k}"
            ] = selected


            for rank, busco in enumerate(
                selected,
                start=1
            ):

                selected_feature_records.append({

                    "Phenotype":
                        phenotype,

                    "Outer_Fold":
                        int(fold),

                    "Feature_Set":
                        f"Phenotype_Selected_Top{k}",

                    "BUSCO":
                        busco,

                    "Selection_Rank":
                        int(rank)
                })


        # ---------------------------------------------------------------------
        # SAVE BASE FEATURE LISTS
        # ---------------------------------------------------------------------

        for set_name, features in (
            fold_feature_sets.items()
        ):

            if set_name.startswith(
                "Phenotype_Selected"
            ):

                continue


            for rank, busco in enumerate(
                features,
                start=1
            ):

                selected_feature_records.append({

                    "Phenotype":
                        phenotype,

                    "Outer_Fold":
                        int(fold),

                    "Feature_Set":
                        set_name,

                    "BUSCO":
                        busco,

                    "Selection_Rank":
                        int(rank)
                })


        print(
            "\nFeature counts for fold:"
        )


        for name, features in (
            fold_feature_sets.items()
        ):

            print(
                f"  {name}: {len(features)}"
            )


        # =====================================================================
        # MODEL DEFINITIONS
        # =====================================================================

        models = get_models()

        grids = get_parameter_grids()


        # =====================================================================
        # MODEL LOOP
        # =====================================================================

        for feature_set_name, features in (
            fold_feature_sets.items()
        ):

            if len(features) == 0:

                print(
                    f"\nSkipping empty feature set: "
                    f"{feature_set_name}"
                )

                continue


            X_train = (

                X_train_all[
                    features
                ]

                .copy()
            )


            X_test = (

                X_test_all[
                    features
                ]

                .copy()
            )


            # -----------------------------------------------------------------
            # CLEAN TRAIN / TEST
            # -----------------------------------------------------------------

            X_train = (

                X_train

                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                .fillna(0)
            )


            X_test = (

                X_test

                .replace(
                    [np.inf, -np.inf],
                    np.nan
                )

                .fillna(0)
            )


            # =================================================================
            # MODEL LOOP
            # =================================================================

            for model_name, model in (
                models.items()
            ):

                print(
                    f"\n{feature_set_name} | "
                    f"{model_name}"
                )


                param_grid = grids[
                    model_name
                ]


                # -------------------------------------------------------------
                # INNER CV
                # -------------------------------------------------------------

                if param_grid:

                    search = GridSearchCV(

                        estimator=model,

                        param_grid=param_grid,

                        scoring=
                        "neg_root_mean_squared_error",

                        cv=N_INNER_FOLDS,

                        # IMPORTANT:
                        # Prevent nested joblib parallelism.
                        n_jobs=GRID_N_JOBS,

                        refit=True
                    )


                    search.fit(

                        X_train,

                        y_train
                    )


                    fitted_model = (
                        search.best_estimator_
                    )


                    best_params = (
                        search.best_params_
                    )

                else:

                    fitted_model = clone(
                        model
                    )


                    fitted_model.fit(

                        X_train,

                        y_train
                    )


                    best_params = {}


                # -------------------------------------------------------------
                # OUTER TEST PREDICTION
                # -------------------------------------------------------------

                pred = fitted_model.predict(
                    X_test
                )


                pred = np.asarray(
                    pred,
                    dtype=float
                )


                # Guard against accidental NaN/Inf predictions

                pred = np.nan_to_num(
                    pred,
                    nan=float(np.nanmean(y_train)),
                    posinf=float(np.nanmax(y_train)),
                    neginf=float(np.nanmin(y_train))
                )


                fold_rmse = rmse(

                    y_test,

                    pred
                )


                fold_mae = safe_float(

                    mean_absolute_error(

                        y_test,

                        pred
                    )
                )


                fold_r2 = safe_r2(

                    y_test,

                    pred
                )


                print(

                    f"RMSE={fold_rmse:.6f} | "

                    f"MAE={fold_mae:.6f} | "

                    f"R2={fold_r2:.6f}"
                )


                # -------------------------------------------------------------
                # STORE OOF PREDICTION
                # -------------------------------------------------------------

                key = (

                    feature_set_name,

                    model_name
                )


                prediction_store[
                    key
                ][
                    "y_true"
                ][
                    test_idx
                ] = y_test


                prediction_store[
                    key
                ][
                    "y_pred"
                ][
                    test_idx
                ] = pred


                # IMPORTANT:
                #
                # Store fold as ordinary float.
                #
                # DO NOT use:
                #
                # stored["fold"].astype("Int64")
                #
                # because stored["fold"] is a NumPy array.
                # -------------------------------------------------------------

                prediction_store[
                    key
                ][
                    "fold"
                ][
                    test_idx
                ] = float(fold)


                # -------------------------------------------------------------
                # SAVE FOLD RESULT
                # -------------------------------------------------------------

                all_results.append({

                    "Phenotype":
                        phenotype,

                    "Feature_Set":
                        feature_set_name,

                    "Model":
                        model_name,

                    "Outer_Fold":
                        int(fold),

                    "N_Train":
                        int(len(train_idx)),

                    "N_Test":
                        int(len(test_idx)),

                    "N_Features":
                        int(len(features)),

                    "RMSE":
                        safe_float(
                            fold_rmse
                        ),

                    "MAE":
                        safe_float(
                            fold_mae
                        ),

                    "R2":
                        safe_float(
                            fold_r2
                        ),

                    "Best_Params":
                        json.dumps(
                            json_safe(
                                best_params
                            )
                        )
                })


    # =========================================================================
    # AGGREGATE OOF PERFORMANCE
    # =========================================================================

    print("\n" + "=" * 80)

    print(
        f"AGGREGATED PERFORMANCE — {phenotype}"
    )

    print("=" * 80)


    phenotype_summary = []


    for (
        feature_set_name,
        model_name
    ), stored in prediction_store.items():


        # ---------------------------------------------------------------------
        # VALID OOF PREDICTIONS
        # ---------------------------------------------------------------------

        valid = (

            ~np.isnan(
                stored["y_pred"]
            )
        )


        if valid.sum() == 0:

            continue


        y_true_oof = (

            stored["y_true"][
                valid
            ]
        )


        y_pred_oof = (

            stored["y_pred"][
                valid
            ]
        )


        # ---------------------------------------------------------------------
        # OVERALL OOF METRICS
        # ---------------------------------------------------------------------

        overall_rmse = rmse(

            y_true_oof,

            y_pred_oof
        )


        overall_mae = safe_float(

            mean_absolute_error(

                y_true_oof,

                y_pred_oof
            )
        )


        overall_r2 = safe_r2(

            y_true_oof,

            y_pred_oof
        )


        # ---------------------------------------------------------------------
        # FOLD-LEVEL METRICS
        # ---------------------------------------------------------------------

        fold_rows = [

            row

            for row in all_results

            if (

                row["Phenotype"]
                == phenotype

                and

                row["Feature_Set"]
                == feature_set_name

                and

                row["Model"]
                == model_name
            )
        ]


        fold_rmse_values = [

            row["RMSE"]

            for row in fold_rows

            if pd.notna(
                row["RMSE"]
            )
        ]


        fold_mae_values = [

            row["MAE"]

            for row in fold_rows

            if pd.notna(
                row["MAE"]
            )
        ]


        fold_r2_values = [

            row["R2"]

            for row in fold_rows

            if pd.notna(
                row["R2"]
            )
        ]


        # ---------------------------------------------------------------------
        # MEANS
        # ---------------------------------------------------------------------

        mean_rmse = safe_float(

            np.mean(
                fold_rmse_values
            )
        )


        if len(fold_rmse_values) > 1:

            sd_rmse = safe_float(

                np.std(
                    fold_rmse_values,
                    ddof=1
                )
            )

        else:

            sd_rmse = np.nan


        mean_mae = safe_float(

            np.mean(
                fold_mae_values
            )
        )


        mean_r2 = (

            safe_float(

                np.mean(
                    fold_r2_values
                )
            )

            if fold_r2_values

            else np.nan
        )


        # ---------------------------------------------------------------------
        # RESULT ROW
        # ---------------------------------------------------------------------

        row = {

            "Phenotype":
                phenotype,

            "Feature_Set":
                feature_set_name,

            "Model":
                model_name,

            "N_Features":
                int(
                    max(
                        r["N_Features"]
                        for r in fold_rows
                    )
                ),

            "Mean_CV_RMSE":
                mean_rmse,

            "SD_CV_RMSE":
                sd_rmse,

            "Mean_CV_MAE":
                mean_mae,

            "Mean_CV_R2":
                mean_r2,

            "Overall_OOF_RMSE":
                safe_float(
                    overall_rmse
                ),

            "Overall_OOF_MAE":
                safe_float(
                    overall_mae
                ),

            "Overall_OOF_R2":
                safe_float(
                    overall_r2
                )
        }


        phenotype_summary.append(
            row
        )


        # =====================================================================
        # OOF PREDICTION TABLE
        # =====================================================================
        #
        # FIX:
        # Never use:
        #
        #     stored["fold"].astype("Int64")
        #
        # because stored["fold"] is a NumPy array.
        #
        # Instead we construct a pandas Series and keep the values numeric.
        # =====================================================================

        fold_values = pd.to_numeric(

            pd.Series(
                stored["fold"]
            ),

            errors="coerce"
        )


        # Keep as normal floating point.
        #
        # Values are 1.0, 2.0, ... and missing values are NaN.
        #
        # This is completely safe for CSV output.

        fold_values = (

            fold_values

            .where(
                fold_values.notna(),
                np.nan
            )
        )


        prediction_df = pd.DataFrame({

            "Species":
                data["Species"].values,

            "Assembly_Accession":
                data["Assembly_Accession"].values,

            "Phenotype":
                phenotype,

            "Feature_Set":
                feature_set_name,

            "Model":
                model_name,

            "Phylogenetic_Fold":
                fold_values.values,

            "Observed":
                stored["y_true"],

            "Predicted":
                stored["y_pred"]
        })


        prediction_path = (

            PREDICTION_OUTPUT

            /

            (
                f"{phenotype}_"
                f"{feature_set_name}_"
                f"{model_name}_OOF_420.csv"
            )
        )


        prediction_df.to_csv(

            prediction_path,

            index=False
        )


    # =========================================================================
    # PHENOTYPE SUMMARY
    # =========================================================================

    summary_df = pd.DataFrame(
        phenotype_summary
    )


    if summary_df.empty:

        raise RuntimeError(
            f"No valid model results were generated "
            f"for phenotype {phenotype}."
        )


    summary_df = (

        summary_df

        .sort_values(

            [
                "Overall_OOF_RMSE",
                "Mean_CV_RMSE"
            ],

            ascending=True
        )
    )


    print(
        summary_df.to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # SAVE PHENOTYPE SUMMARY
    # -------------------------------------------------------------------------

    summary_path = (

        OUTPUT_ROOT

        /

        (
            f"{phenotype}_"
            "feature_model_comparison_420.csv"
        )
    )


    summary_df.to_csv(

        summary_path,

        index=False
    )


    # -------------------------------------------------------------------------
    # BEST CONFIGURATION
    # -------------------------------------------------------------------------

    best_row = (

        summary_df

        .sort_values(

            [
                "Overall_OOF_RMSE",
                "Mean_CV_RMSE"
            ],

            ascending=True
        )

        .iloc[0]
    )


    best_model_records.append(
        best_row.to_dict()
    )


    print("\n" + "-" * 80)

    print(
        f"BEST CONFIGURATION — {phenotype}"
    )

    print("-" * 80)


    print(
        "Feature set:",
        best_row["Feature_Set"]
    )


    print(
        "Model:",
        best_row["Model"]
    )


    print(
        "Features:",
        int(
            best_row["N_Features"]
        )
    )


    print(
        "Mean CV RMSE:",
        f"{best_row['Mean_CV_RMSE']:.6f}"
    )


    print(
        "Mean CV R2:",
        f"{best_row['Mean_CV_R2']:.6f}"
    )


    print(
        "Overall OOF RMSE:",
        f"{best_row['Overall_OOF_RMSE']:.6f}"
    )


    print(
        "Overall OOF R2:",
        f"{best_row['Overall_OOF_R2']:.6f}"
    )


# =============================================================================
# SAVE ALL OUTER-FOLD RESULTS
# =============================================================================

print("\n\n" + "=" * 80)
print("SAVING GLOBAL RESULTS")
print("=" * 80)


all_results_df = pd.DataFrame(
    all_results
)


all_results_path = (

    OUTPUT_ROOT

    /

    "all_outer_fold_results_420.csv"
)


all_results_df.to_csv(

    all_results_path,

    index=False
)


print(
    "\n✓ Outer-fold results written:"
)

print(
    all_results_path
)


# =============================================================================
# SAVE SELECTED FEATURES
# =============================================================================

selected_features_df = pd.DataFrame(
    selected_feature_records
)


selected_features_path = (

    FEATURE_OUTPUT

    /

    "all_selected_features_by_fold_420.csv"
)


selected_features_df.to_csv(

    selected_features_path,

    index=False
)


print(
    "\n✓ Fold-specific feature selections written:"
)

print(
    selected_features_path
)


# =============================================================================
# SAVE BEST MODEL CONFIGURATIONS
# =============================================================================

best_models_df = pd.DataFrame(
    best_model_records
)


best_models_path = (

    OUTPUT_ROOT

    /

    "best_feature_model_combinations_420.csv"
)


best_models_df.to_csv(

    best_models_path,

    index=False
)


print(
    "\n✓ Best configurations written:"
)

print(
    best_models_path
)


# =============================================================================
# GLOBAL COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("FINAL FEATURE-SET / MODEL COMPARISON")
print("=" * 80)


grouped = (

    all_results_df

    .groupby(

        [
            "Phenotype",
            "Feature_Set",
            "Model"
        ]
    )

    .agg(

        Mean_CV_RMSE=(
            "RMSE",
            "mean"
        ),

        SD_CV_RMSE=(
            "RMSE",
            "std"
        ),

        Mean_CV_MAE=(
            "MAE",
            "mean"
        ),

        Mean_CV_R2=(
            "R2",
            "mean"
        ),

        N_Features=(
            "N_Features",
            "max"
        )
    )

    .reset_index()
)


# =============================================================================
# ADD OVERALL OOF METRICS TO GLOBAL COMPARISON
# =============================================================================

oof_global_records = []


for phenotype in PHENOTYPES:

    phenotype_data = all_results_df[
        all_results_df["Phenotype"] == phenotype
    ]


    # Reconstruct OOF metrics from saved prediction files.

    for feature_set_name in [

        "Robust_79",
        "All_Variable",
        "Filtered_Variable",
        "Phenotype_Selected_Top50",
        "Phenotype_Selected_Top100",
        "Phenotype_Selected_Top200"

    ]:

        for model_name in get_models().keys():

            prediction_path = (

                PREDICTION_OUTPUT

                /

                (
                    f"{phenotype}_"
                    f"{feature_set_name}_"
                    f"{model_name}_OOF_420.csv"
                )
            )


            if not prediction_path.exists():

                continue


            pred_df = pd.read_csv(
                prediction_path
            )


            valid_pred = (

                pred_df["Predicted"]
                .notna()
            )


            if valid_pred.sum() == 0:

                continue


            y_true = (

                pred_df.loc[
                    valid_pred,
                    "Observed"
                ]

                .astype(float)
                .values
            )


            y_pred = (

                pred_df.loc[
                    valid_pred,
                    "Predicted"
                ]

                .astype(float)
                .values
            )


            oof_global_records.append({

                "Phenotype":
                    phenotype,

                "Feature_Set":
                    feature_set_name,

                "Model":
                    model_name,

                "Overall_OOF_RMSE":
                    rmse(
                        y_true,
                        y_pred
                    ),

                "Overall_OOF_MAE":
                    safe_float(
                        mean_absolute_error(
                            y_true,
                            y_pred
                        )
                    ),

                "Overall_OOF_R2":
                    safe_r2(
                        y_true,
                        y_pred
                    )
            })


oof_global_df = pd.DataFrame(
    oof_global_records
)


if not oof_global_df.empty:

    grouped = grouped.merge(

        oof_global_df,

        on=[
            "Phenotype",
            "Feature_Set",
            "Model"
        ],

        how="left"
    )


grouped.to_csv(

    OUTPUT_ROOT
    /
    "aggregated_outer_cv_results_420.csv",

    index=False
)


print(

    grouped

    .sort_values(

        [
            "Phenotype",
            "Mean_CV_RMSE"
        ]
    )

    .to_string(
        index=False
    )
)


# =============================================================================
# BASELINE COMPARISON
# =============================================================================

print("\n" + "=" * 80)
print("BASELINE COMPARISON")
print("=" * 80)


baseline_records = []


for phenotype in PHENOTYPES:

    y_baseline = pd.to_numeric(

        ml[phenotype],

        errors="coerce"

    ).dropna()


    # -------------------------------------------------------------------------
    # GLOBAL MEAN PREDICTOR
    # -------------------------------------------------------------------------

    mean_prediction = np.repeat(

        y_baseline.mean(),

        len(y_baseline)
    )


    baseline_rmse = rmse(

        y_baseline.values,

        mean_prediction
    )


    baseline_mae = safe_float(

        mean_absolute_error(

            y_baseline.values,

            mean_prediction
        )
    )


    baseline_r2 = safe_r2(

        y_baseline.values,

        mean_prediction
    )


    baseline_records.append({

        "Phenotype":
            phenotype,

        "Baseline":
            "Global_Mean",

        "RMSE":
            baseline_rmse,

        "MAE":
            baseline_mae,

        "R2":
            baseline_r2
    })


baseline_df = pd.DataFrame(
    baseline_records
)


baseline_path = (

    OUTPUT_ROOT

    /

    "baseline_comparison_420.csv"
)


baseline_df.to_csv(

    baseline_path,

    index=False
)


print(

    baseline_df.to_string(
        index=False
    )
)


# =============================================================================
# FINAL OPTIMIZATION SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("FINAL OPTIMIZATION SUMMARY")
print("=" * 80)


recommendation_records = []


for phenotype in PHENOTYPES:

    phenotype_rows = (

        grouped[
            grouped["Phenotype"] == phenotype
        ]

        .sort_values(
            "Mean_CV_RMSE",
            ascending=True
        )
    )


    if phenotype_rows.empty:

        continue


    # -------------------------------------------------------------------------
    # BEST OVERALL CONFIGURATION
    # -------------------------------------------------------------------------

    best = phenotype_rows.iloc[0]


    # -------------------------------------------------------------------------
    # ROBUST 79 BASELINE
    # -------------------------------------------------------------------------

    robust_rows = phenotype_rows[
        phenotype_rows["Feature_Set"]
        == "Robust_79"
    ]


    robust_best = None


    if not robust_rows.empty:

        robust_best = (

            robust_rows

            .sort_values(
                "Mean_CV_RMSE"
            )

            .iloc[0]
        )


    # -------------------------------------------------------------------------
    # SUMMARY RECORD
    # -------------------------------------------------------------------------

    record = {

        "Phenotype":
            phenotype,

        "Best_Feature_Set":
            best["Feature_Set"],

        "Best_Model":
            best["Model"],

        "Best_N_Features":
            int(
                best["N_Features"]
            ),

        "Best_Mean_CV_RMSE":
            safe_float(
                best["Mean_CV_RMSE"]
            ),

        "Best_Mean_CV_R2":
            safe_float(
                best["Mean_CV_R2"]
            ),

        "Best_Overall_OOF_RMSE":
            safe_float(
                best.get(
                    "Overall_OOF_RMSE",
                    np.nan
                )
            ),

        "Best_Overall_OOF_R2":
            safe_float(
                best.get(
                    "Overall_OOF_R2",
                    np.nan
                )
            )
    }


    if robust_best is not None:

        record[
            "Robust_79_Best_Model"
        ] = robust_best["Model"]


        record[
            "Robust_79_Mean_CV_RMSE"
        ] = safe_float(

            robust_best[
                "Mean_CV_RMSE"
            ]
        )


        record[
            "Robust_79_Mean_CV_R2"
        ] = safe_float(

            robust_best[
                "Mean_CV_R2"
            ]
        )


        # Positive means optimized configuration has lower RMSE.

        record[
            "RMSE_Improvement_vs_Robust"
        ] = safe_float(

            robust_best[
                "Mean_CV_RMSE"
            ]

            -

            best[
                "Mean_CV_RMSE"
            ]
        )


    recommendation_records.append(
        record
    )


recommendation_df = pd.DataFrame(
    recommendation_records
)


recommendation_path = (

    OUTPUT_ROOT

    /

    "feature_optimization_summary_420.csv"
)


recommendation_df.to_csv(

    recommendation_path,

    index=False
)


print(

    recommendation_df.to_string(
        index=False
    )
)


print(
    "\n✓ Optimization summary written:"
)

print(
    recommendation_path
)


# =============================================================================
# SAVE RUN METADATA
# =============================================================================

metadata = {

    "project":
        "Y1000+",

    "stage":
        "Stage 6A - Predictive Feature-Set Optimization",

    "n_taxa":
        int(len(ml)),

    "n_robust_buscos":
        int(len(robust_available)),

    "n_variable_buscos":
        int(len(variable_buscos)),

    "phenotypes":
        PHENOTYPES,

    "phylogenetic_folds":
        [
            int(x)
            for x in folds
        ],

    "fold_sizes":
        {
            str(k):
                int(v)

            for k, v in (

                ml[
                    "Phylogenetic_Fold"
                ]

                .value_counts()

                .sort_index()

                .items()
            )
        },

    "feature_sets":
        {

            "Robust_79":
                int(
                    len(
                        robust_available
                    )
                ),

            "All_Variable":
                int(
                    len(
                        variable_buscos
                    )
                ),

            "Filtered_Variable":
                "Training-fold prevalence 0.05-0.95",

            "Phenotype_Selected":
                TOP_K_VALUES
        },

    "models":
        list(
            get_models().keys()
        ),

    "random_state":
        int(RANDOM_STATE),

    "outer_validation":
        "5-fold phylogenetic CV",

    "inner_validation":
        "3-fold CV for hyperparameter tuning",

    "feature_selection":
        "Performed inside outer training folds only",

    "existing_final_models_overwritten":
        False,

    "gridsearch_n_jobs":
        int(GRID_N_JOBS),

    "tree_model_n_jobs":
        int(TREE_N_JOBS),

    "fold_dtype_fix":
        (
            "OOF phylogenetic fold stored as numeric float; "
            "pandas nullable Int64 conversion removed"
        )
}


metadata_path = (

    OUTPUT_ROOT

    /

    "optimization_run_metadata_420.json"
)


with open(

    metadata_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        json_safe(
            metadata
        ),

        f,

        indent=4
    )


# =============================================================================
# COMPLETION
# =============================================================================

print("\n" + "=" * 80)
print("STAGE 6A COMPLETE")
print("=" * 80)


print(
    "\nThe existing final models were NOT overwritten."
)


print(
    "\nMain output directory:"
)


print(
    OUTPUT_ROOT
)


print(
    "\nUse this file to determine whether broader BUSCO "
    "feature sets improve prediction:"
)


print(
    recommendation_path
)


print(
    "\nNext stage after reviewing these results:"
)


print(
    "Select the validated feature/model configuration "
    "→ final refit → diagnostics → SHAP/XAI"
)


print("=" * 80)