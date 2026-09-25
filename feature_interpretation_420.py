# =============================================================================
# FEATURE-LEVEL INTERPRETATION / XAI — Y1000+ PHYLOGENY-AWARE ML
# =============================================================================
#
# Primary analysis:
#   Carbon_Breadth
#   Feature set: Robust_79
#   Model: SVR_RBF
#
# Interpretation method:
#   Phylogeny-aware permutation importance
#
# Additional analyses:
#   - fold-wise importance
#   - global importance
#   - feature stability
#   - direction/correlation summaries
#   - top BUSCO candidate table
#   - exploratory analysis for other selected phenotypes
#
# IMPORTANT:
#   Existing models and OOF files are NOT overwritten.
#
# FIX:
#   final_model_busco_features_420.csv is ONLY a BUSCO feature LIST.
#   The actual 420-taxon genomic matrix is:
#
#   final_model_training_matrix_420.csv
#
# =============================================================================


import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from sklearn.ensemble import GradientBoostingRegressor

from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path(
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml\phylogeny_cv\model_training"
)

OPT_DIR = BASE / "performance_optimization_420"

OOF_DIR = (
    OPT_DIR /
    "oof_predictions"
)

FINAL_DIR = (
    BASE /
    "final_models_420"
)

OUTPUT_DIR = (
    BASE /
    "feature_interpretation_420"
)

TABLE_DIR = (
    OUTPUT_DIR /
    "tables"
)

FIG_DIR = (
    OUTPUT_DIR /
    "figures"
)


TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# PRIMARY CONFIGURATION
# =============================================================================

PRIMARY_PHENOTYPE = "Carbon_Breadth"

PRIMARY_FEATURE_SET = "Robust_79"

PRIMARY_MODEL = "SVR_RBF"

N_PERMUTATIONS = 100

RANDOM_STATE = 42


# =============================================================================
# SELECTED CONFIGURATIONS FROM OPTIMIZATION
# =============================================================================

CONFIGURATIONS = [

    {
        "phenotype": "Carbon_Breadth",
        "feature_set": "Robust_79",
        "model": "SVR_RBF",
        "primary": True
    },

    {
        "phenotype": "Nitrogen_Breadth",
        "feature_set": "Filtered_Variable",
        "model": "SVR_RBF",
        "primary": False
    },

    {
        "phenotype": "Utilized_Median_Growth",
        "feature_set": "Phenotype_Selected_Top50",
        "model": "GradientBoosting",
        "primary": False
    }

]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def banner(text):

    print("\n" + "=" * 80)

    print(text)

    print("=" * 80)


# =============================================================================
# FIND OOF FILE
# =============================================================================

def find_oof_file(
    phenotype,
    feature_set,
    model
):

    exact_name = (
        f"{phenotype}_{feature_set}_{model}_OOF_420.csv"
    )

    exact_path = (
        OOF_DIR /
        exact_name
    )

    if exact_path.exists():

        return exact_path


    patterns = [

        f"{phenotype}_{feature_set}_{model}_OOF*.csv",

        f"{phenotype}*{feature_set}*{model}*OOF*.csv"

    ]


    for pattern in patterns:

        matches = list(
            OOF_DIR.glob(pattern)
        )

        if matches:

            return matches[0]


    return None


# =============================================================================
# LOCATE ACTUAL 420-TAXON GENOMIC MATRIX
# =============================================================================

def locate_feature_matrix():

    """
    Locate the ACTUAL 420-taxon genomic training matrix.

    IMPORTANT:

    final_model_busco_features_420.csv
        = feature-name list only

    final_model_training_matrix_420.csv
        = actual 420 × genomic-feature matrix
    """

    candidates = [

        FINAL_DIR /
        "final_model_training_matrix_420.csv",

        FINAL_DIR /
        "final_model_training_matrix.csv"

    ]


    for path in candidates:

        if not path.exists():

            continue


        try:

            test = pd.read_csv(
                path,
                nrows=3
            )

        except Exception:

            continue


        required = {
            "Species",
            "Assembly_Accession"
        }


        if required.issubset(
            set(test.columns)
        ):

            return path


    # -------------------------------------------------------------------------
    # Recursive fallback
    # -------------------------------------------------------------------------

    matches = list(
        BASE.rglob(
            "final_model_training_matrix_420.csv"
        )
    )


    for path in matches:

        try:

            test = pd.read_csv(
                path,
                nrows=3
            )

        except Exception:

            continue


        required = {
            "Species",
            "Assembly_Accession"
        }


        if required.issubset(
            set(test.columns)
        ):

            return path


    raise FileNotFoundError(
        "\nCould not locate the actual 420-taxon genomic "
        "training matrix.\n\n"
        "Expected:\n"
        "final_model_training_matrix_420.csv"
    )


# =============================================================================
# LOCATE ROBUST BUSCO FEATURE LIST
# =============================================================================

def locate_robust_feature_list():

    candidates = [

        FINAL_DIR /
        "final_model_busco_features_420.csv",

        FINAL_DIR /
        "final_model_busco_features.csv"

    ]


    for path in candidates:

        if path.exists():

            return path


    matches = list(
        BASE.rglob(
            "final_model_busco_features_420.csv"
        )
    )


    if matches:

        return matches[0]


    raise FileNotFoundError(
        "Could not locate final_model_busco_features_420.csv"
    )


# =============================================================================
# LOCATE PHYLOGENETIC ASSIGNMENTS
# =============================================================================

def locate_phylogenetic_assignments():

    """
    Locate a phylogenetic assignment file containing:

        Species
        Assembly_Accession
        Phylogenetic_Fold
    """


    preferred_patterns = [

        "*phylogenetic*fold*.csv",

        "*phylogeny*fold*.csv",

        "*phylogenetic*assignment*.csv",

        "*phylogeny*assignment*.csv",

        "all_phylogenetic_oof_predictions_420.csv"

    ]


    candidates = []


    for pattern in preferred_patterns:

        candidates.extend(
            BASE.rglob(pattern)
        )


    unique = []


    for x in candidates:

        if x not in unique:

            unique.append(x)


    required = {
        "Species",
        "Assembly_Accession",
        "Phylogenetic_Fold"
    }


    for path in unique:

        try:

            df = pd.read_csv(
                path,
                nrows=5
            )

        except Exception:

            continue


        if required.issubset(
            set(df.columns)
        ):

            return path


    # -------------------------------------------------------------------------
    # Broader fallback
    # -------------------------------------------------------------------------

    for path in BASE.rglob("*.csv"):

        try:

            df = pd.read_csv(
                path,
                nrows=3
            )

        except Exception:

            continue


        if required.issubset(
            set(df.columns)
        ):

            return path


    raise FileNotFoundError(
        "Could not locate a phylogenetic assignment file containing "
        "Species, Assembly_Accession and Phylogenetic_Fold."
    )


# =============================================================================
# LOAD ACTUAL 420-TAXON GENOMIC MATRIX
# =============================================================================

banner(
    "LOADING ACTUAL 420-TAXON GENOMIC TRAINING MATRIX"
)


feature_path = locate_feature_matrix()


print(
    "\nSelected genomic matrix:"
)

print(
    feature_path
)


feature_df = pd.read_csv(
    feature_path
)


print(
    "\nShape:"
)

print(
    feature_df.shape
)


print(
    "\nFirst columns:"
)

print(
    feature_df.columns.tolist()[:20]
)


# =============================================================================
# CHECK MATRIX STRUCTURE
# =============================================================================

required_matrix_columns = [

    "Species",

    "Assembly_Accession"

]


missing_matrix_columns = [

    c

    for c in required_matrix_columns

    if c not in feature_df.columns

]


if missing_matrix_columns:

    raise RuntimeError(
        "The selected genomic matrix is missing required columns: "
        f"{missing_matrix_columns}"
    )


print(
    "\n✓ Actual genomic matrix contains "
    "Species and Assembly_Accession."
)


# =============================================================================
# LOAD ROBUST BUSCO LIST
# =============================================================================

banner(
    "LOADING ROBUST 79 BUSCO FEATURE LIST"
)


robust_feature_path = (
    locate_robust_feature_list()
)


print(
    "\nRobust BUSCO feature-list file:"
)

print(
    robust_feature_path
)


robust_df = pd.read_csv(
    robust_feature_path
)


print(
    "\nShape:"
)

print(
    robust_df.shape
)


print(
    "\nColumns:"
)

print(
    robust_df.columns.tolist()
)


if "BUSCO" not in robust_df.columns:

    raise RuntimeError(
        "Robust BUSCO feature-list file does not contain "
        "a BUSCO column."
    )


robust_buscos = (

    robust_df["BUSCO"]

    .dropna()

    .astype(str)

    .str.strip()

    .tolist()

)


robust_buscos = list(
    dict.fromkeys(
        robust_buscos
    )
)


print(
    "\nRobust BUSCO count:"
)

print(
    len(robust_buscos)
)


# =============================================================================
# CHECK ROBUST BUSCO AVAILABILITY
# =============================================================================

available_buscos = [

    b

    for b in robust_buscos

    if b in feature_df.columns

]


missing_buscos = [

    b

    for b in robust_buscos

    if b not in feature_df.columns

]


print(
    "\nBUSCO FEATURE AVAILABILITY"
)

print(
    "-" * 80
)


print(
    "Robust BUSCOs:",
    len(robust_buscos)
)


print(
    "Available in genomic matrix:",
    len(available_buscos)
)


print(
    "Missing from genomic matrix:",
    len(missing_buscos)
)


if missing_buscos:

    print(
        "\nMissing BUSCOs:"
    )

    for b in missing_buscos[:50]:

        print(
            " ",
            b
        )


if len(available_buscos) == 0:

    raise RuntimeError(
        "None of the robust BUSCO features "
        "are present in the genomic matrix."
    )


# =============================================================================
# DEFINE GENOMIC FEATURE COLUMNS
# =============================================================================

feature_cols = available_buscos.copy()


print(
    "\nNumber of robust genomic features:"
)

print(
    len(feature_cols)
)


print(
    "\nFirst features:"
)

print(
    feature_cols[:20]
)


# =============================================================================
# LOAD PHYLOGENETIC CV ASSIGNMENTS
# =============================================================================

banner(
    "LOADING PHYLOGENETIC CV ASSIGNMENTS"
)


phylo_path = (
    locate_phylogenetic_assignments()
)


print(
    "\nSelected phylogenetic assignment file:"
)

print(
    phylo_path
)


phylo_df = pd.read_csv(
    phylo_path
)


print(
    "\nShape:"
)

print(
    phylo_df.shape
)


print(
    "\nColumns:"
)

print(
    phylo_df.columns.tolist()
)


required_phylo = [

    "Species",

    "Assembly_Accession",

    "Phylogenetic_Fold"

]


missing = [

    c

    for c in required_phylo

    if c not in phylo_df.columns

]


if missing:

    raise ValueError(
        f"Phylogenetic assignment file missing columns: {missing}"
    )


# =============================================================================
# CLEAN PHYLOGENETIC ASSIGNMENTS
# =============================================================================

phylo_df = phylo_df[
    required_phylo
].copy()


phylo_df[
    "Assembly_Accession"
] = (

    phylo_df[
        "Assembly_Accession"
    ]

    .astype(str)

    .str.strip()

)


phylo_df[
    "Species"
] = (

    phylo_df[
        "Species"
    ]

    .astype(str)

    .str.strip()

)


phylo_df[
    "Phylogenetic_Fold"
] = pd.to_numeric(

    phylo_df[
        "Phylogenetic_Fold"
    ],

    errors="coerce"

)


phylo_df = phylo_df.dropna(
    subset=[
        "Assembly_Accession",
        "Phylogenetic_Fold"
    ]
)


phylo_df = phylo_df.drop_duplicates(
    subset=[
        "Assembly_Accession"
    ]
)


phylo_df[
    "Phylogenetic_Fold"
] = (

    phylo_df[
        "Phylogenetic_Fold"
    ]

    .round()

    .astype(int)

)


print(
    "\nPhylogenetic folds:"
)

print(
    phylo_df[
        "Phylogenetic_Fold"
    ]
    .value_counts()
    .sort_index()
)


# =============================================================================
# CLEAN GENOMIC MATRIX IDENTIFIERS
# =============================================================================

feature_df[
    "Assembly_Accession"
] = (

    feature_df[
        "Assembly_Accession"
    ]

    .astype(str)

    .str.strip()

)


feature_df[
    "Species"
] = (

    feature_df[
        "Species"
    ]

    .astype(str)

    .str.strip()

)


# =============================================================================
# RECONCILE GENOMIC MATRIX WITH PHYLOGENETIC CV
# =============================================================================

banner(
    "RECONCILING GENOMIC FEATURES WITH PHYLOGENETIC CV"
)


# Check duplicate accessions

if feature_df[
    "Assembly_Accession"
].duplicated().any():

    duplicate_count = int(

        feature_df[
            "Assembly_Accession"
        ]
        .duplicated()
        .sum()

    )

    raise RuntimeError(
        f"Genomic matrix contains "
        f"{duplicate_count} duplicate Assembly_Accession values."
    )


merged = feature_df.merge(

    phylo_df,

    on="Assembly_Accession",

    how="inner",

    suffixes=(
        "",
        "_phylo"
    )

)


print(
    "\nMerged shape:"
)

print(
    merged.shape
)


if merged.empty:

    raise RuntimeError(
        "No rows remained after merging the genomic matrix "
        "with phylogenetic assignments."
    )


print(
    "\nRows with phylogenetic fold:"
)

print(
    merged[
        "Phylogenetic_Fold"
    ]
    .notna()
    .sum()
)


print(
    "\nRows without phylogenetic fold:"
)

print(
    merged[
        "Phylogenetic_Fold"
    ]
    .isna()
    .sum()
)


if merged[
    "Phylogenetic_Fold"
].isna().any():

    missing_taxa = merged.loc[
        merged[
            "Phylogenetic_Fold"
        ].isna(),
        [
            "Species",
            "Assembly_Accession"
        ]
    ]

    print(
        missing_taxa.to_string(
            index=False
        )
    )

    raise RuntimeError(
        "Some genomic rows do not have phylogenetic fold assignments."
    )


# =============================================================================
# HANDLE SPECIES / FOLD COLUMNS
# =============================================================================

if "Species_phylo" in merged.columns:

    merged["Species"] = (

        merged["Species"]

        .fillna(
            merged["Species_phylo"]
        )

    )


if "Phylogenetic_Fold_phylo" in merged.columns:

    merged["Phylogenetic_Fold"] = (

        merged[
            "Phylogenetic_Fold_phylo"
        ]

    )


merged[
    "Phylogenetic_Fold"
] = pd.to_numeric(

    merged[
        "Phylogenetic_Fold"
    ],

    errors="coerce"

)


merged = merged.dropna(
    subset=[
        "Phylogenetic_Fold"
    ]
)


merged[
    "Phylogenetic_Fold"
] = (

    merged[
        "Phylogenetic_Fold"
    ]

    .round()

    .astype(int)

)


print(
    "\nFinal fold counts:"
)

print(
    merged[
        "Phylogenetic_Fold"
    ]
    .value_counts()
    .sort_index()
)


# =============================================================================
# VALIDATE ROBUST FEATURES
# =============================================================================

usable_feature_cols = []


for col in feature_cols:

    if col not in merged.columns:

        continue


    values = pd.to_numeric(

        merged[col],

        errors="coerce"

    )


    if values.notna().sum() == 0:

        continue


    usable_feature_cols.append(
        col
    )


feature_cols = usable_feature_cols


print(
    "\nUsable robust genomic features:"
)

print(
    len(feature_cols)
)


if len(feature_cols) == 0:

    raise RuntimeError(
        "No usable robust BUSCO features remain."
    )


if len(feature_cols) != len(robust_buscos):

    print(
        "\nWARNING:"
    )

    print(
        f"Expected {len(robust_buscos)} robust BUSCOs, "
        f"but {len(feature_cols)} are usable."
    )


# =============================================================================
# FEATURE IMPORTANCE FUNCTION
# =============================================================================

def calculate_fold_importance(

    X_train,

    y_train,

    X_test,

    y_test,

    feature_names,

    fold,

    phenotype,

    model_name

):


    print(
        "\n" + "-" * 80
    )

    print(
        f"PHYLOGENETIC FOLD {fold}"
    )

    print(
        "-" * 80
    )


    # -------------------------------------------------------------------------
    # Numeric conversion
    # -------------------------------------------------------------------------

    X_train = X_train.apply(

        pd.to_numeric,

        errors="coerce"

    )


    X_test = X_test.apply(

        pd.to_numeric,

        errors="coerce"

    )


    # -------------------------------------------------------------------------
    # Training-only median imputation
    # -------------------------------------------------------------------------

    train_medians = X_train.median()


    X_train = X_train.fillna(
        train_medians
    )


    X_test = X_test.fillna(
        train_medians
    )


    # -------------------------------------------------------------------------
    # Target conversion
    # -------------------------------------------------------------------------

    y_train = pd.to_numeric(

        y_train,

        errors="coerce"

    )


    y_test = pd.to_numeric(

        y_test,

        errors="coerce"

    )


    valid_train = y_train.notna()

    valid_test = y_test.notna()


    X_train = X_train.loc[
        valid_train
    ]


    y_train = y_train.loc[
        valid_train
    ]


    X_test = X_test.loc[
        valid_test
    ]


    y_test = y_test.loc[
        valid_test
    ]


    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    if model_name == "SVR_RBF":

        model = Pipeline(

            [

                (
                    "scaler",

                    StandardScaler()
                ),

                (
                    "model",

                    SVR(

                        kernel="rbf",

                        C=1.0,

                        epsilon=0.1,

                        gamma="scale"

                    )
                )

            ]

        )


    elif model_name == "GradientBoosting":

        model = GradientBoostingRegressor(

            n_estimators=300,

            learning_rate=0.03,

            max_depth=2,

            random_state=RANDOM_STATE

        )


    else:

        raise ValueError(
            f"Unsupported model: {model_name}"
        )


    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    model.fit(

        X_train,

        y_train

    )


    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    pred = model.predict(
        X_test
    )


    baseline_rmse = np.sqrt(

        mean_squared_error(

            y_test,

            pred

        )

    )


    baseline_mae = mean_absolute_error(

        y_test,

        pred

    )


    baseline_r2 = r2_score(

        y_test,

        pred

    )


    print(
        f"Baseline RMSE: {baseline_rmse:.6f}"
    )

    print(
        f"Baseline MAE : {baseline_mae:.6f}"
    )

    print(
        f"Baseline R2  : {baseline_r2:.6f}"
    )


    # -------------------------------------------------------------------------
    # Permutation importance
    # -------------------------------------------------------------------------

    print(
        f"\nCalculating permutation importance "
        f"({N_PERMUTATIONS} permutations)..."
    )


    perm = permutation_importance(

        model,

        X_test,

        y_test,

        scoring="neg_root_mean_squared_error",

        n_repeats=N_PERMUTATIONS,

        random_state=(
            RANDOM_STATE + fold
        ),

        n_jobs=-1

    )


    importance_df = pd.DataFrame(

        {

            "Feature":
                feature_names,

            "Permutation_Importance":
                perm.importances_mean,

            "Permutation_SD":
                perm.importances_std

        }

    )


    importance_df[
        "Phenotype"
    ] = phenotype


    importance_df[
        "Model"
    ] = model_name


    importance_df[
        "Phylogenetic_Fold"
    ] = fold


    importance_df[
        "Baseline_RMSE"
    ] = baseline_rmse


    importance_df[
        "Baseline_MAE"
    ] = baseline_mae


    importance_df[
        "Baseline_R2"
    ] = baseline_r2


    importance_df = (

        importance_df

        .sort_values(

            "Permutation_Importance",

            ascending=False

        )

    )


    return (

        importance_df,

        model,

        baseline_rmse,

        baseline_mae,

        baseline_r2

    )


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

all_importance = []

fold_metrics = []


for config in CONFIGURATIONS:


    phenotype = config[
        "phenotype"
    ]


    feature_set = config[
        "feature_set"
    ]


    model_name = config[
        "model"
    ]


    banner(
        f"FEATURE INTERPRETATION — {phenotype}"
    )


    print(
        "\nFeature set:"
    )

    print(
        feature_set
    )


    print(
        "\nModel:"
    )

    print(
        model_name
    )


    # =========================================================================
    # Locate OOF
    # =========================================================================

    oof_path = find_oof_file(

        phenotype,

        feature_set,

        model_name

    )


    if oof_path is None:

        print(
            "\nWARNING: OOF file not found."
        )

        print(
            "Skipping configuration."
        )

        continue


    print(
        "\nOOF file:"
    )

    print(
        oof_path
    )


    oof = pd.read_csv(
        oof_path
    )


    print(
        "\nOOF shape:"
    )

    print(
        oof.shape
    )


    required_oof = [

        "Species",

        "Assembly_Accession",

        "Phenotype",

        "Feature_Set",

        "Model",

        "Phylogenetic_Fold",

        "Observed",

        "Predicted"

    ]


    missing_oof = [

        c

        for c in required_oof

        if c not in oof.columns

    ]


    if missing_oof:

        print(
            f"WARNING: Missing OOF columns: "
            f"{missing_oof}"
        )

        continue


    # =========================================================================
    # Determine feature set
    # =========================================================================

    if feature_set == "Robust_79":

        selected_features = [

            c

            for c in feature_cols

            if c in merged.columns

        ]


    else:

        # ---------------------------------------------------------------------
        # For optimized exploratory feature sets, attempt to recover the
        # feature list from optimization outputs.
        # ---------------------------------------------------------------------

        selected_features = []


        # Candidate feature-list filenames

        candidate_patterns = [

            f"*{phenotype}*{feature_set}*.csv",

            f"*{feature_set}*{phenotype}*.csv",

            f"*{feature_set}*feature*.csv",

            f"*{phenotype}*feature*.csv"

        ]


        possible_files = []


        for pattern in candidate_patterns:

            possible_files.extend(

                OPT_DIR.rglob(
                    pattern
                )

            )


        # Remove duplicate paths

        unique_files = []


        for path in possible_files:

            if path not in unique_files:

                unique_files.append(path)


        # ---------------------------------------------------------------------
        # Inspect possible feature-list files
        # ---------------------------------------------------------------------

        for path in unique_files:

            try:

                temp = pd.read_csv(
                    path
                )

            except Exception:

                continue


            candidate_columns = [

                c

                for c in temp.columns

                if c in merged.columns

            ]


            # A feature list should contain multiple actual BUSCO columns

            if len(candidate_columns) >= 5:

                selected_features = (
                    candidate_columns
                )

                print(
                    "\nRecovered feature list:"
                )

                print(
                    path
                )

                break


        # ---------------------------------------------------------------------
        # Fallback based on expected feature count
        # ---------------------------------------------------------------------

        expected_counts = {

            "Filtered_Variable":
                740,

            "Phenotype_Selected_Top50":
                50

        }


        expected_n = expected_counts.get(
            feature_set
        )


        if (

            len(selected_features) == 0

            and expected_n is not None

        ):

            print(
                "\nWARNING:"
            )

            print(
                f"Could not find an explicit feature-list file "
                f"for {feature_set}."
            )

            print(
                "The configuration will be skipped rather than "
                "silently using the wrong feature set."
            )

            continue


    # =========================================================================
    # Validate selected features
    # =========================================================================

    selected_features = [

        c

        for c in selected_features

        if c in merged.columns

    ]


    if len(selected_features) == 0:

        print(
            "\nWARNING: No genomic features available "
            f"for {feature_set}."
        )

        continue


    print(
        "\nNumber of features used:"
    )

    print(
        len(selected_features)
    )


    # =========================================================================
    # Merge OOF with genomic matrix
    # =========================================================================

    analysis = oof[

        [

            "Assembly_Accession",

            "Phylogenetic_Fold",

            "Observed",

            "Predicted"

        ]

    ].copy()


    analysis = analysis.merge(

        merged[
            [
                "Assembly_Accession"
            ]

            + selected_features

        ],

        on="Assembly_Accession",

        how="inner"

    )


    print(
        "\nAnalysis matrix shape:"
    )

    print(
        analysis.shape
    )


    if len(analysis) == 0:

        print(
            "WARNING: No rows after merging."
        )

        continue


    # =========================================================================
    # Check fold consistency
    # =========================================================================

    analysis[
        "Phylogenetic_Fold"
    ] = pd.to_numeric(

        analysis[
            "Phylogenetic_Fold"
        ],

        errors="coerce"

    )


    analysis = analysis.dropna(

        subset=[
            "Phylogenetic_Fold"
        ]

    )


    analysis[
        "Phylogenetic_Fold"
    ] = (

        analysis[
            "Phylogenetic_Fold"
        ]

        .round()

        .astype(int)

    )


    # =========================================================================
    # Fold-wise model interpretation
    # =========================================================================

    for fold in sorted(

        analysis[
            "Phylogenetic_Fold"
        ]
        .unique()

    ):


        train = analysis[

            analysis[
                "Phylogenetic_Fold"
            ]

            != fold

        ].copy()


        test = analysis[

            analysis[
                "Phylogenetic_Fold"
            ]

            == fold

        ].copy()


        X_train = train[
            selected_features
        ]


        X_test = test[
            selected_features
        ]


        y_train = train[
            "Observed"
        ]


        y_test = test[
            "Observed"
        ]


        (

            importance,

            fitted_model,

            rmse,

            mae,

            r2

        ) = calculate_fold_importance(

            X_train,

            y_train,

            X_test,

            y_test,

            selected_features,

            int(fold),

            phenotype,

            model_name

        )


        importance[
            "Feature_Set"
        ] = feature_set


        importance[
            "Primary_Analysis"
        ] = (

            phenotype == PRIMARY_PHENOTYPE

            and

            feature_set == PRIMARY_FEATURE_SET

        )


        all_importance.append(
            importance
        )


        fold_metrics.append(

            {

                "Phenotype":
                    phenotype,

                "Feature_Set":
                    feature_set,

                "Model":
                    model_name,

                "Phylogenetic_Fold":
                    int(fold),

                "N_Train":
                    len(train),

                "N_Test":
                    len(test),

                "N_Features":
                    len(selected_features),

                "RMSE":
                    rmse,

                "MAE":
                    mae,

                "R2":
                    r2

            }

        )


# =============================================================================
# COMBINE IMPORTANCE RESULTS
# =============================================================================

banner(
    "AGGREGATING FEATURE IMPORTANCE"
)


if len(all_importance) == 0:

    raise RuntimeError(
        "No feature importance results were generated."
    )


importance_all = pd.concat(

    all_importance,

    ignore_index=True

)


fold_metrics_df = pd.DataFrame(
    fold_metrics
)


# =============================================================================
# SAVE FOLD-LEVEL RESULTS
# =============================================================================

fold_path = (

    TABLE_DIR /

    "feature_importance_fold_level_420.csv"

)


fold_metrics_df.to_csv(

    fold_path,

    index=False

)


print(
    "\nFold-level results saved:"
)

print(
    fold_path
)


# =============================================================================
# GLOBAL FEATURE STABILITY
# =============================================================================

banner(
    "FEATURE STABILITY ACROSS PHYLOGENETIC FOLDS"
)


group_cols = [

    "Phenotype",

    "Feature_Set",

    "Model",

    "Feature"

]


stability = (

    importance_all

    .groupby(
        group_cols
    )

    .agg(

        Mean_Importance=(

            "Permutation_Importance",

            "mean"

        ),

        SD_Importance=(

            "Permutation_Importance",

            "std"

        ),

        Median_Importance=(

            "Permutation_Importance",

            "median"

        ),

        Min_Importance=(

            "Permutation_Importance",

            "min"

        ),

        Max_Importance=(

            "Permutation_Importance",

            "max"

        ),

        Mean_Baseline_R2=(

            "Baseline_R2",

            "mean"

        )

    )

    .reset_index()

)


# =============================================================================
# POSITIVE IMPORTANCE COUNTS
# =============================================================================

positive_counts = (

    importance_all

    .assign(

        Positive=lambda x:

        x[
            "Permutation_Importance"
        ] > 0

    )

    .groupby(
        group_cols
    )[
        "Positive"
    ]

    .sum()

    .reset_index(

        name="Positive_Importance_Folds"

    )

)


stability = stability.merge(

    positive_counts,

    on=group_cols,

    how="left"

)


stability[
    "Fold_Stability"
] = (

    stability[
        "Positive_Importance_Folds"
    ]

    / 5.0

)


# =============================================================================
# IMPORTANCE RANK
# =============================================================================

stability[
    "Importance_Rank"
] = (

    stability

    .groupby(

        [

            "Phenotype",

            "Feature_Set",

            "Model"

        ]

    )[

        "Mean_Importance"

    ]

    .rank(

        ascending=False,

        method="min"

    )

)


stability = stability.sort_values(

    [

        "Phenotype",

        "Mean_Importance"

    ],

    ascending=[

        True,

        False

    ]

)


# =============================================================================
# SAVE STABILITY TABLE
# =============================================================================

stability_path = (

    TABLE_DIR /

    "feature_importance_stability_420.csv"

)


stability.to_csv(

    stability_path,

    index=False

)


print(
    "\nFeature stability table saved:"
)

print(
    stability_path
)


# =============================================================================
# PRIMARY CARBON BREADTH RESULTS
# =============================================================================

primary = stability[

    (

        stability[
            "Phenotype"
        ]

        == PRIMARY_PHENOTYPE

    )

    &

    (

        stability[
            "Feature_Set"
        ]

        == PRIMARY_FEATURE_SET

    )

    &

    (

        stability[
            "Model"
        ]

        == PRIMARY_MODEL

    )

].copy()


if primary.empty:

    raise RuntimeError(
        "No primary Carbon_Breadth feature-importance "
        "results were generated."
    )


primary = primary.sort_values(

    "Mean_Importance",

    ascending=False

)


primary_top = primary.head(
    20
).copy()


# =============================================================================
# SAVE TOP 20
# =============================================================================

top20_path = (

    TABLE_DIR /

    "Carbon_Breadth_top20_BUSCO_features_420.csv"

)


primary_top.to_csv(

    top20_path,

    index=False

)


print(
    "\nTop 20 Carbon_Breadth BUSCO candidates saved:"
)

print(
    top20_path
)


print(
    "\nTop 20 features:"
)


print(

    primary_top[

        [

            "Feature",

            "Mean_Importance",

            "SD_Importance",

            "Fold_Stability",

            "Positive_Importance_Folds",

            "Importance_Rank"

        ]

    ]

    .to_string(
        index=False
    )

)


# =============================================================================
# PRIMARY FEATURE IMPORTANCE FIGURE
# =============================================================================

banner(
    "GENERATING PRIMARY FEATURE IMPORTANCE FIGURE"
)


plot_df = primary_top.copy()


plot_df = plot_df.sort_values(

    "Mean_Importance"

)


plt.figure(
    figsize=(10, 8)
)


plt.barh(

    plot_df[
        "Feature"
    ],

    plot_df[
        "Mean_Importance"
    ],

    xerr=plot_df[
        "SD_Importance"
    ],

    capsize=3

)


plt.xlabel(
    "Mean permutation importance\n"
    "(decrease in negative RMSE score)"
)


plt.ylabel(
    "BUSCO feature"
)


plt.title(
    "Carbon Breadth — "
    "Phylogeny-Aware Feature Importance"
)


plt.tight_layout()


fig_path = (

    FIG_DIR /

    "Carbon_Breadth_top20_permutation_importance_420.png"

)


plt.savefig(

    fig_path,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nFigure saved:"
)

print(
    fig_path
)


# =============================================================================
# FEATURE STABILITY FIGURE
# =============================================================================

stable_plot = primary_top.copy()


stable_plot = stable_plot.sort_values(

    "Fold_Stability"

)


plt.figure(
    figsize=(10, 8)
)


plt.barh(

    stable_plot[
        "Feature"
    ],

    stable_plot[
        "Fold_Stability"
    ]

)


plt.xlabel(
    "Fold stability"
)


plt.ylabel(
    "BUSCO feature"
)


plt.title(
    "Carbon Breadth — "
    "Cross-Phylogenetic-Fold Feature Stability"
)


plt.xlim(
    0,
    1.05
)


plt.tight_layout()


stability_fig = (

    FIG_DIR /

    "Carbon_Breadth_feature_stability_420.png"

)


plt.savefig(

    stability_fig,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print(
    "\nStability figure saved:"
)

print(
    stability_fig
)


# =============================================================================
# FEATURE-TARGET ASSOCIATIONS
# =============================================================================

banner(
    "CALCULATING FEATURE–PHENOTYPE ASSOCIATIONS"
)


association_records = []


primary_analysis = merged.copy()


if PRIMARY_PHENOTYPE in primary_analysis.columns:

    y = pd.to_numeric(

        primary_analysis[
            PRIMARY_PHENOTYPE
        ],

        errors="coerce"

    )


else:

    carbon_oof_path = find_oof_file(

        PRIMARY_PHENOTYPE,

        PRIMARY_FEATURE_SET,

        PRIMARY_MODEL

    )


    if carbon_oof_path is None:

        raise RuntimeError(
            "Could not locate Carbon_Breadth OOF file "
            "for phenotype associations."
        )


    carbon_oof = pd.read_csv(
        carbon_oof_path
    )


    primary_analysis = primary_analysis.merge(

        carbon_oof[

            [

                "Assembly_Accession",

                "Observed"

            ]

        ],

        on="Assembly_Accession",

        how="inner"

    )


    y = pd.to_numeric(

        primary_analysis[
            "Observed"
        ],

        errors="coerce"

    )


for feature in feature_cols:


    if feature not in primary_analysis.columns:

        continue


    x = pd.to_numeric(

        primary_analysis[
            feature
        ],

        errors="coerce"

    )


    valid = (

        x.notna()

        &

        y.notna()

    )


    if valid.sum() < 10:

        continue


    try:

        pearson_r, pearson_p = pearsonr(

            x[valid],

            y[valid]

        )

    except Exception:

        pearson_r = np.nan

        pearson_p = np.nan


    try:

        spearman_r, spearman_p = spearmanr(

            x[valid],

            y[valid]

        )

    except Exception:

        spearman_r = np.nan

        spearman_p = np.nan


    association_records.append(

        {

            "Phenotype":
                PRIMARY_PHENOTYPE,

            "Feature":
                feature,

            "N":
                int(valid.sum()),

            "Pearson_r":
                pearson_r,

            "Pearson_p":
                pearson_p,

            "Spearman_rho":
                spearman_r,

            "Spearman_p":
                spearman_p

        }

    )


association_df = pd.DataFrame(
    association_records
)


# =============================================================================
# MERGE IMPORTANCE + ASSOCIATION
# =============================================================================

primary_interpretation = primary.merge(

    association_df,

    on=[

        "Phenotype",

        "Feature"

    ],

    how="left"

)


primary_interpretation = (

    primary_interpretation

    .sort_values(

        "Mean_Importance",

        ascending=False

    )

)


interpretation_path = (

    TABLE_DIR /

    "Carbon_Breadth_BUSCO_interpretation_420.csv"

)


primary_interpretation.to_csv(

    interpretation_path,

    index=False

)


print(
    "\nCombined interpretation table saved:"
)

print(
    interpretation_path
)


# =============================================================================
# STABLE HIGH-IMPORTANCE CANDIDATES
# =============================================================================

banner(
    "IDENTIFYING STABLE HIGH-IMPORTANCE BUSCO CANDIDATES"
)


# Conservative reproducibility criterion:
#
#   1. top 20 by mean importance
#   2. positive importance in >= 4/5 folds
#
# This is NOT a biological causality claim.

stable_candidates = primary[

    (

        primary[
            "Importance_Rank"
        ]

        <= 20

    )

    &

    (

        primary[
            "Positive_Importance_Folds"
        ]

        >= 4

    )

].copy()


stable_candidates = (

    stable_candidates

    .sort_values(

        "Mean_Importance",

        ascending=False

    )

)


candidate_path = (

    TABLE_DIR /

    "Carbon_Breadth_stable_BUSCO_candidates_420.csv"

)


stable_candidates.to_csv(

    candidate_path,

    index=False

)


print(
    "\nStable candidates:"
)


print(

    stable_candidates[

        [

            "Feature",

            "Mean_Importance",

            "SD_Importance",

            "Fold_Stability",

            "Positive_Importance_Folds"

        ]

    ]

    .to_string(
        index=False
    )

)


print(
    "\nSaved:"
)

print(
    candidate_path
)


# =============================================================================
# MACHINE-READABLE SUMMARY
# =============================================================================

summary = {

    "analysis":
        "Phylogeny-aware feature interpretation",

    "n_taxa":
        int(
            merged[
                "Assembly_Accession"
            ].nunique()
        ),

    "primary_phenotype":
        PRIMARY_PHENOTYPE,

    "primary_feature_set":
        PRIMARY_FEATURE_SET,

    "primary_model":
        PRIMARY_MODEL,

    "n_features":
        len(feature_cols),

    "n_permutations":
        N_PERMUTATIONS,

    "n_phylogenetic_folds":
        int(
            merged[
                "Phylogenetic_Fold"
            ].nunique()
        ),

    "stable_candidate_count":
        int(
            len(stable_candidates)
        ),

    "interpretation_note":
        (
            "Permutation importance measures the change "
            "in predictive performance after feature "
            "permutation. It does not by itself establish "
            "biological causality."
        )

}


summary_path = (

    TABLE_DIR /

    "feature_interpretation_summary_420.json"

)


with open(

    summary_path,

    "w",

    encoding="utf-8"

) as f:

    json.dump(

        summary,

        f,

        indent=4

    )


# =============================================================================
# WRITE INTERPRETATION NOTES
# =============================================================================

interpretation_text = f"""

PHYLOGENY-AWARE FEATURE INTERPRETATION
======================================

Primary phenotype:
{PRIMARY_PHENOTYPE}

Feature set:
{PRIMARY_FEATURE_SET}

Model:
{PRIMARY_MODEL}

Number of taxa:
{summary["n_taxa"]}

Number of genomic features:
{summary["n_features"]}

Number of phylogenetic folds:
{summary["n_phylogenetic_folds"]}

Permutation repetitions:
{N_PERMUTATIONS}


Interpretation
--------------

Permutation importance identifies genomic features whose
permutation reduces predictive performance on held-out
phylogenetic folds.

A positive permutation importance indicates that disrupting
that feature tends to worsen prediction under the evaluation
procedure.

The analysis is predictive rather than causal.


Stable candidate definition
----------------------------

The most reproducible candidates were defined as features that:

1. ranked within the top 20 by mean permutation importance; and

2. had positive permutation importance in at least 4 of 5
   phylogenetic folds.


Stable candidate count:
{len(stable_candidates)}


Important limitation
--------------------

Feature importance should not be interpreted as proof that
a BUSCO directly causes the phenotype.

Functional annotation, pathway analysis, and independent
biological evidence are required before assigning mechanistic
interpretation.

The genomic signal should also be interpreted in the context
of the previously completed phylogeny-versus-genomic diagnostic
analysis.
"""


text_path = (

    TABLE_DIR /

    "feature_interpretation_notes_420.txt"

)


with open(

    text_path,

    "w",

    encoding="utf-8"

) as f:

    f.write(
        interpretation_text
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

banner(
    "FEATURE INTERPRETATION COMPLETE"
)


print(
    "\nPrimary phenotype:"
)

print(
    PRIMARY_PHENOTYPE
)


print(
    "\nPrimary feature set:"
)

print(
    PRIMARY_FEATURE_SET
)


print(
    "\nPrimary model:"
)

print(
    PRIMARY_MODEL
)


print(
    "\nActual genomic matrix used:"
)

print(
    feature_path
)


print(
    "\nRobust BUSCO feature-list used:"
)

print(
    robust_feature_path
)


print(
    "\nMain output directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nTables:"
)

print(
    TABLE_DIR
)


print(
    "\nFigures:"
)

print(
    FIG_DIR
)


print(
    "\nTop-20 BUSCO table:"
)

print(
    top20_path
)


print(
    "\nStable candidate table:"
)

print(
    candidate_path
)


print(
    "\nInterpretation table:"
)

print(
    interpretation_path
)


print(
    "\nJSON summary:"
)

print(
    summary_path
)


print(
    "\nInterpretation notes:"
)

print(
    text_path
)


print(
    "\nExisting models and OOF predictions were NOT overwritten."
)


print(
    "\nNext biological stage:"
)

print(
    "Functional annotation → pathway mapping → "
    "network analysis → candidate chassis interpretation"
)


print(
    "\nIMPORTANT:"
)

print(
    "Feature importance is predictive evidence, "
    "not proof of causality."
)

print(
    "\n" + "=" * 80
)