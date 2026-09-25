# ================================================================
# train_phylogeny_aware_ml_420.py
#
# PHYLOGENY-AWARE ML MODEL TRAINING
#
# Dataset:
#   420 taxa
#   1988 variable BUSCO features
#
# CV:
#   1. Phylogenetic CV
#   2. Random CV baseline
#
# Models:
#   Ridge Regression
#   Random Forest Regression
#   Gradient Boosting Regression
#
# Target:
#   Utilized_Median_Growth
#
# IMPORTANT:
#   Uses the EXISTING CV assignments.
#   Does NOT recreate or modify folds.
#   Does NOT modify the original ML matrix or tree.
# ================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT = Path(
    r"C:\Y1000_chassis_project"
)

BASE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
)

ML_MATRIX = (
    BASE
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)

# ------------------------------------------------
# EXISTING CV DIRECTORY
# ------------------------------------------------

CV_DIR = (
    BASE
    / "phylogeny_cv"
)

PHYLO_ASSIGNMENTS = (
    CV_DIR
    / "phylogenetic_cv_assignments_420.csv"
)

RANDOM_ASSIGNMENTS = (
    CV_DIR
    / "random_cv_assignments_420.csv"
)

# ------------------------------------------------
# MODEL OUTPUT DIRECTORY
# ------------------------------------------------

OUTPUT_DIR = (
    CV_DIR
    / "model_training"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ================================================================
# CONSTANTS
# ================================================================

EXPECTED_TAXA = 420
EXPECTED_BUSCO = 1988
EXPECTED_FOLDS = 5

TARGET = (
    "Utilized_Median_Growth"
)

IDENTIFIER_COLUMNS = [
    "Species",
    "Assembly_Accession"
]

PHENOTYPE_COLUMNS = [
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD"
]

METADATA_COLUMNS = [
    "Species",
    "Assembly_Accession",
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
    "Phenotype_Source_Species"
]


# ================================================================
# HELPER
# ================================================================

def section(title):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def calculate_metrics(
    y_true,
    y_pred
):

    return {
        "R2": r2_score(
            y_true,
            y_pred
        ),

        "RMSE": np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        ),

        "MAE": mean_absolute_error(
            y_true,
            y_pred
        )
    }


# ================================================================
# CHECK INPUTS
# ================================================================

section(
    "CHECKING INPUT FILES"
)

required_files = [
    ML_MATRIX,
    PHYLO_ASSIGNMENTS,
    RANDOM_ASSIGNMENTS
]

for filepath in required_files:

    if not filepath.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n"
            f"{filepath}"
        )

    print(
        f"✓ {filepath}"
    )


# ================================================================
# LOAD ML MATRIX
# ================================================================

section(
    "LOADING FINAL PHYLOGENY-AWARE ML MATRIX"
)

df = pd.read_csv(
    ML_MATRIX
)

print(
    f"Rows:    {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)


# ================================================================
# MATRIX QC
# ================================================================

section(
    "FINAL ML MATRIX QC"
)

if len(df) != EXPECTED_TAXA:

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} taxa "
        f"but found {len(df)}."
    )

print(
    f"✓ Exactly {EXPECTED_TAXA} taxa"
)


for col in IDENTIFIER_COLUMNS:

    if col not in df.columns:

        raise RuntimeError(
            f"Missing identifier column: {col}"
        )

    print(
        f"✓ {col}"
    )


for col in PHENOTYPE_COLUMNS:

    if col not in df.columns:

        raise RuntimeError(
            f"Missing phenotype column: {col}"
        )

    print(
        f"✓ {col}"
    )


# ================================================================
# TAXON QC
# ================================================================

section(
    "TAXON QC"
)

duplicate_species = (
    df["Species"]
    .duplicated()
    .sum()
)

duplicate_accessions = (
    df["Assembly_Accession"]
    .duplicated()
    .sum()
)

print(
    f"Taxa:                 {len(df)}"
)

print(
    f"Duplicate species:    {duplicate_species}"
)

print(
    f"Duplicate accessions: {duplicate_accessions}"
)

if duplicate_species != 0:

    raise RuntimeError(
        "Duplicate species detected."
    )

if duplicate_accessions != 0:

    raise RuntimeError(
        "Duplicate assembly accessions detected."
    )

print(
    "✓ Species are unique"
)

print(
    "✓ Assembly accessions are unique"
)


# ================================================================
# IDENTIFY BUSCO FEATURES
# ================================================================

section(
    "IDENTIFYING BUSCO FEATURES"
)

busco_features = [
    c
    for c in df.columns
    if c not in METADATA_COLUMNS
]

print(
    f"BUSCO features detected: "
    f"{len(busco_features)}"
)

if len(busco_features) != EXPECTED_BUSCO:

    raise RuntimeError(
        f"Expected {EXPECTED_BUSCO} BUSCO features "
        f"but detected {len(busco_features)}."
    )

print(
    f"✓ Correct number of BUSCO features: "
    f"{EXPECTED_BUSCO}"
)


# ================================================================
# BUSCO QC
# ================================================================

section(
    "BUSCO VALUE QC"
)

busco = df[
    busco_features
]

missing_busco = (
    busco.isna()
    .sum()
    .sum()
)

print(
    f"Missing BUSCO cells: "
    f"{missing_busco}"
)

if missing_busco != 0:

    raise RuntimeError(
        "Missing BUSCO cells detected."
    )


unique_values = set()

for col in busco_features:

    values = (
        busco[col]
        .astype(str)
        .str.strip()
        .unique()
    )

    unique_values.update(
        values
    )

print(
    "Unique BUSCO values:"
)

for value in sorted(
    unique_values
):

    print(
        f"  {value}"
    )


if not unique_values.issubset(
    {"0", "1"}
):

    raise RuntimeError(
        "BUSCO matrix contains "
        f"values other than 0/1: "
        f"{unique_values}"
    )

print(
    "✓ BUSCO matrix is binary"
)

print(
    "✓ No missing BUSCO values"
)


# ================================================================
# CONVERT BUSCO MATRIX TO NUMERIC
# ================================================================

X = (
    busco
    .astype(str)
    .apply(
        lambda col:
        pd.to_numeric(
            col.str.strip(),
            errors="raise"
        )
    )
)


# ================================================================
# TARGET QC
# ================================================================

section(
    "PHENOTYPE / TARGET QC"
)

for col in [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
]:

    missing = (
        df[col]
        .isna()
        .sum()
    )

    print(
        f"{col}: missing = {missing}"
    )

    if missing != 0:

        raise RuntimeError(
            f"Missing values in {col}"
        )

print(
    "✓ No missing target values"
)

y = pd.to_numeric(
    df[TARGET],
    errors="raise"
)


# ================================================================
# LOAD EXISTING FOLD ASSIGNMENTS
# ================================================================

def load_assignment(
    filepath,
    strategy_name
):

    print()
    print(
        f"Loading {strategy_name} assignments:"
    )

    print(
        filepath
    )

    assignment = pd.read_csv(
        filepath
    )

    print(
        f"Rows: {len(assignment)}"
    )

    print(
        f"Columns: "
        f"{list(assignment.columns)}"
    )

    # ------------------------------------------------------------
    # FIND ACCESSION COLUMN
    # ------------------------------------------------------------

    accession_candidates = [
        "Assembly_Accession",
        "assembly_accession",
        "Accession"
    ]

    accession_col = None

    for candidate in accession_candidates:

        if candidate in assignment.columns:

            accession_col = candidate
            break

    if accession_col is None:

        raise RuntimeError(
            f"Could not identify accession column "
            f"in {filepath}"
        )

    # ------------------------------------------------------------
    # FIND FOLD COLUMN
    # ------------------------------------------------------------

    fold_candidates = [
        "Fold",
        "fold",
        "CV_Fold",
        "cv_fold",
        "Phylogenetic_Fold",
        "Random_Fold"
    ]

    fold_col = None

    for candidate in fold_candidates:

        if candidate in assignment.columns:

            fold_col = candidate
            break

    if fold_col is None:

        raise RuntimeError(
            f"Could not identify fold column "
            f"in {filepath}\n"
            f"Available columns:\n"
            f"{list(assignment.columns)}"
        )

    result = assignment[
        [
            accession_col,
            fold_col
        ]
    ].copy()

    result.columns = [
        "Assembly_Accession",
        "Fold"
    ]

    result[
        "Assembly_Accession"
    ] = (
        result[
            "Assembly_Accession"
        ]
        .astype(str)
        .str.strip()
    )

    result[
        "Fold"
    ] = pd.to_numeric(
        result["Fold"],
        errors="raise"
    ).astype(int)

    # ------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------

    if len(result) != EXPECTED_TAXA:

        raise RuntimeError(
            f"{strategy_name} assignments contain "
            f"{len(result)} rows; expected "
            f"{EXPECTED_TAXA}."
        )

    if (
        result[
            "Assembly_Accession"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"Duplicate accessions found in "
            f"{strategy_name} assignments."
        )

    folds = sorted(
        result["Fold"]
        .unique()
        .tolist()
    )

    if folds != [
        1, 2, 3, 4, 5
    ]:

        raise RuntimeError(
            f"Expected folds 1-5 for "
            f"{strategy_name}, found {folds}"
        )

    print(
        f"✓ {strategy_name} assignments validated"
    )

    print(
        "Fold sizes:"
    )

    print(
        result[
            "Fold"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return result


# ================================================================
# LOAD PHYLOGENETIC FOLDS
# ================================================================

section(
    "LOADING PHYLOGENETIC FOLDS"
)

phylo_folds = load_assignment(
    PHYLO_ASSIGNMENTS,
    "Phylogenetic"
)


# ================================================================
# LOAD RANDOM FOLDS
# ================================================================

section(
    "LOADING RANDOM FOLDS"
)

random_folds = load_assignment(
    RANDOM_ASSIGNMENTS,
    "Random"
)


# ================================================================
# RECONCILE FOLDS WITH MATRIX
# ================================================================

section(
    "CV ASSIGNMENT ↔ ML MATRIX RECONCILIATION"
)

matrix_accessions = set(
    df[
        "Assembly_Accession"
    ]
    .astype(str)
    .str.strip()
)

phylo_accessions = set(
    phylo_folds[
        "Assembly_Accession"
    ]
)

random_accessions = set(
    random_folds[
        "Assembly_Accession"
    ]
)

print(
    f"Matrix accessions: "
    f"{len(matrix_accessions)}"
)

print(
    f"Phylogenetic accessions: "
    f"{len(phylo_accessions)}"
)

print(
    f"Random accessions: "
    f"{len(random_accessions)}"
)

if matrix_accessions != phylo_accessions:

    raise RuntimeError(
        "Phylogenetic fold accessions do not "
        "exactly match ML matrix."
    )

if matrix_accessions != random_accessions:

    raise RuntimeError(
        "Random fold accessions do not "
        "exactly match ML matrix."
    )

print(
    "✓ Phylogenetic folds match all 420 taxa"
)

print(
    "✓ Random folds match all 420 taxa"
)


# ================================================================
# MODEL DEFINITIONS
# ================================================================

section(
    "INITIALIZING MODELS"
)

models = {

    "Ridge": Pipeline(
        [
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
                    alpha=1.0
                )
            )
        ]
    ),

    "RandomForest": Pipeline(
        [
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
                    random_state=42,
                    n_jobs=-1,
                    max_features="sqrt"
                )
            )
        ]
    ),

    "GradientBoosting": Pipeline(
        [
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
                    max_depth=3,
                    random_state=42
                )
            )
        ]
    )
}

for name in models:

    print(
        f"✓ {name}"
    )


# ================================================================
# CV FUNCTION
# ================================================================

def run_cv(
    df,
    X,
    y,
    fold_table,
    fold_column,
    strategy_name
):

    section(
        f"RUNNING {strategy_name.upper()} CV"
    )

    results = []

    prediction_list = []

    for model_name, model in models.items():

        print()
        print(
            "-" * 70
        )

        print(
            f"MODEL: {model_name}"
        )

        print(
            "-" * 70
        )

        for fold in range(
            1,
            EXPECTED_FOLDS + 1
        ):

            validation_accessions = set(
                fold_table.loc[
                    fold_table[
                        fold_column
                    ] == fold,
                    "Assembly_Accession"
                ]
            )

            valid_mask = (
                df[
                    "Assembly_Accession"
                ]
                .astype(str)
                .str.strip()
                .isin(
                    validation_accessions
                )
            )

            train_mask = ~valid_mask

            X_train = X.loc[
                train_mask
            ]

            X_valid = X.loc[
                valid_mask
            ]

            y_train = y.loc[
                train_mask
            ]

            y_valid = y.loc[
                valid_mask
            ]

            print(
                f"Fold {fold}: "
                f"Train={len(X_train)}, "
                f"Validation={len(X_valid)}"
            )

            if len(X_train) != 336:

                raise RuntimeError(
                    f"Fold {fold}: expected "
                    f"336 training taxa, got "
                    f"{len(X_train)}"
                )

            if len(X_valid) != 84:

                raise RuntimeError(
                    f"Fold {fold}: expected "
                    f"84 validation taxa, got "
                    f"{len(X_valid)}"
                )

            # ----------------------------------------------------
            # FIT
            # ----------------------------------------------------

            model.fit(
                X_train,
                y_train
            )

            # ----------------------------------------------------
            # PREDICT
            # ----------------------------------------------------

            y_pred = model.predict(
                X_valid
            )

            metrics = calculate_metrics(
                y_valid,
                y_pred
            )

            results.append(
                {
                    "CV_Strategy":
                        strategy_name,

                    "Model":
                        model_name,

                    "Fold":
                        fold,

                    "Train_N":
                        len(X_train),

                    "Validation_N":
                        len(X_valid),

                    "R2":
                        metrics["R2"],

                    "RMSE":
                        metrics["RMSE"],

                    "MAE":
                        metrics["MAE"]
                }
            )

            # ----------------------------------------------------
            # PREDICTION TABLE
            # ----------------------------------------------------

            valid_rows = df.loc[
                valid_mask,
                [
                    "Species",
                    "Assembly_Accession"
                ]
            ].copy()

            valid_rows[
                "CV_Strategy"
            ] = strategy_name

            valid_rows[
                "Model"
            ] = model_name

            valid_rows[
                "Fold"
            ] = fold

            valid_rows[
                "Observed_Growth"
            ] = y_valid.values

            valid_rows[
                "Predicted_Growth"
            ] = y_pred

            valid_rows[
                "Residual"
            ] = (
                y_valid.values
                - y_pred
            )

            prediction_list.append(
                valid_rows
            )

            print(
                f"  R²   = "
                f"{metrics['R2']:.4f}"
            )

            print(
                f"  RMSE = "
                f"{metrics['RMSE']:.6f}"
            )

            print(
                f"  MAE  = "
                f"{metrics['MAE']:.6f}"
            )

    return (
        pd.DataFrame(results),
        pd.concat(
            prediction_list,
            ignore_index=True
        )
    )


# ================================================================
# CREATE FOLD TABLES
# ================================================================

phylo_fold_table = (
    phylo_folds.copy()
)

random_fold_table = (
    random_folds.copy()
)


# ================================================================
# RUN PHYLOGENETIC CV
# ================================================================

phylo_results, phylo_predictions = run_cv(
    df=df,
    X=X,
    y=y,
    fold_table=phylo_fold_table,
    fold_column="Fold",
    strategy_name="Phylogenetic"
)


# ================================================================
# RUN RANDOM CV
# ================================================================

random_results, random_predictions = run_cv(
    df=df,
    X=X,
    y=y,
    fold_table=random_fold_table,
    fold_column="Fold",
    strategy_name="Random"
)


# ================================================================
# COMBINE RESULTS
# ================================================================

section(
    "COMBINING MODEL RESULTS"
)

all_results = pd.concat(
    [
        phylo_results,
        random_results
    ],
    ignore_index=True
)

all_predictions = pd.concat(
    [
        phylo_predictions,
        random_predictions
    ],
    ignore_index=True
)


# ================================================================
# SAVE FOLD RESULTS
# ================================================================

fold_results_file = (
    OUTPUT_DIR
    / "phylogeny_aware_ml_fold_results_420.csv"
)

all_results.to_csv(
    fold_results_file,
    index=False
)

print(
    f"✓ Fold results written:\n"
    f"{fold_results_file}"
)


# ================================================================
# SAVE PREDICTIONS
# ================================================================

prediction_file = (
    OUTPUT_DIR
    / "phylogeny_aware_ml_predictions_420.csv"
)

all_predictions.to_csv(
    prediction_file,
    index=False
)

print(
    f"✓ Predictions written:\n"
    f"{prediction_file}"
)


# ================================================================
# MODEL SUMMARY
# ================================================================

section(
    "MODEL PERFORMANCE SUMMARY"
)

summary = (
    all_results
    .groupby(
        [
            "CV_Strategy",
            "Model"
        ]
    )
    .agg(
        R2_Mean=("R2", "mean"),
        R2_SD=("R2", "std"),

        RMSE_Mean=("RMSE", "mean"),
        RMSE_SD=("RMSE", "std"),

        MAE_Mean=("MAE", "mean"),
        MAE_SD=("MAE", "std")
    )
    .reset_index()
)

print(
    summary.to_string(
        index=False
    )
)


summary_file = (
    OUTPUT_DIR
    / "phylogeny_aware_ml_model_summary_420.csv"
)

summary.to_csv(
    summary_file,
    index=False
)

print(
    f"\n✓ Model summary written:\n"
    f"{summary_file}"
)


# ================================================================
# PHYLOGENETIC VS RANDOM COMPARISON
# ================================================================

section(
    "PHYLOGENETIC VS RANDOM CV COMPARISON"
)

comparison = []

for model_name in models:

    phy = summary[
        (
            summary[
                "CV_Strategy"
            ]
            == "Phylogenetic"
        )
        &
        (
            summary[
                "Model"
            ]
            == model_name
        )
    ]

    rnd = summary[
        (
            summary[
                "CV_Strategy"
            ]
            == "Random"
        )
        &
        (
            summary[
                "Model"
            ]
            == model_name
        )
    ]

    if len(phy) != 1:
        continue

    if len(rnd) != 1:
        continue

    phy_r2 = (
        phy[
            "R2_Mean"
        ]
        .iloc[0]
    )

    rnd_r2 = (
        rnd[
            "R2_Mean"
        ]
        .iloc[0]
    )

    phy_rmse = (
        phy[
            "RMSE_Mean"
        ]
        .iloc[0]
    )

    rnd_rmse = (
        rnd[
            "RMSE_Mean"
        ]
        .iloc[0]
    )

    phy_mae = (
        phy[
            "MAE_Mean"
        ]
        .iloc[0]
    )

    rnd_mae = (
        rnd[
            "MAE_Mean"
        ]
        .iloc[0]
    )

    comparison.append(
        {
            "Model":
                model_name,

            "Phylogenetic_R2":
                phy_r2,

            "Random_R2":
                rnd_r2,

            "R2_Difference":
                phy_r2 - rnd_r2,

            "Phylogenetic_RMSE":
                phy_rmse,

            "Random_RMSE":
                rnd_rmse,

            "RMSE_Difference":
                phy_rmse - rnd_rmse,

            "Phylogenetic_MAE":
                phy_mae,

            "Random_MAE":
                rnd_mae,

            "MAE_Difference":
                phy_mae - rnd_mae
        }
    )


comparison = pd.DataFrame(
    comparison
)

print(
    comparison.to_string(
        index=False
    )
)


comparison_file = (
    OUTPUT_DIR
    / "phylogenetic_vs_random_model_comparison_420.csv"
)

comparison.to_csv(
    comparison_file,
    index=False
)

print(
    f"\n✓ Comparison written:\n"
    f"{comparison_file}"
)


# ================================================================
# FOLD-LEVEL RESULTS
# ================================================================

section(
    "FOLD-LEVEL PERFORMANCE"
)

for strategy in [
    "Phylogenetic",
    "Random"
]:

    print()
    print(
        strategy
    )

    temp = all_results[
        all_results[
            "CV_Strategy"
        ]
        == strategy
    ]

    print(
        temp[
            [
                "Model",
                "Fold",
                "R2",
                "RMSE",
                "MAE"
            ]
        ]
        .to_string(
            index=False
        )
    )


# ================================================================
# FINAL VALIDATION
# ================================================================

section(
    "FINAL MODELING VALIDATION"
)

expected_result_rows = (
    2
    * 3
    * 5
)

expected_prediction_rows = (
    2
    * 3
    * 420
)

if len(all_results) != expected_result_rows:

    raise RuntimeError(
        f"Expected {expected_result_rows} "
        f"result rows but found "
        f"{len(all_results)}."
    )

if len(all_predictions) != expected_prediction_rows:

    raise RuntimeError(
        f"Expected {expected_prediction_rows} "
        f"prediction rows but found "
        f"{len(all_predictions)}."
    )

print(
    "✓ All 30 model/fold evaluations completed"
)

print(
    "✓ All 2,520 prediction records generated"
)

print(
    "✓ All 420 taxa evaluated"
)

print(
    "✓ No input files modified"
)


# ================================================================
# FINAL OUTPUT
# ================================================================

section(
    "PHYLOGENY-AWARE ML MODEL TRAINING COMPLETE"
)

print(
    f"Final taxa:             {EXPECTED_TAXA}"
)

print(
    f"Variable BUSCOs:        {EXPECTED_BUSCO}"
)

print(
    "Models:                 3"
)

print(
    "CV strategies:          2"
)

print(
    "Folds per strategy:     5"
)

print()
print(
    "OUTPUT DIRECTORY:"
)

print(
    OUTPUT_DIR
)

print()
print(
    "FOLD RESULTS:"
)

print(
    fold_results_file
)

print()
print(
    "PREDICTIONS:"
)

print(
    prediction_file
)

print()
print(
    "MODEL SUMMARY:"
)

print(
    summary_file
)

print()
print(
    "PHYLOGENETIC VS RANDOM:"
)

print(
    comparison_file
)

print()
print(
    "✓ Existing phylogenetic folds were used"
)

print(
    "✓ Existing random folds were used"
)

print(
    "✓ 420 taxa retained"
)

print(
    "✓ 1988 BUSCO predictors retained"
)

print(
    "✓ Original ML matrix was NOT modified"
)

print(
    "✓ Original ASTRAL tree was NOT modified"
)

print(
    "=" * 80
)