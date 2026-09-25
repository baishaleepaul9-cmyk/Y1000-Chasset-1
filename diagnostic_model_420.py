# =============================================================================
# DIAGNOSTIC MODEL — PHYLOGENY-AWARE GENOME → PHENOTYPE ANALYSIS
# Y1000+ / 420 TAXA
#
# FIXED VERSION
#
# Purpose:
#   Determine whether genomic BUSCO features provide predictive information
#   beyond phylogenetic structure.
#
# IMPORTANT:
#   Feature selection is reconstructed INSIDE each outer phylogenetic fold.
#   This prevents feature-selection leakage.
#
# Comparisons:
#   1. Global mean baseline
#   2. Phylogenetic baseline
#   3. Existing validated genomic OOF model
#   4. Diagnostic genomic model
#   5. Permutation/null diagnostic
#
# Existing models and OOF files are NOT overwritten.
# =============================================================================

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path(
    r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset"
)

ML_BASE = (
    BASE
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

OPT_BASE = (
    ML_BASE
    / "performance_optimization_420"
)

OUTPUT = (
    ML_BASE
    / "diagnostic_model_420"
)

TABLE_DIR = OUTPUT / "tables"
FIG_DIR = OUTPUT / "figures"
FOLD_DIR = OUTPUT / "fold_results"

for directory in [
    OUTPUT,
    TABLE_DIR,
    FIG_DIR,
    FOLD_DIR
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# ACTUAL GENOMIC MATRIX
# =============================================================================

GENOMIC_MATRIX = (
    BASE
    / "phylogeny_aware_ml"
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)

# Fallback because your uploaded matrix is actually stored directly under
# stage5_phylogeny_ml_dataset in the current project structure.
if not GENOMIC_MATRIX.exists():

    GENOMIC_MATRIX = (
        BASE
        / "y1000_420_phylogeny_aware_ml_matrix.csv"
    )


# =============================================================================
# BUSCO FEATURE LISTS
# =============================================================================

VARIABLE_BUSCO_FILE = (
    BASE
    / "phylogeny_aware_ml"
    / "y1000_420_variable_busco_feature_list.csv"
)

if not VARIABLE_BUSCO_FILE.exists():

    VARIABLE_BUSCO_FILE = (
        BASE
        / "y1000_420_variable_busco_feature_list.csv"
    )


ROBUST_BUSCO_CANDIDATES = [
    ML_BASE
    / "final_models_420"
    / "final_model_busco_features_420.csv",

    ML_BASE
    / "final_models_420"
    / "final_model_busco_features.csv"
]


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

CONFIGS = {

    "Carbon_Breadth": {
        "feature_set": "Robust_79",
        "model": "SVR_RBF",
        "n_features": 79
    },

    "Nitrogen_Breadth": {
        "feature_set": "Filtered_Variable",
        "model": "SVR_RBF",
        "n_features": 740
    },

    "Utilized_Median_Growth": {
        "feature_set": "Phenotype_Selected_Top50",
        "model": "GradientBoosting",
        "n_features": 50
    }
}


# =============================================================================
# FEATURE-SELECTION PARAMETERS
# =============================================================================

MIN_PREVALENCE = 0.05
MAX_PREVALENCE = 0.95

TOP_K = 50

RANDOM_STATE = 42

N_PERMUTATIONS = 100


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def rmse(y_true, y_pred):

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred
            )
        )
    )


def evaluate(y_true, y_pred):

    return {

        "N":
            int(len(y_true)),

        "RMSE":
            rmse(
                y_true,
                y_pred
            ),

        "MAE":
            float(
                mean_absolute_error(
                    y_true,
                    y_pred
                )
            ),

        "R2":
            float(
                r2_score(
                    y_true,
                    y_pred
                )
            )
    }


def safe_corr(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    if (
        np.std(y_true) == 0
        or
        np.std(y_pred) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            y_true,
            y_pred
        )[0, 1]
    )


def make_diagnostic_model():

    return RandomForestRegressor(

        n_estimators=500,

        max_features="sqrt",

        min_samples_leaf=3,

        random_state=RANDOM_STATE,

        n_jobs=-1
    )


# =============================================================================
# LOAD ACTUAL GENOMIC MATRIX
# =============================================================================

print("=" * 80)
print("DIAGNOSTIC PHYLOGENY-AWARE MODEL — FIXED")
print("=" * 80)

print("\nLoading actual genomic matrix:")

print(
    GENOMIC_MATRIX
)

if not GENOMIC_MATRIX.exists():

    raise FileNotFoundError(
        "\nCould not locate the actual Y1000+ genomic matrix.\n"
        "Expected:\n"
        f"{GENOMIC_MATRIX}"
    )


ml = pd.read_csv(
    GENOMIC_MATRIX
)

print("\nGenomic matrix shape:")

print(
    ml.shape
)

print("\nFirst columns:")

print(
    ml.columns[:15].tolist()
)


# =============================================================================
# BASIC MATRIX CHECKS
# =============================================================================

required_ids = [
    "Species",
    "Assembly_Accession"
]

for col in required_ids:

    if col not in ml.columns:

        raise ValueError(
            f"Required column missing from genomic matrix: {col}"
        )


ml["Species"] = (
    ml["Species"]
    .astype(str)
    .str.strip()
)

ml["Assembly_Accession"] = (
    ml["Assembly_Accession"]
    .astype(str)
    .str.strip()
)


# =============================================================================
# LOAD VARIABLE BUSCO LIST
# =============================================================================

print("\n" + "=" * 80)
print("LOADING VARIABLE BUSCO LIST")
print("=" * 80)

if not VARIABLE_BUSCO_FILE.exists():

    raise FileNotFoundError(
        f"Variable BUSCO list not found:\n"
        f"{VARIABLE_BUSCO_FILE}"
    )


variable_df = pd.read_csv(
    VARIABLE_BUSCO_FILE
)

print("\nVariable BUSCO file:")

print(
    VARIABLE_BUSCO_FILE
)

print("\nShape:")

print(
    variable_df.shape
)


if "BUSCO_ID" not in variable_df.columns:

    raise ValueError(
        "Variable BUSCO file must contain BUSCO_ID."
    )


variable_buscos = (
    variable_df["BUSCO_ID"]
    .astype(str)
    .str.strip()
    .tolist()
)


# Keep only BUSCOs actually present in genomic matrix.

busco_columns = [
    c
    for c in ml.columns
    if c not in {
        "Species",
        "Assembly_Accession",
        "N_Strains",
        "Carbon_Breadth",
        "Nitrogen_Breadth",
        "Utilized_Median_Growth",
        "Carbon_Breadth_SD",
        "Nitrogen_Breadth_SD",
        "Phenotype_Source_Species",
        "Phylogenetic_Fold"
    }
]

busco_columns = [
    c
    for c in busco_columns
    if c in ml.columns
]


variable_buscos = [
    b
    for b in variable_buscos
    if b in busco_columns
]


print(
    "\nVariable BUSCOs available in matrix:"
)

print(
    len(variable_buscos)
)


# =============================================================================
# LOAD ROBUST 79 FEATURE LIST
# =============================================================================

print("\n" + "=" * 80)
print("LOADING ROBUST BUSCO LIST")
print("=" * 80)

robust_file = None

for candidate in ROBUST_BUSCO_CANDIDATES:

    if candidate.exists():

        robust_file = candidate

        break


if robust_file is None:

    raise FileNotFoundError(
        "\nCould not locate Robust_79 BUSCO feature list.\n"
        "Checked:\n"
        +
        "\n".join(
            str(x)
            for x in ROBUST_BUSCO_CANDIDATES
        )
    )


robust_df = pd.read_csv(
    robust_file
)

print("\nRobust feature file:")

print(
    robust_file
)

print("\nShape:")

print(
    robust_df.shape
)


# The robust file may use BUSCO or BUSCO_ID.

if "BUSCO" in robust_df.columns:

    robust_col = "BUSCO"

elif "BUSCO_ID" in robust_df.columns:

    robust_col = "BUSCO_ID"

else:

    # The known final feature-list file can be a one-column file.
    if robust_df.shape[1] == 1:

        robust_col = robust_df.columns[0]

    else:

        raise ValueError(
            "Could not identify BUSCO column in Robust_79 file."
        )


robust_buscos = (
    robust_df[robust_col]
    .astype(str)
    .str.strip()
    .tolist()
)


robust_buscos = [
    b
    for b in robust_buscos
    if b in busco_columns
]


print(
    "\nRobust BUSCOs available:"
)

print(
    len(robust_buscos)
)


if len(robust_buscos) == 0:

    raise ValueError(
        "No Robust BUSCOs were found in the genomic matrix."
    )


# =============================================================================
# LOAD SELECTED OOF FILE
# =============================================================================

def locate_oof(
    phenotype,
    feature_set,
    model
):

    oof_dir = (
        OPT_BASE
        / "oof_predictions"
    )

    expected = (
        f"{phenotype}_"
        f"{feature_set}_"
        f"{model}_"
        f"OOF_420.csv"
    )

    exact = (
        oof_dir
        / expected
    )

    if exact.exists():

        return exact


    matches = list(
        oof_dir.glob(
            f"{phenotype}_"
            f"{feature_set}_"
            f"{model}_"
            f"OOF*.csv"
        )
    )


    if matches:

        return matches[0]


    raise FileNotFoundError(
        f"\nCould not find OOF file for:\n"
        f"Phenotype = {phenotype}\n"
        f"Feature set = {feature_set}\n"
        f"Model = {model}\n"
        f"Directory = {oof_dir}"
    )


# =============================================================================
# RECONSTRUCT FEATURE SET INSIDE TRAINING FOLD
# =============================================================================

def construct_features(
    train_df,
    phenotype,
    feature_set
):

    # ---------------------------------------------------------
    # Robust 79
    # ---------------------------------------------------------

    if feature_set == "Robust_79":

        features = [
            f
            for f in robust_buscos
            if f in train_df.columns
        ]

        return features


    # ---------------------------------------------------------
    # All variable BUSCOs
    # ---------------------------------------------------------

    variable = [
        f
        for f in variable_buscos
        if f in train_df.columns
    ]


    # ---------------------------------------------------------
    # Filtered variable BUSCOs
    # ---------------------------------------------------------

    if feature_set == "Filtered_Variable":

        selected = []

        for feature in variable:

            values = pd.to_numeric(
                train_df[feature],
                errors="coerce"
            ).fillna(0).values

            prevalence = np.mean(
                values > 0
            )

            if (
                MIN_PREVALENCE
                <= prevalence
                <= MAX_PREVALENCE
            ):

                selected.append(
                    feature
                )

        return selected


    # ---------------------------------------------------------
    # Phenotype-associated Top50
    # ---------------------------------------------------------

    if feature_set == "Phenotype_Selected_Top50":

        filtered = construct_features(
            train_df,
            phenotype,
            "Filtered_Variable"
        )

        if len(filtered) == 0:

            return []


        X = (
            train_df[filtered]
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

        y = pd.to_numeric(
            train_df[phenotype],
            errors="coerce"
        )


        valid = y.notna()

        X = X.loc[valid]

        y = y.loc[valid]


        if len(X) == 0:

            return []


        scores, pvalues = f_regression(
            X,
            y
        )


        score_df = pd.DataFrame({

            "BUSCO":
                filtered,

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


        return (
            score_df
            .head(TOP_K)
            ["BUSCO"]
            .tolist()
        )


    raise ValueError(
        f"Unknown feature set: {feature_set}"
    )


# =============================================================================
# START ANALYSIS
# =============================================================================

all_summary = []
all_fold_results = []


for phenotype, config in CONFIGS.items():

    feature_set = config[
        "feature_set"
    ]

    model_name = config[
        "model"
    ]


    print("\n\n")

    print("=" * 80)

    print(
        f"DIAGNOSTIC ANALYSIS — {phenotype}"
    )

    print("=" * 80)


    # =================================================================
    # LOAD OOF
    # =================================================================

    oof_file = locate_oof(
        phenotype,
        feature_set,
        model_name
    )


    print("\nOOF file:")

    print(
        oof_file
    )


    oof = pd.read_csv(
        oof_file
    )


    print("\nOOF shape:")

    print(
        oof.shape
    )


    required_oof = [

        "Species",

        "Assembly_Accession",

        "Phylogenetic_Fold",

        "Observed",

        "Predicted"
    ]


    missing = [
        c
        for c in required_oof
        if c not in oof.columns
    ]


    if missing:

        raise ValueError(
            f"OOF file missing: {missing}"
        )


    # =================================================================
    # CLEAN
    # =================================================================

    oof["Species"] = (
        oof["Species"]
        .astype(str)
        .str.strip()
    )


    oof["Assembly_Accession"] = (
        oof["Assembly_Accession"]
        .astype(str)
        .str.strip()
    )


    oof["Phylogenetic_Fold"] = pd.to_numeric(
        oof["Phylogenetic_Fold"],
        errors="coerce"
    )


    oof["Observed"] = pd.to_numeric(
        oof["Observed"],
        errors="coerce"
    )


    oof["Predicted"] = pd.to_numeric(
        oof["Predicted"],
        errors="coerce"
    )


    oof = oof.dropna(
        subset=[
            "Assembly_Accession",
            "Phylogenetic_Fold",
            "Observed",
            "Predicted"
        ]
    ).copy()


    oof["Phylogenetic_Fold"] = (
        oof["Phylogenetic_Fold"]
        .astype(int)
    )


    # =================================================================
    # CHECK DUPLICATES
    # =================================================================

    duplicate_ids = (
        oof["Assembly_Accession"]
        .duplicated()
        .sum()
    )


    if duplicate_ids > 0:

        raise ValueError(
            f"{duplicate_ids} duplicate assembly accessions "
            f"found in OOF file."
        )


    # =================================================================
    # MATCH OOF TO GENOMIC MATRIX
    # =================================================================

    merged_base = oof.merge(

        ml,

        on=[
            "Species",
            "Assembly_Accession"
        ],

        how="inner",

        suffixes=(
            "_OOF",
            "_ML"
        )
    )


    if len(merged_base) != len(oof):

        missing_n = (
            len(oof)
            -
            len(merged_base)
        )

        raise ValueError(
            f"{missing_n} OOF taxa could not be matched "
            f"to the genomic matrix."
        )


    print("\nRows after genomic matrix matching:")

    print(
        len(merged_base)
    )


    # =================================================================
    # USE OOF VALUES AS TARGET
    # =================================================================

    y = (
        merged_base["Observed"]
        .astype(float)
        .values
    )


    genomic_oof_pred = (
        merged_base["Predicted"]
        .astype(float)
        .values
    )


    groups = (
        merged_base["Phylogenetic_Fold"]
        .astype(int)
        .values
    )


    # =================================================================
    # 1. EXISTING MODEL
    # =================================================================

    existing_metrics = evaluate(
        y,
        genomic_oof_pred
    )


    existing_metrics["Pearson"] = safe_corr(
        y,
        genomic_oof_pred
    )


    # =================================================================
    # 2. GLOBAL MEAN BASELINE
    # =================================================================

    global_pred = np.repeat(
        np.mean(y),
        len(y)
    )


    global_metrics = evaluate(
        y,
        global_pred
    )


    global_metrics["Pearson"] = safe_corr(
        y,
        global_pred
    )


    # =================================================================
    # 3. PHYLOGENETIC BASELINE
    # =================================================================

    phylo_pred = np.zeros(
        len(y),
        dtype=float
    )


    for fold in sorted(
        np.unique(groups)
    ):

        train_idx = np.where(
            groups != fold
        )[0]

        test_idx = np.where(
            groups == fold
        )[0]


        prediction = np.mean(
            y[train_idx]
        )


        phylo_pred[
            test_idx
        ] = prediction


    phylo_metrics = evaluate(
        y,
        phylo_pred
    )


    phylo_metrics["Pearson"] = safe_corr(
        y,
        phylo_pred
    )


    # =================================================================
    # 4. DIAGNOSTIC GENOMIC MODEL
    # =================================================================
    #
    # Feature selection is reconstructed inside each training fold.
    #
    # =================================================================

    diagnostic_pred = np.zeros(
        len(y),
        dtype=float
    )


    fold_feature_records = []


    print("\n")

    print(
        "Running fold-specific diagnostic genomic model..."
    )


    for fold in sorted(
        np.unique(groups)
    ):

        print(
            f"\nDiagnostic fold {fold}"
        )


        train_idx = np.where(
            groups != fold
        )[0]

        test_idx = np.where(
            groups == fold
        )[0]


        train_data = merged_base.iloc[
            train_idx
        ].copy()


        test_data = merged_base.iloc[
            test_idx
        ].copy()


        # -------------------------------------------------------------
        # Reconstruct features using training data ONLY.
        # -------------------------------------------------------------

        features = construct_features(

            train_data,

            phenotype,

            feature_set
        )


        if len(features) == 0:

            raise RuntimeError(
                f"No features available for "
                f"{phenotype}, fold {fold}."
            )


        print(
            f"Features selected: {len(features)}"
        )


        for rank, feature in enumerate(
            features,
            start=1
        ):

            fold_feature_records.append({

                "Phenotype":
                    phenotype,

                "Phylogenetic_Fold":
                    int(fold),

                "Feature_Set":
                    feature_set,

                "BUSCO":
                    feature,

                "Rank":
                    int(rank)
            })


        X_train = (
            train_data[features]
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


        X_test = (
            test_data[features]
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


        y_train = (
            train_data["Observed"]
            .astype(float)
            .values
        )


        y_test = (
            test_data["Observed"]
            .astype(float)
            .values
        )


        model = make_diagnostic_model()


        model.fit(
            X_train,
            y_train
        )


        diagnostic_pred[
            test_idx
        ] = model.predict(
            X_test
        )


        fold_metric = evaluate(
            y_test,
            diagnostic_pred[test_idx]
        )


        fold_metric.update({

            "Phenotype":
                phenotype,

            "Feature_Set":
                feature_set,

            "Fold":
                int(fold),

            "N_Features":
                int(len(features))
        })


        all_fold_results.append(
            fold_metric
        )


    # =================================================================
    # DIAGNOSTIC METRICS
    # =================================================================

    diagnostic_metrics = evaluate(
        y,
        diagnostic_pred
    )


    diagnostic_metrics["Pearson"] = safe_corr(
        y,
        diagnostic_pred
    )


    # =================================================================
    # DELTAS
    # =================================================================

    delta_existing_vs_global = (
        existing_metrics["R2"]
        -
        global_metrics["R2"]
    )


    delta_existing_vs_phylo = (
        existing_metrics["R2"]
        -
        phylo_metrics["R2"]
    )


    delta_diagnostic_vs_phylo = (
        diagnostic_metrics["R2"]
        -
        phylo_metrics["R2"]
    )


    delta_diagnostic_vs_existing = (
        diagnostic_metrics["R2"]
        -
        existing_metrics["R2"]
    )


    # =================================================================
    # PERMUTATION DIAGNOSTIC
    # =================================================================

    print(
        "\nRunning permutation diagnostic..."
    )


    rng = np.random.default_rng(
        RANDOM_STATE
    )


    null_r2 = []


    for i in range(
        N_PERMUTATIONS
    ):

        shuffled_y = rng.permutation(
            y
        )


        score = r2_score(
            shuffled_y,
            genomic_oof_pred
        )


        null_r2.append(
            float(score)
        )


    null_r2 = np.asarray(
        null_r2
    )


    observed_r2 = (
        existing_metrics["R2"]
    )


    permutation_p = (
        (
            np.sum(
                null_r2 >= observed_r2
            )
            + 1
        )
        /
        (
            len(null_r2)
            + 1
        )
    )


    # =================================================================
    # PRINT RESULTS
    # =================================================================

    print("\n")

    print("=" * 80)

    print(
        f"DIAGNOSTIC RESULTS — {phenotype}"
    )

    print("=" * 80)


    print("\nGlobal mean baseline:")

    print(
        global_metrics
    )


    print("\nPhylogenetic baseline:")

    print(
        phylo_metrics
    )


    print("\nExisting genomic model:")

    print(
        existing_metrics
    )


    print("\nDiagnostic genomic model:")

    print(
        diagnostic_metrics
    )


    print("\n")

    print(
        f"Existing genomic ΔR² vs phylogeny: "
        f"{delta_existing_vs_phylo:.6f}"
    )


    print(
        f"Diagnostic genomic ΔR² vs phylogeny: "
        f"{delta_diagnostic_vs_phylo:.6f}"
    )


    print(
        f"Diagnostic ΔR² vs existing model: "
        f"{delta_diagnostic_vs_existing:.6f}"
    )


    print(
        f"Permutation p-value: "
        f"{permutation_p:.6f}"
    )


    # =================================================================
    # SAVE FEATURE RECORDS
    # =================================================================

    feature_record_df = pd.DataFrame(
        fold_feature_records
    )


    feature_record_path = (
        TABLE_DIR
        /
        f"{phenotype}_diagnostic_fold_features_420.csv"
    )


    feature_record_df.to_csv(
        feature_record_path,
        index=False
    )


    # =================================================================
    # SAVE PREDICTIONS
    # =================================================================

    prediction_df = merged_base[
        [
            "Species",
            "Assembly_Accession",
            "Phylogenetic_Fold"
        ]
    ].copy()


    prediction_df[
        "Observed"
    ] = y


    prediction_df[
        "Existing_Genomic_Predicted"
    ] = genomic_oof_pred


    prediction_df[
        "Phylogenetic_Baseline_Predicted"
    ] = phylo_pred


    prediction_df[
        "Diagnostic_Genomic_Predicted"
    ] = diagnostic_pred


    prediction_path = (
        FOLD_DIR
        /
        f"{phenotype}_diagnostic_predictions_420.csv"
    )


    prediction_df.to_csv(
        prediction_path,
        index=False
    )


    # =================================================================
    # OBSERVED VS PREDICTED
    # =================================================================

    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        y,
        diagnostic_pred,
        alpha=0.7
    )


    min_value = min(
        np.min(y),
        np.min(diagnostic_pred)
    )


    max_value = max(
        np.max(y),
        np.max(diagnostic_pred)
    )


    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )


    plt.xlabel(
        "Observed"
    )


    plt.ylabel(
        "Diagnostic genomic prediction"
    )


    plt.title(
        f"{phenotype} — Diagnostic Genomic Model"
    )


    plt.tight_layout()


    fig_path = (
        FIG_DIR
        /
        f"{phenotype}_diagnostic_observed_vs_predicted_420.png"
    )


    plt.savefig(
        fig_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =================================================================
    # RESIDUAL DIAGNOSTIC
    # =================================================================

    residuals = (
        y
        -
        diagnostic_pred
    )


    plt.figure(
        figsize=(7, 6)
    )


    plt.scatter(
        diagnostic_pred,
        residuals,
        alpha=0.7
    )


    plt.axhline(
        0,
        linestyle="--"
    )


    plt.xlabel(
        "Predicted"
    )


    plt.ylabel(
        "Residual"
    )


    plt.title(
        f"{phenotype} — Residual Diagnostics"
    )


    plt.tight_layout()


    residual_path = (
        FIG_DIR
        /
        f"{phenotype}_diagnostic_residuals_420.png"
    )


    plt.savefig(
        residual_path,
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    # =================================================================
    # SUMMARY ROW
    # =================================================================

    summary_row = {

        "Phenotype":
            phenotype,

        "Feature_Set":
            feature_set,

        "Existing_Model":
            model_name,

        "N_Taxa":
            len(oof),

        "Global_RMSE":
            global_metrics["RMSE"],

        "Global_MAE":
            global_metrics["MAE"],

        "Global_R2":
            global_metrics["R2"],

        "Phylogeny_RMSE":
            phylo_metrics["RMSE"],

        "Phylogeny_MAE":
            phylo_metrics["MAE"],

        "Phylogeny_R2":
            phylo_metrics["R2"],

        "Existing_Genomic_RMSE":
            existing_metrics["RMSE"],

        "Existing_Genomic_MAE":
            existing_metrics["MAE"],

        "Existing_Genomic_R2":
            existing_metrics["R2"],

        "Existing_Genomic_Pearson":
            existing_metrics["Pearson"],

        "Diagnostic_Genomic_RMSE":
            diagnostic_metrics["RMSE"],

        "Diagnostic_Genomic_MAE":
            diagnostic_metrics["MAE"],

        "Diagnostic_Genomic_R2":
            diagnostic_metrics["R2"],

        "Diagnostic_Genomic_Pearson":
            diagnostic_metrics["Pearson"],

        "Delta_R2_Existing_vs_Phylogeny":
            delta_existing_vs_phylo,

        "Delta_R2_Diagnostic_vs_Phylogeny":
            delta_diagnostic_vs_phylo,

        "Delta_R2_Diagnostic_vs_Existing":
            delta_diagnostic_vs_existing,

        "Permutation_N":
            len(null_r2),

        "Permutation_P":
            permutation_p,

        "Permutation_Null_Mean_R2":
            float(
                np.mean(null_r2)
            ),

        "Permutation_Null_95th_R2":
            float(
                np.percentile(
                    null_r2,
                    95
                )
            )
    }


    all_summary.append(
        summary_row
    )


# =============================================================================
# SAVE FINAL SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    all_summary
)


summary_path = (
    TABLE_DIR
    /
    "diagnostic_model_summary_420.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# =============================================================================
# SAVE FOLD RESULTS
# =============================================================================

fold_df = pd.DataFrame(
    all_fold_results
)


fold_path = (
    TABLE_DIR
    /
    "diagnostic_fold_results_420.csv"
)


fold_df.to_csv(
    fold_path,
    index=False
)


# =============================================================================
# PRINT FINAL SUMMARY
# =============================================================================

print("\n\n")

print("=" * 80)

print(
    "FINAL DIAGNOSTIC SUMMARY"
)

print("=" * 80)


print(
    summary_df.to_string(
        index=False
    )
)


print("\n")

print("=" * 80)

print(
    "DIAGNOSTIC MODEL COMPLETE"
)

print("=" * 80)


print("\nMain output directory:")

print(
    OUTPUT
)


print("\nSummary:")

print(
    summary_path
)


print("\nFold-level results:")

print(
    fold_path
)


print("\nFigures:")

print(
    FIG_DIR
)


print("\nFeature-selection records:")

print(
    TABLE_DIR
)


print("\nExisting models and OOF files were NOT overwritten.")

print("=" * 80)