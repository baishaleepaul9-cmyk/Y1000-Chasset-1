#!/usr/bin/env python3

"""
===============================================================================
PUBLICATION-GRADE VALIDATION — Y1000+ 420 TAXA
===============================================================================

Purpose
-------
Validate the FINAL SELECTED genome -> phenotype configurations using the
already-generated phylogenetic out-of-fold (OOF) predictions.

Selected configurations
-----------------------

Carbon_Breadth
    Feature set : Robust_79
    Model       : SVR_RBF

Nitrogen_Breadth
    Feature set : Filtered_Variable
    Model       : SVR_RBF

Utilized_Median_Growth
    Feature set : Phenotype_Selected_Top50
    Model       : GradientBoosting

IMPORTANT
---------
The OOF files contain:

    Species
    Assembly_Accession
    Phenotype
    Feature_Set
    Model
    Phylogenetic_Fold
    Observed
    Predicted

Therefore:

    Observed  -> actual phenotype
    Predicted -> model prediction

The script NEVER assumes that a column called Carbon_Breadth,
Nitrogen_Breadth, or Utilized_Median_Growth exists inside the OOF file.

The script validates:
    1. Configuration identity
    2. 420 taxa
    3. Five phylogenetic folds
    4. Fold balance
    5. No sample leakage / duplicate taxa
    6. OOF RMSE
    7. OOF MAE
    8. OOF R2
    9. Pearson correlation
   10. Spearman correlation
   11. Bootstrap confidence intervals
   12. Fold-level performance
   13. Observed vs predicted plots
   14. Residual diagnostics
   15. Prediction tables
   16. Publication summary

Existing model files are NOT modified.
Existing optimization results are NOT modified.
"""

# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


warnings.filterwarnings("ignore")


# =============================================================================
# PROJECT PATHS
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
# OPTIMIZATION OUTPUT
# =============================================================================

OPTIMIZATION_ROOT = (
    RESULTS_ROOT
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "performance_optimization_420"
)


OOF_ROOT = (
    OPTIMIZATION_ROOT
    / "oof_predictions"
)


# =============================================================================
# PUBLICATION VALIDATION OUTPUT
# =============================================================================

OUTPUT_ROOT = (
    OPTIMIZATION_ROOT
    / "publication_validation_420"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


FIGURE_ROOT = (
    OUTPUT_ROOT
    / "figures"
)

FIGURE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


TABLE_ROOT = (
    OUTPUT_ROOT
    / "tables"
)

TABLE_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_TAXA = 420

EXPECTED_FOLDS = 5

RANDOM_STATE = 42

N_BOOTSTRAP = 5000


# =============================================================================
# FINAL SELECTED CONFIGURATIONS
# =============================================================================

FINAL_CONFIGURATIONS = {

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
# EXPECTED OOF COLUMNS
# =============================================================================

REQUIRED_OOF_COLUMNS = [
    "Species",
    "Assembly_Accession",
    "Phenotype",
    "Feature_Set",
    "Model",
    "Phylogenetic_Fold",
    "Observed",
    "Predicted",
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def banner(text):

    print()
    print("=" * 80)
    print(text)
    print("=" * 80)


def require_file(path, description):

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {description}:\n{path}"
        )


def json_safe(value):

    if isinstance(
        value,
        dict
    ):

        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(
        value,
        (list, tuple)
    ):

        return [
            json_safe(v)
            for v in value
        ]

    if isinstance(
        value,
        np.ndarray
    ):

        return [
            json_safe(v)
            for v in value.tolist()
        ]

    if isinstance(
        value,
        np.integer
    ):

        return int(value)

    if isinstance(
        value,
        np.floating
    ):

        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(
        value,
        np.bool_
    ):

        return bool(value)

    if isinstance(
        value,
        pd.Timestamp
    ):

        return value.isoformat()

    if value is None:
        return None

    return value


def safe_r2(
    observed,
    predicted
):

    try:

        if len(np.unique(observed)) < 2:
            return np.nan

        return float(
            r2_score(
                observed,
                predicted
            )
        )

    except Exception:

        return np.nan


def calculate_metrics(
    observed,
    predicted
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    valid = (
        np.isfinite(observed)
        &
        np.isfinite(predicted)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    if len(observed) == 0:

        raise ValueError(
            "No valid observed/predicted pairs."
        )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                observed,
                predicted
            )
        )
    )

    mae = float(
        mean_absolute_error(
            observed,
            predicted
        )
    )

    r2 = safe_r2(
        observed,
        predicted
    )

    pearson_r = np.nan
    pearson_p = np.nan

    spearman_rho = np.nan
    spearman_p = np.nan

    if len(observed) >= 3:

        try:

            pearson_r, pearson_p = pearsonr(
                observed,
                predicted
            )

        except Exception:
            pass

        try:

            spearman_rho, spearman_p = spearmanr(
                observed,
                predicted
            )

        except Exception:
            pass

    return {

        "N": int(len(observed)),

        "RMSE": rmse,

        "MAE": mae,

        "R2": r2,

        "Pearson_r": (
            float(pearson_r)
            if np.isfinite(pearson_r)
            else np.nan
        ),

        "Pearson_p": (
            float(pearson_p)
            if np.isfinite(pearson_p)
            else np.nan
        ),

        "Spearman_rho": (
            float(spearman_rho)
            if np.isfinite(spearman_rho)
            else np.nan
        ),

        "Spearman_p": (
            float(spearman_p)
            if np.isfinite(spearman_p)
            else np.nan
        ),
    }


def bootstrap_metric_ci(
    observed,
    predicted,
    metric_name,
    n_boot=N_BOOTSTRAP,
    seed=RANDOM_STATE
):

    observed = np.asarray(
        observed,
        dtype=float
    )

    predicted = np.asarray(
        predicted,
        dtype=float
    )

    valid = (
        np.isfinite(observed)
        &
        np.isfinite(predicted)
    )

    observed = observed[valid]
    predicted = predicted[valid]

    if len(observed) < 2:

        return (
            np.nan,
            np.nan
        )

    rng = np.random.default_rng(
        seed
    )

    values = []

    n = len(observed)

    for _ in range(n_boot):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        y_true = observed[
            indices
        ]

        y_pred = predicted[
            indices
        ]

        try:

            if metric_name == "RMSE":

                value = np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred
                    )
                )

            elif metric_name == "MAE":

                value = mean_absolute_error(
                    y_true,
                    y_pred
                )

            elif metric_name == "R2":

                if len(
                    np.unique(y_true)
                ) < 2:

                    continue

                value = r2_score(
                    y_true,
                    y_pred
                )

            elif metric_name == "Pearson_r":

                if (
                    len(
                        np.unique(y_true)
                    ) < 2
                    or
                    len(
                        np.unique(y_pred)
                    ) < 2
                ):

                    continue

                value = pearsonr(
                    y_true,
                    y_pred
                )[0]

            elif metric_name == "Spearman_rho":

                if (
                    len(
                        np.unique(y_true)
                    ) < 2
                    or
                    len(
                        np.unique(y_pred)
                    ) < 2
                ):

                    continue

                value = spearmanr(
                    y_true,
                    y_pred
                )[0]

            else:

                raise ValueError(
                    f"Unknown metric: {metric_name}"
                )

            if np.isfinite(value):

                values.append(
                    float(value)
                )

        except Exception:

            continue

    if len(values) == 0:

        return (
            np.nan,
            np.nan
        )

    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


def locate_oof_file(
    phenotype,
    feature_set,
    model
):

    """
    Locate the OOF file using the actual filename convention generated
    by the optimization workflow.

    Expected:

        Phenotype_FeatureSet_Model_OOF_420.csv
    """

    exact_name = (
        f"{phenotype}_"
        f"{feature_set}_"
        f"{model}_"
        f"OOF_420.csv"
    )

    exact_path = (
        OOF_ROOT
        / exact_name
    )

    if exact_path.exists():

        return exact_path


    # -------------------------------------------------------------------------
    # Fallback: search files rather than silently selecting the wrong phenotype
    # -------------------------------------------------------------------------

    candidates = sorted(
        OOF_ROOT.glob(
            "*.csv"
        )
    )

    matching = []

    for path in candidates:

        name = path.name

        if (
            phenotype in name
            and feature_set in name
            and model in name
            and "OOF_420" in name
        ):

            matching.append(
                path
            )


    if len(matching) == 1:

        return matching[0]


    if len(matching) > 1:

        raise RuntimeError(
            "\nMultiple possible OOF files found for:\n"
            f"Phenotype = {phenotype}\n"
            f"Feature set = {feature_set}\n"
            f"Model = {model}\n\n"
            "Candidates:\n"
            +
            "\n".join(
                str(x)
                for x in matching
            )
        )


    raise FileNotFoundError(
        "\nCould not locate OOF prediction file for:\n"
        f"Phenotype = {phenotype}\n"
        f"Feature set = {feature_set}\n"
        f"Model = {model}\n\n"
        f"OOF directory:\n{OOF_ROOT}"
    )


def validate_oof_identity(
    df,
    phenotype,
    feature_set,
    model,
    path
):

    """
    Critical protection against accidentally validating the wrong
    phenotype/model file.
    """

    if "Phenotype" not in df.columns:

        raise ValueError(
            f"\nMissing Phenotype column in:\n{path}"
        )

    if "Feature_Set" not in df.columns:

        raise ValueError(
            f"\nMissing Feature_Set column in:\n{path}"
        )

    if "Model" not in df.columns:

        raise ValueError(
            f"\nMissing Model column in:\n{path}"
        )


    observed_phenotypes = (
        df["Phenotype"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    observed_feature_sets = (
        df["Feature_Set"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    observed_models = (
        df["Model"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    if observed_phenotypes != [phenotype]:

        raise RuntimeError(
            "\nPHENOTYPE MISMATCH\n"
            f"Expected : {phenotype}\n"
            f"Found    : {observed_phenotypes}\n"
            f"File     : {path}"
        )


    if observed_feature_sets != [feature_set]:

        raise RuntimeError(
            "\nFEATURE SET MISMATCH\n"
            f"Expected : {feature_set}\n"
            f"Found    : {observed_feature_sets}\n"
            f"File     : {path}"
        )


    if observed_models != [model]:

        raise RuntimeError(
            "\nMODEL MISMATCH\n"
            f"Expected : {model}\n"
            f"Found    : {observed_models}\n"
            f"File     : {path}"
        )


def load_oof_configuration(
    phenotype,
    feature_set,
    model
):

    banner(
        f"LOADING OOF — "
        f"{phenotype} — "
        f"{feature_set} — "
        f"{model}"
    )


    path = locate_oof_file(
        phenotype,
        feature_set,
        model
    )


    print()
    print("Selected OOF file:")
    print(path)


    df = pd.read_csv(
        path
    )


    print()
    print("Shape:")
    print(df.shape)


    print()
    print("Columns:")
    print(df.columns.tolist())


    missing = [
        col
        for col in REQUIRED_OOF_COLUMNS
        if col not in df.columns
    ]


    if missing:

        raise ValueError(
            "\nMissing required OOF columns:\n"
            +
            "\n".join(
                missing
            )
        )


    validate_oof_identity(
        df,
        phenotype,
        feature_set,
        model,
        path
    )


    # -------------------------------------------------------------------------
    # Convert prediction fields to numeric
    # -------------------------------------------------------------------------

    df["Observed"] = pd.to_numeric(
        df["Observed"],
        errors="coerce"
    )

    df["Predicted"] = pd.to_numeric(
        df["Predicted"],
        errors="coerce"
    )


    df["Phylogenetic_Fold"] = pd.to_numeric(
        df["Phylogenetic_Fold"],
        errors="coerce"
    )


    # -------------------------------------------------------------------------
    # Remove invalid prediction records
    # -------------------------------------------------------------------------

    valid = (
        df["Observed"].notna()
        &
        df["Predicted"].notna()
        &
        df["Phylogenetic_Fold"].notna()
    )


    invalid_count = (
        len(df)
        -
        int(valid.sum())
    )


    if invalid_count > 0:

        print()
        print(
            f"Removing {invalid_count} invalid OOF rows."
        )


    df = df.loc[
        valid
    ].copy()


    df["Phylogenetic_Fold"] = (
        df["Phylogenetic_Fold"]
        .astype(int)
    )


    return df, path


# =============================================================================
# DATA INTEGRITY CHECKS
# =============================================================================

def validate_dataset_integrity(
    df,
    phenotype
):

    banner(
        f"DATA INTEGRITY — {phenotype}"
    )


    # -------------------------------------------------------------------------
    # Taxa count
    # -------------------------------------------------------------------------

    n_rows = len(df)

    print(
        f"OOF rows: {n_rows}"
    )


    if n_rows != EXPECTED_TAXA:

        raise RuntimeError(
            f"\nExpected {EXPECTED_TAXA} OOF rows "
            f"but found {n_rows}."
        )


    # -------------------------------------------------------------------------
    # Unique accessions
    # -------------------------------------------------------------------------

    if (
        df["Assembly_Accession"]
        .duplicated()
        .any()
    ):

        duplicates = (
            df.loc[
                df["Assembly_Accession"].duplicated(
                    keep=False
                ),
                "Assembly_Accession"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "\nDuplicate assembly accessions detected:\n"
            +
            "\n".join(
                duplicates
            )
        )


    print(
        "Unique assembly accessions: "
        f"{df['Assembly_Accession'].nunique()}"
    )


    # -------------------------------------------------------------------------
    # Phylogenetic folds
    # -------------------------------------------------------------------------

    folds = sorted(
        df["Phylogenetic_Fold"]
        .dropna()
        .unique()
        .tolist()
    )


    print()
    print(
        "Phylogenetic folds:"
    )

    print(
        folds
    )


    if folds != list(
        range(
            1,
            EXPECTED_FOLDS + 1
        )
    ):

        raise RuntimeError(
            "\nUnexpected phylogenetic fold structure:\n"
            f"{folds}"
        )


    fold_sizes = (
        df["Phylogenetic_Fold"]
        .value_counts()
        .sort_index()
    )


    print()
    print(
        "Phylogenetic fold sizes:"
    )

    print(
        fold_sizes
    )


    if (
        fold_sizes
        != fold_sizes.iloc[0]
    ).any():

        print(
            "\nWARNING: folds are not exactly equal."
        )


    # -------------------------------------------------------------------------
    # Check phenotype consistency
    # -------------------------------------------------------------------------

    phenotype_values = (
        df["Phenotype"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )


    if phenotype_values != [phenotype]:

        raise RuntimeError(
            "\nPhenotype inconsistency detected."
        )


    print()
    print(
        "✓ Dataset integrity checks passed."
    )


# =============================================================================
# PERFORMANCE ANALYSIS
# =============================================================================

def calculate_publication_metrics(
    df,
    phenotype
):

    banner(
        f"OVERALL OOF PERFORMANCE — {phenotype}"
    )


    observed = df[
        "Observed"
    ].to_numpy(
        dtype=float
    )


    predicted = df[
        "Predicted"
    ].to_numpy(
        dtype=float
    )


    metrics = calculate_metrics(
        observed,
        predicted
    )


    print()
    print(
        f"N       : {metrics['N']}"
    )

    print(
        f"RMSE    : {metrics['RMSE']:.6f}"
    )

    print(
        f"MAE     : {metrics['MAE']:.6f}"
    )

    print(
        f"R2      : {metrics['R2']:.6f}"
    )

    print(
        f"Pearson : {metrics['Pearson_r']:.6f}"
    )

    print(
        f"Spearman: {metrics['Spearman_rho']:.6f}"
    )


    # -------------------------------------------------------------------------
    # Bootstrap CIs
    # -------------------------------------------------------------------------

    bootstrap_results = {}


    for metric_name in [
        "RMSE",
        "MAE",
        "R2",
        "Pearson_r",
        "Spearman_rho",
    ]:

        ci_low, ci_high = (
            bootstrap_metric_ci(
                observed,
                predicted,
                metric_name
            )
        )


        bootstrap_results[
            metric_name
        ] = {

            "CI_95_Lower":
                ci_low,

            "CI_95_Upper":
                ci_high,
        }


    return (
        metrics,
        bootstrap_results
    )


# =============================================================================
# FOLD PERFORMANCE
# =============================================================================

def calculate_fold_performance(
    df,
    phenotype
):

    banner(
        f"FOLD-LEVEL PERFORMANCE — {phenotype}"
    )


    records = []


    for fold in sorted(
        df["Phylogenetic_Fold"]
        .unique()
    ):

        fold_df = df[
            df["Phylogenetic_Fold"]
            == fold
        ]


        observed = fold_df[
            "Observed"
        ].to_numpy(
            dtype=float
        )


        predicted = fold_df[
            "Predicted"
        ].to_numpy(
            dtype=float
        )


        metrics = calculate_metrics(
            observed,
            predicted
        )


        record = {

            "Phenotype":
                phenotype,

            "Phylogenetic_Fold":
                int(fold),

            "N":
                int(len(fold_df)),

            "RMSE":
                metrics["RMSE"],

            "MAE":
                metrics["MAE"],

            "R2":
                metrics["R2"],

            "Pearson_r":
                metrics["Pearson_r"],

            "Spearman_rho":
                metrics["Spearman_rho"],
        }


        records.append(
            record
        )


        print(
            f"Fold {fold}: "
            f"N={len(fold_df)} | "
            f"RMSE={metrics['RMSE']:.6f} | "
            f"MAE={metrics['MAE']:.6f} | "
            f"R2={metrics['R2']:.6f}"
        )


    fold_df = pd.DataFrame(
        records
    )


    return fold_df


# =============================================================================
# RESIDUAL ANALYSIS
# =============================================================================

def create_residual_table(
    df,
    phenotype
):

    residual_df = df[
        [
            "Species",
            "Assembly_Accession",
            "Phenotype",
            "Feature_Set",
            "Model",
            "Phylogenetic_Fold",
            "Observed",
            "Predicted",
        ]
    ].copy()


    residual_df[
        "Residual"
    ] = (
        residual_df["Observed"]
        -
        residual_df["Predicted"]
    )


    residual_df[
        "Absolute_Error"
    ] = (
        residual_df["Residual"]
        .abs()
    )


    residual_df[
        "Squared_Error"
    ] = (
        residual_df["Residual"]
        ** 2
    )


    residual_df[
        "Relative_Error"
    ] = np.where(
        residual_df["Observed"].abs() > 0,
        residual_df["Absolute_Error"]
        /
        residual_df["Observed"].abs(),
        np.nan
    )


    output_path = (
        TABLE_ROOT
        /
        f"{phenotype}_OOF_residuals_420.csv"
    )


    residual_df.to_csv(
        output_path,
        index=False
    )


    return residual_df


# =============================================================================
# FIGURE — OBSERVED VS PREDICTED
# =============================================================================

def plot_observed_vs_predicted(
    df,
    phenotype,
    metrics
):

    observed = df[
        "Observed"
    ].to_numpy(
        dtype=float
    )


    predicted = df[
        "Predicted"
    ].to_numpy(
        dtype=float
    )


    fig, ax = plt.subplots(
        figsize=(7, 7)
    )


    ax.scatter(
        observed,
        predicted,
        s=35,
        alpha=0.70
    )


    minimum = min(
        observed.min(),
        predicted.min()
    )


    maximum = max(
        observed.max(),
        predicted.max()
    )


    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=1.5
    )


    ax.set_xlabel(
        f"Observed {phenotype}",
        fontsize=12
    )


    ax.set_ylabel(
        f"Predicted {phenotype}",
        fontsize=12
    )


    ax.set_title(
        f"Observed vs Predicted — {phenotype}",
        fontsize=14,
        fontweight="bold"
    )


    text = (
        f"RMSE = {metrics['RMSE']:.3f}\n"
        f"MAE = {metrics['MAE']:.3f}\n"
        f"R² = {metrics['R2']:.3f}\n"
        f"Spearman ρ = {metrics['Spearman_rho']:.3f}"
    )


    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    plt.tight_layout()


    output_path = (
        FIGURE_ROOT
        /
        f"{phenotype}_observed_vs_predicted_420.png"
    )


    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print()
    print(
        "✓ Observed-vs-predicted figure:"
    )
    print(
        output_path
    )


# =============================================================================
# FIGURE — RESIDUALS
# =============================================================================

def plot_residuals(
    residual_df,
    phenotype
):

    observed = residual_df[
        "Observed"
    ].to_numpy(
        dtype=float
    )


    residuals = residual_df[
        "Residual"
    ].to_numpy(
        dtype=float
    )


    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    ax.scatter(
        observed,
        residuals,
        s=32,
        alpha=0.70
    )


    ax.axhline(
        0,
        linestyle="--",
        linewidth=1.5
    )


    ax.set_xlabel(
        f"Observed {phenotype}",
        fontsize=12
    )


    ax.set_ylabel(
        "Residual",
        fontsize=12
    )


    ax.set_title(
        f"Residual Diagnostics — {phenotype}",
        fontsize=14,
        fontweight="bold"
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    plt.tight_layout()


    output_path = (
        FIGURE_ROOT
        /
        f"{phenotype}_residual_diagnostics_420.png"
    )


    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


    print(
        "✓ Residual figure:"
    )
    print(
        output_path
    )


# =============================================================================
# FIGURE — FOLD PERFORMANCE
# =============================================================================

def plot_fold_performance(
    fold_df,
    phenotype
):

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )


    x = np.arange(
        len(fold_df)
    )


    ax.bar(
        x,
        fold_df["RMSE"]
    )


    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        [
            f"Fold {x}"
            for x in fold_df[
                "Phylogenetic_Fold"
            ]
        ]
    )


    ax.set_ylabel(
        "RMSE",
        fontsize=12
    )


    ax.set_xlabel(
        "Phylogenetic fold",
        fontsize=12
    )


    ax.set_title(
        f"Phylogenetic CV RMSE — {phenotype}",
        fontsize=14,
        fontweight="bold"
    )


    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )


    plt.tight_layout()


    output_path = (
        FIGURE_ROOT
        /
        f"{phenotype}_phylogenetic_fold_RMSE_420.png"
    )


    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )


    plt.close(
        fig
    )


# =============================================================================
# VALIDATE ONE CONFIGURATION
# =============================================================================

def validate_configuration(
    phenotype,
    feature_set,
    model
):

    banner(
        f"PUBLICATION VALIDATION — {phenotype}"
    )


    print()
    print(
        "Expected configuration:"
    )

    print(
        f"Phenotype  : {phenotype}"
    )

    print(
        f"Feature set: {feature_set}"
    )

    print(
        f"Model      : {model}"
    )


    # -------------------------------------------------------------------------
    # Load correct OOF file
    # -------------------------------------------------------------------------

    df, oof_path = (
        load_oof_configuration(
            phenotype,
            feature_set,
            model
        )
    )


    # -------------------------------------------------------------------------
    # Dataset integrity
    # -------------------------------------------------------------------------

    validate_dataset_integrity(
        df,
        phenotype
    )


    # -------------------------------------------------------------------------
    # Overall metrics
    # -------------------------------------------------------------------------

    metrics, bootstrap_results = (
        calculate_publication_metrics(
            df,
            phenotype
        )
    )


    # -------------------------------------------------------------------------
    # Fold metrics
    # -------------------------------------------------------------------------

    fold_df = (
        calculate_fold_performance(
            df,
            phenotype
        )
    )


    # -------------------------------------------------------------------------
    # Residuals
    # -------------------------------------------------------------------------

    residual_df = (
        create_residual_table(
            df,
            phenotype
        )
    )


    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    plot_observed_vs_predicted(
        df,
        phenotype,
        metrics
    )


    plot_residuals(
        residual_df,
        phenotype
    )


    plot_fold_performance(
        fold_df,
        phenotype
    )


    # -------------------------------------------------------------------------
    # Save fold performance
    # -------------------------------------------------------------------------

    fold_path = (
        TABLE_ROOT
        /
        f"{phenotype}_fold_performance_420.csv"
    )


    fold_df.to_csv(
        fold_path,
        index=False
    )


    # -------------------------------------------------------------------------
    # Save complete OOF table
    # -------------------------------------------------------------------------

    oof_output_path = (
        TABLE_ROOT
        /
        f"{phenotype}_validated_OOF_predictions_420.csv"
    )


    df.to_csv(
        oof_output_path,
        index=False
    )


    # -------------------------------------------------------------------------
    # Summary record
    # -------------------------------------------------------------------------

    summary = {

        "Phenotype":
            phenotype,

        "Feature_Set":
            feature_set,

        "Model":
            model,

        "N_Taxa":
            metrics["N"],

        "N_Phylogenetic_Folds":
            EXPECTED_FOLDS,

        "RMSE":
            metrics["RMSE"],

        "RMSE_CI95_Lower":
            bootstrap_results[
                "RMSE"
            ][
                "CI_95_Lower"
            ],

        "RMSE_CI95_Upper":
            bootstrap_results[
                "RMSE"
            ][
                "CI_95_Upper"
            ],

        "MAE":
            metrics["MAE"],

        "MAE_CI95_Lower":
            bootstrap_results[
                "MAE"
            ][
                "CI_95_Lower"
            ],

        "MAE_CI95_Upper":
            bootstrap_results[
                "MAE"
            ][
                "CI_95_Upper"
            ],

        "R2":
            metrics["R2"],

        "R2_CI95_Lower":
            bootstrap_results[
                "R2"
            ][
                "CI_95_Lower"
            ],

        "R2_CI95_Upper":
            bootstrap_results[
                "R2"
            ][
                "CI_95_Upper"
            ],

        "Pearson_r":
            metrics["Pearson_r"],

        "Pearson_CI95_Lower":
            bootstrap_results[
                "Pearson_r"
            ][
                "CI_95_Lower"
            ],

        "Pearson_CI95_Upper":
            bootstrap_results[
                "Pearson_r"
            ][
                "CI_95_Upper"
            ],

        "Pearson_p":
            metrics["Pearson_p"],

        "Spearman_rho":
            metrics["Spearman_rho"],

        "Spearman_CI95_Lower":
            bootstrap_results[
                "Spearman_rho"
            ][
                "CI_95_Lower"
            ],

        "Spearman_CI95_Upper":
            bootstrap_results[
                "Spearman_rho"
            ][
                "CI_95_Upper"
            ],

        "Spearman_p":
            metrics["Spearman_p"],

        "OOF_File":
            str(oof_path),
    }


    print()
    print(
        "Publication validation summary:"
    )

    print(
        f"RMSE = {summary['RMSE']:.6f}"
    )

    print(
        f"MAE  = {summary['MAE']:.6f}"
    )

    print(
        f"R2   = {summary['R2']:.6f}"
    )

    print(
        f"Pearson r = {summary['Pearson_r']:.6f}"
    )

    print(
        f"Spearman rho = {summary['Spearman_rho']:.6f}"
    )


    return summary


# =============================================================================
# MAIN
# =============================================================================

def main():

    banner(
        "Y1000+ PUBLICATION-GRADE PHYLOGENETIC VALIDATION"
    )


    print()
    print(
        "Project:"
    )
    print(
        PROJECT_ROOT
    )


    print()
    print(
        "Optimization directory:"
    )
    print(
        OPTIMIZATION_ROOT
    )


    print()
    print(
        "OOF directory:"
    )
    print(
        OOF_ROOT
    )


    # -------------------------------------------------------------------------
    # Input checks
    # -------------------------------------------------------------------------

    if not OPTIMIZATION_ROOT.exists():

        raise FileNotFoundError(
            "\nOptimization output directory not found:\n"
            f"{OPTIMIZATION_ROOT}"
        )


    if not OOF_ROOT.exists():

        raise FileNotFoundError(
            "\nOOF prediction directory not found:\n"
            f"{OOF_ROOT}"
        )


    # -------------------------------------------------------------------------
    # List available OOF files before doing anything
    # -------------------------------------------------------------------------

    banner(
        "AVAILABLE OOF FILES"
    )


    available_files = sorted(
        OOF_ROOT.glob(
            "*.csv"
        )
    )


    if len(available_files) == 0:

        raise FileNotFoundError(
            "\nNo OOF CSV files found in:\n"
            f"{OOF_ROOT}"
        )


    for path in available_files:

        print(
            path.name
        )


    print()
    print(
        f"Total OOF CSV files found: "
        f"{len(available_files)}"
    )


    # -------------------------------------------------------------------------
    # Validate each selected configuration
    # -------------------------------------------------------------------------

    summaries = []


    for phenotype in [
        "Carbon_Breadth",
        "Nitrogen_Breadth",
        "Utilized_Median_Growth",
    ]:

        configuration = (
            FINAL_CONFIGURATIONS[
                phenotype
            ]
        )


        summary = (
            validate_configuration(
                phenotype=phenotype,
                feature_set=configuration[
                    "feature_set"
                ],
                model=configuration[
                    "model"
                ],
            )
        )


        summaries.append(
            summary
        )


    # -------------------------------------------------------------------------
    # Combined publication summary
    # -------------------------------------------------------------------------

    banner(
        "COMBINED PUBLICATION VALIDATION SUMMARY"
    )


    summary_df = pd.DataFrame(
        summaries
    )


    summary_path = (
        TABLE_ROOT
        /
        "publication_grade_validation_summary_420.csv"
    )


    summary_df.to_csv(
        summary_path,
        index=False
    )


    print()
    print(
        summary_df[
            [
                "Phenotype",
                "Feature_Set",
                "Model",
                "N_Taxa",
                "RMSE",
                "MAE",
                "R2",
                "Pearson_r",
                "Spearman_rho",
            ]
        ].to_string(
            index=False
        )
    )


    # -------------------------------------------------------------------------
    # Bootstrap summary
    # -------------------------------------------------------------------------

    bootstrap_records = []


    for row in summaries:

        bootstrap_records.append({

            "Phenotype":
                row["Phenotype"],

            "Feature_Set":
                row["Feature_Set"],

            "Model":
                row["Model"],

            "RMSE":
                row["RMSE"],

            "RMSE_95CI_Lower":
                row["RMSE_CI95_Lower"],

            "RMSE_95CI_Upper":
                row["RMSE_CI95_Upper"],

            "MAE":
                row["MAE"],

            "MAE_95CI_Lower":
                row["MAE_CI95_Lower"],

            "MAE_95CI_Upper":
                row["MAE_CI95_Upper"],

            "R2":
                row["R2"],

            "R2_95CI_Lower":
                row["R2_CI95_Lower"],

            "R2_95CI_Upper":
                row["R2_CI95_Upper"],

            "Pearson_r":
                row["Pearson_r"],

            "Pearson_95CI_Lower":
                row["Pearson_CI95_Lower"],

            "Pearson_95CI_Upper":
                row["Pearson_CI95_Upper"],

            "Spearman_rho":
                row["Spearman_rho"],

            "Spearman_95CI_Lower":
                row["Spearman_CI95_Lower"],

            "Spearman_95CI_Upper":
                row["Spearman_CI95_Upper"],

        })


    bootstrap_df = pd.DataFrame(
        bootstrap_records
    )


    bootstrap_path = (
        TABLE_ROOT
        /
        "publication_bootstrap_confidence_intervals_420.csv"
    )


    bootstrap_df.to_csv(
        bootstrap_path,
        index=False
    )


    # -------------------------------------------------------------------------
    # Save machine-readable metadata
    # -------------------------------------------------------------------------

    metadata = {

        "project":
            "Y1000+",

        "dataset_size":
            EXPECTED_TAXA,

        "phylogenetic_folds":
            EXPECTED_FOLDS,

        "random_state":
            RANDOM_STATE,

        "bootstrap_iterations":
            N_BOOTSTRAP,

        "selected_configurations":
            FINAL_CONFIGURATIONS,

        "output_directory":
            str(OUTPUT_ROOT),

        "summary_file":
            str(summary_path),

        "bootstrap_file":
            str(bootstrap_path),

    }


    metadata_path = (
        OUTPUT_ROOT
        /
        "publication_validation_metadata_420.json"
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            json_safe(metadata),
            handle,
            indent=4
        )


    # -------------------------------------------------------------------------
    # Final checks
    # -------------------------------------------------------------------------

    banner(
        "FINAL VALIDATION CHECK"
    )


    if len(summary_df) != 3:

        raise RuntimeError(
            "Expected validation results for all 3 phenotypes."
        )


    if not summary_df[
        "N_Taxa"
    ].eq(
        EXPECTED_TAXA
    ).all():

        raise RuntimeError(
            "At least one phenotype does not contain "
            "420 validated taxa."
        )


    print(
        "✓ Carbon_Breadth validated."
    )

    print(
        "✓ Nitrogen_Breadth validated."
    )

    print(
        "✓ Utilized_Median_Growth validated."
    )


    print()
    print(
        "✓ All three selected configurations passed "
        "the structural publication-validation checks."
    )


    banner(
        "PUBLICATION VALIDATION COMPLETE"
    )


    print()
    print(
        "Main output directory:"
    )

    print(
        OUTPUT_ROOT
    )


    print()
    print(
        "Summary:"
    )

    print(
        summary_path
    )


    print()
    print(
        "Bootstrap confidence intervals:"
    )

    print(
        bootstrap_path
    )


    print()
    print(
        "Figures:"
    )

    print(
        FIGURE_ROOT
    )


    print()
    print(
        "Tables:"
    )

    print(
        TABLE_ROOT
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()