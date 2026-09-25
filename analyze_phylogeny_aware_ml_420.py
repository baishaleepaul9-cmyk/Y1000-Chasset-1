# ================================================================
# analyze_phylogeny_aware_ml_420.py
#
# OPTIMIZED FINAL PERFORMANCE + FEATURE IMPORTANCE + PHYLOGENETIC
# CONCENTRATION ANALYSIS
#
# Project:
#   Y1000 chassis project
#
# IMPORTANT:
#   - Original ML matrix is NOT modified
#   - Original ASTRAL tree is NOT modified
#   - Exactly 420 taxa are expected
#   - Exactly 1988 BUSCO predictors are expected
#   - Phenotype_Source_Species is metadata
#   - No ete3 dependency
#   - Uses Bio.Phylo
#   - PHYLOGENETIC DISTANCE MATRIX IS CALCULATED ONCE
#     instead of calling tree.distance() repeatedly for every BUSCO
# ================================================================

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from Bio import Phylo

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

ML_FILE = (
    PROJECT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    / "y1000_420_phylogeny_aware_ml_matrix.csv"
)

TREE_FILE = (
    PROJECT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_qc"
    / "ASTRAL_420_taxon_pruned.nwk"
)

TREE_FILE_ALT = (
    PROJECT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    / "phylogeny_qc"
    / "ASTRAL_420_taxon_pruned.nwk"
)

MODEL_DIR = (
    PROJECT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    / r"phylogeny_cv\model_training"
)

FEATURE_LIST_FILE = (
    PROJECT
    / r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    / "y1000_420_variable_busco_feature_list.csv"
)

OUTPUT_DIR = (
    MODEL_DIR
    / "feature_importance_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ================================================================
# CONSTANTS
# ================================================================

EXPECTED_TAXA = 420
EXPECTED_BUSCOS = 1988

BOOTSTRAP_ITERATIONS = 5000
RANDOM_SEED = 42

PHENOTYPE_VARIABLES = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
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
# UTILITY FUNCTIONS
# ================================================================

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


def clean_string(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def accession_from_tip(tip):
    """
    Expected tree labels:

        Species__GCA_XXXXXXXXX.X
    """

    tip = clean_string(tip)

    if "__" in tip:
        return tip.rsplit("__", 1)[-1].strip()

    return tip.strip()


def species_from_tip(tip):

    tip = clean_string(tip)

    if "__" in tip:
        return (
            tip.rsplit("__", 1)[0]
            .replace("_", " ")
        )

    return tip.replace("_", " ")


def bootstrap_ci(
    values,
    n_boot=5000,
    seed=42
):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return (
            float(values[0]),
            float(values[0])
        )

    rng = np.random.default_rng(seed)

    boot_means = np.empty(
        n_boot,
        dtype=float
    )

    for i in range(n_boot):

        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        boot_means[i] = np.mean(
            sample
        )

    return (
        float(
            np.percentile(
                boot_means,
                2.5
            )
        ),
        float(
            np.percentile(
                boot_means,
                97.5
            )
        )
    )


# ================================================================
# START
# ================================================================

banner(
    "OPTIMIZED PHYLOGENY-AWARE ML PERFORMANCE + "
    "FEATURE IMPORTANCE ANALYSIS"
)

print("Project:")
print(PROJECT)

print()
print("Output directory:")
print(OUTPUT_DIR)

# ================================================================
# INPUT CHECK
# ================================================================

banner("CHECKING INPUT FILES")

require_file(
    ML_FILE,
    "final phylogeny-aware ML matrix"
)

if TREE_FILE.exists():

    ACTIVE_TREE_FILE = TREE_FILE

elif TREE_FILE_ALT.exists():

    ACTIVE_TREE_FILE = TREE_FILE_ALT

else:

    raise FileNotFoundError(
        "\nCould not find the validated 420-taxon ASTRAL tree.\n"
        f"Tried:\n{TREE_FILE}\n{TREE_FILE_ALT}"
    )

require_file(
    ACTIVE_TREE_FILE,
    "final 420-taxon ASTRAL tree"
)

require_file(
    MODEL_DIR /
    "phylogeny_aware_ml_fold_results_420.csv",
    "ML fold results"
)

require_file(
    MODEL_DIR /
    "phylogeny_aware_ml_predictions_420.csv",
    "ML predictions"
)

print("✓ Final ML matrix found")
print("✓ Final ASTRAL tree found")
print("✓ Fold results found")
print("✓ Prediction file found")

# ================================================================
# LOAD MATRIX
# ================================================================

banner("LOADING FINAL ML MATRIX")

df = pd.read_csv(
    ML_FILE
)

print(
    f"Rows:    {df.shape[0]}"
)

print(
    f"Columns: {df.shape[1]}"
)

if len(df) != EXPECTED_TAXA:

    raise RuntimeError(
        f"Expected {EXPECTED_TAXA} taxa "
        f"but found {len(df)}."
    )

for col in [
    "Species",
    "Assembly_Accession"
]:

    if col not in df.columns:

        raise RuntimeError(
            f"Required column missing: {col}"
        )

for phenotype in PHENOTYPE_VARIABLES:

    if phenotype not in df.columns:

        raise RuntimeError(
            f"Required phenotype column missing: "
            f"{phenotype}"
        )

print("✓ Exactly 420 taxa")
print("✓ Required identifier columns present")
print("✓ Required phenotype variables present")

# ================================================================
# TAXON QC
# ================================================================

banner("TAXON QC")

duplicate_species = (
    df["Species"]
    .astype(str)
    .duplicated()
    .sum()
)

duplicate_accessions = (
    df["Assembly_Accession"]
    .astype(str)
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

print("✓ 420 unique taxa")
print("✓ Species are unique")
print("✓ Assembly accessions are unique")

# ================================================================
# IDENTIFY BUSCO FEATURES
# ================================================================

banner("IDENTIFYING BUSCO FEATURES")

BUSCO_FEATURES = [
    c for c in df.columns
    if c not in METADATA_COLUMNS
]

print(
    f"BUSCO features detected: "
    f"{len(BUSCO_FEATURES)}"
)

if len(BUSCO_FEATURES) != EXPECTED_BUSCOS:

    print()
    print("Last 20 detected BUSCOs:")

    for c in BUSCO_FEATURES[-20:]:
        print(c)

    raise RuntimeError(
        f"Expected {EXPECTED_BUSCOS} BUSCO features "
        f"but detected {len(BUSCO_FEATURES)}."
    )

print(
    f"✓ Correct number of BUSCO features: "
    f"{EXPECTED_BUSCOS}"
)

# ================================================================
# BUSCO ID QC
# ================================================================

banner("BUSCO ID QC")

duplicate_buscos = (
    pd.Series(BUSCO_FEATURES)
    .duplicated()
    .sum()
)

print(
    f"BUSCO features: {len(BUSCO_FEATURES)}"
)

print(
    f"Duplicate BUSCO IDs: "
    f"{duplicate_buscos}"
)

if duplicate_buscos != 0:

    raise RuntimeError(
        "Duplicate BUSCO IDs detected."
    )

print("✓ BUSCO IDs are unique")

# ================================================================
# BUSCO VALUE QC
# ================================================================

banner("BUSCO VALUE QC")

busco = df[
    BUSCO_FEATURES
].copy()

missing_busco = int(
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
        "BUSCO matrix contains missing values."
    )

unique_values = sorted(
    set(
        str(v).strip()
        for v in busco.to_numpy().ravel()
    )
)

print("Unique BUSCO values:")

for value in unique_values:
    print(
        f"  {repr(value)}"
    )

if not set(unique_values).issubset(
    {"0", "1"}
):

    raise RuntimeError(
        "BUSCO matrix contains "
        "values other than 0/1."
    )

busco = busco.astype(
    np.int8
)

print("✓ BUSCO matrix is binary")
print("✓ No missing BUSCO values")

# ================================================================
# PHENOTYPE QC
# ================================================================

banner("PHENOTYPE QC")

for phenotype in PHENOTYPE_VARIABLES:

    missing = int(
        df[phenotype]
        .isna()
        .sum()
    )

    print(
        f"{phenotype}: "
        f"missing = {missing}"
    )

    if missing != 0:

        raise RuntimeError(
            f"Missing values in "
            f"{phenotype}"
        )

print("✓ No missing target values")

# ================================================================
# MODEL PERFORMANCE ANALYSIS
# ================================================================

banner(
    "MODEL PERFORMANCE: MEAN ± SD + BOOTSTRAP CI"
)

fold_file = (
    MODEL_DIR /
    "phylogeny_aware_ml_fold_results_420.csv"
)

fold_results = pd.read_csv(
    fold_file
)

required_fold_columns = [
    "Model",
    "Fold",
    "R2",
    "RMSE",
    "MAE"
]

missing_fold_columns = [
    c
    for c in required_fold_columns
    if c not in fold_results.columns
]

if missing_fold_columns:

    raise RuntimeError(
        f"Missing fold-result columns: "
        f"{missing_fold_columns}"
    )

summary_rows = []

group_columns = ["Model"]

if "CV_Strategy" in fold_results.columns:
    group_columns.append(
        "CV_Strategy"
    )

for keys, group in fold_results.groupby(
    group_columns
):

    if not isinstance(keys, tuple):
        keys = (keys,)

    row = dict(
        zip(
            group_columns,
            keys
        )
    )

    for metric in [
        "R2",
        "RMSE",
        "MAE"
    ]:

        values = (
            pd.to_numeric(
                group[metric],
                errors="coerce"
            )
            .dropna()
            .values
        )

        mean_value = float(
            np.mean(values)
        )

        sd_value = (
            float(
                np.std(
                    values,
                    ddof=1
                )
            )
            if len(values) > 1
            else 0.0
        )

        ci_low, ci_high = (
            bootstrap_ci(
                values,
                n_boot=BOOTSTRAP_ITERATIONS,
                seed=RANDOM_SEED
            )
        )

        row[
            f"{metric}_Mean"
        ] = mean_value

        row[
            f"{metric}_SD"
        ] = sd_value

        row[
            f"{metric}_Mean_SD"
        ] = (
            f"{mean_value:.6f} ± "
            f"{sd_value:.6f}"
        )

        row[
            f"{metric}_Bootstrap95CI_Lower"
        ] = ci_low

        row[
            f"{metric}_Bootstrap95CI_Upper"
        ] = ci_high

    summary_rows.append(row)

performance_summary = pd.DataFrame(
    summary_rows
)

performance_summary_file = (
    OUTPUT_DIR /
    "model_performance_mean_sd_bootstrap_420.csv"
)

performance_summary.to_csv(
    performance_summary_file,
    index=False
)

print(
    "✓ Performance summary written:"
)

print(
    performance_summary_file
)

# ================================================================
# PERFORMANCE DISTRIBUTIONS
# ================================================================

banner(
    "FOLD-LEVEL PERFORMANCE DISTRIBUTIONS"
)

distribution_rows = []

for keys, group in fold_results.groupby(
    group_columns
):

    if not isinstance(keys, tuple):
        keys = (keys,)

    row = dict(
        zip(
            group_columns,
            keys
        )
    )

    for metric in [
        "R2",
        "RMSE",
        "MAE"
    ]:

        values = pd.to_numeric(
            group[metric],
            errors="coerce"
        ).dropna()

        row[
            f"{metric}_Min"
        ] = values.min()

        row[
            f"{metric}_Q1"
        ] = values.quantile(
            0.25
        )

        row[
            f"{metric}_Median"
        ] = values.median()

        row[
            f"{metric}_Q3"
        ] = values.quantile(
            0.75
        )

        row[
            f"{metric}_Max"
        ] = values.max()

    distribution_rows.append(
        row
    )

performance_distribution = (
    pd.DataFrame(
        distribution_rows
    )
)

distribution_file = (
    OUTPUT_DIR /
    "fold_performance_distributions_420.csv"
)

performance_distribution.to_csv(
    distribution_file,
    index=False
)

print(
    "✓ Fold-performance distribution "
    "table written:"
)

print(distribution_file)

# ================================================================
# LOAD PREDICTIONS
# ================================================================

banner("LOADING MODEL PREDICTIONS")

prediction_file = (
    MODEL_DIR /
    "phylogeny_aware_ml_predictions_420.csv"
)

predictions = pd.read_csv(
    prediction_file
)

print(
    f"Prediction records: "
    f"{len(predictions)}"
)

# ================================================================
# FULL-DATASET FEATURE IMPORTANCE
# ================================================================

banner(
    "TRAINING FULL-DATASET MODELS "
    "FOR FEATURE IMPORTANCE"
)

X = busco.copy()

feature_importance_results = []

for phenotype in PHENOTYPE_VARIABLES:

    print()
    print(
        f"Phenotype: {phenotype}"
    )

    y = pd.to_numeric(
        df[phenotype],
        errors="coerce"
    ).values

    # ------------------------------------------------------------
    # Ridge
    # ------------------------------------------------------------

    ridge = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            Ridge(alpha=1.0)
        )
    ])

    ridge.fit(
        X,
        y
    )

    ridge_model = (
        ridge.named_steps["model"]
    )

    ridge_importance = np.abs(
        ridge_model.coef_
    )

    for feature, importance in zip(
        BUSCO_FEATURES,
        ridge_importance
    ):

        feature_importance_results.append({
            "Phenotype": phenotype,
            "Model": "Ridge",
            "BUSCO": feature,
            "Importance": float(
                importance
            )
        })

    # ------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------

    rf = RandomForestRegressor(
        n_estimators=500,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        max_features="sqrt"
    )

    rf.fit(
        X,
        y
    )

    for feature, importance in zip(
        BUSCO_FEATURES,
        rf.feature_importances_
    ):

        feature_importance_results.append({
            "Phenotype": phenotype,
            "Model": "RandomForest",
            "BUSCO": feature,
            "Importance": float(
                importance
            )
        })

    # ------------------------------------------------------------
    # Gradient Boosting
    # ------------------------------------------------------------

    gb = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=RANDOM_SEED
    )

    gb.fit(
        X,
        y
    )

    for feature, importance in zip(
        BUSCO_FEATURES,
        gb.feature_importances_
    ):

        feature_importance_results.append({
            "Phenotype": phenotype,
            "Model": "GradientBoosting",
            "BUSCO": feature,
            "Importance": float(
                importance
            )
        })

importance_df = pd.DataFrame(
    feature_importance_results
)

importance_file = (
    OUTPUT_DIR /
    "full_dataset_busco_feature_importance_420.csv"
)

importance_df.to_csv(
    importance_file,
    index=False
)

print()
print(
    "✓ Full-dataset feature importance written:"
)

print(
    importance_file
)

# ================================================================
# TOP BUSCO FEATURES
# ================================================================

banner(
    "TOP BUSCO FEATURES BY MODEL "
    "AND PHENOTYPE"
)

top_rows = []

for (
    phenotype,
    model
), group in importance_df.groupby(
    [
        "Phenotype",
        "Model"
    ]
):

    top = (
        group
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(50)
        .copy()
    )

    top["Rank"] = np.arange(
        1,
        len(top) + 1
    )

    top_rows.append(
        top
    )

top_features = pd.concat(
    top_rows,
    ignore_index=True
)

top_features_file = (
    OUTPUT_DIR /
    "top_50_busco_features_by_model_phenotype_420.csv"
)

top_features.to_csv(
    top_features_file,
    index=False
)

print(
    "✓ Top BUSCO feature table written:"
)

print(
    top_features_file
)

# ================================================================
# REPEATED IMPORTANCE
# ================================================================

banner(
    "IDENTIFYING REPEATEDLY IMPORTANT BUSCOS"
)

repeated_rows = []

for phenotype in PHENOTYPE_VARIABLES:

    sub = importance_df[
        importance_df[
            "Phenotype"
        ] == phenotype
    ]

    for model in sub[
        "Model"
    ].unique():

        model_sub = sub[
            sub["Model"] == model
        ]

        threshold = (
            model_sub[
                "Importance"
            ].quantile(0.95)
        )

        important = (
            model_sub[
                model_sub[
                    "Importance"
                ] >= threshold
            ][
                [
                    "BUSCO",
                    "Importance"
                ]
            ]
            .copy()
        )

        important[
            "Phenotype"
        ] = phenotype

        important[
            "Model"
        ] = model

        repeated_rows.append(
            important
        )

repeated_df = pd.concat(
    repeated_rows,
    ignore_index=True
)

repeat_counts = (
    repeated_df
    .groupby("BUSCO")
    .agg(
        N_Model_Phenotypes=(
            "Model",
            "count"
        ),
        Mean_Importance=(
            "Importance",
            "mean"
        ),
        Max_Importance=(
            "Importance",
            "max"
        )
    )
    .reset_index()
    .sort_values(
        [
            "N_Model_Phenotypes",
            "Mean_Importance"
        ],
        ascending=False
    )
)

repeated_file = (
    OUTPUT_DIR /
    "repeatedly_important_buscos_420.csv"
)

repeat_counts.to_csv(
    repeated_file,
    index=False
)

print(
    "✓ Repeatedly important BUSCO "
    "table written:"
)

print(
    repeated_file
)

# ================================================================
# BUSCO ↔ PHENOTYPE ASSOCIATIONS
# ================================================================

banner(
    "BUSCO ↔ PHENOTYPE ASSOCIATION ANALYSIS"
)

association_rows = []

# Vectorized rank-correlation approach.
# This is considerably faster than calling .corr()
# separately for every BUSCO.

busco_numeric = busco.astype(
    float
)

for phenotype in PHENOTYPE_VARIABLES:

    y = pd.to_numeric(
        df[phenotype],
        errors="coerce"
    )

    y_rank = y.rank(
        method="average"
    )

    for busco_id in BUSCO_FEATURES:

        x = busco_numeric[
            busco_id
        ]

        if x.nunique() < 2:

            rho = np.nan

        else:

            rho = x.corr(
                y_rank,
                method="pearson"
            )

        association_rows.append({
            "Phenotype": phenotype,
            "BUSCO": busco_id,
            "Spearman_rho": rho,
            "Absolute_Spearman_rho": (
                abs(rho)
                if pd.notna(rho)
                else np.nan
            )
        })

association_df = pd.DataFrame(
    association_rows
)

association_file = (
    OUTPUT_DIR /
    "important_busco_phenotype_associations_420.csv"
)

association_df.to_csv(
    association_file,
    index=False
)

print(
    "✓ BUSCO ↔ phenotype table written:"
)

print(
    association_file
)

# ================================================================
# INTEGRATED BUSCO PRIORITY TABLE
# ================================================================

banner(
    "CREATING INTEGRATED BUSCO PRIORITY TABLE"
)

mean_importance = (
    importance_df
    .groupby(
        [
            "Phenotype",
            "BUSCO"
        ]
    )[
        "Importance"
    ]
    .mean()
    .reset_index()
    .rename(
        columns={
            "Importance":
            "Mean_Model_Importance"
        }
    )
)

integrated = (
    mean_importance
    .merge(
        association_df,
        on=[
            "Phenotype",
            "BUSCO"
        ],
        how="left"
    )
)

integrated[
    "Importance_Rank"
] = (
    integrated
    .groupby(
        "Phenotype"
    )[
        "Mean_Model_Importance"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

integrated[
    "Association_Rank"
] = (
    integrated
    .groupby(
        "Phenotype"
    )[
        "Absolute_Spearman_rho"
    ]
    .rank(
        ascending=False,
        method="min"
    )
)

integrated_file = (
    OUTPUT_DIR /
    "integrated_busco_importance_phenotype_analysis_420.csv"
)

integrated.to_csv(
    integrated_file,
    index=False
)

print(
    "✓ Integrated BUSCO analysis written:"
)

print(
    integrated_file
)

# ================================================================
# PHYLOGENETIC TREE
# ================================================================

banner(
    "PHYLOGENETIC CONCENTRATION OF IMPORTANT BUSCOS"
)

print("Tree file:")
print(ACTIVE_TREE_FILE)

tree = Phylo.read(
    str(ACTIVE_TREE_FILE),
    "newick"
)

terminal_clades = (
    tree.get_terminals()
)

print(
    f"ASTRAL tree tips detected: "
    f"{len(terminal_clades)}"
)

if len(terminal_clades) != EXPECTED_TAXA:

    raise RuntimeError(
        "\n"
        "The supplied validated ASTRAL tree "
        "does not contain 420 tips.\n"
        f"Expected: {EXPECTED_TAXA}\n"
        f"Detected: {len(terminal_clades)}\n"
        f"Tree: {ACTIVE_TREE_FILE}"
    )

print(
    "✓ Correct 420-taxon ASTRAL tree loaded"
)

# ================================================================
# TREE ACCESSION MAP
# ================================================================

tree_tip_map = {}

for clade in terminal_clades:

    label = clean_string(
        clade.name
    )

    accession = accession_from_tip(
        label
    )

    if accession:
        tree_tip_map[
            accession
        ] = label

if len(tree_tip_map) != EXPECTED_TAXA:

    raise RuntimeError(
        "Could not construct a unique "
        "420-accession tree mapping."
    )

matrix_accessions = set(
    df[
        "Assembly_Accession"
    ]
    .astype(str)
    .str.strip()
)

tree_accessions = set(
    tree_tip_map.keys()
)

shared = (
    matrix_accessions
    & tree_accessions
)

if len(shared) != EXPECTED_TAXA:

    raise RuntimeError(
        "Tree/matrix accession mismatch."
    )

print(
    "✓ Exact 420/420 accession "
    "correspondence confirmed"
)

# ================================================================
# IMPORTANT BUSCO SET
# ================================================================

important_buscos = set()

for phenotype in PHENOTYPE_VARIABLES:

    sub = importance_df[
        importance_df[
            "Phenotype"
        ] == phenotype
    ]

    for model in sub[
        "Model"
    ].unique():

        model_sub = sub[
            sub["Model"] == model
        ]

        cutoff = (
            model_sub[
                "Importance"
            ].quantile(0.95)
        )

        important = (
            model_sub[
                model_sub[
                    "Importance"
                ] >= cutoff
            ][
                "BUSCO"
            ]
        )

        important_buscos.update(
            important.tolist()
        )

print(
    f"Unique important BUSCOs selected: "
    f"{len(important_buscos)}"
)

# ================================================================
# OPTIMIZED PHYLOGENETIC DISTANCE MATRIX
# ================================================================

banner(
    "BUILDING 420 × 420 PHYLOGENETIC "
    "DISTANCE MATRIX"
)

"""
IMPORTANT OPTIMIZATION

The old script performed:

    tree.distance(A, B)

again and again for every BUSCO.

This means the same tree distances were repeatedly
recalculated.

Instead:

1. Build parent relationships once.
2. Calculate root-to-node depths once.
3. Build ancestor paths once.
4. Calculate every terminal pair distance once.
5. Store the result in a 420 × 420 NumPy matrix.

After this, every BUSCO uses simple NumPy indexing.
"""

# ------------------------------------------------
# Parent map
# ------------------------------------------------

parent_map = {}

for parent in tree.find_clades(
    order="level"
):

    for child in parent.clades:

        parent_map[
            id(child)
        ] = parent

# ------------------------------------------------
# Root-to-node distances
# ------------------------------------------------

root = tree.root

depth_map = {
    id(root): 0.0
}

for clade in tree.find_clades(
    order="level"
):

    parent_depth = depth_map.get(
        id(clade),
        0.0
    )

    for child in clade.clades:

        branch_length = (
            child.branch_length
            if child.branch_length
            is not None
            else 0.0
        )

        depth_map[
            id(child)
        ] = (
            parent_depth
            + float(branch_length)
        )

# ------------------------------------------------
# Terminal ordering
# ------------------------------------------------

terminal_labels = [
    clean_string(
        clade.name
    )
    for clade in terminal_clades
]

terminal_accessions = [
    accession_from_tip(label)
    for label in terminal_labels
]

if len(
    set(terminal_accessions)
) != EXPECTED_TAXA:

    raise RuntimeError(
        "Terminal accessions are not unique."
    )

accession_to_index = {
    acc: i
    for i, acc in enumerate(
        terminal_accessions
    )
}

# ------------------------------------------------
# Ancestor paths
# ------------------------------------------------

ancestor_paths = {}

for terminal in terminal_clades:

    path = []

    current = terminal

    while current is not None:

        path.append(
            current
        )

        if current is root:
            break

        current = parent_map.get(
            id(current)
        )

    ancestor_paths[
        id(terminal)
    ] = path

# ------------------------------------------------
# Pairwise patristic distance matrix
# ------------------------------------------------

n_taxa = len(
    terminal_clades
)

distance_matrix = np.zeros(
    (
        n_taxa,
        n_taxa
    ),
    dtype=np.float64
)

print(
    f"Calculating {n_taxa} × {n_taxa} "
    "distance matrix..."
)

for i in range(n_taxa):

    terminal_i = terminal_clades[i]

    path_i = ancestor_paths[
        id(terminal_i)
    ]

    ancestors_i = {
        id(node): node
        for node in path_i
    }

    depth_i = depth_map[
        id(terminal_i)
    ]

    for j in range(
        i + 1,
        n_taxa
    ):

        terminal_j = (
            terminal_clades[j]
        )

        depth_j = depth_map[
            id(terminal_j)
        ]

        # Find first common ancestor
        # from j's terminal toward root.
        lca = None

        for node in ancestor_paths[
            id(terminal_j)
        ]:

            if id(node) in ancestors_i:

                lca = node
                break

        if lca is None:

            raise RuntimeError(
                "Could not identify LCA "
                f"for terminals {i} and {j}."
            )

        lca_depth = depth_map[
            id(lca)
        ]

        d = (
            depth_i
            + depth_j
            - 2.0 * lca_depth
        )

        distance_matrix[
            i,
            j
        ] = d

        distance_matrix[
            j,
            i
        ] = d

print(
    "✓ 420 × 420 phylogenetic "
    "distance matrix calculated"
)

# ================================================================
# ACCESSION → DISTANCE-MATRIX INDEX
# ================================================================

tree_index_by_accession = {
    acc: i
    for i, acc in enumerate(
        terminal_accessions
    )
}

# ================================================================
# PHYLOGENETIC CONCENTRATION
# ================================================================

banner(
    "CALCULATING PHYLOGENETIC "
    "CONCENTRATION OF IMPORTANT BUSCOS"
)

phylo_rows = []

for counter, busco_id in enumerate(
    sorted(important_buscos),
    start=1
):

    if busco_id not in df.columns:
        continue

    # ------------------------------------------------------------
    # Presence taxa
    # ------------------------------------------------------------

    presence_mask = (
        busco[
            busco_id
        ].to_numpy()
        == 1
    )

    presence_accessions = (
        df.loc[
            presence_mask,
            "Assembly_Accession"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    presence_accessions = [
        acc
        for acc in presence_accessions
        if acc in tree_index_by_accession
    ]

    n_present = len(
        presence_accessions
    )

    if n_present == 0:
        continue

    # ------------------------------------------------------------
    # Distance calculation
    # ------------------------------------------------------------

    indices = np.array(
        [
            tree_index_by_accession[
                acc
            ]
            for acc in presence_accessions
        ],
        dtype=int
    )

    if len(indices) >= 2:

        submatrix = (
            distance_matrix[
                np.ix_(
                    indices,
                    indices
                )
            ]
        )

        upper_values = (
            submatrix[
                np.triu_indices(
                    len(indices),
                    k=1
                )
            ]
        )

        upper_values = (
            upper_values[
                np.isfinite(
                    upper_values
                )
            ]
        )

        if len(upper_values) > 0:

            mean_distance = float(
                np.mean(
                    upper_values
                )
            )

            median_distance = float(
                np.median(
                    upper_values
                )
            )

            max_distance = float(
                np.max(
                    upper_values
                )
            )

        else:

            mean_distance = np.nan
            median_distance = np.nan
            max_distance = np.nan

    else:

        mean_distance = np.nan
        median_distance = np.nan
        max_distance = np.nan

    prevalence = (
        n_present
        / EXPECTED_TAXA
    )

    phylo_rows.append({

        "BUSCO": busco_id,

        "N_Present_Taxa":
            n_present,

        "Prevalence":
            prevalence,

        "Mean_Pairwise_Tree_Distance":
            mean_distance,

        "Median_Pairwise_Tree_Distance":
            median_distance,

        "Maximum_Pairwise_Tree_Distance":
            max_distance
    })

    if counter % 25 == 0:

        print(
            f"Processed "
            f"{counter}/"
            f"{len(important_buscos)} "
            f"important BUSCOs..."
        )

# ================================================================
# SAVE PHYLOGENETIC CONCENTRATION
# ================================================================

phylo_concentration = pd.DataFrame(
    phylo_rows
)

if not phylo_concentration.empty:

    phylo_concentration = (
        phylo_concentration
        .sort_values(
            [
                "Mean_Pairwise_Tree_Distance",
                "Median_Pairwise_Tree_Distance"
            ],
            ascending=True
        )
    )

phylo_file = (
    OUTPUT_DIR /
    "important_busco_phylogenetic_concentration_420.csv"
)

phylo_concentration.to_csv(
    phylo_file,
    index=False
)

print()
print(
    "✓ Phylogenetic concentration "
    "table written:"
)

print(
    phylo_file
)

# ================================================================
# TOP PHYLOGENETICALLY CONCENTRATED BUSCOS
# ================================================================

if not phylo_concentration.empty:

    top_phylo = (
        phylo_concentration
        .head(100)
        .copy()
    )

else:

    top_phylo = (
        phylo_concentration
    )

top_phylo_file = (
    OUTPUT_DIR /
    "top_phylogenetically_concentrated_buscos_420.csv"
)

top_phylo.to_csv(
    top_phylo_file,
    index=False
)

print(
    "✓ Top phylogenetically "
    "concentrated BUSCOs written:"
)

print(
    top_phylo_file
)

# ================================================================
# ADD NORMALIZED PHYLOGENETIC CONCENTRATION
# ================================================================

banner(
    "CALCULATING NORMALIZED PHYLOGENETIC "
    "CONCENTRATION"
)

if not phylo_concentration.empty:

    # Global mean pairwise distance
    # among all 420 taxa.

    global_upper = (
        distance_matrix[
            np.triu_indices(
                EXPECTED_TAXA,
                k=1
            )
        ]
    )

    global_upper = (
        global_upper[
            np.isfinite(
                global_upper
            )
        ]
    )

    global_mean_distance = float(
        np.mean(
            global_upper
        )
    )

    phylo_concentration[
        "Global_Mean_Tree_Distance"
    ] = global_mean_distance

    phylo_concentration[
        "Normalized_Mean_Distance"
    ] = (
        phylo_concentration[
            "Mean_Pairwise_Tree_Distance"
        ]
        / global_mean_distance
    )

    phylo_concentration[
        "Phylogenetic_Concentration_Index"
    ] = (
        1.0
        -
        phylo_concentration[
            "Normalized_Mean_Distance"
        ]
    )

    phylo_concentration.to_csv(
        phylo_file,
        index=False
    )

    top_phylo = (
        phylo_concentration
        .sort_values(
            "Phylogenetic_Concentration_Index",
            ascending=False
        )
        .head(100)
        .copy()
    )

    top_phylo.to_csv(
        top_phylo_file,
        index=False
    )

    print(
        "✓ Normalized phylogenetic "
        "concentration calculated"
    )

# ================================================================
# FINAL SUMMARY
# ================================================================

banner(
    "CREATING FINAL ANALYSIS SUMMARY"
)

summary = {

    "Taxa":
        EXPECTED_TAXA,

    "Total_Matrix_Columns":
        df.shape[1],

    "BUSCO_Features":
        len(BUSCO_FEATURES),

    "Phenotype_Variables":
        len(PHENOTYPE_VARIABLES),

    "Missing_BUSCO_Cells":
        missing_busco,

    "Tree_Tips":
        len(terminal_clades),

    "Shared_Accessions":
        len(shared),

    "Unique_Important_BUSCOs":
        len(important_buscos),

    "Phylogenetic_Distance_Matrix":
        "420x420",

    "Bootstrap_Iterations":
        BOOTSTRAP_ITERATIONS,

    "Random_Seed":
        RANDOM_SEED
}

summary_df = pd.DataFrame(
    [summary]
)

summary_file = (
    OUTPUT_DIR /
    "feature_importance_analysis_summary_420.csv"
)

summary_df.to_csv(
    summary_file,
    index=False
)

# ================================================================
# FINAL OUTPUT
# ================================================================

banner(
    "OPTIMIZED FEATURE IMPORTANCE + "
    "PHYLOGENETIC ANALYSIS COMPLETE"
)

print(
    f"Final taxa:                  "
    f"{EXPECTED_TAXA}"
)

print(
    f"BUSCO predictors:            "
    f"{EXPECTED_BUSCOS}"
)

print(
    f"Phenotypes analyzed:         "
    f"{len(PHENOTYPE_VARIABLES)}"
)

print(
    f"ASTRAL tree tips:             "
    f"{len(terminal_clades)}"
)

print(
    f"Important BUSCOs identified: "
    f"{len(important_buscos)}"
)

print()
print("OUTPUT FILES:")

print()
print(
    "Performance mean ± SD + bootstrap:"
)

print(
    performance_summary_file
)

print()
print(
    "Fold performance distributions:"
)

print(
    distribution_file
)

print()
print(
    "Full BUSCO feature importance:"
)

print(
    importance_file
)

print()
print(
    "Top BUSCO features:"
)

print(
    top_features_file
)

print()
print(
    "Repeatedly important BUSCOs:"
)

print(
    repeated_file
)

print()
print(
    "BUSCO ↔ phenotype associations:"
)

print(
    association_file
)

print()
print(
    "Integrated BUSCO analysis:"
)

print(
    integrated_file
)

print()
print(
    "Phylogenetic concentration:"
)

print(
    phylo_file
)

print()
print(
    "Top phylogenetically concentrated BUSCOs:"
)

print(
    top_phylo_file
)

print()
print(
    "Overall analysis summary:"
)

print(
    summary_file
)

print()
print(
    "✓ Original phylogeny-aware ML matrix "
    "was NOT modified."
)

print(
    "✓ Original ASTRAL tree "
    "was NOT modified."
)

print(
    "✓ Exactly 1988 BUSCO predictors "
    "were analyzed."
)

print(
    "✓ Exactly 420 taxa "
    "were analyzed."
)

print(
    "✓ No ete3 dependency."
)

print(
    "✓ Phylogenetic distances were "
    "calculated once and reused."
)

print(
    "✓ Optimized phylogenetic "
    "concentration analysis completed."
)

print("=" * 80)