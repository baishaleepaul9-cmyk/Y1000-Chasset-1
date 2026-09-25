# ================================================================
# CANDIDATE YEAST CHASSIS IDENTIFICATION
# Y1000+ | VALIDATED PHYLOGENY-AWARE ML + PARETO + FUNCTIONAL EVIDENCE
# ================================================================

import os
import re
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ================================================================
# 1. PATHS
# ================================================================

BASE = r"C:\Y1000_chassis_project"

PARETO_FILE = os.path.join(
    BASE,
    r"results\stage2B_pareto_analysis\pareto_frontier.csv"
)

ML_DIR = os.path.join(
    BASE,
    r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training"
)

FEATURE_DIR = os.path.join(
    ML_DIR,
    r"feature_interpretation_420"
)

FUNCTIONAL_DIR = os.path.join(
    FEATURE_DIR,
    r"functional_annotation_420\BUSCO20_annotation"
)

BIO_DIR = os.path.join(
    FUNCTIONAL_DIR,
    r"FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V3"
)

OUTPUT_DIR = os.path.join(
    BASE,
    r"results\stage6_candidate_yeast_chassis_420"
)

TABLE_DIR = os.path.join(OUTPUT_DIR, "tables")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ================================================================
# 2. HELPERS
# ================================================================

def clean_id(x):
    """
    Standardize BUSCO IDs so that numeric/string mismatches
    cannot break merges.
    """
    if pd.isna(x):
        return ""

    s = str(x).strip()

    # Remove accidental .0 from numeric conversion
    if s.endswith(".0"):
        s = s[:-2]

    return s


def find_file(root, filename):
    matches = glob.glob(
        os.path.join(root, "**", filename),
        recursive=True
    )

    if matches:
        return matches[0]

    return None


def find_feature_file():
    """
    Locate the already-generated validated ML feature-importance
    table. No model is retrained here.
    """

    preferred = [
        os.path.join(
            FEATURE_DIR,
            "tables",
            "feature_importance_stability_420.csv"
        ),
        os.path.join(
            FEATURE_DIR,
            "feature_importance_stability_420.csv"
        )
    ]

    for p in preferred:
        if os.path.exists(p):
            return p

    candidates = glob.glob(
        os.path.join(
            FEATURE_DIR,
            "**",
            "*feature*importance*.csv"
        ),
        recursive=True
    )

    if candidates:
        return candidates[0]

    return None


def safe_numeric(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce"
            )

    return df


# ================================================================
# 3. HEADER
# ================================================================

print("=" * 80)
print("CANDIDATE YEAST CHASSIS IDENTIFICATION")
print("Y1000+ | VALIDATED PHYLOGENY-AWARE ML + PARETO")
print("=" * 80)

# ================================================================
# 4. LOAD PARETO
# ================================================================

print("\n" + "=" * 80)
print("LOADING PARETO FRONTIER")
print("=" * 80)

if not os.path.exists(PARETO_FILE):
    raise FileNotFoundError(
        f"\nPareto frontier not found:\n{PARETO_FILE}"
    )

pareto = pd.read_csv(PARETO_FILE)

print(f"Pareto rows    : {len(pareto)}")
print(f"Pareto columns : {list(pareto.columns)}")

if "Species" not in pareto.columns:
    raise ValueError(
        "Pareto file does not contain the required Species column."
    )

pareto["Species"] = pareto["Species"].astype(str).str.strip()

# Numeric columns
pareto = safe_numeric(
    pareto,
    [
        "N_Strains",
        "Carbon_Breadth",
        "Nitrogen_Breadth",
        "Utilized_Median_Growth",
        "Carbon_Breadth_SD",
        "Nitrogen_Breadth_SD",
        "Pareto_Optimal",
        "Number_of_Dominating_Species",
        "Pareto_Rank"
    ]
)

# ================================================================
# 5. PARETO CANDIDATES
# ================================================================

if "Pareto_Optimal" in pareto.columns:

    pareto["Pareto_Status"] = np.where(
        pareto["Pareto_Optimal"] == 1,
        "Pareto_Optimal",
        "Non_Pareto"
    )

else:
    pareto["Pareto_Status"] = "Unknown"


pareto_candidates = pareto[
    pareto["Pareto_Status"] == "Pareto_Optimal"
].copy()

print("\nPareto-optimal candidates:",
      len(pareto_candidates))

print(
    pareto_candidates[
        [
            c for c in [
                "Species",
                "N_Strains",
                "Carbon_Breadth",
                "Nitrogen_Breadth",
                "Utilized_Median_Growth",
                "Pareto_Rank"
            ]
            if c in pareto_candidates.columns
        ]
    ].to_string(index=False)
)

# ================================================================
# 6. LOAD VALIDATED ML FEATURE IMPORTANCE
# ================================================================

print("\n" + "=" * 80)
print("LOADING VALIDATED ML FEATURE EVIDENCE")
print("=" * 80)

ML_FILE = find_feature_file()

if ML_FILE is None:
    raise FileNotFoundError(
        "\nCould not locate the existing validated ML "
        "feature-importance output."
    )

print("ML feature file:")
print(ML_FILE)

ml = pd.read_csv(ML_FILE)

print(f"\nML rows    : {len(ml)}")
print(f"ML columns : {list(ml.columns)}")

required_ml = [
    "Phenotype",
    "Feature",
    "Mean_Importance",
    "Fold_Stability",
    "Importance_Rank"
]

missing_ml = [
    c for c in required_ml
    if c not in ml.columns
]

if missing_ml:
    raise ValueError(
        f"Missing required ML columns: {missing_ml}"
    )

ml["Feature"] = ml["Feature"].astype(str).str.strip()

# ================================================================
# 7. VALIDATED / STABLE ML FEATURES
# ================================================================

print("\n" + "=" * 80)
print("IDENTIFYING STABLE ML FEATURES")
print("=" * 80)

# We do NOT retrain anything.
# We use the already calculated feature stability.

ml["Mean_Importance"] = pd.to_numeric(
    ml["Mean_Importance"],
    errors="coerce"
)

ml["Fold_Stability"] = pd.to_numeric(
    ml["Fold_Stability"],
    errors="coerce"
)

ml["Importance_Rank"] = pd.to_numeric(
    ml["Importance_Rank"],
    errors="coerce"
)

# Existing validated feature evidence
stable_ml = ml[
    (ml["Fold_Stability"] >= 0.60) &
    (ml["Mean_Importance"] > 0)
].copy()

print(
    f"Stable positive ML feature records: {len(stable_ml)}"
)

print(
    f"Unique stable ML features: "
    f"{stable_ml['Feature'].nunique()}"
)

# ================================================================
# 8. LOAD FINAL BUSCO BIOLOGICAL SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("LOADING BUSCO BIOLOGICAL EVIDENCE")
print("=" * 80)

busco_file = os.path.join(
    BIO_DIR,
    "tables",
    "Carbon_Breadth_final_BUSCO_biological_summary_420_v3.csv"
)

if not os.path.exists(busco_file):

    # fallback search
    busco_file = find_file(
        BIO_DIR,
        "Carbon_Breadth_final_BUSCO_biological_summary_420_v3.csv"
    )

if busco_file is None:
    raise FileNotFoundError(
        "\nBUSCO biological summary not found."
    )

busco = pd.read_csv(busco_file)

print(f"BUSCO rows    : {len(busco)}")
print(f"BUSCO columns : {list(busco.columns)}")

if "BUSCO_ID" not in busco.columns:
    raise ValueError(
        "BUSCO biological summary does not contain BUSCO_ID."
    )

busco["BUSCO_ID"] = busco["BUSCO_ID"].apply(clean_id)

# ================================================================
# 9. LOAD NORMALIZED PATHWAY SUMMARY
# ================================================================

pathway_file = os.path.join(
    BIO_DIR,
    "tables",
    "Carbon_Breadth_final_normalized_pathway_summary_420_v3.csv"
)

if os.path.exists(pathway_file):

    pathways = pd.read_csv(pathway_file)

    print(
        f"Normalized pathway rows: {len(pathways)}"
    )

else:

    pathways = pd.DataFrame()

    print(
        "Normalized pathway summary not found; "
        "continuing with BUSCO biological summary."
    )

# ================================================================
# 10. MAP ML FEATURES → BUSCO
# ================================================================

print("\n" + "=" * 80)
print("MAPPING ML FEATURES TO BUSCO EVIDENCE")
print("=" * 80)

busco_ids = set(busco["BUSCO_ID"])

stable_ml["BUSCO_ID"] = stable_ml["Feature"].apply(
    clean_id
)

stable_ml["BUSCO_Mapped"] = stable_ml["BUSCO_ID"].isin(
    busco_ids
)

mapped_ml = stable_ml[
    stable_ml["BUSCO_Mapped"]
].copy()

print(
    f"Stable ML records mapped to BUSCOs: "
    f"{len(mapped_ml)}"
)

print(
    f"Unique mapped BUSCOs: "
    f"{mapped_ml['BUSCO_ID'].nunique()}"
)

# ================================================================
# 11. BUSCO EVIDENCE SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("BUILDING BUSCO EVIDENCE SUMMARY")
print("=" * 80)

if len(mapped_ml) > 0:

    busco_ml = (
        mapped_ml
        .groupby("BUSCO_ID")
        .agg(
            ML_Phenotypes=(
                "Phenotype",
                lambda x: "; ".join(
                    sorted(set(map(str, x)))
                )
            ),
            ML_Feature_Count=(
                "Feature",
                "count"
            ),
            ML_Max_Importance=(
                "Mean_Importance",
                "max"
            ),
            ML_Mean_Importance=(
                "Mean_Importance",
                "mean"
            ),
            ML_Max_Fold_Stability=(
                "Fold_Stability",
                "max"
            ),
            ML_Best_Rank=(
                "Importance_Rank",
                "min"
            )
        )
        .reset_index()
    )

else:

    busco_ml = pd.DataFrame(
        columns=[
            "BUSCO_ID",
            "ML_Phenotypes",
            "ML_Feature_Count",
            "ML_Max_Importance",
            "ML_Mean_Importance",
            "ML_Max_Fold_Stability",
            "ML_Best_Rank"
        ]
    )

# ================================================================
# 12. MERGE BUSCO BIOLOGICAL INFORMATION
# ================================================================

busco_evidence = busco.merge(
    busco_ml,
    on="BUSCO_ID",
    how="left"
)

# Fill missing ML evidence
for c in [
    "ML_Feature_Count",
    "ML_Max_Importance",
    "ML_Mean_Importance",
    "ML_Max_Fold_Stability",
    "ML_Best_Rank"
]:
    if c in busco_evidence.columns:
        busco_evidence[c] = busco_evidence[c].fillna(0)

if "ML_Phenotypes" in busco_evidence.columns:
    busco_evidence["ML_Phenotypes"] = (
        busco_evidence["ML_Phenotypes"]
        .fillna("")
    )

# ================================================================
# 13. BIOLOGICAL EVIDENCE FLAGS
# ================================================================

busco_evidence["ML_Supported"] = (
    busco_evidence["ML_Feature_Count"] > 0
)

if "Pathway_Mapped" in busco_evidence.columns:

    busco_evidence["Pathway_Supported"] = (
        busco_evidence["Pathway_Mapped"]
        .astype(str)
        .str.lower()
        .isin(["true", "yes", "1"])
    )

else:

    busco_evidence["Pathway_Supported"] = False


if "Normalized_Pathway_Count" in busco_evidence.columns:

    busco_evidence["Pathway_Count"] = pd.to_numeric(
        busco_evidence["Normalized_Pathway_Count"],
        errors="coerce"
    ).fillna(0)

else:

    busco_evidence["Pathway_Count"] = 0


busco_evidence["Integrated_Genomic_Evidence"] = (
    busco_evidence["ML_Supported"].astype(int)
    +
    busco_evidence["Pathway_Supported"].astype(int)
)

# ================================================================
# 14. SAVE BUSCO EVIDENCE
# ================================================================

busco_output = os.path.join(
    TABLE_DIR,
    "candidate_chassis_BUSCO_evidence_420.csv"
)

busco_evidence.to_csv(
    busco_output,
    index=False
)

print(
    f"Saved BUSCO evidence:\n{busco_output}"
)

# ================================================================
# 15. CONNECT GENOMIC EVIDENCE TO PARETO SPECIES
# ================================================================

print("\n" + "=" * 80)
print("INTEGRATING PARETO + GENOMIC EVIDENCE")
print("=" * 80)

# ------------------------------------------------
# Important:
# BUSCO/ML evidence is gene-family evidence.
# It cannot automatically be assigned to a species
# unless the species-to-feature mapping exists.
#
# Therefore we explicitly preserve two evidence
# layers:
#
# 1. Species-level Pareto evidence
# 2. Genomic feature-level evidence
# ------------------------------------------------

pareto_output = pareto.copy()

# Overall Pareto evidence count
pareto_output["Pareto_Evidence"] = np.where(
    pareto_output["Pareto_Status"] ==
    "Pareto_Optimal",
    1,
    0
)

# ================================================================
# 16. SPECIES-LEVEL CHASSIS EVIDENCE
# ================================================================

# Normalize trait values
for c in [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD"
]:
    if c in pareto_output.columns:
        pareto_output[c] = pd.to_numeric(
            pareto_output[c],
            errors="coerce"
        )

# ================================================================
# 17. PARETO-BASED EVIDENCE SCORE
# ================================================================

pareto_output["Chassis_Pareto_Score"] = 0.0

# Pareto membership is the primary chassis-level criterion
pareto_output.loc[
    pareto_output["Pareto_Status"] ==
    "Pareto_Optimal",
    "Chassis_Pareto_Score"
] = 1.0

# ================================================================
# 18. TRAIT PROFILE
# ================================================================

def percentile_score(series):
    if series.notna().sum() <= 1:
        return pd.Series(
            np.ones(len(series)),
            index=series.index
        )

    return series.rank(
        pct=True,
        method="average"
    )


if "Carbon_Breadth" in pareto_output.columns:
    pareto_output["Carbon_Profile_Score"] = (
        percentile_score(
            pareto_output["Carbon_Breadth"]
        )
    )
else:
    pareto_output["Carbon_Profile_Score"] = 0.0


if "Nitrogen_Breadth" in pareto_output.columns:
    pareto_output["Nitrogen_Profile_Score"] = (
        percentile_score(
            pareto_output["Nitrogen_Breadth"]
        )
    )
else:
    pareto_output["Nitrogen_Profile_Score"] = 0.0


if "Utilized_Median_Growth" in pareto_output.columns:
    pareto_output["Growth_Profile_Score"] = (
        percentile_score(
            pareto_output["Utilized_Median_Growth"]
        )
    )
else:
    pareto_output["Growth_Profile_Score"] = 0.0

# ================================================================
# 19. MULTI-TRAIT PROFILE SCORE
# ================================================================

profile_cols = [
    "Carbon_Profile_Score",
    "Nitrogen_Profile_Score",
    "Growth_Profile_Score"
]

pareto_output["Multi_Trait_Profile_Score"] = (
    pareto_output[profile_cols]
    .mean(axis=1)
)

# ================================================================
# 20. CHASSIS CANDIDATE STATUS
# ================================================================

pareto_output["Candidate_Chassis"] = np.where(
    pareto_output["Pareto_Status"] ==
    "Pareto_Optimal",
    "Candidate_Yeast_Chassis",
    "Supporting_Candidate"
)

# ================================================================
# 21. FINAL CHASSIS TABLE
# ================================================================

final_species_cols = [
    "Species",
    "N_Strains",
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth",
    "Carbon_Breadth_SD",
    "Nitrogen_Breadth_SD",
    "Pareto_Status",
    "Pareto_Rank",
    "Number_of_Dominating_Species",
    "Carbon_Profile_Score",
    "Nitrogen_Profile_Score",
    "Growth_Profile_Score",
    "Multi_Trait_Profile_Score",
    "Chassis_Pareto_Score",
    "Candidate_Chassis"
]

final_species_cols = [
    c for c in final_species_cols
    if c in pareto_output.columns
]

final_chassis = pareto_output[
    final_species_cols
].copy()

# Sort Pareto first, then profile
final_chassis = final_chassis.sort_values(
    [
        "Chassis_Pareto_Score",
        "Multi_Trait_Profile_Score"
    ],
    ascending=[False, False]
)

species_output = os.path.join(
    TABLE_DIR,
    "candidate_yeast_chassis_species_420.csv"
)

final_chassis.to_csv(
    species_output,
    index=False
)

# ================================================================
# 22. SAVE PARETO CANDIDATE TABLE
# ================================================================

pareto_candidate_output = os.path.join(
    TABLE_DIR,
    "pareto_optimal_yeast_chassis_candidates_420.csv"
)

pareto_candidates_final = final_chassis[
    final_chassis["Pareto_Status"] ==
    "Pareto_Optimal"
].copy()

pareto_candidates_final.to_csv(
    pareto_candidate_output,
    index=False
)

# ================================================================
# 23. SAVE STABLE ML FEATURES
# ================================================================

ml_output = os.path.join(
    TABLE_DIR,
    "validated_stable_ML_features_420.csv"
)

stable_ml.sort_values(
    [
        "Mean_Importance",
        "Fold_Stability"
    ],
    ascending=[False, False]
).to_csv(
    ml_output,
    index=False
)

# ================================================================
# 24. FINAL REPORT
# ================================================================

report_file = os.path.join(
    REPORT_DIR,
    "candidate_yeast_chassis_420_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("FINAL CANDIDATE YEAST CHASSIS ANALYSIS\n")
    f.write("Y1000+ | VALIDATED PHYLOGENY-AWARE ML + PARETO\n")
    f.write("=" * 80 + "\n\n")

    f.write("PARETO ANALYSIS\n")
    f.write("-" * 80 + "\n")
    f.write(
        f"Total species evaluated : {len(pareto)}\n"
    )
    f.write(
        f"Pareto-optimal species  : "
        f"{len(pareto_candidates_final)}\n"
    )

    f.write("\nPARETO-OPTIMAL CANDIDATES\n")
    f.write("-" * 80 + "\n")

    for _, row in pareto_candidates_final.iterrows():

        f.write(
            f"{row['Species']} | "
            f"Carbon={row.get('Carbon_Breadth', np.nan)} | "
            f"Nitrogen={row.get('Nitrogen_Breadth', np.nan)} | "
            f"Growth={row.get('Utilized_Median_Growth', np.nan)} | "
            f"Pareto Rank={row.get('Pareto_Rank', np.nan)}\n"
        )

    f.write("\n\nVALIDATED ML EVIDENCE\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"Stable positive ML records : "
        f"{len(stable_ml)}\n"
    )

    f.write(
        f"Stable unique features     : "
        f"{stable_ml['Feature'].nunique()}\n"
    )

    f.write(
        f"Stable features mapped to BUSCOs : "
        f"{mapped_ml['BUSCO_ID'].nunique()}\n"
    )

    f.write("\nTOP STABLE ML FEATURES\n")
    f.write("-" * 80 + "\n")

    top_features = (
        stable_ml
        .sort_values(
            [
                "Mean_Importance",
                "Fold_Stability"
            ],
            ascending=[False, False]
        )
        .head(20)
    )

    for _, row in top_features.iterrows():

        f.write(
            f"{row['Feature']} | "
            f"Phenotype={row['Phenotype']} | "
            f"Importance={row['Mean_Importance']:.6f} | "
            f"Stability={row['Fold_Stability']:.2f} | "
            f"Rank={row['Importance_Rank']}\n"
        )

    f.write("\n\nFUNCTIONAL EVIDENCE\n")
    f.write("-" * 80 + "\n")

    f.write(
        f"BUSCO candidates: {len(busco_evidence)}\n"
    )

    f.write(
        f"ML-supported BUSCOs: "
        f"{busco_evidence['ML_Supported'].sum()}\n"
    )

    f.write(
        f"Pathway-supported BUSCOs: "
        f"{busco_evidence['Pathway_Supported'].sum()}\n"
    )

    f.write("\n\nINTERPRETATION\n")
    f.write("-" * 80 + "\n")

    f.write(
        "Candidate chassis status is defined at the species level "
        "primarily from the existing Pareto analysis. The validated "
        "phylogeny-aware ML outputs and downstream BUSCO/pathway "
        "annotations are retained as genomic/functional evidence and "
        "are not used to retrain the model in this step.\n"
    )

    f.write(
        "Pareto-optimal species therefore represent the phenotype-level "
        "candidate chassis set, while stable ML features and functional "
        "annotations provide mechanistic genomic evidence associated "
        "with the modeled industrial traits.\n"
    )

# ================================================================
# 25. FINAL CONSOLE OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("CANDIDATE YEAST CHASSIS ANALYSIS COMPLETE")
print("=" * 80)

print(
    f"\nTotal species evaluated       : {len(pareto)}"
)

print(
    f"Pareto-optimal candidates    : "
    f"{len(pareto_candidates_final)}"
)

print(
    f"Stable ML features           : "
    f"{stable_ml['Feature'].nunique()}"
)

print(
    f"BUSCO candidates              : "
    f"{len(busco_evidence)}"
)

print(
    f"ML-supported BUSCOs           : "
    f"{busco_evidence['ML_Supported'].sum()}"
)

print(
    f"Pathway-supported BUSCOs      : "
    f"{busco_evidence['Pathway_Supported'].sum()}"
)

print("\n" + "=" * 80)
print("PARETO-OPTIMAL CANDIDATE YEAST CHASSIS")
print("=" * 80)

print(
    pareto_candidates_final.to_string(
        index=False
    )
)

print("\n" + "=" * 80)
print("OUTPUT FILES")
print("=" * 80)

print(species_output)
print(pareto_candidate_output)
print(ml_output)
print(busco_output)
print(report_file)

print("\n" + "=" * 80)
print("NEXT PIPELINE STEP:")
print("INDEPENDENT VALIDATION")
print("=" * 80)