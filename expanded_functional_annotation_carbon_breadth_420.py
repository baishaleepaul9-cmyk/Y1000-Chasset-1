# ================================================================
# EXPANDED FUNCTIONAL ANNOTATION — CARBON BREADTH
# Y1000+ PHYLOGENY-AWARE ML PROJECT
#
# Purpose:
#   Expand biological interpretation of the 20 ML-derived
#   Carbon_Breadth BUSCO candidates.
#
# IMPORTANT:
#   - Does NOT modify models
#   - Does NOT modify OOF predictions
#   - Does NOT rerun ML
#   - Does NOT invent pathway assignments
#   - Existing outputs are preserved
#
# Evidence tiers:
#   Tier 1 = ML + pathway / strong curated functional evidence
#   Tier 2 = ML + reliable domain/function evidence
#   Tier 3 = ML + conserved/functionally informative annotation
#   Tier 4 = ML candidate with insufficient external evidence
# ================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json
import re
import time
from collections import Counter

# ================================================================
# PROJECT PATHS
# ================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

FEATURE_ROOT = MODEL_ROOT / "feature_interpretation_420"

FUNCTIONAL_ROOT = (
    FEATURE_ROOT
    / "functional_annotation_420"
)

PATHWAY_ROOT = (
    FUNCTIONAL_ROOT
    / "pathway_mapping_420"
)

EXPANDED_ROOT = (
    FUNCTIONAL_ROOT
    / "expanded_annotation_420"
)

TABLE_DIR = EXPANDED_ROOT / "tables"
FIGURE_DIR = EXPANDED_ROOT / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# INPUT FILES
# ================================================================

STABLE_CANDIDATES = (
    FEATURE_ROOT
    / "tables"
    / "Carbon_Breadth_stable_BUSCO_candidates_420.csv"
)

FUNCTIONAL_ANNOTATION = (
    FUNCTIONAL_ROOT
    / "tables"
    / "Carbon_Breadth_BUSCO_functional_annotation_420.csv"
)

CANDIDATE_SUMMARY = (
    FUNCTIONAL_ROOT
    / "tables"
    / "Carbon_Breadth_BUSCO_candidate_summary_420.csv"
)

EXTERNAL_ANNOTATION = (
    PATHWAY_ROOT
    / "external_annotation_420"
    / "tables"
    / "Carbon_Breadth_external_functional_annotation_420.csv"
)

GO_TABLE = (
    PATHWAY_ROOT
    / "external_annotation_420"
    / "tables"
    / "Carbon_Breadth_GO_annotated_candidates_420.csv"
)

INTERPRO_TABLE = (
    PATHWAY_ROOT
    / "external_annotation_420"
    / "tables"
    / "Carbon_Breadth_InterPro_annotated_candidates_420.csv"
)

PATHWAY_TABLE = (
    PATHWAY_ROOT
    / "external_annotation_420"
    / "tables"
    / "Carbon_Breadth_pathway_supported_candidates_420.csv"
)

INTEGRATED_TABLE = (
    PATHWAY_ROOT
    / "pathway_level_integration_420"
    / "tables"
    / "Carbon_Breadth_pathway_level_integrated_candidates_420.csv"
)


# ================================================================
# HELPERS
# ================================================================

def safe_read(path, required=False):
    path = Path(path)

    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        if required:
            raise RuntimeError(
                f"Could not read required file:\n{path}\n\n{e}"
            )
        return pd.DataFrame()


def clean_string(x):
    if pd.isna(x):
        return ""

    x = str(x).strip()

    if x.lower() in {
        "nan",
        "none",
        "null",
        "na",
        "n/a",
        ""
    }:
        return ""

    return x


def first_existing_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def unique_join(values, sep=" | "):
    out = []

    for x in values:
        x = clean_string(x)

        if x and x not in out:
            out.append(x)

    return sep.join(out)


def yes(value):
    value = clean_string(value).lower()

    return value in {
        "yes",
        "true",
        "1",
        "complete",
        "supported",
        "strong",
        "moderate"
    }


def text_contains(value, patterns):
    text = clean_string(value).lower()

    if not text:
        return False

    for pattern in patterns:
        if re.search(pattern, text):
            return True

    return False


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("EXPANDED FUNCTIONAL ANNOTATION — CARBON BREADTH")
print("=" * 80)

print("\nProject:")
print(PROJECT_ROOT)

print("\nOutput:")
print(EXPANDED_ROOT)


# ================================================================
# LOAD STABLE ML CANDIDATES
# ================================================================

print("\n" + "=" * 80)
print("LOADING STABLE ML CANDIDATES")
print("=" * 80)

stable = safe_read(STABLE_CANDIDATES, required=True)

print("Shape:", stable.shape)
print("Columns:")
print(stable.columns.tolist())

busco_col = first_existing_column(
    stable,
    ["Feature", "BUSCO_ID", "Busco_ID", "BUSCO"]
)

if busco_col is None:
    raise ValueError(
        "Could not identify BUSCO column in stable candidate table."
    )

stable["BUSCO_ID"] = stable[busco_col].astype(str).str.strip()

candidate_ids = sorted(
    stable["BUSCO_ID"]
    .dropna()
    .unique()
    .tolist()
)

print("\nNumber of stable candidates:", len(candidate_ids))

for x in candidate_ids:
    print(x)


# ================================================================
# LOAD PRIMARY BUSCO FUNCTIONAL ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("LOADING PRIMARY BUSCO FUNCTIONAL ANNOTATION")
print("=" * 80)

primary = safe_read(
    FUNCTIONAL_ANNOTATION,
    required=True
)

print("Shape:", primary.shape)

print("\nColumns:")
print(primary.columns.tolist())

primary_busco = first_existing_column(
    primary,
    ["BUSCO_ID", "Feature", "Busco_ID"]
)

if primary_busco is None:
    raise ValueError(
        "Primary annotation table does not contain BUSCO_ID."
    )

primary["BUSCO_ID"] = (
    primary[primary_busco]
    .astype(str)
    .str.strip()
)


# ================================================================
# LOAD EXTERNAL ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("LOADING EXTERNAL FUNCTIONAL ANNOTATION")
print("=" * 80)

external = safe_read(EXTERNAL_ANNOTATION)

if external.empty:
    print("No external annotation table found.")
else:
    print("Shape:", external.shape)
    print("Columns:")
    print(external.columns.tolist())

    if "BUSCO_ID" in external.columns:
        external["BUSCO_ID"] = (
            external["BUSCO_ID"]
            .astype(str)
            .str.strip()
        )


# ================================================================
# LOAD GO / INTERPRO / PATHWAY TABLES
# ================================================================

print("\n" + "=" * 80)
print("LOADING EXTERNAL EVIDENCE TABLES")
print("=" * 80)

go = safe_read(GO_TABLE)
interpro = safe_read(INTERPRO_TABLE)
pathway = safe_read(PATHWAY_TABLE)
integrated = safe_read(INTEGRATED_TABLE)

print("GO rows:", len(go))
print("InterPro rows:", len(interpro))
print("Pathway rows:", len(pathway))
print("Integrated rows:", len(integrated))


def evidence_ids(df):
    if df.empty:
        return set()

    col = first_existing_column(
        df,
        ["BUSCO_ID", "Feature", "Busco_ID"]
    )

    if col is None:
        return set()

    return set(
        df[col]
        .dropna()
        .astype(str)
        .str.strip()
    )


go_ids = evidence_ids(go)
interpro_ids = evidence_ids(interpro)
pathway_ids = evidence_ids(pathway)


# ================================================================
# CREATE ONE RECORD PER BUSCO
# ================================================================

print("\n" + "=" * 80)
print("BUILDING CANDIDATE-LEVEL MASTER TABLE")
print("=" * 80)

records = []

for busco in candidate_ids:

    record = {
        "BUSCO_ID": busco
    }

    # ------------------------------------------------------------
    # ML INFORMATION
    # ------------------------------------------------------------

    s = stable[
        stable["BUSCO_ID"] == busco
    ].copy()

    if not s.empty:

        row = s.iloc[0]

        for col in [
            "Phenotype",
            "Feature_Set",
            "Model",
            "Mean_Importance",
            "SD_Importance",
            "Median_Importance",
            "Min_Importance",
            "Max_Importance",
            "Mean_Baseline_R2",
            "Positive_Importance_Folds",
            "Fold_Stability",
            "Importance_Rank"
        ]:
            if col in s.columns:
                record[col] = row[col]

    # ------------------------------------------------------------
    # PRIMARY FUNCTIONAL ANNOTATION
    # ------------------------------------------------------------

    p = primary[
        primary["BUSCO_ID"] == busco
    ].copy()

    if not p.empty:

        desc_col = first_existing_column(
            p,
            [
                "Functional_Description",
                "Description",
                "Annotation"
            ]
        )

        if desc_col:
            record["Functional_Description"] = unique_join(
                p[desc_col].tolist()
            )
        else:
            record["Functional_Description"] = ""

        for col in [
            "Status",
            "Length",
            "Score",
            "Sequence",
            "OrthoDB_URL"
        ]:
            if col in p.columns:
                record[col] = unique_join(
                    p[col].tolist()
                )

    else:

        record["Functional_Description"] = ""

    # ------------------------------------------------------------
    # EXTERNAL ANNOTATION
    # ------------------------------------------------------------

    e = external[
        external["BUSCO_ID"] == busco
    ].copy() if not external.empty and "BUSCO_ID" in external.columns else pd.DataFrame()

    if not e.empty:

        erow = e.iloc[0]

        for col in [
            "OrthoDB_Query_Success",
            "HTTP_Status",
            "OrthoDB_Name",
            "Functional_Category",
            "GO_Molecular_Function",
            "GO_Biological_Process",
            "GO_Cellular_Component",
            "EC_Number",
            "KEGG_Pathway",
            "InterPro_Domains",
            "Phyletic_Profile",
            "Evolutionary_Rate",
            "External_Annotation_Status",
            "Pathway_Evidence_Status",
            "InterPro_Evidence_Status",
            "Publication_Interpretation_Level",
            "Query_Error"
        ]:
            if col in e.columns:
                record[col] = erow[col]

    # ------------------------------------------------------------
    # EXTERNAL EVIDENCE FLAGS
    # ------------------------------------------------------------

    record["GO_Supported"] = (
        busco in go_ids
        or any(
            clean_string(record.get(c, ""))
            for c in [
                "GO_Molecular_Function",
                "GO_Biological_Process",
                "GO_Cellular_Component"
            ]
        )
    )

    record["InterPro_Supported"] = (
        busco in interpro_ids
        or bool(
            clean_string(
                record.get("InterPro_Domains", "")
            )
        )
    )

    record["Pathway_Supported"] = (
        busco in pathway_ids
        or bool(
            clean_string(
                record.get("KEGG_Pathway", "")
            )
        )
    )

    # ------------------------------------------------------------
    # FUNCTIONAL DESCRIPTION CLASSIFICATION
    # ------------------------------------------------------------

    desc = clean_string(
        record.get(
            "Functional_Description",
            ""
        )
    )

    ext_desc = clean_string(
        record.get(
            "OrthoDB_Name",
            ""
        )
    )

    combined = (
        desc + " | " + ext_desc
    ).lower()

    record["Functional_Evidence_Text"] = (
        desc if desc else ext_desc
    )

    # ------------------------------------------------------------
    # FUNCTIONAL GROUP
    # ------------------------------------------------------------

    if re.search(
        r"helicase|rna helicase|rna processing|methyltransferase|"
        r"ribosomal|snf7|rna|mrna|transcription|translation",
        combined
    ):
        category = "RNA / gene expression"

    elif re.search(
        r"aconitase|metabolic|metabolism|enzyme|dehydratase|"
        r"phosphatase|biosynthesis|porphobilinogen",
        combined
    ):
        category = "Metabolism / enzymatic activity"

    elif re.search(
        r"gpi|membrane|trafficking|vesicle|armadillo|"
        r"protein yip|pdz",
        combined
    ):
        category = "Membrane / protein trafficking"

    elif re.search(
        r"kinesin|motor|cytoskeleton|wd40",
        combined
    ):
        category = "Cytoskeleton / cellular organization"

    elif re.search(
        r"domain|regulatory|smr|sant|myb|pdz",
        combined
    ):
        category = "Protein domain / regulatory"

    else:
        category = "Other / unclear"

    record["Expanded_Functional_Category"] = category

    # ------------------------------------------------------------
    # EVIDENCE TIER
    # ------------------------------------------------------------

    # Tier 1:
    # pathway evidence OR strong curated external evidence
    pathway_text = clean_string(
        record.get("KEGG_Pathway", "")
    )

    publication_level = clean_string(
        record.get(
            "Publication_Interpretation_Level",
            ""
        )
    ).lower()

    if (
        record["Pathway_Supported"]
        and (
            record["GO_Supported"]
            or record["InterPro_Supported"]
            or publication_level in {
                "strong",
                "strong evidence"
            }
        )
    ):
        tier = "Tier 1 — Strong functional/pathway evidence"

    # Tier 2:
    # domain or GO evidence
    elif (
        record["GO_Supported"]
        or record["InterPro_Supported"]
    ):
        tier = "Tier 2 — Functional/domain evidence"

    # Tier 3:
    # informative conserved annotation
    elif desc or ext_desc:
        tier = "Tier 3 — Conserved/functionally informative annotation"

    # Tier 4:
    else:
        tier = "Tier 4 — ML candidate; limited external evidence"

    record["Evidence_Tier"] = tier

    # ------------------------------------------------------------
    # EVIDENCE SUMMARY
    # ------------------------------------------------------------

    evidence_list = []

    if record["GO_Supported"]:
        evidence_list.append("GO")

    if record["InterPro_Supported"]:
        evidence_list.append("InterPro")

    if record["Pathway_Supported"]:
        evidence_list.append("Pathway")

    if desc or ext_desc:
        evidence_list.append("BUSCO/OrthoDB function")

    record["Evidence_Sources"] = (
        "; ".join(evidence_list)
        if evidence_list
        else "ML only"
    )

    records.append(record)


master = pd.DataFrame(records)


# ================================================================
# SAVE MASTER TABLE
# ================================================================

MASTER_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_expanded_functional_annotation_420.csv"
)

master.to_csv(
    MASTER_OUT,
    index=False
)

print("\nMaster expanded annotation:")
print(MASTER_OUT)

print("\nShape:")
print(master.shape)


# ================================================================
# EVIDENCE SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("EVIDENCE SUMMARY")
print("=" * 80)

tier_counts = (
    master["Evidence_Tier"]
    .value_counts()
    .sort_index()
)

print(tier_counts)


summary = {
    "Total_ML_Candidates": int(len(master)),
    "Tier_1_Strong": int(
        (master["Evidence_Tier"]
         == "Tier 1 — Strong functional/pathway evidence")
        .sum()
    ),
    "Tier_2_Functional_Domain": int(
        (master["Evidence_Tier"]
         == "Tier 2 — Functional/domain evidence")
        .sum()
    ),
    "Tier_3_Informative_Annotation": int(
        (master["Evidence_Tier"]
         == "Tier 3 — Conserved/functionally informative annotation")
        .sum()
    ),
    "Tier_4_ML_Only": int(
        (master["Evidence_Tier"]
         == "Tier 4 — ML candidate; limited external evidence")
        .sum()
    ),
    "GO_Supported": int(
        master["GO_Supported"].sum()
    ),
    "InterPro_Supported": int(
        master["InterPro_Supported"].sum()
    ),
    "Pathway_Supported": int(
        master["Pathway_Supported"].sum()
    )
}

SUMMARY_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_expanded_evidence_summary_420.json"
)

with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=4
    )

print("\nSummary:")
print(json.dumps(summary, indent=4))


# ================================================================
# FUNCTIONAL GROUP SUMMARY
# ================================================================

print("\n" + "=" * 80)
print("FUNCTIONAL GROUP SUMMARY")
print("=" * 80)

group_summary = (
    master
    .groupby("Expanded_Functional_Category")
    .agg(
        Candidate_Count=("BUSCO_ID", "count"),
        Mean_Importance=("Mean_Importance", "mean"),
        Mean_Fold_Stability=("Fold_Stability", "mean"),
        GO_Supported=("GO_Supported", "sum"),
        InterPro_Supported=("InterPro_Supported", "sum"),
        Pathway_Supported=("Pathway_Supported", "sum")
    )
    .reset_index()
    .sort_values(
        "Candidate_Count",
        ascending=False
    )
)

print(group_summary.to_string(index=False))

GROUP_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_expanded_functional_group_summary_420.csv"
)

group_summary.to_csv(
    GROUP_OUT,
    index=False
)


# ================================================================
# EVIDENCE-TIER TABLE
# ================================================================

tier_table = master[
    [
        "BUSCO_ID",
        "Mean_Importance",
        "Fold_Stability",
        "Importance_Rank",
        "Functional_Evidence_Text",
        "Expanded_Functional_Category",
        "GO_Supported",
        "InterPro_Supported",
        "Pathway_Supported",
        "Evidence_Sources",
        "Evidence_Tier"
    ]
].copy()

tier_table = tier_table.sort_values(
    [
        "Evidence_Tier",
        "Mean_Importance"
    ],
    ascending=[True, False]
)

TIER_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_candidate_evidence_tiers_420.csv"
)

tier_table.to_csv(
    TIER_OUT,
    index=False
)


# ================================================================
# TOP CANDIDATES WITH FUNCTIONAL INFORMATION
# ================================================================

functional_candidates = master[
    master["Evidence_Tier"].isin([
        "Tier 1 — Strong functional/pathway evidence",
        "Tier 2 — Functional/domain evidence",
        "Tier 3 — Conserved/functionally informative annotation"
    ])
].copy()

functional_candidates = functional_candidates.sort_values(
    "Mean_Importance",
    ascending=False
)

FUNCTIONAL_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_functionally_annotated_candidates_420.csv"
)

functional_candidates.to_csv(
    FUNCTIONAL_OUT,
    index=False
)

print("\n" + "=" * 80)
print("FUNCTIONALLY ANNOTATED CANDIDATES")
print("=" * 80)

if functional_candidates.empty:

    print(
        "No candidate currently has sufficient external "
        "evidence beyond ML."
    )

else:

    display_cols = [
        "BUSCO_ID",
        "Mean_Importance",
        "Fold_Stability",
        "Functional_Evidence_Text",
        "Expanded_Functional_Category",
        "GO_Supported",
        "InterPro_Supported",
        "Pathway_Supported",
        "Evidence_Tier"
    ]

    display_cols = [
        c for c in display_cols
        if c in functional_candidates.columns
    ]

    print(
        functional_candidates[
            display_cols
        ].to_string(index=False)
    )


# ================================================================
# HIGH-IMPORTANCE / HIGH-STABILITY CANDIDATES
# ================================================================

high_confidence = master[
    (
        pd.to_numeric(
            master["Fold_Stability"],
            errors="coerce"
        ).fillna(0) >= 0.8
    )
].copy()

high_confidence = high_confidence.sort_values(
    "Mean_Importance",
    ascending=False
)

HIGH_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_high_stability_candidates_420.csv"
)

high_confidence.to_csv(
    HIGH_OUT,
    index=False
)


# ================================================================
# CANDIDATE PRIORITY TABLE
# ================================================================

# This is NOT a scientific "ranking" of political-style decisions;
# it is simply a reproducible research prioritization based on
# predefined evidence fields.

def priority_label(row):

    importance = pd.to_numeric(
        row.get("Mean_Importance", np.nan),
        errors="coerce"
    )

    stability = pd.to_numeric(
        row.get("Fold_Stability", np.nan),
        errors="coerce"
    )

    if pd.isna(importance):
        importance = 0

    if pd.isna(stability):
        stability = 0

    tier = clean_string(
        row.get("Evidence_Tier", "")
    )

    if tier.startswith("Tier 1"):
        evidence_score = 3

    elif tier.startswith("Tier 2"):
        evidence_score = 2

    elif tier.startswith("Tier 3"):
        evidence_score = 1

    else:
        evidence_score = 0

    if stability >= 0.8 and evidence_score >= 2:
        return "High-priority follow-up"

    elif stability >= 0.8 and evidence_score >= 1:
        return "Follow-up candidate"

    elif evidence_score >= 2:
        return "Functional candidate"

    else:
        return "ML candidate requiring validation"


master["Followup_Category"] = master.apply(
    priority_label,
    axis=1
)

PRIORITY_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_candidate_followup_categories_420.csv"
)

master.sort_values(
    [
        "Followup_Category",
        "Mean_Importance"
    ],
    ascending=[True, False]
).to_csv(
    PRIORITY_OUT,
    index=False
)


# ================================================================
# TEXT REPORT
# ================================================================

REPORT_OUT = (
    TABLE_DIR
    / "Carbon_Breadth_expanded_annotation_report_420.txt"
)

with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "CARBON BREADTH — EXPANDED FUNCTIONAL ANNOTATION\n"
    )
    f.write("=" * 70 + "\n\n")

    f.write(
        f"Total ML candidates: {len(master)}\n\n"
    )

    f.write(
        "Evidence tier counts:\n"
    )

    for tier, count in tier_counts.items():

        f.write(
            f"  {tier}: {count}\n"
        )

    f.write("\n")

    f.write(
        f"GO-supported: {summary['GO_Supported']}\n"
    )

    f.write(
        f"InterPro-supported: "
        f"{summary['InterPro_Supported']}\n"
    )

    f.write(
        f"Pathway-supported: "
        f"{summary['Pathway_Supported']}\n"
    )

    f.write("\n")

    f.write(
        "IMPORTANT INTERPRETATION:\n"
    )

    f.write(
        "The 20 BUSCOs remain ML-derived candidates. "
        "Absence of GO, InterPro, or pathway evidence does "
        "not imply absence of biological function. It indicates "
        "that the current external annotation workflow did not "
        "provide independent corroboration.\n"
    )

    f.write(
        "\nFeature importance represents predictive association "
        "and should not be interpreted as causal evidence.\n"
    )


# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("EXPANDED FUNCTIONAL ANNOTATION COMPLETE")
print("=" * 80)

print("\nTotal candidates:")
print(len(master))

print("\nEvidence tiers:")
print(tier_counts.to_string())

print("\nGO-supported:", summary["GO_Supported"])
print("InterPro-supported:", summary["InterPro_Supported"])
print("Pathway-supported:", summary["Pathway_Supported"])

print("\n" + "-" * 80)
print("OUTPUT FILES")
print("-" * 80)

print("\nMaster annotation:")
print(MASTER_OUT)

print("\nEvidence summary:")
print(SUMMARY_OUT)

print("\nFunctional group summary:")
print(GROUP_OUT)

print("\nEvidence tiers:")
print(TIER_OUT)

print("\nFunctionally annotated candidates:")
print(FUNCTIONAL_OUT)

print("\nHigh-stability candidates:")
print(HIGH_OUT)

print("\nFollow-up categories:")
print(PRIORITY_OUT)

print("\nReport:")
print(REPORT_OUT)

print("\nExisting models and OOF predictions were NOT modified.")

print("\nNEXT SCIENTIFIC STAGE:")
print(
    "Literature/functional verification → "
    "network integration → Pareto/chassis integration → "
    "independent validation"
)

print("=" * 80)