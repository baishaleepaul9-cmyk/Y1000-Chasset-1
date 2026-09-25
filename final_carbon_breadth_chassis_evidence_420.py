# =============================================================================
# FINAL CARBON BREADTH CHASSIS EVIDENCE INTEGRATION - 420
# =============================================================================
#
# Purpose:
#   Integrate the completed Carbon Breadth analysis into a final
#   computational candidate-chassis evidence framework.
#
# Existing evidence integrated:
#   1. Pareto candidate information
#   2. ML/feature candidate information
#   3. BUSCO functional annotation
#   4. KEGG pathway annotation
#   5. Normalized biological pathways
#   6. Biological themes
#
# Important:
#   This script does NOT redo:
#       - Pareto analysis
#       - ML training
#       - SHAP
#       - BUSCO annotation
#       - KEGG annotation
#
#   It only integrates already-generated outputs.
#
# Output:
#   FINAL_CARBON_BREADTH_CHASSIS_EVIDENCE_420
#
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


# =============================================================================
# 1. PROJECT DIRECTORIES
# =============================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "functional_annotation_420"
    / "BUSCO20_annotation"
    / "FINAL_CARBON_BREADTH_CHASSIS_EVIDENCE_420"
)

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = OUTPUT_DIR / "reports"

for d in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def normalize_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def normalize_colname(x):
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def find_csv_files(root):
    return list(root.rglob("*.csv"))


def score_file(path, keywords):
    """
    Score filenames according to keyword matches.
    """
    name = path.name.lower()

    score = 0

    for keyword, weight in keywords:
        if keyword.lower() in name:
            score += weight

    return score


def select_best_file(root, keywords, required_columns=None):
    """
    Automatically identify the most relevant CSV.

    required_columns:
        list of possible column-name fragments
    """

    candidates = find_csv_files(root)

    scored = []

    for path in candidates:

        try:
            header = pd.read_csv(path, nrows=0)
            columns = [normalize_colname(c) for c in header.columns]
        except Exception:
            continue

        score = score_file(path, keywords)

        if required_columns:

            column_hits = 0

            for group in required_columns:

                found = False

                for col in columns:

                    for token in group:

                        if token in col:
                            found = True
                            break

                    if found:
                        break

                if found:
                    column_hits += 1

            score += column_hits * 10

        if score > 0:
            scored.append((score, path))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)

    return scored[0][1]


def find_column(df, patterns):

    normalized = {
        normalize_colname(c): c
        for c in df.columns
    }

    for pattern in patterns:

        pattern = pattern.lower()

        for norm, original in normalized.items():

            if pattern in norm:
                return original

    return None


def unique_join(series):

    values = []

    for x in series:

        if pd.isna(x):
            continue

        text = str(x).strip()

        if not text:
            continue

        parts = re.split(r"\s*[;|]\s*", text)

        for p in parts:

            p = p.strip()

            if p and p not in values:
                values.append(p)

    return "; ".join(values)


def numeric_value(x, default=0):

    try:
        if pd.isna(x):
            return default

        return float(x)

    except Exception:
        return default


# =============================================================================
# 3. FIND COMPLETED CARBON BREADTH BIOLOGICAL OUTPUT
# =============================================================================

print("=" * 80)
print("FINAL CARBON BREADTH CHASSIS EVIDENCE INTEGRATION - 420")
print("=" * 80)

print("\nSearching completed Carbon Breadth biological interpretation...")


v3_root = (
    RESULTS_ROOT
    / "functional_annotation_420"
    / "BUSCO20_annotation"
    / "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V3"
)


v3_summary = (
    v3_root
    / "tables"
    / "Carbon_Breadth_final_BUSCO_biological_summary_420_v3.csv"
)

v3_pathway = (
    v3_root
    / "tables"
    / "Carbon_Breadth_final_normalized_pathway_summary_420_v3.csv"
)

v3_theme = (
    v3_root
    / "tables"
    / "Carbon_Breadth_final_biological_theme_summary_420_v3.csv"
)

v3_matrix = (
    v3_root
    / "tables"
    / "Carbon_Breadth_final_BUSCO_pathway_matrix_420_v3.csv"
)

coverage_file = (
    v3_root
    / "tables"
    / "Carbon_Breadth_final_annotation_coverage_420_v3.csv"
)


# =============================================================================
# 4. LOAD BIOLOGICAL SUMMARY
# =============================================================================

if not v3_summary.exists():

    print("\nERROR:")
    print("Carbon Breadth V3 biological summary was not found:")
    print(v3_summary)
    print("\nMake sure final_carbon_breadth_biological_interpretation_420_v3.py")
    print("has completed successfully.")

    raise SystemExit(1)


busco_df = pd.read_csv(v3_summary)

print("\nCarbon Breadth biological summary loaded.")
print("Rows:", len(busco_df))
print("Columns:", len(busco_df.columns))

print("\nColumns:")
for c in busco_df.columns:
    print(" ", c)


# =============================================================================
# 5. LOAD PATHWAY SUMMARY
# =============================================================================

if v3_pathway.exists():

    pathway_df = pd.read_csv(v3_pathway)

else:

    pathway_df = pd.DataFrame()

    print("\nWARNING:")
    print("Normalized pathway summary not found.")
    print(v3_pathway)


# =============================================================================
# 6. LOAD BIOLOGICAL THEME SUMMARY
# =============================================================================

if v3_theme.exists():

    theme_df = pd.read_csv(v3_theme)

else:

    theme_df = pd.DataFrame()

    print("\nWARNING:")
    print("Biological theme summary not found.")
    print(v3_theme)


# =============================================================================
# 7. LOAD BUSCO × PATHWAY MATRIX
# =============================================================================

if v3_matrix.exists():

    matrix_df = pd.read_csv(v3_matrix)

else:

    matrix_df = pd.DataFrame()

    print("\nWARNING:")
    print("BUSCO × pathway matrix not found.")


# =============================================================================
# 8. FIND PARETO OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("SEARCHING FOR COMPLETED PARETO OUTPUT")
print("=" * 80)


pareto_keywords = [

    ("pareto", 50),
    ("chassis", 20),
    ("candidate", 10),
    ("optimal", 10),
    ("multi_trait", 10),
    ("multitrait", 10),
    ("front", 5),

]


pareto_file = select_best_file(
    PROJECT_ROOT / "results",
    pareto_keywords,
    required_columns=[
        ["assembly", "genome", "strain", "species", "taxon"],
        ["pareto", "optimal", "front", "candidate"]
    ]
)


if pareto_file is not None:

    print("\nPareto file detected:")
    print(pareto_file)

    try:
        pareto_df = pd.read_csv(pareto_file)

        print("Rows:", len(pareto_df))
        print("Columns:", list(pareto_df.columns))

    except Exception as e:

        print("\nWARNING: Could not load Pareto file.")
        print(e)

        pareto_df = pd.DataFrame()

else:

    print("\nWARNING:")
    print("No Pareto candidate CSV was automatically detected.")

    print(
        "\nThe biological evidence analysis will continue, "
        "but Pareto integration will remain unavailable."
    )

    pareto_df = pd.DataFrame()


# =============================================================================
# 9. FIND ML / FEATURE CANDIDATE OUTPUT
# =============================================================================

print("\n" + "=" * 80)
print("SEARCHING FOR ML / FEATURE CANDIDATE OUTPUT")
print("=" * 80)


ml_keywords = [

    ("shap", 40),
    ("feature", 30),
    ("importance", 20),
    ("candidate", 20),
    ("carbon", 15),
    ("420", 5),

]


ml_file = select_best_file(
    RESULTS_ROOT,
    ml_keywords,
    required_columns=[
        ["busco", "gene", "feature"],
        ["importance", "shap", "score"]
    ]
)


if ml_file is not None:

    print("\nML/feature file detected:")
    print(ml_file)

    try:

        ml_df = pd.read_csv(ml_file)

        print("Rows:", len(ml_df))
        print("Columns:", list(ml_df.columns))

    except Exception as e:

        print("\nWARNING: Could not load ML file.")
        print(e)

        ml_df = pd.DataFrame()

else:

    print("\nWARNING:")
    print("No ML feature file automatically detected.")

    ml_df = pd.DataFrame()


# =============================================================================
# 10. IDENTIFY BUSCO COLUMN
# =============================================================================

busco_col = find_column(
    busco_df,
    [
        "busco_id",
        "busco",
        "candidate_busco"
    ]
)


if busco_col is None:

    print("\nERROR:")
    print("BUSCO column could not be identified.")

    raise SystemExit(1)


print("\nBUSCO column detected:", busco_col)


# =============================================================================
# 11. IDENTIFY BIOLOGICAL INFORMATION COLUMNS
# =============================================================================

function_col = find_column(
    busco_df,
    [
        "functional_description",
        "function",
        "description",
        "annotation"
    ]
)


theme_col = find_column(
    busco_df,
    [
        "biological_theme",
        "theme",
        "functional_theme"
    ]
)


pathway_col = find_column(
    busco_df,
    [
        "normalized_pathway",
        "pathway"
    ]
)


print("\nDetected biological columns:")

print("Function :", function_col)
print("Theme    :", theme_col)
print("Pathway  :", pathway_col)


# =============================================================================
# 12. BUILD FINAL BUSCO EVIDENCE TABLE
# =============================================================================

print("\n" + "=" * 80)
print("BUILDING FINAL BUSCO EVIDENCE TABLE")
print("=" * 80)


evidence_df = pd.DataFrame()

evidence_df["BUSCO_ID"] = busco_df[busco_col].astype(str)


if function_col:

    evidence_df["Functional_Description"] = (
        busco_df[function_col].fillna("").astype(str)
    )

else:

    evidence_df["Functional_Description"] = ""


if theme_col:

    evidence_df["Biological_Theme"] = (
        busco_df[theme_col].fillna("").astype(str)
    )

else:

    evidence_df["Biological_Theme"] = ""


if pathway_col:

    evidence_df["Normalized_Pathways"] = (
        busco_df[pathway_col].fillna("").astype(str)
    )

else:

    evidence_df["Normalized_Pathways"] = ""


# =============================================================================
# 13. ADD GENERAL ANNOTATION COUNTS
# =============================================================================

for candidate_col, output_col in [

    (["kegg_gene_count", "kegg_genes"], "KEGG_Gene_Count"),

    (["kegg_pathway_count", "kegg_pathways"], "KEGG_Pathway_Count"),

    (["go_count", "go_terms"], "GO_Count"),

    (["ec_count", "ec"], "EC_Count"),

]:

    col = find_column(busco_df, candidate_col)

    if col:

        evidence_df[output_col] = pd.to_numeric(
            busco_df[col],
            errors="coerce"
        ).fillna(0)


# =============================================================================
# 14. ADD PATHWAY RECORD INFORMATION
# =============================================================================

if not pathway_df.empty:

    p_busco_col = find_column(
        pathway_df,
        ["busco_id", "busco"]
    )

    p_pathway_col = find_column(
        pathway_df,
        [
            "normalized_pathway",
            "pathway_name",
            "pathway"
        ]
    )

    if p_busco_col and p_pathway_col:

        pathway_group = (
            pathway_df
            .groupby(p_busco_col)[p_pathway_col]
            .apply(unique_join)
            .reset_index()
        )

        pathway_group.columns = [
            "BUSCO_ID",
            "Pathway_Evidence"
        ]

        evidence_df = evidence_df.merge(
            pathway_group,
            on="BUSCO_ID",
            how="left"
        )

    else:

        evidence_df["Pathway_Evidence"] = ""

else:

    evidence_df["Pathway_Evidence"] = ""


evidence_df["Pathway_Evidence"] = (
    evidence_df["Pathway_Evidence"]
    .fillna("")
)


# =============================================================================
# 15. ADD ML EVIDENCE WHEN POSSIBLE
# =============================================================================

print("\nIntegrating ML/feature evidence...")


evidence_df["ML_Evidence"] = "Not available"
evidence_df["ML_Score"] = np.nan


if not ml_df.empty:

    ml_busco_col = find_column(
        ml_df,
        [
            "busco_id",
            "busco",
            "feature",
            "gene"
        ]
    )

    ml_score_col = find_column(
        ml_df,
        [
            "shap",
            "importance",
            "score",
            "mean_abs"
        ]
    )

    if ml_busco_col:

        ml_temp = ml_df.copy()

        ml_temp["BUSCO_ID"] = (
            ml_temp[ml_busco_col]
            .astype(str)
            .str.strip()
        )

        if ml_score_col:

            ml_temp["ML_Score"] = pd.to_numeric(
                ml_temp[ml_score_col],
                errors="coerce"
            )

            ml_temp["ML_Evidence"] = np.where(
                ml_temp["ML_Score"].notna(),
                "ML feature-supported",
                "ML candidate"
            )

        else:

            ml_temp["ML_Evidence"] = "ML candidate"
            ml_temp["ML_Score"] = np.nan


        ml_keep = [
            "BUSCO_ID",
            "ML_Evidence",
            "ML_Score"
        ]

        ml_temp = ml_temp[
            [c for c in ml_keep if c in ml_temp.columns]
        ]

        ml_temp = ml_temp.drop_duplicates(
            subset=["BUSCO_ID"]
        )

        evidence_df = evidence_df.merge(
            ml_temp,
            on="BUSCO_ID",
            how="left",
            suffixes=("", "_ML")
        )

        if "ML_Evidence_ML" in evidence_df.columns:

            evidence_df["ML_Evidence"] = (
                evidence_df["ML_Evidence_ML"]
                .fillna(evidence_df["ML_Evidence"])
            )

            evidence_df.drop(
                columns=["ML_Evidence_ML"],
                inplace=True
            )

        if "ML_Score_ML" in evidence_df.columns:

            evidence_df["ML_Score"] = (
                evidence_df["ML_Score_ML"]
                .fillna(evidence_df["ML_Score"])
            )

            evidence_df.drop(
                columns=["ML_Score_ML"],
                inplace=True
            )


# =============================================================================
# 16. DETERMINE PATHWAY STATUS
# =============================================================================

evidence_df["Pathway_Mapped"] = (
    evidence_df["Pathway_Evidence"]
    .astype(str)
    .str.strip()
    .ne("")
)


# =============================================================================
# 17. BIOLOGICAL EVIDENCE SCORE
# =============================================================================
#
# This is NOT a statistical significance score.
#
# It is a transparent evidence-count indicator:
#
#   ML evidence             = 1
#   KEGG pathway evidence   = 1
#   biological theme        = 1
#   functional annotation  = 1
#
# Maximum = 4
#
# It is used only to summarize how much annotation evidence is available.
# =============================================================================

evidence_df["Functional_Evidence"] = np.where(
    evidence_df["Functional_Description"]
    .astype(str)
    .str.strip()
    .ne(""),
    1,
    0
)


evidence_df["Theme_Evidence"] = np.where(
    evidence_df["Biological_Theme"]
    .astype(str)
    .str.strip()
    .ne(""),
    1,
    0
)


evidence_df["ML_Evidence_Flag"] = np.where(
    evidence_df["ML_Evidence"]
    .astype(str)
    .str.contains(
        "supported|candidate",
        case=False,
        regex=True
    ),
    1,
    0
)


evidence_df["Pathway_Evidence_Flag"] = (
    evidence_df["Pathway_Mapped"].astype(int)
)


evidence_df["Evidence_Count"] = (
    evidence_df["Functional_Evidence"]
    + evidence_df["Theme_Evidence"]
    + evidence_df["ML_Evidence_Flag"]
    + evidence_df["Pathway_Evidence_Flag"]
)


# =============================================================================
# 18. ADD PARETO INFORMATION
# =============================================================================

print("\nIntegrating Pareto candidate information...")


evidence_df["Pareto_Evidence"] = "Not directly linked"


if not pareto_df.empty:

    print("\nPareto columns:")

    for c in pareto_df.columns:
        print(" ", c)

    pareto_text = pareto_df.astype(str).agg(
        " ".join,
        axis=1
    )

    evidence_df["Pareto_Search_Tokens"] = (
        evidence_df["BUSCO_ID"].astype(str)
    )

    # Direct BUSCO match if available
    pareto_busco_col = find_column(
        pareto_df,
        [
            "busco_id",
            "busco"
        ]
    )

    if pareto_busco_col:

        pareto_ids = set(
            pareto_df[pareto_busco_col]
            .dropna()
            .astype(str)
            .str.strip()
        )

        evidence_df["Pareto_Evidence"] = np.where(
            evidence_df["BUSCO_ID"].isin(pareto_ids),
            "Direct Pareto-linked",
            "Not directly linked"
        )

    else:

        # No direct BUSCO-to-Pareto relationship exists.
        # Do NOT fabricate a link.
        evidence_df["Pareto_Evidence"] = (
            "Pareto completed; BUSCO linkage unavailable"
        )


evidence_df.drop(
    columns=["Pareto_Search_Tokens"],
    errors="ignore",
    inplace=True
)


# =============================================================================
# 19. FINAL CANDIDATE CLASSIFICATION
# =============================================================================

def classify_candidate(row):

    ml = row["ML_Evidence_Flag"]
    pathway = row["Pathway_Evidence_Flag"]
    function = row["Functional_Evidence"]
    theme = row["Theme_Evidence"]

    total = ml + pathway + function + theme

    if total >= 4:
        return "Strong multi-layer evidence"

    elif total == 3:
        return "High annotation support"

    elif total == 2:
        return "Moderate annotation support"

    elif total == 1:
        return "Limited annotation support"

    else:
        return "Annotation unresolved"


evidence_df["Evidence_Class"] = evidence_df.apply(
    classify_candidate,
    axis=1
)


# =============================================================================
# 20. SORT FINAL EVIDENCE TABLE
# =============================================================================

evidence_df = evidence_df.sort_values(
    by=[
        "Evidence_Count",
        "Pathway_Evidence_Flag",
        "Functional_Evidence"
    ],
    ascending=False
)


# =============================================================================
# 21. SAVE FINAL BUSCO EVIDENCE
# =============================================================================

final_busco_file = (
    TABLE_DIR
    / "Carbon_Breadth_final_BUSCO_chassis_evidence_420.csv"
)

evidence_df.to_csv(
    final_busco_file,
    index=False
)


# =============================================================================
# 22. BUILD PATHWAY EVIDENCE TABLE
# =============================================================================

print("\nBuilding pathway evidence table...")


if not pathway_df.empty:

    pathway_out = pathway_df.copy()

else:

    pathway_out = pd.DataFrame(
        columns=[
            "Normalized_Pathway",
            "BUSCO_ID"
        ]
    )


pathway_file = (
    TABLE_DIR
    / "Carbon_Breadth_final_pathway_chassis_evidence_420.csv"
)

pathway_out.to_csv(
    pathway_file,
    index=False
)


# =============================================================================
# 23. BUILD BIOLOGICAL THEME TABLE
# =============================================================================

print("Building biological theme evidence table...")


if not theme_df.empty:

    theme_out = theme_df.copy()

else:

    theme_out = pd.DataFrame()


theme_file = (
    TABLE_DIR
    / "Carbon_Breadth_final_biological_theme_evidence_420.csv"
)

theme_out.to_csv(
    theme_file,
    index=False
)


# =============================================================================
# 24. BUILD PARETO SUMMARY
# =============================================================================

pareto_summary_file = (
    TABLE_DIR
    / "Carbon_Breadth_Pareto_candidate_integration_420.csv"
)


if not pareto_df.empty:

    pareto_df.to_csv(
        pareto_summary_file,
        index=False
    )

else:

    pd.DataFrame(
        columns=["Pareto_information_unavailable"]
    ).to_csv(
        pareto_summary_file,
        index=False
    )


# =============================================================================
# 25. BUILD FINAL CANDIDATE PRIORITIZATION TABLE
# =============================================================================

print("\nBuilding final computational candidate evidence table...")


priority_cols = [
    "BUSCO_ID",
    "Functional_Description",
    "Biological_Theme",
    "Normalized_Pathways",
    "Pathway_Evidence",
    "ML_Evidence",
    "ML_Score",
    "Pareto_Evidence",
    "Evidence_Count",
    "Evidence_Class"
]


priority_cols = [
    c for c in priority_cols
    if c in evidence_df.columns
]


priority_df = evidence_df[priority_cols].copy()


priority_file = (
    TABLE_DIR
    / "Carbon_Breadth_final_candidate_prioritization_420.csv"
)


priority_df.to_csv(
    priority_file,
    index=False
)


# =============================================================================
# 26. ANNOTATION COVERAGE SUMMARY
# =============================================================================

coverage = {

    "Total_Candidate_BUSCOs":
        len(evidence_df),

    "Functional_Annotation":
        int(evidence_df["Functional_Evidence"].sum()),

    "Pathway_Mapped":
        int(evidence_df["Pathway_Evidence_Flag"].sum()),

    "ML_Evidence":
        int(evidence_df["ML_Evidence_Flag"].sum()),

    "Biological_Theme":
        int(evidence_df["Theme_Evidence"].sum()),

    "Strong_Multi_Layer_Evidence":
        int(
            (
                evidence_df["Evidence_Class"]
                == "Strong multi-layer evidence"
            ).sum()
        )

}


coverage_df = pd.DataFrame(
    list(coverage.items()),
    columns=["Metric", "Count"]
)


coverage_file_out = (
    TABLE_DIR
    / "Carbon_Breadth_final_chassis_evidence_coverage_420.csv"
)


coverage_df.to_csv(
    coverage_file_out,
    index=False
)


# =============================================================================
# 27. FIGURE 1 - EVIDENCE COVERAGE
# =============================================================================

print("\nGenerating figures...")


plt.figure(figsize=(9, 6))


labels = [
    "Total candidates",
    "Functional annotation",
    "Pathway mapped",
    "ML evidence",
    "Biological theme"
]


values = [
    coverage["Total_Candidate_BUSCOs"],
    coverage["Functional_Annotation"],
    coverage["Pathway_Mapped"],
    coverage["ML_Evidence"],
    coverage["Biological_Theme"]
]


plt.bar(labels, values)

plt.ylabel("Number of BUSCO candidates")
plt.title(
    "Carbon Breadth Candidate Evidence Coverage"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()


figure1 = (
    FIGURE_DIR
    / "Carbon_Breadth_candidate_evidence_coverage_420.png"
)

plt.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 28. FIGURE 2 - EVIDENCE CLASS
# =============================================================================

class_counts = (
    evidence_df["Evidence_Class"]
    .value_counts()
)


plt.figure(figsize=(9, 6))

plt.bar(
    class_counts.index,
    class_counts.values
)

plt.ylabel("Number of BUSCO candidates")

plt.title(
    "Carbon Breadth Candidate Evidence Classes"
)

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()


figure2 = (
    FIGURE_DIR
    / "Carbon_Breadth_candidate_evidence_classes_420.png"
)

plt.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 29. FIGURE 3 - PATHWAY REPRESENTATION
# =============================================================================

if not pathway_df.empty:

    pathway_name_col = find_column(
        pathway_df,
        [
            "normalized_pathway",
            "pathway_name",
            "pathway"
        ]
    )

    if pathway_name_col:

        pathway_counts = (
            pathway_df[pathway_name_col]
            .dropna()
            .astype(str)
            .value_counts()
            .head(20)
        )

        if len(pathway_counts) > 0:

            plt.figure(figsize=(10, 8))

            plt.barh(
                pathway_counts.index[::-1],
                pathway_counts.values[::-1]
            )

            plt.xlabel(
                "Number of pathway records"
            )

            plt.ylabel(
                "Normalized biological pathway"
            )

            plt.title(
                "Carbon Breadth Candidate Pathway Evidence"
            )

            plt.tight_layout()

            figure3 = (
                FIGURE_DIR
                / "Carbon_Breadth_candidate_pathway_evidence_420.png"
            )

            plt.savefig(
                figure3,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()


# =============================================================================
# 30. BUILD FINAL REPORT
# =============================================================================

report_file = (
    REPORT_DIR
    / "Carbon_Breadth_final_chassis_evidence_report_420.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write("FINAL CARBON BREADTH CHASSIS EVIDENCE INTEGRATION - 420\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "This report integrates the completed Carbon Breadth "
        "computational evidence generated from ML candidate selection, "
        "functional annotation, KEGG pathway mapping, normalized "
        "biological pathways and biological themes.\n\n"
    )

    f.write("EVIDENCE COVERAGE\n")
    f.write("-" * 80 + "\n")

    for key, value in coverage.items():

        f.write(
            f"{key:<40}: {value}\n"
        )

    f.write("\n")

    f.write("EVIDENCE CLASSIFICATION\n")
    f.write("-" * 80 + "\n")

    for cls, count in class_counts.items():

        f.write(
            f"{cls:<40}: {count}\n"
        )

    f.write("\n")

    f.write("PARETO INTEGRATION\n")
    f.write("-" * 80 + "\n")

    if not pareto_df.empty:

        f.write(
            "A Pareto output file was detected and loaded.\n"
        )

        f.write(
            f"Pareto file: {pareto_file}\n"
        )

        f.write(
            f"Pareto rows: {len(pareto_df)}\n"
        )

        f.write(
            "\nNote: BUSCO-level Pareto linkage was only assigned "
            "when an explicit BUSCO identifier was available.\n"
        )

    else:

        f.write(
            "No Pareto CSV was automatically detected.\n"
        )

    f.write("\n")

    f.write("IMPORTANT BIOLOGICAL INTERPRETATION\n")
    f.write("-" * 80 + "\n")

    f.write(
        "The evidence classes in this analysis represent the amount "
        "of computational annotation support available for each "
        "candidate BUSCO. They are not statistical significance values, "
        "effect sizes, or experimental validation scores.\n\n"
    )

    f.write(
        "Pathway presence indicates annotation coverage and should not "
        "be interpreted as pathway enrichment without an appropriate "
        "statistical enrichment analysis and background set.\n\n"
    )

    f.write(
        "Candidates without recovered pathway annotations should not "
        "be interpreted as biologically irrelevant. Lack of pathway "
        "mapping may reflect annotation limitations, database coverage, "
        "orthology limitations or pathway representation.\n\n"
    )

    f.write(
        "The final output represents computationally supported "
        "candidate evidence and does not constitute independent "
        "experimental validation of a yeast chassis.\n\n"
    )

    f.write("FINAL TABLES\n")
    f.write("-" * 80 + "\n")

    f.write(
        str(final_busco_file) + "\n"
    )

    f.write(
        str(pathway_file) + "\n"
    )

    f.write(
        str(theme_file) + "\n"
    )

    f.write(
        str(pareto_summary_file) + "\n"
    )

    f.write(
        str(priority_file) + "\n"
    )

    f.write(
        str(coverage_file_out) + "\n"
    )

    f.write("\nFINAL FIGURES\n")
    f.write("-" * 80 + "\n")

    f.write(
        str(figure1) + "\n"
    )

    f.write(
        str(figure2) + "\n"
    )

    if not pathway_df.empty:
        f.write(
            str(
                FIGURE_DIR
                / "Carbon_Breadth_candidate_pathway_evidence_420.png"
            )
            + "\n"
        )


# =============================================================================
# 31. FINAL CONSOLE OUTPUT
# =============================================================================

print("\n")
print("=" * 80)
print("FINAL CARBON BREADTH CHASSIS EVIDENCE INTEGRATION COMPLETE")
print("=" * 80)

print("\nCANDIDATE EVIDENCE")
print("-" * 80)

for key, value in coverage.items():

    print(
        f"{key:<40}: {value}"
    )


print("\nEVIDENCE CLASSES")
print("-" * 80)

for cls, count in class_counts.items():

    print(
        f"{cls:<40}: {count}"
    )


print("\nOUTPUT DIRECTORY")
print("-" * 80)
print(OUTPUT_DIR)


print("\nTABLES")
print("-" * 80)

print(final_busco_file)
print(pathway_file)
print(theme_file)
print(pareto_summary_file)
print(priority_file)
print(coverage_file_out)


print("\nFIGURES")
print("-" * 80)

print(figure1)
print(figure2)

if not pathway_df.empty:
    print(
        FIGURE_DIR
        / "Carbon_Breadth_candidate_pathway_evidence_420.png"
    )


print("\nREPORT")
print("-" * 80)

print(report_file)

print("\n")
print("=" * 80)
print("NEXT STAGE: INDEPENDENT CANDIDATE-CHASSIS VALIDATION")
print("=" * 80)