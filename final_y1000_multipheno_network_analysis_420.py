# =============================================================================
# FINAL Y1000+ MULTI-PHENOTYPE NETWORK ANALYSIS
# 420 TAXA / PHYLOGENY-AWARE ML
#
# FIXED / ROBUST VERSION
#
# IMPORTANT:
#   - Does NOT retrain models
#   - Does NOT rerun permutations
#   - Does NOT change feature sets
#   - Does NOT modify candidate chassis
#   - Does NOT fabricate BUSCO/pathway mappings
#   - Handles phenotype-specific biological annotation availability
#   - Handles empty CSV files safely
# =============================================================================

import os
import glob
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = r"C:\Y1000_chassis_project"

RESULTS = os.path.join(
    BASE,
    "results"
)

STAGE5 = os.path.join(
    RESULTS,
    "stage5_phylogeny_ml_dataset",
    "phylogeny_aware_ml"
)

STAGE6 = os.path.join(
    RESULTS,
    "stage6_candidate_yeast_chassis_420"
)

OUT = os.path.join(
    RESULTS,
    "stage7_final_multiphenotype_network_analysis_420"
)

TABLES = os.path.join(OUT, "tables")
FIGURES = os.path.join(OUT, "figures")
REPORTS = os.path.join(OUT, "reports")

for d in [TABLES, FIGURES, REPORTS]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def safe_read_csv(path, label="CSV"):
    """
    Safely read a CSV.

    Returns:
        DataFrame if valid
        None if missing/empty/unreadable
    """

    if path is None:
        return None

    if not os.path.exists(path):
        print(f"{label}: NOT FOUND")
        return None

    if os.path.getsize(path) == 0:
        print(f"{label}: EMPTY FILE - SKIPPED")
        return None

    try:
        df = pd.read_csv(path)

        if df.empty:
            print(f"{label}: EMPTY DATAFRAME - SKIPPED")
            return None

        print(f"{label}:")
        print(f"  {path}")
        print(f"  Rows: {len(df)}")
        print(f"  Columns: {list(df.columns)}")

        return df

    except pd.errors.EmptyDataError:
        print(f"{label}: EMPTY CSV - SKIPPED")
        return None

    except Exception as e:
        print(f"{label}: FAILED TO READ")
        print(f"  Reason: {e}")
        return None


def detect_column(df, candidates):

    if df is None:
        return None

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:

        key = candidate.lower()

        if key in lookup:
            return lookup[key]

    return None


def unique_clean(values):

    out = []

    for x in values:

        if pd.isna(x):
            continue

        x = str(x).strip()

        if not x:
            continue

        if x not in out:
            out.append(x)

    return out


# =============================================================================
# PHENOTYPE CONFIGURATION
# =============================================================================

PHENOTYPES = [
    "Carbon_Breadth",
    "Nitrogen_Breadth",
    "Utilized_Median_Growth"
]


# =============================================================================
# FIND EXISTING BIOLOGICAL ANNOTATION FILES
# =============================================================================

banner("SEARCHING FOR PHENOTYPE-SPECIFIC BIOLOGICAL EVIDENCE")

annotation_candidates = {
    "Carbon_Breadth": [],
    "Nitrogen_Breadth": [],
    "Utilized_Median_Growth": []
}


# Search recursively through Stage 5
for root, dirs, files in os.walk(STAGE5):

    for filename in files:

        lower = filename.lower()

        if not lower.endswith(".csv"):
            continue

        full = os.path.join(root, filename)

        # Biological annotation indicators
        biological_words = [
            "busco",
            "pathway",
            "kegg",
            "annotation",
            "theme",
            "functional"
        ]

        if not any(x in lower for x in biological_words):
            continue

        # Assign only when phenotype is actually represented
        for phenotype in PHENOTYPES:

            if phenotype.lower() in lower:

                annotation_candidates[phenotype].append(full)


# =============================================================================
# PRINT DISCOVERY RESULTS
# =============================================================================

for phenotype in PHENOTYPES:

    print(f"\n{phenotype}")

    candidates = annotation_candidates[phenotype]

    if not candidates:
        print("  No phenotype-specific biological annotation file found.")
    else:

        for f in candidates[:10]:
            print(" ", f)


# =============================================================================
# LOAD VALIDATED ML FEATURE EVIDENCE
# =============================================================================

banner("LOADING VALIDATED ML FEATURE EVIDENCE")

stable_ml_file = os.path.join(
    STAGE6,
    "tables",
    "validated_stable_ML_features_420.csv"
)

stable_ml = safe_read_csv(
    stable_ml_file,
    "Validated stable ML features"
)


if stable_ml is None:

    # Fallback to feature stability table
    fallback_ml = os.path.join(
        STAGE5,
        "phylogeny_cv",
        "model_training",
        "feature_interpretation_420",
        "tables",
        "feature_importance_stability_420.csv"
    )

    stable_ml = safe_read_csv(
        fallback_ml,
        "Fallback ML feature stability table"
    )


# =============================================================================
# IDENTIFY PHENOTYPE / FEATURE / BUSCO COLUMNS
# =============================================================================

if stable_ml is not None:

    phenotype_col = detect_column(
        stable_ml,
        [
            "Phenotype",
            "phenotype"
        ]
    )

    feature_col = detect_column(
        stable_ml,
        [
            "Feature",
            "BUSCO_ID",
            "BUSCO",
            "Feature_ID"
        ]
    )

else:

    phenotype_col = None
    feature_col = None


# =============================================================================
# BUILD ML FEATURE SUMMARY
# =============================================================================

banner("BUILDING MULTI-PHENOTYPE ML FEATURE SUMMARY")

ml_summary_rows = []

if stable_ml is not None:

    for phenotype in PHENOTYPES:

        if phenotype_col is not None:

            subset = stable_ml[
                stable_ml[phenotype_col].astype(str)
                == phenotype
            ].copy()

        else:

            subset = stable_ml.copy()

        if feature_col is not None:

            features = unique_clean(
                subset[feature_col].tolist()
            )

        else:

            features = []

        ml_summary_rows.append({

            "Phenotype": phenotype,

            "Stable_ML_Feature_Count":
                len(features),

            "Stable_ML_Features":
                "; ".join(features),

            "ML_Evidence_Available":
                len(features) > 0

        })

else:

    for phenotype in PHENOTYPES:

        ml_summary_rows.append({

            "Phenotype": phenotype,

            "Stable_ML_Feature_Count": 0,

            "Stable_ML_Features": "",

            "ML_Evidence_Available": False

        })


ml_summary = pd.DataFrame(
    ml_summary_rows
)


# =============================================================================
# BIOLOGICAL ANNOTATION PROCESSOR
# =============================================================================

def process_annotation_file(
    phenotype,
    filepath
):

    df = safe_read_csv(
        filepath,
        f"{phenotype} biological annotation"
    )

    if df is None:
        return None

    busco_col = detect_column(
        df,
        [
            "BUSCO_ID",
            "BUSCO",
            "Busco_ID",
            "Feature"
        ]
    )

    pathway_col = detect_column(
        df,
        [
            "Normalized_Pathways",
            "Pathways",
            "Pathway",
            "KEGG_Pathways",
            "KEGG"
        ]
    )

    theme_col = detect_column(
        df,
        [
            "Biological_Theme",
            "Theme",
            "Functional_Theme"
        ]
    )

    # -------------------------------------------------------------------------
    # REQUIRED BUSCO COLUMN
    # -------------------------------------------------------------------------

    if busco_col is None:

        print(
            f"\n{phenotype}: "
            "No BUSCO identifier found. "
            "File will NOT be used for network construction."
        )

        return None

    # -------------------------------------------------------------------------
    # NORMALIZE BUSCO IDs
    # -------------------------------------------------------------------------

    df = df.copy()

    df["BUSCO_ID_STANDARD"] = (
        df[busco_col]
        .astype(str)
        .str.strip()
    )

    # Remove invalid identifiers
    df = df[
        df["BUSCO_ID_STANDARD"].notna()
    ]

    # -------------------------------------------------------------------------
    # PATHWAY EXPANSION
    # -------------------------------------------------------------------------

    pathway_edges = []

    if pathway_col is not None:

        for _, row in df.iterrows():

            busco = row["BUSCO_ID_STANDARD"]

            value = row[pathway_col]

            if pd.isna(value):
                continue

            pathways = str(value).split(";")

            for pathway in pathways:

                pathway = pathway.strip()

                if not pathway:
                    continue

                pathway_edges.append({

                    "Phenotype": phenotype,

                    "BUSCO_ID": busco,

                    "Pathway": pathway

                })

    # -------------------------------------------------------------------------
    # THEME EXPANSION
    # -------------------------------------------------------------------------

    theme_edges = []

    if theme_col is not None:

        for _, row in df.iterrows():

            busco = row["BUSCO_ID_STANDARD"]

            value = row[theme_col]

            if pd.isna(value):
                continue

            themes = str(value).split(";")

            for theme in themes:

                theme = theme.strip()

                if not theme:
                    continue

                theme_edges.append({

                    "Phenotype": phenotype,

                    "BUSCO_ID": busco,

                    "Biological_Theme": theme

                })

    return {

        "data": df,

        "busco_col": busco_col,

        "pathway_col": pathway_col,

        "theme_col": theme_col,

        "pathway_edges":
            pd.DataFrame(pathway_edges),

        "theme_edges":
            pd.DataFrame(theme_edges)

    }


# =============================================================================
# SELECT BEST ANNOTATION FILE
# =============================================================================

banner("SELECTING VALID BIOLOGICAL ANNOTATION")

annotation_results = {}

for phenotype in PHENOTYPES:

    candidates = annotation_candidates[phenotype]

    selected = None

    for filepath in candidates:

        result = process_annotation_file(
            phenotype,
            filepath
        )

        if result is None:
            continue

        # Prefer files that actually contain pathway information
        pathway_edges = result["pathway_edges"]

        if (
            pathway_edges is not None
            and not pathway_edges.empty
        ):

            selected = result
            selected["filepath"] = filepath
            break

        # Otherwise retain as fallback
        if selected is None:

            result["filepath"] = filepath
            selected = result

    annotation_results[phenotype] = selected


# =============================================================================
# BUILD PHENOTYPE SUMMARY
# =============================================================================

banner("BUILDING BIOLOGICAL EVIDENCE SUMMARY")

summary_rows = []

all_pathway_edges = []
all_theme_edges = []

for phenotype in PHENOTYPES:

    result = annotation_results.get(phenotype)

    ml_row = ml_summary[
        ml_summary["Phenotype"] == phenotype
    ]

    if not ml_row.empty:

        ml_count = int(
            ml_row.iloc[0]["Stable_ML_Feature_Count"]
        )

    else:

        ml_count = 0

    if result is None:

        summary_rows.append({

            "Phenotype": phenotype,

            "Biological_Annotation_Available":
                False,

            "ML_BUSCO_Count":
                ml_count,

            "Mapped_BUSCO_Count":
                0,

            "Pathway_Count":
                0,

            "BUSCO_Pathway_Edges":
                0,

            "Theme_Count":
                0,

            "BUSCO_Theme_Edges":
                0,

            "Network_Analysis_Status":
                "No_valid_biological_annotation"

        })

        continue

    df = result["data"]

    mapped_buscos = set(
        df["BUSCO_ID_STANDARD"]
        .dropna()
        .astype(str)
    )

    pathway_edges = result["pathway_edges"]

    theme_edges = result["theme_edges"]

    pathway_count = 0
    pathway_edge_count = 0

    theme_count = 0
    theme_edge_count = 0

    if (
        pathway_edges is not None
        and not pathway_edges.empty
    ):

        pathway_count = (
            pathway_edges["Pathway"]
            .nunique()
        )

        pathway_edge_count = len(
            pathway_edges
        )

        all_pathway_edges.append(
            pathway_edges
        )

    if (
        theme_edges is not None
        and not theme_edges.empty
    ):

        theme_count = (
            theme_edges[
                "Biological_Theme"
            ].nunique()
        )

        theme_edge_count = len(
            theme_edges
        )

        all_theme_edges.append(
            theme_edges
        )

    status = "Annotation_available"

    if pathway_edge_count == 0:

        status = (
            "Annotation_available_but_no_pathway_edges"
        )

    summary_rows.append({

        "Phenotype": phenotype,

        "Biological_Annotation_Available":
            True,

        "ML_BUSCO_Count":
            ml_count,

        "Mapped_BUSCO_Count":
            len(mapped_buscos),

        "Pathway_Count":
            pathway_count,

        "BUSCO_Pathway_Edges":
            pathway_edge_count,

        "Theme_Count":
            theme_count,

        "BUSCO_Theme_Edges":
            theme_edge_count,

        "Network_Analysis_Status":
            status

    })


phenotype_summary = pd.DataFrame(
    summary_rows
)


# =============================================================================
# SAVE PHENOTYPE SUMMARY
# =============================================================================

summary_file = os.path.join(
    TABLES,
    "multi_phenotype_network_summary_420.csv"
)

phenotype_summary.to_csv(
    summary_file,
    index=False
)

print("\nSaved:")
print(summary_file)


# =============================================================================
# COMBINE VALID PATHWAY EDGES
# =============================================================================

banner("BUILDING CROSS-PHENOTYPE BUSCO-PATHWAY NETWORK")

if all_pathway_edges:

    pathway_network = pd.concat(
        all_pathway_edges,
        ignore_index=True
    )

    pathway_network = (
        pathway_network
        .drop_duplicates()
        .reset_index(drop=True)
    )

else:

    pathway_network = pd.DataFrame(
        columns=[
            "Phenotype",
            "BUSCO_ID",
            "Pathway"
        ]
    )


print(
    f"Valid BUSCO-pathway edges: "
    f"{len(pathway_network)}"
)

print(
    f"Phenotypes represented: "
    f"{pathway_network['Phenotype'].nunique() if not pathway_network.empty else 0}"
)

print(
    f"Unique BUSCOs: "
    f"{pathway_network['BUSCO_ID'].nunique() if not pathway_network.empty else 0}"
)

print(
    f"Unique pathways: "
    f"{pathway_network['Pathway'].nunique() if not pathway_network.empty else 0}"
)


pathway_network_file = os.path.join(
    TABLES,
    "cross_phenotype_BUSCO_pathway_network_edges_420.csv"
)

pathway_network.to_csv(
    pathway_network_file,
    index=False
)


# =============================================================================
# BUSCO CONNECTIVITY
# =============================================================================

banner("CALCULATING BUSCO NETWORK CONNECTIVITY")

if not pathway_network.empty:

    busco_connectivity = (
        pathway_network
        .groupby(
            ["Phenotype", "BUSCO_ID"],
            as_index=False
        )
        .agg(
            Pathway_Degree=(
                "Pathway",
                "nunique"
            )
        )
    )

    pathway_lists = (
        pathway_network
        .groupby(
            ["Phenotype", "BUSCO_ID"]
        )["Pathway"]
        .apply(
            lambda x:
            "; ".join(
                sorted(
                    set(x)
                )
            )
        )
        .reset_index(
            name="Connected_Pathways"
        )
    )

    busco_connectivity = busco_connectivity.merge(
        pathway_lists,
        on=[
            "Phenotype",
            "BUSCO_ID"
        ],
        how="left"
    )

else:

    busco_connectivity = pd.DataFrame(
        columns=[
            "Phenotype",
            "BUSCO_ID",
            "Pathway_Degree",
            "Connected_Pathways"
        ]
    )


busco_connectivity_file = os.path.join(
    TABLES,
    "cross_phenotype_BUSCO_network_connectivity_420.csv"
)

busco_connectivity.to_csv(
    busco_connectivity_file,
    index=False
)


# =============================================================================
# PATHWAY CONNECTIVITY
# =============================================================================

banner("CALCULATING PATHWAY CONNECTIVITY")

if not pathway_network.empty:

    pathway_connectivity = (
        pathway_network
        .groupby(
            ["Phenotype", "Pathway"],
            as_index=False
        )
        .agg(
            BUSCO_Degree=(
                "BUSCO_ID",
                "nunique"
            )
        )
    )

    busco_lists = (
        pathway_network
        .groupby(
            ["Phenotype", "Pathway"]
        )["BUSCO_ID"]
        .apply(
            lambda x:
            "; ".join(
                sorted(
                    set(x)
                )
            )
        )
        .reset_index(
            name="Connected_BUSCOs"
        )
    )

    pathway_connectivity = pathway_connectivity.merge(
        busco_lists,
        on=[
            "Phenotype",
            "Pathway"
        ],
        how="left"
    )

else:

    pathway_connectivity = pd.DataFrame(
        columns=[
            "Phenotype",
            "Pathway",
            "BUSCO_Degree",
            "Connected_BUSCOs"
        ]
    )


pathway_connectivity_file = os.path.join(
    TABLES,
    "cross_phenotype_pathway_network_connectivity_420.csv"
)

pathway_connectivity.to_csv(
    pathway_connectivity_file,
    index=False
)


# =============================================================================
# BIOLOGICAL THEME NETWORK
# =============================================================================

banner("BUILDING CROSS-PHENOTYPE BIOLOGICAL THEME NETWORK")

if all_theme_edges:

    theme_network = pd.concat(
        all_theme_edges,
        ignore_index=True
    )

    theme_network = (
        theme_network
        .drop_duplicates()
        .reset_index(drop=True)
    )

else:

    theme_network = pd.DataFrame(
        columns=[
            "Phenotype",
            "BUSCO_ID",
            "Biological_Theme"
        ]
    )


theme_network_file = os.path.join(
    TABLES,
    "cross_phenotype_BUSCO_theme_network_edges_420.csv"
)

theme_network.to_csv(
    theme_network_file,
    index=False
)


# =============================================================================
# THEME CONNECTIVITY
# =============================================================================

if not theme_network.empty:

    theme_connectivity = (
        theme_network
        .groupby(
            [
                "Phenotype",
                "Biological_Theme"
            ],
            as_index=False
        )
        .agg(
            BUSCO_Count=(
                "BUSCO_ID",
                "nunique"
            )
        )
    )

else:

    theme_connectivity = pd.DataFrame(
        columns=[
            "Phenotype",
            "Biological_Theme",
            "BUSCO_Count"
        ]
    )


theme_connectivity_file = os.path.join(
    TABLES,
    "cross_phenotype_theme_connectivity_420.csv"
)

theme_connectivity.to_csv(
    theme_connectivity_file,
    index=False
)


# =============================================================================
# CROSS-PHENOTYPE BUSCO OVERLAP
# =============================================================================

banner("CALCULATING CROSS-PHENOTYPE BUSCO OVERLAP")

phenotype_buscos = {}

for phenotype in PHENOTYPES:

    result = annotation_results.get(phenotype)

    if result is None:

        phenotype_buscos[phenotype] = set()

        continue

    phenotype_buscos[phenotype] = set(
        result["data"][
            "BUSCO_ID_STANDARD"
        ]
        .dropna()
        .astype(str)
    )


overlap_rows = []

for i, phenotype_a in enumerate(PHENOTYPES):

    for phenotype_b in PHENOTYPES[i + 1:]:

        a = phenotype_buscos[phenotype_a]
        b = phenotype_buscos[phenotype_b]

        overlap = sorted(
            a.intersection(b)
        )

        overlap_rows.append({

            "Phenotype_A":
                phenotype_a,

            "Phenotype_B":
                phenotype_b,

            "BUSCO_Overlap_Count":
                len(overlap),

            "Shared_BUSCOs":
                "; ".join(overlap)

        })


overlap_df = pd.DataFrame(
    overlap_rows
)

overlap_file = os.path.join(
    TABLES,
    "cross_phenotype_BUSCO_overlap_420.csv"
)

overlap_df.to_csv(
    overlap_file,
    index=False
)


# =============================================================================
# LINK NETWORK TO CANDIDATE CHASSIS
# =============================================================================

banner("LINKING NETWORK EVIDENCE TO CANDIDATE CHASSIS")

candidate_file = os.path.join(
    STAGE6,
    "tables",
    "candidate_yeast_chassis_species_420.csv"
)

pareto_file = os.path.join(
    STAGE6,
    "tables",
    "pareto_optimal_yeast_chassis_candidates_420.csv"
)

candidate_df = safe_read_csv(
    candidate_file,
    "Candidate chassis"
)

pareto_df = safe_read_csv(
    pareto_file,
    "Pareto candidates"
)


# =============================================================================
# CANDIDATE NETWORK CONTEXT
# =============================================================================

if candidate_df is not None:

    species_col = detect_column(
        candidate_df,
        [
            "Species",
            "species"
        ]
    )

    if species_col is not None:

        candidate_context = candidate_df.copy()

        # Network evidence is global BUSCO/pathway evidence rather than
        # species-specific expression unless an explicit species column exists.
        candidate_context[
            "Network_Evidence_Available"
        ] = (
            not pathway_network.empty
            or not theme_network.empty
        )

        candidate_context[
            "Network_Evidence_Note"
        ] = (
            "Candidate chassis linked to "
            "validated multi-phenotype biological "
            "network evidence; no species-level "
            "network attribution was inferred."
        )

    else:

        candidate_context = pd.DataFrame()

else:

    candidate_context = pd.DataFrame()


candidate_context_file = os.path.join(
    TABLES,
    "candidate_chassis_multi_phenotype_network_context_420.csv"
)

candidate_context.to_csv(
    candidate_context_file,
    index=False
)


# =============================================================================
# VALIDATION STATUS
# =============================================================================

banner("FINAL MULTI-PHENOTYPE NETWORK STATUS")

for _, row in phenotype_summary.iterrows():

    print(
        f"\n{row['Phenotype']}"
    )

    print(
        "  ML BUSCO features:",
        row["ML_BUSCO_Count"]
    )

    print(
        "  Mapped BUSCOs:",
        row["Mapped_BUSCO_Count"]
    )

    print(
        "  Pathways:",
        row["Pathway_Count"]
    )

    print(
        "  BUSCO-pathway edges:",
        row["BUSCO_Pathway_Edges"]
    )

    print(
        "  Network status:",
        row["Network_Analysis_Status"]
    )


# =============================================================================
# FINAL REPORT
# =============================================================================

banner("GENERATING FINAL MULTI-PHENOTYPE NETWORK REPORT")

report_file = os.path.join(
    REPORTS,
    "final_y1000_multiphenotype_network_analysis_420_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "FINAL Y1000+ MULTI-PHENOTYPE NETWORK ANALYSIS\n"
    )

    f.write(
        "420 TAXA / PHYLOGENY-AWARE ML\n"
    )

    f.write("=" * 80 + "\n\n")

    f.write(
        "This analysis integrates existing validated ML "
        "and biological annotation evidence.\n\n"
    )

    f.write(
        "NO MODEL RETRAINING\n"
    )

    f.write(
        "NO PERMUTATION RERUN\n"
    )

    f.write(
        "NO FEATURE SELECTION\n"
    )

    f.write(
        "NO CHASSIS MODIFICATION\n\n"
    )

    f.write(
        "IMPORTANT INTERPRETATION RULE:\n"
    )

    f.write(
        "Network edges were constructed only where an "
        "actual BUSCO-to-pathway or BUSCO-to-theme "
        "annotation was available. Missing biological "
        "annotations were not inferred or fabricated.\n\n"
    )

    f.write(
        phenotype_summary.to_string(
            index=False
        )
    )

    f.write("\n\n")

    f.write(
        "CROSS-PHENOTYPE BUSCO OVERLAP\n"
    )

    f.write("-" * 80 + "\n")

    if overlap_df.empty:

        f.write(
            "No cross-phenotype biological annotation "
            "overlap could be calculated.\n"
        )

    else:

        f.write(
            overlap_df.to_string(
                index=False
            )
        )

    f.write("\n\n")

    f.write(
        "INTERPRETATION\n"
    )

    f.write("-" * 80 + "\n")

    for _, row in phenotype_summary.iterrows():

        phenotype = row["Phenotype"]

        if row[
            "Network_Analysis_Status"
        ] == "No_valid_biological_annotation":

            f.write(
                f"{phenotype}: validated ML evidence is "
                f"available where indicated, but no valid "
                f"phenotype-specific BUSCO biological "
                f"annotation was available for network "
                f"construction.\n"
            )

        elif row[
            "BUSCO_Pathway_Edges"
        ] == 0:

            f.write(
                f"{phenotype}: biological annotation was "
                f"detected, but no BUSCO-pathway edges were "
                f"available. No network interpretation was "
                f"inferred.\n"
            )

        else:

            f.write(
                f"{phenotype}: BUSCO-pathway network "
                f"constructed from {int(row['Mapped_BUSCO_Count'])} "
                f"mapped BUSCOs and "
                f"{int(row['Pathway_Count'])} pathways.\n"
            )

    f.write("\n")

    f.write(
        "This analysis does not imply that a BUSCO is "
        "causally responsible for a phenotype. Network "
        "connectivity represents functional annotation "
        "context only.\n"
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

banner("FINAL MULTI-PHENOTYPE NETWORK ANALYSIS COMPLETE")

print("\nTABLES:")

for f in [
    summary_file,
    pathway_network_file,
    busco_connectivity_file,
    pathway_connectivity_file,
    theme_network_file,
    theme_connectivity_file,
    overlap_file,
    candidate_context_file
]:

    print(f)


print("\nREPORT:")
print(report_file)

print(
    "\nIMPORTANT:"
)

print(
    "The script did NOT invent pathway/network evidence "
    "for Nitrogen_Breadth or Utilized_Median_Growth."
)

print(
    "Phenotypes without valid BUSCO biological annotation "
    "remain explicitly documented as such."
)

print(
    "\nNo ML models retrained."
)

print(
    "No permutation tests rerun."
)

print(
    "No feature set changed."
)

print(
    "No candidate chassis modified."
)

print(
    "No Pareto candidates modified."
)