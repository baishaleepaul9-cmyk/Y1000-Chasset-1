# =============================================================================
# EXTERNAL FUNCTIONAL ANNOTATION
# Carbon_Breadth — Y1000+ Yeast Chassis Project
#
# Input:
#   20 Carbon_Breadth BUSCO/OrthoDB candidates
#
# Output:
#   OrthoDB functional annotation
#   GO terms
#   EC numbers
#   KEGG pathway information
#   InterPro domains
#   Functional categories
#
# IMPORTANT:
#   No unsupported GO/KEGG/pathway assignments are invented.
#   Missing annotations remain missing.
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import requests
import time
import json
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(r"C:\Y1000_chassis_project")

BASE_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
)

INPUT_FILE = (
    BASE_DIR
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "pathway_mapping_420"
    / "tables"
    / "Carbon_Breadth_pathway_ready_candidates_420.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "pathway_mapping_420"
    / "external_annotation_420"
)

TABLE_DIR = OUTPUT_DIR / "tables"
RAW_DIR = OUTPUT_DIR / "raw_orthodb"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Current OrthoDB API
ORTHODB_BASE = "https://data.orthodb.org/v12"

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.4

# =============================================================================
# PRINT HELPERS
# =============================================================================

def header(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def safe_text(value):
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip()


# =============================================================================
# LOAD INPUT
# =============================================================================

header("EXTERNAL FUNCTIONAL ANNOTATION — CARBON_BREADTH")

print("Input:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )

df = pd.read_csv(INPUT_FILE)

print("\nInput shape:")
print(df.shape)

print("\nInput columns:")
print(list(df.columns))

# =============================================================================
# IDENTIFY BUSCO COLUMN
# =============================================================================

busco_col = None

for col in df.columns:

    if str(col).lower() in {
        "busco_id",
        "busco",
        "feature"
    }:
        busco_col = col
        break

if busco_col is None:

    for col in df.columns:

        if "busco" in str(col).lower():
            busco_col = col
            break

if busco_col is None:
    raise ValueError(
        "Could not identify BUSCO_ID column."
    )

print("\nBUSCO column:")
print(busco_col)

# =============================================================================
# GET UNIQUE CANDIDATES
# =============================================================================

buscos = (
    df[busco_col]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

buscos = sorted(buscos)

header("CANDIDATES")

print(f"Number of candidates: {len(buscos)}")

for busco in buscos:
    print(busco)

# =============================================================================
# ORTHODB REQUEST
# =============================================================================

session = requests.Session()

session.headers.update({
    "User-Agent": (
        "Y1000-Chassis-Project/"
        "Carbon-Breadth-Functional-Annotation"
    ),
    "Accept": "application/json"
})


def query_orthodb_group(orthodb_id):

    url = f"{ORTHODB_BASE}/group"

    params = {
        "id": orthodb_id
    }

    try:

        response = session.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        result = {
            "orthodb_id": orthodb_id,
            "http_status": response.status_code,
            "success": False,
            "error": "",
            "data": None
        }

        if response.status_code != 200:

            result["error"] = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            return result

        try:
            data = response.json()

        except Exception:

            result["error"] = (
                "Response was not valid JSON."
            )

            return result

        result["success"] = True
        result["data"] = data

        return result

    except Exception as e:

        return {
            "orthodb_id": orthodb_id,
            "http_status": None,
            "success": False,
            "error": str(e),
            "data": None
        }


# =============================================================================
# NORMALIZATION FUNCTIONS
# =============================================================================

def recursive_find(data, keys):
    """
    Search nested JSON for any of the requested keys.
    """

    if isinstance(data, dict):

        for key in keys:

            if key in data:
                return data[key]

        for value in data.values():

            result = recursive_find(
                value,
                keys
            )

            if result is not None:
                return result

    elif isinstance(data, list):

        for item in data:

            result = recursive_find(
                item,
                keys
            )

            if result is not None:
                return result

    return None


def flatten_annotation(value):

    """
    Convert nested/list annotation structures
    into a clean semicolon-separated representation.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, list):

        parts = []

        for item in value:

            if isinstance(item, dict):

                # Try useful identifiers/names first
                preferred = [
                    "id",
                    "name",
                    "description",
                    "term",
                    "pathway",
                    "accession"
                ]

                found = []

                for key in preferred:

                    if key in item:

                        val = item[key]

                        if val is not None:
                            found.append(str(val))

                if found:
                    parts.append(
                        " | ".join(found)
                    )

                else:
                    parts.append(
                        json.dumps(
                            item,
                            ensure_ascii=False
                        )
                    )

            else:
                parts.append(str(item))

        return "; ".join(
            sorted(
                set(
                    x.strip()
                    for x in parts
                    if x.strip()
                )
            )
        )

    if isinstance(value, dict):

        return json.dumps(
            value,
            ensure_ascii=False
        )

    return str(value)


# =============================================================================
# QUERY ALL CANDIDATES
# =============================================================================

header("QUERYING ORTHODB")

results = []

for i, busco in enumerate(buscos, start=1):

    print(
        f"[{i}/{len(buscos)}] "
        f"Querying {busco}"
    )

    result = query_orthodb_group(busco)

    # Save raw response
    raw_file = RAW_DIR / f"{busco}.json"

    with open(
        raw_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    row = {
        "BUSCO_ID": busco,
        "OrthoDB_Query_Success": result["success"],
        "HTTP_Status": result["http_status"],
        "Query_Error": result["error"]
    }

    data = result["data"]

    if data is not None:

        # -------------------------------------------------------------
        # GENERAL FUNCTIONAL DESCRIPTION
        # -------------------------------------------------------------

        row["OrthoDB_Name"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "name",
                    "description"
                ]
            )
        )

        row["Functional_Category"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "functional_category",
                    "functional_categories"
                ]
            )
        )

        # -------------------------------------------------------------
        # GO
        # -------------------------------------------------------------

        row["GO_Molecular_Function"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "molecular_function",
                    "molfunction_go",
                    "molecular_function_go"
                ]
            )
        )

        row["GO_Biological_Process"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "biological_process",
                    "bioprocess_go",
                    "biological_process_go"
                ]
            )
        )

        row["GO_Cellular_Component"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "cellular_component",
                    "cellular_component_go"
                ]
            )
        )

        # -------------------------------------------------------------
        # EC
        # -------------------------------------------------------------

        row["EC_Number"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "ECnumber",
                    "ec",
                    "ec_number"
                ]
            )
        )

        # -------------------------------------------------------------
        # KEGG
        # -------------------------------------------------------------

        row["KEGG_Pathway"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "KEGGpathway",
                    "kegg",
                    "kegg_pathway",
                    "KEGG"
                ]
            )
        )

        # -------------------------------------------------------------
        # INTERPRO
        # -------------------------------------------------------------

        row["InterPro_Domains"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "interpro_domains",
                    "interpro",
                    "InterPro"
                ]
            )
        )

        # -------------------------------------------------------------
        # OTHER
        # -------------------------------------------------------------

        row["Phyletic_Profile"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "phyletic_profile"
                ]
            )
        )

        row["Evolutionary_Rate"] = flatten_annotation(
            recursive_find(
                data,
                [
                    "evolutionary_rate"
                ]
            )
        )
    results.append(row)

    time.sleep(SLEEP_SECONDS)

# =============================================================================
# CREATE RESULT TABLE
# =============================================================================

header("BUILDING ANNOTATION TABLE")

external = pd.DataFrame(results)

print("Result shape:")
print(external.shape)

# =============================================================================
# MERGE WITH EXISTING ML EVIDENCE
# =============================================================================

print("\nMerging with existing candidate evidence...")

candidate_columns = [
    "BUSCO_ID",
    "Mean_Importance",
    "SD_Importance",
    "Fold_Stability",
    "Positive_Importance_Folds",
    "Importance_Rank"
]

available_candidate_columns = [
    c for c in candidate_columns
    if c in df.columns
]

existing_ml = df[
    available_candidate_columns
].drop_duplicates(
    subset=["BUSCO_ID"]
)

merged = external.merge(
    existing_ml,
    on="BUSCO_ID",
    how="left"
)

# =============================================================================
# ANNOTATION STATUS
# =============================================================================

def annotation_status(row):

    success = row["OrthoDB_Query_Success"]

    if not success:
        return "OrthoDB_query_failed"

    fields = [
        "GO_Molecular_Function",
        "GO_Biological_Process",
        "GO_Cellular_Component",
        "EC_Number",
        "KEGG_Pathway",
        "InterPro_Domains"
    ]

    available = 0

    for field in fields:

        value = safe_text(
            row.get(field, "")
        )

        if value:
            available += 1

    if available >= 2:
        return "Multi-source_functional_annotation"

    if available == 1:
        return "Partial_functional_annotation"

    return "No_external_functional_annotation"


merged[
    "External_Annotation_Status"
] = merged.apply(
    annotation_status,
    axis=1
)

# =============================================================================
# PATHWAY SUPPORT
# =============================================================================

def pathway_status(row):

    kegg = safe_text(
        row.get("KEGG_Pathway", "")
    )

    ec = safe_text(
        row.get("EC_Number", "")
    )

    description = safe_text(
        row.get("OrthoDB_Name", "")
    )

    if kegg:
        return "KEGG_pathway_supported"

    if ec:
        return "EC_supported_no_KEGG_pathway"

    if description:
        return "Functional_description_only"

    return "No_pathway_evidence"


merged[
    "Pathway_Evidence_Status"
] = merged.apply(
    pathway_status,
    axis=1
)

# =============================================================================
# INTERPRO / GO SUPPORT
# =============================================================================

def domain_support(row):

    interpro = safe_text(
        row.get("InterPro_Domains", "")
    )

    if interpro:
        return "InterPro_supported"

    return "No_InterPro_mapping_retrieved"


merged[
    "InterPro_Evidence_Status"
] = merged.apply(
    domain_support,
    axis=1
)

# =============================================================================
# PUBLICATION INTERPRETATION LEVEL
# =============================================================================

def interpretation_level(row):

    status = row[
        "External_Annotation_Status"
    ]

    pathway = row[
        "Pathway_Evidence_Status"
    ]

    stability = row.get(
        "Fold_Stability",
        np.nan
    )

    if (
        status == "Multi-source_functional_annotation"
        and
        pathway == "KEGG_pathway_supported"
        and
        pd.notna(stability)
        and
        stability >= 0.8
    ):
        return "Strong_candidate_support"

    if (
        status in [
            "Multi-source_functional_annotation",
            "Partial_functional_annotation"
        ]
        and
        pd.notna(stability)
        and
        stability >= 0.6
    ):
        return "Moderate_candidate_support"

    if status != "OrthoDB_query_failed":
        return "Annotation_requires_interpretation"

    return "Requires_manual_review"


merged[
    "Publication_Interpretation_Level"
] = merged.apply(
    interpretation_level,
    axis=1
)

# =============================================================================
# REORDER COLUMNS
# =============================================================================

priority_columns = [
    "BUSCO_ID",
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
    "Mean_Importance",
    "SD_Importance",
    "Fold_Stability",
    "Positive_Importance_Folds",
    "Importance_Rank",
    "Publication_Interpretation_Level",
    "Query_Error"
]

priority_columns = [
    c for c in priority_columns
    if c in merged.columns
]

remaining_columns = [
    c for c in merged.columns
    if c not in priority_columns
]

merged = merged[
    priority_columns + remaining_columns
]

# =============================================================================
# SAVE MASTER TABLE
# =============================================================================

master_file = (
    TABLE_DIR
    / "Carbon_Breadth_external_functional_annotation_420.csv"
)

merged.to_csv(
    master_file,
    index=False
)

print("\nMASTER ANNOTATION TABLE:")
print(master_file)

# =============================================================================
# SAVE PATHWAY-SUPPORTED CANDIDATES
# =============================================================================

pathway_supported = merged[
    merged[
        "Pathway_Evidence_Status"
    ].isin(
        [
            "KEGG_pathway_supported",
            "EC_supported_no_KEGG_pathway"
        ]
    )
].copy()

pathway_file = (
    TABLE_DIR
    / "Carbon_Breadth_pathway_supported_candidates_420.csv"
)

pathway_supported.to_csv(
    pathway_file,
    index=False
)

print("\nPATHWAY-SUPPORTED CANDIDATES:")
print(pathway_file)

# =============================================================================
# SAVE GO-ANNOTATED CANDIDATES
# =============================================================================

go_mask = (
    merged[
        [
            "GO_Molecular_Function",
            "GO_Biological_Process",
            "GO_Cellular_Component"
        ]
    ]
    .fillna("")
    .astype(str)
    .apply(
        lambda row: any(
            x.strip() != ""
            for x in row
        ),
        axis=1
    )
)

go_supported = merged[
    go_mask
].copy()

go_file = (
    TABLE_DIR
    / "Carbon_Breadth_GO_annotated_candidates_420.csv"
)

go_supported.to_csv(
    go_file,
    index=False
)

print("\nGO-ANNOTATED CANDIDATES:")
print(go_file)

# =============================================================================
# SAVE INTERPRO-ANNOTATED CANDIDATES
# =============================================================================

interpro_supported = merged[
    merged[
        "InterPro_Domains"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    != ""
].copy()

interpro_file = (
    TABLE_DIR
    / "Carbon_Breadth_InterPro_annotated_candidates_420.csv"
)

interpro_supported.to_csv(
    interpro_file,
    index=False
)

print("\nINTERPRO-ANNOTATED CANDIDATES:")
print(interpro_file)

# =============================================================================
# SAVE MANUAL REVIEW TABLE
# =============================================================================

manual_review = merged[
    merged[
        "Publication_Interpretation_Level"
    ].isin(
        [
            "Annotation_requires_interpretation",
            "Requires_manual_review"
        ]
    )
].copy()

manual_file = (
    TABLE_DIR
    / "Carbon_Breadth_annotation_manual_review_420.csv"
)

manual_review.to_csv(
    manual_file,
    index=False
)

print("\nMANUAL REVIEW TABLE:")
print(manual_file)

# =============================================================================
# SUMMARY
# =============================================================================

header("ANNOTATION SUMMARY")

print(
    merged[
        [
            "BUSCO_ID",
            "OrthoDB_Name",
            "External_Annotation_Status",
            "Pathway_Evidence_Status",
            "Publication_Interpretation_Level"
        ]
    ].to_string(index=False)
)

print("\n\nAnnotation status counts:")
print(
    merged[
        "External_Annotation_Status"
    ].value_counts(
        dropna=False
    )
)

print("\nPathway evidence counts:")
print(
    merged[
        "Pathway_Evidence_Status"
    ].value_counts(
        dropna=False
    )
)

print("\nPublication interpretation:")
print(
    merged[
        "Publication_Interpretation_Level"
    ].value_counts(
        dropna=False
    )
)

# =============================================================================
# JSON SUMMARY
# =============================================================================

summary = {

    "phenotype": "Carbon_Breadth",

    "candidate_count": int(
        len(merged)
    ),

    "orthodb_success_count": int(
        merged[
            "OrthoDB_Query_Success"
        ].sum()
    ),

    "GO_annotated_count": int(
        len(go_supported)
    ),

    "InterPro_annotated_count": int(
        len(interpro_supported)
    ),

    "pathway_supported_count": int(
        len(pathway_supported)
    ),

    "manual_review_count": int(
        len(manual_review)
    ),

    "annotation_status_counts":
        merged[
            "External_Annotation_Status"
        ]
        .value_counts()
        .to_dict(),

    "pathway_status_counts":
        merged[
            "Pathway_Evidence_Status"
        ]
        .value_counts()
        .to_dict(),

    "important_note": (
        "GO, KEGG, EC and InterPro annotations are retained "
        "only when returned by the external annotation source. "
        "No pathway assignments are inferred solely from keywords."
    ),

    "source": (
        "OrthoDB v12 group annotation API"
    )
}

summary_file = (
    TABLE_DIR
    / "Carbon_Breadth_external_annotation_summary_420.json"
)

with open(
    summary_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2,
        ensure_ascii=False
    )

print("\nJSON summary:")
print(summary_file)

# =============================================================================
# FINAL
# =============================================================================

header("EXTERNAL ANNOTATION COMPLETE")

print(f"""
Candidates processed:
{len(merged)}

OrthoDB queries successful:
{int(merged["OrthoDB_Query_Success"].sum())}

GO annotated:
{len(go_supported)}

InterPro annotated:
{len(interpro_supported)}

Pathway-supported:
{len(pathway_supported)}

Manual review:
{len(manual_review)}

MASTER TABLE:
{master_file}

PATHWAY TABLE:
{pathway_file}

GO TABLE:
{go_file}

INTERPRO TABLE:
{interpro_file}

MANUAL REVIEW:
{manual_file}

SUMMARY:
{summary_file}

NEXT STAGE:
Pathway interpretation → pathway-level grouping
→ network construction → candidate chassis interpretation
""")
        
