# ================================================================
# CARBON BREADTH 420
# BUSCO -> PROTEIN/SEQUENCE -> FUNCTIONAL ANNOTATION
# NO NCBI / NO NCBI EMAIL REQUIRED
# ================================================================

import os
import re
import json
import time
import requests
import pandas as pd
from collections import defaultdict

# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage5_phylogeny_ml_dataset",
    "phylogeny_aware_ml",
    "phylogeny_cv",
    "model_training",
    "feature_interpretation_420"
)

TOP20_FILE = os.path.join(
    FEATURE_DIR,
    "tables",
    "Carbon_Breadth_top20_BUSCO_features_420.csv"
)

MAPPING_FILE = os.path.join(
    FEATURE_DIR,
    "functional_annotation_420",
    "busco_gene_mapping_420",
    "tables",
    "Carbon_Breadth_BUSCO_gene_protein_mapping_420.csv"
)

OUT_DIR = os.path.join(
    FEATURE_DIR,
    "functional_annotation_420",
    "gene_level_annotation_420"
)

TABLE_DIR = os.path.join(OUT_DIR, "tables")
REPORT_DIR = os.path.join(OUT_DIR, "reports")

os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MASTER_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_gene_functional_annotation_420.csv"
)

SUMMARY_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_functional_annotation_summary_420.csv"
)

GO_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_GO_annotations_420.csv"
)

PATHWAY_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_pathway_annotations_420.csv"
)

INTERPRO_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_InterPro_annotations_420.csv"
)

FUNCTIONAL_CANDIDATES_OUT = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_functional_candidates_420.csv"
)

REPORT_OUT = os.path.join(
    REPORT_DIR,
    "Carbon_Breadth_BUSCO_functional_annotation_report_420.txt"
)

# UniProt REST
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

# Avoid aggressive requests
REQUEST_DELAY = 0.25

# ================================================================
# HELPERS
# ================================================================

def clean(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def split_values(value):
    """
    Split annotation fields while retaining meaningful values.
    """
    value = clean(value)

    if not value:
        return []

    parts = re.split(r"[;,|]+", value)

    result = []

    for p in parts:
        p = p.strip()

        if p and p.lower() not in {
            "none",
            "nan",
            "na",
            "n/a",
            "-",
            "unknown",
            "not available"
        }:
            result.append(p)

    return list(dict.fromkeys(result))


def looks_like_protein_accession(value):
    """
    Conservative accession detection.

    Examples that may match:
        P12345
        Q9ABC1
        A0A1234567
        XP_123456.1
        WP_123456789.1
    """

    value = clean(value)

    patterns = [
        r"\b[OPQ][0-9][A-Z0-9]{3}[0-9]\b",
        r"\b[A-Z][0-9]{5}\b",
        r"\b[A-Z0-9]{6,10}\.[0-9]+\b",
        r"\b(?:XP|WP|YP|NP|AP|SP|TP)_?[0-9]+\.[0-9]+\b",
        r"\b(?:XP|WP|YP|NP|AP|SP|TP)_?[0-9]+\b"
    ]

    for pattern in patterns:
        match = re.search(pattern, value)

        if match:
            return match.group(0)

    return ""


def extract_accession_from_row(row):
    """
    Search existing mapping table for accession-like identifiers.
    No external database is required for this step.
    """

    priority_columns = [
        "Protein_ID",
        "Protein",
        "Protein_IDs",
        "Protein_Accession",
        "Protein_Accession_ID",
        "Sequence_ID",
        "Sequence",
        "Sequence_IDs",
        "Gene_ID",
        "Gene",
        "Gene_IDs",
        "Locus",
        "Locus_Tag",
        "Primary_Locus",
        "Product",
        "Product_Name",
        "Functional_Description"
    ]

    for col in priority_columns:

        if col not in row.index:
            continue

        value = clean(row[col])

        if not value:
            continue

        accession = looks_like_protein_accession(value)

        if accession:
            return accession

    # Search every column as a fallback
    for col in row.index:

        value = clean(row[col])

        if not value:
            continue

        accession = looks_like_protein_accession(value)

        if accession:
            return accession

    return ""


def unique_join(values):
    cleaned = []

    for value in values:

        value = clean(value)

        if not value:
            continue

        if value not in cleaned:
            cleaned.append(value)

    return "; ".join(cleaned)


# ================================================================
# HEADER
# ================================================================

print("=" * 80)
print("CARBON BREADTH — BUSCO FUNCTIONAL ANNOTATION")
print("NO NCBI / NO NCBI EMAIL REQUIRED")
print("=" * 80)

# ================================================================
# CHECK INPUTS
# ================================================================

if not os.path.exists(TOP20_FILE):
    raise FileNotFoundError(
        f"\nTop-20 BUSCO file not found:\n{TOP20_FILE}"
    )

if not os.path.exists(MAPPING_FILE):
    raise FileNotFoundError(
        f"\nBUSCO mapping file not found:\n{MAPPING_FILE}"
    )

# ================================================================
# LOAD TOP 20
# ================================================================

print("\n" + "=" * 80)
print("LOADING TOP-20 BUSCO CANDIDATES")
print("=" * 80)

top20 = pd.read_csv(TOP20_FILE)

print("Shape:", top20.shape)
print("Columns:", list(top20.columns))

if "Feature" in top20.columns:
    candidate_column = "Feature"
elif "BUSCO_ID" in top20.columns:
    candidate_column = "BUSCO_ID"
else:
    raise KeyError(
        "Could not identify BUSCO candidate column in top-20 file."
    )

candidate_buscos = (
    top20[candidate_column]
    .astype(str)
    .str.strip()
    .dropna()
    .unique()
    .tolist()
)

print("\nTop-20 BUSCO candidates:")
for x in candidate_buscos:
    print(" ", x)

print("\nUnique candidates:", len(candidate_buscos))

# ================================================================
# LOAD MAPPING
# ================================================================

print("\n" + "=" * 80)
print("LOADING EXISTING BUSCO GENE/PROTEIN MAPPING")
print("=" * 80)

mapping = pd.read_csv(MAPPING_FILE, low_memory=False)

print("Mapping shape:", mapping.shape)

print("\nMapping columns:")
print(list(mapping.columns))

# ================================================================
# IDENTIFY BUSCO COLUMN
# ================================================================

busco_col = first_existing(
    mapping,
    [
        "BUSCO_ID",
        "Busco_ID",
        "BUSCO",
        "busco_id"
    ]
)

if busco_col is None:
    raise KeyError(
        "BUSCO_ID column not found in mapping table."
    )

mapping[busco_col] = mapping[busco_col].astype(str).str.strip()

mapping = mapping[
    mapping[busco_col].isin(candidate_buscos)
].copy()

print("\nRows belonging to top-20 BUSCOs:", len(mapping))

# ================================================================
# INSPECT WHAT WE ACTUALLY HAVE
# ================================================================

print("\n" + "=" * 80)
print("INSPECTING AVAILABLE IDENTIFIERS")
print("=" * 80)

identifier_columns = [
    "Sequence_ID",
    "Sequence",
    "Gene_ID",
    "Gene",
    "Gene_Name",
    "Locus",
    "Locus_Tag",
    "Primary_Locus",
    "Protein_ID",
    "Protein",
    "Protein_Accession",
    "Product",
    "Product_Name",
    "Functional_Description"
]

available_identifier_columns = [
    c for c in identifier_columns
    if c in mapping.columns
]

print("Available relevant columns:")

for c in available_identifier_columns:
    print(" ", c)

# ================================================================
# EXTRACT EXISTING ACCESSIONS
# ================================================================

print("\n" + "=" * 80)
print("EXTRACTING ACCESSION-LIKE IDENTIFIERS")
print("=" * 80)

mapping["Recovered_Accession"] = mapping.apply(
    extract_accession_from_row,
    axis=1
)

accessions = (
    mapping["Recovered_Accession"]
    .replace("", pd.NA)
    .dropna()
    .unique()
    .tolist()
)

print("Recovered accession-like IDs:", len(accessions))

if len(accessions) > 0:

    print("\nExample IDs:")
    for x in accessions[:20]:
        print(" ", x)

else:

    print(
        "\nWARNING:"
        "\nNo accession-like protein IDs were found in the existing mapping."
    )

    print(
        "\nThe script will still create a complete evidence table "
        "using the functional descriptions already recovered."
    )

# ================================================================
# UNIPROT QUERY
# ================================================================

def query_uniprot(accession):

    if not accession:
        return {}

    params = {
        "query": f"accession:{accession}",
        "format": "json",
        "size": 1
    }

    try:

        response = requests.get(
            UNIPROT_SEARCH_URL,
            params=params,
            timeout=30
        )

        if response.status_code != 200:
            return {
                "Query_Status": "HTTP_ERROR",
                "HTTP_Status": response.status_code
            }

        data = response.json()

        results = data.get("results", [])

        if not results:
            return {
                "Query_Status": "NO_RESULT",
                "HTTP_Status": response.status_code
            }

        entry = results[0]

        # --------------------------------------------------------
        # Protein name
        # --------------------------------------------------------

        protein_name = ""

        try:
            protein_name = (
                entry
                .get("proteinDescription", {})
                .get("recommendedName", {})
                .get("fullName", {})
                .get("value", "")
            )
        except Exception:
            pass

        # --------------------------------------------------------
        # Gene names
        # --------------------------------------------------------

        gene_names = []

        for gene in entry.get("genes", []):

            if "geneName" in gene:

                value = gene["geneName"].get("value")

                if value:
                    gene_names.append(value)

        # --------------------------------------------------------
        # GO
        # --------------------------------------------------------

        go_mf = []
        go_bp = []
        go_cc = []

        # --------------------------------------------------------
        # EC
        # --------------------------------------------------------

        ec_numbers = []

        # --------------------------------------------------------
        # InterPro
        # --------------------------------------------------------

        interpro = []

        # --------------------------------------------------------
        # Pathways
        # --------------------------------------------------------

        pathways = []

        for dbref in entry.get("uniProtKBCrossReferences", []):

            database = dbref.get("database", "")
            primary_id = dbref.get("id", "")

            properties = dbref.get("properties", [])

            property_dict = {
                p.get("key", ""): p.get("value", "")
                for p in properties
            }

            if database == "GO":

                term = property_dict.get("GoTerm", "")

                if term.startswith("F:"):
                    go_mf.append(term[2:])

                elif term.startswith("P:"):
                    go_bp.append(term[2:])

                elif term.startswith("C:"):
                    go_cc.append(term[2:])

            elif database == "InterPro":

                interpro.append(
                    f"{primary_id}: "
                    f"{property_dict.get('EntryName', '')}"
                )

            elif database == "EC":

                ec_numbers.append(primary_id)

            elif database in {
                "KEGG",
                "Reactome",
                "BioCyc",
                "PathwayCommons"
            }:

                pathways.append(
                    f"{database}:{primary_id}"
                )

        return {
            "Query_Status": "SUCCESS",
            "HTTP_Status": response.status_code,
            "UniProt_Accession": entry.get("primaryAccession", ""),
            "Protein_Name": protein_name,
            "Gene_Names": unique_join(gene_names),
            "GO_Molecular_Function": unique_join(go_mf),
            "GO_Biological_Process": unique_join(go_bp),
            "GO_Cellular_Component": unique_join(go_cc),
            "EC_Number": unique_join(ec_numbers),
            "InterPro_Domains": unique_join(interpro),
            "Pathways": unique_join(pathways),
            "Organism": (
                entry.get("organism", {})
                .get("scientificName", "")
            )
        }

    except Exception as e:

        return {
            "Query_Status": "ERROR",
            "HTTP_Status": "",
            "Query_Error": str(e)
        }


# ================================================================
# RUN UNIPROT ONLY IF ACCESSIONS EXIST
# ================================================================

print("\n" + "=" * 80)
print("EXTERNAL FUNCTIONAL ANNOTATION")
print("=" * 80)

uniprot_results = {}

if accessions:

    total = len(accessions)

    for i, accession in enumerate(accessions, 1):

        print(
            f"[{i}/{total}] UniProt: {accession}"
        )

        result = query_uniprot(accession)

        uniprot_results[accession] = result

        time.sleep(REQUEST_DELAY)

else:

    print(
        "No accession IDs available."
    )

    print(
        "Skipping UniProt queries."
    )

# ================================================================
# MERGE UNIPROT ANNOTATIONS
# ================================================================

print("\n" + "=" * 80)
print("MERGING FUNCTIONAL ANNOTATION")
print("=" * 80)

annotation_columns = [
    "Query_Status",
    "HTTP_Status",
    "UniProt_Accession",
    "Protein_Name",
    "Gene_Names",
    "GO_Molecular_Function",
    "GO_Biological_Process",
    "GO_Cellular_Component",
    "EC_Number",
    "InterPro_Domains",
    "Pathways",
    "Organism"
]

for col in annotation_columns:
    mapping[col] = ""

for idx, row in mapping.iterrows():

    accession = clean(row["Recovered_Accession"])

    if accession and accession in uniprot_results:

        result = uniprot_results[accession]

        for col, value in result.items():

            if col in mapping.columns:
                mapping.at[idx, col] = value

# ================================================================
# FUNCTIONAL EVIDENCE FROM EXISTING BUSCO ANNOTATION
# ================================================================

print("\n" + "=" * 80)
print("RECOVERING EXISTING BUSCO FUNCTIONAL EVIDENCE")
print("=" * 80)

description_col = first_existing(
    mapping,
    [
        "Functional_Description",
        "Product",
        "Product_Name",
        "Description"
    ]
)

if description_col:

    mapping["Existing_Functional_Description"] = (
        mapping[description_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

else:

    mapping["Existing_Functional_Description"] = ""

# ================================================================
# EVIDENCE FLAGS
# ================================================================

def has_value(x):
    return bool(clean(x))


mapping["GO_Supported"] = (
    mapping[
        [
            "GO_Molecular_Function",
            "GO_Biological_Process",
            "GO_Cellular_Component"
        ]
    ]
    .fillna("")
    .astype(str)
    .apply(
        lambda row: any(has_value(x) for x in row),
        axis=1
    )
    .astype(int)
)

mapping["InterPro_Supported"] = (
    mapping["InterPro_Domains"]
    .fillna("")
    .astype(str)
    .apply(has_value)
    .astype(int)
)

mapping["KEGG_Pathway_Supported"] = (
    mapping["Pathways"]
    .fillna("")
    .astype(str)
    .str.contains(
        "KEGG",
        case=False,
        na=False
    )
    .astype(int)
)

mapping["EC_Supported"] = (
    mapping["EC_Number"]
    .fillna("")
    .astype(str)
    .apply(has_value)
    .astype(int)
)

mapping["Existing_Functional_Description_Supported"] = (
    mapping["Existing_Functional_Description"]
    .fillna("")
    .astype(str)
    .apply(has_value)
    .astype(int)
)

mapping["External_Evidence_Count"] = (
    mapping["GO_Supported"]
    + mapping["InterPro_Supported"]
    + mapping["KEGG_Pathway_Supported"]
    + mapping["EC_Supported"]
)

mapping["Any_Functional_Evidence"] = (
    (
        (mapping["External_Evidence_Count"] > 0)
        |
        (
            mapping[
                "Existing_Functional_Description_Supported"
            ] == 1
        )
    )
    .astype(int)
)

# ================================================================
# COLLAPSE TO BUSCO LEVEL
# ================================================================

print("\n" + "=" * 80)
print("COLLAPSING EVIDENCE TO BUSCO LEVEL")
print("=" * 80)

# Candidate-level predictive information
predictive_cols = [
    "Mean_Importance",
    "SD_Importance",
    "Median_Importance",
    "Min_Importance",
    "Max_Importance",
    "Mean_Baseline_R2",
    "Positive_Importance_Folds",
    "Fold_Stability",
    "Importance_Rank"
]

available_predictive_cols = [
    c for c in predictive_cols
    if c in mapping.columns
]

# Aggregation dictionary
agg = {}

for col in available_predictive_cols:
    agg[col] = "first"

for col in [
    "Recovered_Accession",
    "UniProt_Accession",
    "Protein_Name",
    "Gene_Names",
    "GO_Molecular_Function",
    "GO_Biological_Process",
    "GO_Cellular_Component",
    "EC_Number",
    "InterPro_Domains",
    "Pathways",
    "Organism",
    "Existing_Functional_Description"
]:
    agg[col] = lambda x: unique_join(x)

agg["GO_Supported"] = "max"
agg["InterPro_Supported"] = "max"
agg["KEGG_Pathway_Supported"] = "max"
agg["EC_Supported"] = "max"
agg["Existing_Functional_Description_Supported"] = "max"
agg["Any_Functional_Evidence"] = "max"

busco_level = (
    mapping
    .groupby(busco_col, as_index=False)
    .agg(agg)
)

busco_level = busco_level.rename(
    columns={busco_col: "BUSCO_ID"}
)

# ================================================================
# COUNT UNIQUE IDENTIFIERS
# ================================================================

print("\n" + "=" * 80)
print("FUNCTIONAL MAPPING SUMMARY")
print("=" * 80)

total_candidates = len(candidate_buscos)

mapped_candidates = busco_level["BUSCO_ID"].nunique()

go_supported = int(
    busco_level["GO_Supported"].sum()
)

interpro_supported = int(
    busco_level["InterPro_Supported"].sum()
)

kegg_supported = int(
    busco_level["KEGG_Pathway_Supported"].sum()
)

ec_supported = int(
    busco_level["EC_Supported"].sum()
)

functional_description_supported = int(
    busco_level[
        "Existing_Functional_Description_Supported"
    ].sum()
)

any_external = int(
    (
        (
            busco_level["GO_Supported"]
            + busco_level["InterPro_Supported"]
            + busco_level["KEGG_Pathway_Supported"]
            + busco_level["EC_Supported"]
        ) > 0
    ).sum()
)

print("Total BUSCO candidates       :", total_candidates)
print("BUSCO candidates represented :", mapped_candidates)
print("GO-supported                 :", go_supported)
print("InterPro-supported           :", interpro_supported)
print("KEGG-supported               :", kegg_supported)
print("EC-supported                 :", ec_supported)
print(
    "Existing functional descriptions:",
    functional_description_supported
)
print(
    "Any external functional evidence:",
    any_external
)

# ================================================================
# CANDIDATE CLASSIFICATION
# ================================================================

def classify(row):

    external_count = (
        int(row["GO_Supported"])
        + int(row["InterPro_Supported"])
        + int(row["KEGG_Pathway_Supported"])
        + int(row["EC_Supported"])
    )

    existing = int(
        row["Existing_Functional_Description_Supported"]
    )

    if external_count >= 2:
        return "Strong external functional support"

    if external_count == 1:
        return "Moderate external functional support"

    if existing == 1:
        return "Functional description available; external validation required"

    return "No functional annotation recovered"


busco_level["Functional_Evidence_Class"] = (
    busco_level.apply(classify, axis=1)
)

# ================================================================
# SAVE MASTER TABLE
# ================================================================

busco_level.to_csv(
    MASTER_OUT,
    index=False
)

# ================================================================
# GO TABLE
# ================================================================

go_table = busco_level[
    busco_level["GO_Supported"] == 1
].copy()

go_table.to_csv(
    GO_OUT,
    index=False
)

# ================================================================
# PATHWAY TABLE
# ================================================================

pathway_table = busco_level[
    busco_level["KEGG_Pathway_Supported"] == 1
].copy()

pathway_table.to_csv(
    PATHWAY_OUT,
    index=False
)

# ================================================================
# INTERPRO TABLE
# ================================================================

interpro_table = busco_level[
    busco_level["InterPro_Supported"] == 1
].copy()

interpro_table.to_csv(
    INTERPRO_OUT,
    index=False
)

# ================================================================
# FUNCTIONAL CANDIDATES
# ================================================================

functional_candidates = busco_level[
    busco_level["External_Evidence_Count"] > 0
].copy()

functional_candidates.to_csv(
    FUNCTIONAL_CANDIDATES_OUT,
    index=False
)

# ================================================================
# SUMMARY TABLE
# ================================================================

summary = pd.DataFrame([
    {
        "Total_BUSCO_Candidates": total_candidates,
        "BUSCO_Candidates_Represented": mapped_candidates,
        "GO_Supported": go_supported,
        "InterPro_Supported": interpro_supported,
        "KEGG_Supported": kegg_supported,
        "EC_Supported": ec_supported,
        "Existing_Functional_Description": functional_description_supported,
        "Any_External_Evidence": any_external
    }
])

summary.to_csv(
    SUMMARY_OUT,
    index=False
)

# ================================================================
# REPORT
# ================================================================

report = f"""
CARBON BREADTH 420
BUSCO FUNCTIONAL ANNOTATION REPORT
===================================

Input BUSCO candidates:
{total_candidates}

BUSCO candidates represented:
{mapped_candidates}

GO-supported:
{go_supported}

InterPro-supported:
{interpro_supported}

KEGG-supported:
{kegg_supported}

EC-supported:
{ec_supported}

Existing functional descriptions:
{functional_description_supported}

Any external functional evidence:
{any_external}

NCBI:
NOT USED

NCBI EMAIL:
NOT REQUIRED

Interpretation:
The BUSCO candidates were retained independently of whether
external annotation was recovered.

External functional evidence was only assigned when supported
by the available annotation.

A missing GO/KEGG/InterPro annotation does NOT mean that the
BUSCO is biologically non-functional. It means that the current
identifier-to-annotation mapping did not recover that evidence.

The next stage should use candidates with recovered identifiers
for deeper pathway analysis and retain unsupported BUSCOs as
predictive candidates requiring additional annotation.
"""

with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)

# ================================================================
# FINAL OUTPUT
# ================================================================

print("\n" + "=" * 80)
print("BUSCO FUNCTIONAL ANNOTATION COMPLETE")
print("=" * 80)

print("\nMaster table:")
print(MASTER_OUT)

print("\nGO table:")
print(GO_OUT)

print("\nPathway table:")
print(PATHWAY_OUT)

print("\nInterPro table:")
print(INTERPRO_OUT)

print("\nFunctional candidates:")
print(FUNCTIONAL_CANDIDATES_OUT)

print("\nSummary:")
print(SUMMARY_OUT)

print("\nReport:")
print(REPORT_OUT)

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

if len(accessions) == 0:

    print(
        "\nNo usable protein accession was recovered."
    )

    print(
        "Therefore external UniProt annotation was NOT queried."
    )

    print(
        "Your existing BUSCO functional descriptions are still retained."
    )

else:

    print(
        f"\nRecovered {len(accessions)} accession-like identifiers."
    )

    print(
        "UniProt annotation was queried for these identifiers."
    )

print(
    "\nNext stage:"
    "\nBUSCO → recovered identifiers"
    "\n→ GO / InterPro / EC / KEGG"
    "\n→ pathway grouping"
    "\n→ network analysis"
    "\n→ candidate chassis interpretation"
)

print("\nExisting predictive models were NOT modified.")
print("=" * 80)