# ================================================================
# CARBON BREADTH - TOP 20 BUSCO UniProt FUNCTIONAL ANNOTATION
# ================================================================
#
# INPUT:
#   Existing 20-BUSCO functional annotation table
#
# OUTPUT:
#   Candidate UniProt matches
#   GO terms
#   EC numbers
#   KEGG cross-references
#   UniProt pathways
#
# IMPORTANT:
#   This script DOES NOT re-extract proteins.
#   This script DOES NOT search GFF/GTF files.
#   This script DOES NOT use the 26-GB eggNOG database.
#
# ================================================================

import os
import re
import time
import requests
import pandas as pd

# ----------------------------------------------------------------
# INPUT
# ----------------------------------------------------------------

INPUT_FILE = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation\Carbon_Breadth_top20_BUSCO_functional_annotation_input_420.csv"

# ----------------------------------------------------------------
# OUTPUT DIRECTORY
# ----------------------------------------------------------------

OUT_DIR = r"C:\Y1000_chassis_project\results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml\phylogeny_cv\model_training\feature_interpretation_420\functional_annotation_420\BUSCO20_annotation\UniProt"

os.makedirs(OUT_DIR, exist_ok=True)

CANDIDATE_OUT = os.path.join(
    OUT_DIR,
    "Carbon_Breadth_top20_UniProt_candidate_matches_420.csv"
)

FINAL_OUT = os.path.join(
    OUT_DIR,
    "Carbon_Breadth_top20_UniProt_functional_annotations_420.csv"
)

REPORT_OUT = os.path.join(
    OUT_DIR,
    "Carbon_Breadth_top20_UniProt_annotation_report_420.txt"
)

# ----------------------------------------------------------------
# UniProt API
# ----------------------------------------------------------------

UNIPROT_URL = "https://rest.uniprot.org/uniprotkb/search"

# Saccharomycetes = NCBI taxonomy 4891
# Your BUSCO IDs are from this taxonomic group.
TAXON_ID = "4891"

FIELDS = ",".join([
    "accession",
    "id",
    "protein_name",
    "gene_names",
    "organism_name",
    "organism_id",
    "go_id",
    "go",
    "go_p",
    "go_f",
    "go_c",
    "ec",
    "cc_function",
    "cc_pathway",
    "xref_kegg",
    "xref_reactome",
    "xref_unipathway",
    "xref_interpro",
    "xref_pfam",
    "annotation_score",
    "reviewed",
])

# ----------------------------------------------------------------
# LOAD INPUT
# ----------------------------------------------------------------

print("=" * 80)
print("CARBON BREADTH TOP-20 BUSCO → UNIPROT ANNOTATION")
print("=" * 80)

print("\nLoading:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE, dtype=str).fillna("")

print("\nInput shape:")
print(df.shape)

required = [
    "BUSCO_ID",
    "Selected_Function"
]

for col in required:
    if col not in df.columns:
        raise ValueError(
            f"Required column missing: {col}"
        )

print("\nBUSCO candidates:")
for x in df["BUSCO_ID"]:
    print(" ", x)

# ----------------------------------------------------------------
# CLEAN SEARCH TERM
# ----------------------------------------------------------------

def clean_function(text):

    text = str(text).strip()

    # Remove overly generic domain wording where possible.
    text = re.sub(
        r"\b(domain|family|superfamily|fold)\b",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ----------------------------------------------------------------
# SEARCH UNIPROT
# ----------------------------------------------------------------

def search_uniprot(function_text):

    """
    Search within Saccharomycetes.

    We deliberately use a broad text search and retrieve
    annotation fields. Results are NOT automatically accepted
    as definitive annotations.
    """

    query = (
        f'taxonomy_id:{TAXON_ID} '
        f'AND "{function_text}"'
    )

    params = {
        "query": query,
        "format": "tsv",
        "fields": FIELDS,
        "size": 10,
    }

    try:

        response = requests.get(
            UNIPROT_URL,
            params=params,
            timeout=60
        )

        if response.status_code != 200:

            print(
                f"  UniProt HTTP error: "
                f"{response.status_code}"
            )

            return pd.DataFrame()

        text = response.text.strip()

        if not text:
            return pd.DataFrame()

        from io import StringIO

        result = pd.read_csv(
            StringIO(text),
            sep="\t",
            dtype=str
        ).fillna("")

        return result

    except Exception as e:

        print(
            f"  UniProt request failed: {e}"
        )

        return pd.DataFrame()


# ----------------------------------------------------------------
# RUN SEARCH
# ----------------------------------------------------------------

all_matches = []

print("\n" + "=" * 80)
print("SEARCHING UNIPROT")
print("=" * 80)

for i, row in df.iterrows():

    busco = row["BUSCO_ID"]
    function = row["Selected_Function"]

    search_term = clean_function(function)

    print(
        f"\n[{i+1}/{len(df)}] {busco}"
    )

    print(
        f"  Function: {function}"
    )

    print(
        f"  Search: {search_term}"
    )

    results = search_uniprot(search_term)

    if results.empty:

        print("  Matches: 0")

        all_matches.append({
            "BUSCO_ID": busco,
            "Original_Function": function,
            "Search_Term": search_term,
            "UniProt_Accession": "",
            "UniProt_ID": "",
            "Protein_Name": "",
            "Gene_Names": "",
            "Organism": "",
            "GO_IDs": "",
            "GO_Terms": "",
            "GO_Biological_Process": "",
            "GO_Molecular_Function": "",
            "GO_Cellular_Component": "",
            "EC_Number": "",
            "Function_Description": "",
            "UniProt_Pathway": "",
            "KEGG": "",
            "Reactome": "",
            "UniPathway": "",
            "InterPro": "",
            "Pfam": "",
            "Annotation_Score": "",
            "Reviewed": "",
            "Match_Status": "NO_MATCH"
        })

    else:

        print(
            f"  Matches retrieved: {len(results)}"
        )

        for _, r in results.iterrows():

            all_matches.append({
                "BUSCO_ID": busco,
                "Original_Function": function,
                "Search_Term": search_term,
                "UniProt_Accession": r.get(
                    "Entry",
                    ""
                ),
                "UniProt_ID": r.get(
                    "Entry Name",
                    ""
                ),
                "Protein_Name": r.get(
                    "Protein names",
                    ""
                ),
                "Gene_Names": r.get(
                    "Gene Names",
                    ""
                ),
                "Organism": r.get(
                    "Organism",
                    ""
                ),
                "GO_IDs": r.get(
                    "Gene Ontology IDs",
                    ""
                ),
                "GO_Terms": r.get(
                    "Gene Ontology (GO)",
                    ""
                ),
                "GO_Biological_Process": r.get(
                    "Gene Ontology (biological process)",
                    ""
                ),
                "GO_Molecular_Function": r.get(
                    "Gene Ontology (molecular function)",
                    ""
                ),
                "GO_Cellular_Component": r.get(
                    "Gene Ontology (cellular component)",
                    ""
                ),
                "EC_Number": r.get(
                    "EC number",
                    ""
                ),
                "Function_Description": r.get(
                    "Function [CC]",
                    ""
                ),
                "UniProt_Pathway": r.get(
                    "Pathway",
                    ""
                ),
                "KEGG": r.get(
                    "KEGG",
                    ""
                ),
                "Reactome": r.get(
                    "Reactome",
                    ""
                ),
                "UniPathway": r.get(
                    "UniPathway",
                    ""
                ),
                "InterPro": r.get(
                    "InterPro",
                    ""
                ),
                "Pfam": r.get(
                    "Pfam",
                    ""
                ),
                "Annotation_Score": r.get(
                    "Annotation",
                    ""
                ),
                "Reviewed": r.get(
                    "Reviewed",
                    ""
                ),
                "Match_Status": "CANDIDATE"
            })

    # Don't hammer the service.
    time.sleep(0.5)


# ----------------------------------------------------------------
# SAVE ALL CANDIDATE MATCHES
# ----------------------------------------------------------------

matches = pd.DataFrame(all_matches)

matches.to_csv(
    CANDIDATE_OUT,
    index=False
)

# ----------------------------------------------------------------
# BUILD BUSCO-CENTRIC SUMMARY
# ----------------------------------------------------------------

summary_rows = []

for busco in df["BUSCO_ID"].tolist():

    sub = matches[
        matches["BUSCO_ID"] == busco
    ].copy()

    original_function = df.loc[
        df["BUSCO_ID"] == busco,
        "Selected_Function"
    ].iloc[0]

    if sub.empty:

        summary_rows.append({
            "BUSCO_ID": busco,
            "Selected_Function": original_function,
            "UniProt_Candidate_Count": 0,
            "Candidate_UniProt_Accessions": "",
            "Candidate_Protein_Names": "",
            "Candidate_GO_IDs": "",
            "Candidate_EC_Numbers": "",
            "Candidate_KEGG_IDs": "",
            "Candidate_Pathways": "",
            "Annotation_Status": "NO_UNIPROT_MATCH"
        })

        continue

    def unique_join(series):

        values = []

        for x in series.astype(str):

            if not x.strip():
                continue

            # UniProt may return semicolon-separated values.
            for item in x.split(";"):

                item = item.strip()

                if item and item not in values:
                    values.append(item)

        return "; ".join(values)

    summary_rows.append({

        "BUSCO_ID": busco,

        "Selected_Function":
            original_function,

        "UniProt_Candidate_Count":
            len(sub),

        "Candidate_UniProt_Accessions":
            unique_join(
                sub["UniProt_Accession"]
            ),

        "Candidate_Protein_Names":
            unique_join(
                sub["Protein_Name"]
            ),

        "Candidate_GO_IDs":
            unique_join(
                sub["GO_IDs"]
            ),

        "Candidate_EC_Numbers":
            unique_join(
                sub["EC_Number"]
            ),

        "Candidate_KEGG_IDs":
            unique_join(
                sub["KEGG"]
            ),

        "Candidate_Pathways":
            unique_join(
                sub["UniProt_Pathway"]
            ),

        "Annotation_Status":
            "CANDIDATE_MATCHES_RETRIEVED"
    })


summary = pd.DataFrame(summary_rows)

summary.to_csv(
    FINAL_OUT,
    index=False
)

# ----------------------------------------------------------------
# REPORT
# ----------------------------------------------------------------

with open(
    REPORT_OUT,
    "w",
    encoding="utf-8"
) as f:

    f.write("=" * 80 + "\n")
    f.write(
        "CARBON BREADTH TOP-20 BUSCO UNIPROT ANNOTATION\n"
    )
    f.write("=" * 80 + "\n\n")

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "UniProt matches are candidate annotations and must be "
        "reviewed before being treated as definitive.\n\n"
    )

    f.write(
        f"BUSCO candidates: {len(df)}\n"
    )

    f.write(
        "BUSCOs with UniProt matches: "
        f"{sum(summary['UniProt_Candidate_Count'] > 0)}\n"
    )

    f.write("\n\n")

    for _, r in summary.iterrows():

        f.write(
            f"{r['BUSCO_ID']}\n"
        )

        f.write(
            f"Function: "
            f"{r['Selected_Function']}\n"
        )

        f.write(
            f"UniProt candidates: "
            f"{r['UniProt_Candidate_Count']}\n"
        )

        f.write(
            f"GO: "
            f"{r['Candidate_GO_IDs']}\n"
        )

        f.write(
            f"EC: "
            f"{r['Candidate_EC_Numbers']}\n"
        )

        f.write(
            f"KEGG: "
            f"{r['Candidate_KEGG_IDs']}\n"
        )

        f.write(
            f"Pathways: "
            f"{r['Candidate_Pathways']}\n"
        )

        f.write("\n")


# ----------------------------------------------------------------
# FINAL CONSOLE OUTPUT
# ----------------------------------------------------------------

print("\n" + "=" * 80)
print("UNIPROT SEARCH COMPLETE")
print("=" * 80)

print(
    "\nBUSCO candidates:",
    len(df)
)

print(
    "BUSCOs with candidate UniProt matches:",
    sum(
        summary["UniProt_Candidate_Count"] > 0
    )
)

print(
    "Total candidate UniProt records:",
    len(matches)
)

print("\nCandidate matches:")
print(CANDIDATE_OUT)

print("\nBUSCO-level annotation summary:")
print(FINAL_OUT)

print("\nReport:")
print(REPORT_OUT)

print("\n" + "=" * 80)
print("IMPORTANT")
print("=" * 80)

print(
    "\nDo NOT yet treat every retrieved GO/KEGG term as confirmed."
)

print(
    "We will use the candidate-match table to select "
    "the biologically appropriate annotation for each BUSCO."
)

print("\nDONE.")