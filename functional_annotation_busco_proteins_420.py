import os
import sys
import shutil
import subprocess
import pandas as pd
import json
from pathlib import Path

# ============================================================
# CARBON BREADTH 420
# BUSCO PROTEIN FUNCTIONAL ANNOTATION
#
# Input:
#   7,162 unique BUSCO-derived protein sequences
#
# Annotation:
#   eggNOG-mapper
#
# Outputs:
#   GO
#   KEGG KO
#   KEGG pathways
#   EC
#   COG
#   Functional descriptions
#   BUSCO-level summaries
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = r"C:\Y1000_chassis_project"

RESULTS = os.path.join(
    BASE,
    r"results\stage5_phylogeny_ml_dataset"
    r"\phylogeny_aware_ml"
    r"\phylogeny_cv"
    r"\model_training"
    r"\feature_interpretation_420"
)

INPUT_DIR = os.path.join(
    RESULTS,
    r"functional_annotation_420"
    r"protein_functional_annotation_420"
)

INPUT_FASTA = os.path.join(
    INPUT_DIR,
    r"fasta\Carbon_Breadth_top20_unique_proteins_420.faa"
)

OUTPUT_DIR = os.path.join(
    RESULTS,
    r"functional_annotation_420"
    r"eggnog_functional_annotation_420"
)

TABLE_DIR = os.path.join(
    OUTPUT_DIR,
    "tables"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

ANNOTATION_DIR = os.path.join(
    OUTPUT_DIR,
    "eggnog_output"
)

for directory in [
    OUTPUT_DIR,
    TABLE_DIR,
    REPORT_DIR,
    ANNOTATION_DIR
]:
    os.makedirs(directory, exist_ok=True)


# ============================================================
# BUSCO CANDIDATES
# ============================================================

TOP_BUSCOS = [
    "36839at4891",
    "5746at4891",
    "23379at4891",
    "6711at4891",
    "2471at4891",
    "29457at4891",
    "27915at4891",
    "1679at4891",
    "18913at4891",
    "22532at4891",
    "28058at4891",
    "4322at4891",
    "33531at4891",
    "3574at4891",
    "12523at4891",
    "12468at4891",
    "2307at4891",
    "24318at4891",
    "9782at4891",
    "34052at4891"
]


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("CARBON BREADTH 420")
print("BUSCO PROTEIN FUNCTIONAL ANNOTATION")
print("=" * 80)


# ============================================================
# CHECK INPUT
# ============================================================

print("\nInput FASTA:")
print(INPUT_FASTA)

if not os.path.exists(INPUT_FASTA):

    raise FileNotFoundError(
        "\nInput FASTA not found:\n"
        + INPUT_FASTA
    )


# ============================================================
# CHECK FASTA
# ============================================================

protein_count = 0

with open(
    INPUT_FASTA,
    "r",
    encoding="utf-8",
    errors="ignore"
) as handle:

    for line in handle:

        if line.startswith(">"):

            protein_count += 1


print("\nProtein sequences detected:")
print(protein_count)

if protein_count == 0:

    raise ValueError(
        "No protein sequences were found in FASTA."
    )


# ============================================================
# CHECK EGGNOG-MAPPER
# ============================================================

print("\n" + "=" * 80)
print("CHECKING EGGNOG-MAPPER")
print("=" * 80)

EMAPPER = shutil.which("emapper.py")

if EMAPPER is None:

    print("""
eggNOG-mapper was not found in PATH.

Install eggNOG-mapper first.

Recommended conda installation:

    conda install -c bioconda eggnog-mapper

Then verify:

    emapper.py --version

After installation, rerun this script.
""")

    raise SystemExit(1)


print("eggNOG-mapper:")
print(EMAPPER)


# ============================================================
# VERSION
# ============================================================

try:

    version_result = subprocess.run(
        [EMAPPER, "--version"],
        capture_output=True,
        text=True
    )

    print(
        "\nVersion:",
        version_result.stdout.strip()
    )

except Exception as e:

    print(
        "Could not determine eggNOG-mapper version:",
        e
    )


# ============================================================
# OUTPUT PREFIX
# ============================================================

OUTPUT_PREFIX = os.path.join(
    ANNOTATION_DIR,
    "Carbon_Breadth_top20"
)


# ============================================================
# RUN EGGNOG-MAPPER
# ============================================================

print("\n" + "=" * 80)
print("RUNNING EGGNOG-MAPPER")
print("=" * 80)

print("""
This is the main functional annotation step.

The protein sequences will be searched for
orthologous functional assignments.

Expected annotations include:

    Functional description
    COG
    GO
    EC
    KEGG KO
    KEGG pathway
    KEGG module
    KEGG reaction

This may take substantially longer than the
previous local parsing steps.
""")


command = [

    EMAPPER,

    "-i",
    INPUT_FASTA,

    "--itype",
    "proteins",

    "--output",
    "Carbon_Breadth_top20",

    "--output_dir",
    ANNOTATION_DIR,

    "--cpu",
    str(max(1, (os.cpu_count() or 4) - 1)),

    "--override",

    "--data_dir",
    os.environ.get(
        "EGGNOG_DATA_DIR",
        ""
    )

]


# Remove empty --data_dir if environment variable isn't set
if command[-1] == "":

    command = command[:-2]


print("\nCommand:")
print(" ".join(command))

print("\nStarting annotation...\n")


try:

    result = subprocess.run(
        command,
        cwd=ANNOTATION_DIR
    )

except KeyboardInterrupt:

    print(
        "\nAnnotation interrupted by user."
    )

    raise SystemExit(1)


if result.returncode != 0:

    raise RuntimeError(
        "\neggNOG-mapper failed.\n"
        "Check the terminal output above."
    )


print("\n" + "=" * 80)
print("EGGNOG-MAPPER COMPLETE")
print("=" * 80)


# ============================================================
# LOCATE ANNOTATION FILE
# ============================================================

annotation_file = os.path.join(
    ANNOTATION_DIR,
    "Carbon_Breadth_top20.emapper.annotations"
)

if not os.path.exists(annotation_file):

    # Search recursively in case eggNOG placed it elsewhere

    candidates = list(
        Path(ANNOTATION_DIR).rglob(
            "*.emapper.annotations"
        )
    )

    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nCould not locate eggNOG annotation file."
        )

    annotation_file = str(
        candidates[0]
    )


print("\nAnnotation file:")
print(annotation_file)


# ============================================================
# READ EGGNOG ANNOTATION
# ============================================================

print("\n" + "=" * 80)
print("READING FUNCTIONAL ANNOTATIONS")
print("=" * 80)


rows = []

with open(
    annotation_file,
    "r",
    encoding="utf-8",
    errors="ignore"
) as handle:

    for line in handle:

        if line.startswith("#"):
            continue

        line = line.rstrip("\n")

        if not line:
            continue

        fields = line.split("\t")

        # eggNOG standard annotation format
        #
        # query
        # seed_ortholog
        # evalue
        # score
        # eggNOG_OGs
        # max_annot_lvl
        # COG_category
        # Description
        # Preferred_name
        # GOs
        # EC
        # KEGG_ko
        # KEGG_Pathway
        # KEGG_Module
        # KEGG_Reaction
        # KEGG_rclass
        # BRITE
        # KEGG_TC
        # CAZy
        # BiGG_Reaction
        # PFAMs

        if len(fields) < 8:
            continue

        while len(fields) < 21:
            fields.append("")

        rows.append(fields[:21])


columns = [

    "query",
    "seed_ortholog",
    "evalue",
    "score",
    "eggNOG_OGs",
    "max_annot_lvl",
    "COG_category",
    "Description",
    "Preferred_name",
    "GOs",
    "EC",
    "KEGG_ko",
    "KEGG_Pathway",
    "KEGG_Module",
    "KEGG_Reaction",
    "KEGG_rclass",
    "BRITE",
    "KEGG_TC",
    "CAZy",
    "BiGG_Reaction",
    "PFAMs"

]


annotation_df = pd.DataFrame(
    rows,
    columns=columns
)


print(
    "\nAnnotated protein records:",
    len(annotation_df)
)


# ============================================================
# NORMALIZE EMPTY VALUES
# ============================================================

annotation_df = annotation_df.fillna("")


# ============================================================
# BUSCO EXTRACTION FROM QUERY
# ============================================================

def extract_busco(query):

    query = str(query)

    # Our FASTA IDs look like:
    #
    # CB420_PROT_00001|BUSCO=1679at4891|Species=...

    match = __import__("re").search(
        r"BUSCO=([^|]+)",
        query
    )

    if match:

        return match.group(1)

    return ""


annotation_df["BUSCO_ID"] = (
    annotation_df["query"]
    .apply(extract_busco)
)


# ============================================================
# CLEAN QUERY ID
# ============================================================

annotation_df["Candidate_Protein_ID"] = (
    annotation_df["query"]
    .astype(str)
    .str.split("|")
    .str[0]
)


# ============================================================
# SAVE COMPLETE ANNOTATION
# ============================================================

complete_annotation = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_protein_functional_annotation_420.csv"
)

annotation_df.to_csv(
    complete_annotation,
    index=False
)


# ============================================================
# GO ANNOTATIONS
# ============================================================

go_df = annotation_df[
    annotation_df["GOs"].astype(str).str.strip() != ""
].copy()


go_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_GO_annotations_420.csv"
)

go_df.to_csv(
    go_path,
    index=False
)


# ============================================================
# KEGG ANNOTATIONS
# ============================================================

kegg_df = annotation_df[
    annotation_df["KEGG_ko"].astype(str).str.strip() != ""
].copy()


kegg_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_KEGG_annotations_420.csv"
)

kegg_df.to_csv(
    kegg_path,
    index=False
)


# ============================================================
# EC ANNOTATIONS
# ============================================================

ec_df = annotation_df[
    annotation_df["EC"].astype(str).str.strip() != ""
].copy()


ec_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_EC_annotations_420.csv"
)

ec_df.to_csv(
    ec_path,
    index=False
)


# ============================================================
# PFAM / INTERPRO-LIKE DOMAIN INFORMATION
# ============================================================

pfam_df = annotation_df[
    annotation_df["PFAMs"].astype(str).str.strip() != ""
].copy()


pfam_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_PFAM_annotations_420.csv"
)

pfam_df.to_csv(
    pfam_path,
    index=False
)


# ============================================================
# FUNCTIONAL DESCRIPTION
# ============================================================

description_df = annotation_df[
    annotation_df["Description"].astype(str).str.strip() != ""
].copy()


description_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_functional_descriptions_420.csv"
)

description_df.to_csv(
    description_path,
    index=False
)


# ============================================================
# BUSCO SUMMARY
# ============================================================

summary_rows = []


for busco in TOP_BUSCOS:

    sub = annotation_df[
        annotation_df["BUSCO_ID"] == busco
    ]

    summary_rows.append({

        "BUSCO_ID":
            busco,

        "Protein_Records_Annotated":
            len(sub),

        "GO_Supported":
            int(
                sub["GOs"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            ),

        "KEGG_KO_Supported":
            int(
                sub["KEGG_ko"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            ),

        "KEGG_Pathway_Supported":
            int(
                sub["KEGG_Pathway"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            ),

        "EC_Supported":
            int(
                sub["EC"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            ),

        "PFAM_Supported":
            int(
                sub["PFAMs"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            ),

        "Functional_Description_Supported":
            int(
                sub["Description"]
                .astype(str)
                .str.strip()
                .ne("")
                .sum()
            )

    })


busco_summary = pd.DataFrame(
    summary_rows
)


summary_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_BUSCO_functional_summary_420.csv"
)


busco_summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# UNIQUE KEGG PATHWAYS
# ============================================================

pathway_rows = []


for _, row in annotation_df.iterrows():

    pathways = str(
        row["KEGG_Pathway"]
    ).strip()

    if not pathways:
        continue

    for pathway in pathways.split(","):

        pathway = pathway.strip()

        if pathway:

            pathway_rows.append({

                "BUSCO_ID":
                    row["BUSCO_ID"],

                "Candidate_Protein_ID":
                    row["Candidate_Protein_ID"],

                "KEGG_Pathway":
                    pathway

            })


pathway_df = pd.DataFrame(
    pathway_rows
)


if len(pathway_df) > 0:

    pathway_df = pathway_df.drop_duplicates()

else:

    pathway_df = pd.DataFrame(
        columns=[
            "BUSCO_ID",
            "Candidate_Protein_ID",
            "KEGG_Pathway"
        ]
    )


pathway_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_KEGG_pathway_assignments_420.csv"
)


pathway_df.to_csv(
    pathway_path,
    index=False
)


# ============================================================
# UNIQUE GO TERMS
# ============================================================

go_rows = []


for _, row in annotation_df.iterrows():

    gos = str(
        row["GOs"]
    ).strip()

    if not gos:
        continue

    for go in gos.split(","):

        go = go.strip()

        if go:

            go_rows.append({

                "BUSCO_ID":
                    row["BUSCO_ID"],

                "Candidate_Protein_ID":
                    row["Candidate_Protein_ID"],

                "GO":
                    go

            })


go_terms_df = pd.DataFrame(
    go_rows
)


if len(go_terms_df) > 0:

    go_terms_df = go_terms_df.drop_duplicates()


go_terms_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_GO_term_assignments_420.csv"
)


go_terms_df.to_csv(
    go_terms_path,
    index=False
)


# ============================================================
# JSON SUMMARY
# ============================================================

json_summary = {

    "BUSCO_candidates":
        len(TOP_BUSCOS),

    "Input_proteins":
        protein_count,

    "Annotated_proteins":
        len(annotation_df),

    "GO_supported_proteins":
        len(go_df),

    "KEGG_KO_supported_proteins":
        len(kegg_df),

    "KEGG_pathway_supported_proteins":
        len(
            annotation_df[
                annotation_df["KEGG_Pathway"]
                .astype(str)
                .str.strip()
                .ne("")
            ]
        ),

    "EC_supported_proteins":
        len(ec_df),

    "PFAM_supported_proteins":
        len(pfam_df),

    "Functional_description_supported":
        len(description_df),

    "Unique_GO_terms":
        int(
            go_terms_df["GO"].nunique()
        )
        if len(go_terms_df) > 0
        else 0,

    "Unique_KEGG_pathways":
        int(
            pathway_df["KEGG_Pathway"].nunique()
        )
        if len(pathway_df) > 0
        else 0,

    "Annotation_method":
        "eggNOG-mapper sequence-based orthology annotation"

}


json_path = os.path.join(
    TABLE_DIR,
    "Carbon_Breadth_functional_annotation_summary_420.json"
)


with open(
    json_path,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        json_summary,
        handle,
        indent=4
    )


# ============================================================
# REPORT
# ============================================================

report_path = os.path.join(
    REPORT_DIR,
    "Carbon_Breadth_functional_annotation_report_420.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "CARBON BREADTH 420\n"
        "BUSCO PROTEIN FUNCTIONAL ANNOTATION\n"
        "=" * 70 + "\n\n"
    )

    handle.write(
        f"BUSCO candidates: {len(TOP_BUSCOS)}\n"
    )

    handle.write(
        f"Input proteins: {protein_count}\n"
    )

    handle.write(
        f"Annotated proteins: {len(annotation_df)}\n"
    )

    handle.write(
        f"GO-supported: {len(go_df)}\n"
    )

    handle.write(
        f"KEGG KO-supported: {len(kegg_df)}\n"
    )

    handle.write(
        f"KEGG pathway-supported: "
        f"{len(pathway_df)} pathway assignments\n"
    )

    handle.write(
        f"EC-supported: {len(ec_df)}\n"
    )

    handle.write(
        f"PFAM-supported: {len(pfam_df)}\n"
    )

    handle.write(
        f"Unique GO terms: "
        f"{json_summary['Unique_GO_terms']}\n"
    )

    handle.write(
        f"Unique KEGG pathways: "
        f"{json_summary['Unique_KEGG_pathways']}\n"
    )


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("FUNCTIONAL ANNOTATION COMPLETE")
print("=" * 80)

print(
    "\nInput proteins:",
    protein_count
)

print(
    "Annotated proteins:",
    len(annotation_df)
)

print(
    "GO-supported proteins:",
    len(go_df)
)

print(
    "KEGG KO-supported proteins:",
    len(kegg_df)
)

print(
    "KEGG pathway assignments:",
    len(pathway_df)
)

print(
    "EC-supported proteins:",
    len(ec_df)
)

print(
    "PFAM-supported proteins:",
    len(pfam_df)
)

print(
    "Unique GO terms:",
    json_summary["Unique_GO_terms"]
)

print(
    "Unique KEGG pathways:",
    json_summary["Unique_KEGG_pathways"]
)


print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\nComplete annotation:")
print(complete_annotation)

print("\nGO annotations:")
print(go_path)

print("\nKEGG annotations:")
print(kegg_path)

print("\nEC annotations:")
print(ec_path)

print("\nPFAM annotations:")
print(pfam_path)

print("\nFunctional descriptions:")
print(description_path)

print("\nBUSCO functional summary:")
print(summary_path)

print("\nKEGG pathway assignments:")
print(pathway_path)

print("\nGO term assignments:")
print(go_terms_path)

print("\nJSON summary:")
print(json_path)

print("\nReport:")
print(report_path)


print("\n" + "=" * 80)
print("NEXT STAGE")
print("=" * 80)

print("""
BUSCO
  ↓
Protein sequences                         [DONE]
  ↓
eggNOG functional annotation              [DONE]
  ↓
GO / KEGG / EC / PFAM                     [DONE]
  ↓
Pathway enrichment / functional grouping  [NEXT]
  ↓
Integrate pathway evidence
with ML importance
  ↓
Candidate carbon-breadth network
  ↓
Candidate chassis interpretation
""")