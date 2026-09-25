import os
import re
import json
import hashlib
import pandas as pd

# ============================================================
# CARBON BREADTH 420
# PREPARE RECOVERED BUSCO PROTEINS FOR FUNCTIONAL ANNOTATION
# ============================================================

BASE = r"C:\Y1000_chassis_project"

RESULTS = os.path.join(
    BASE,
    r"results\stage5_phylogeny_ml_dataset\phylogeny_aware_ml"
    r"\phylogeny_cv\model_training\feature_interpretation_420"
)

PROTEIN_DIR = os.path.join(
    RESULTS,
    r"functional_annotation_420\busco_gene_mapping_420"
    r"\candidate_proteins_420"
)

FASTA = os.path.join(
    PROTEIN_DIR,
    r"fasta\Carbon_Breadth_candidate_BUSCO_proteins_420.faa"
)

OUT_DIR = os.path.join(
    RESULTS,
    r"functional_annotation_420\protein_functional_annotation_420"
)

OUT_TABLE = os.path.join(OUT_DIR, "tables")
OUT_FASTA = os.path.join(OUT_DIR, "fasta")
OUT_REPORT = os.path.join(OUT_DIR, "reports")

for d in [OUT_TABLE, OUT_FASTA, OUT_REPORT]:
    os.makedirs(d, exist_ok=True)


# ============================================================
# TOP 20 BUSCO CANDIDATES
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

TOP_SET = set(TOP_BUSCOS)


print("=" * 80)
print("CARBON BREADTH 420")
print("PREPARING RECOVERED BUSCO PROTEINS")
print("=" * 80)


# ============================================================
# CHECK FASTA
# ============================================================

print("\n" + "=" * 80)
print("LOADING RECOVERED PROTEIN FASTA")
print("=" * 80)

print("FASTA:")
print(FASTA)

if not os.path.exists(FASTA):
    raise FileNotFoundError(
        "\nProtein FASTA not found:\n" + FASTA
    )


# ============================================================
# FASTA PARSER
# ============================================================

records = []

current_header = None
current_sequence = []

with open(
    FASTA,
    "r",
    encoding="utf-8",
    errors="ignore"
) as handle:

    for line in handle:

        line = line.strip()

        if not line:
            continue

        if line.startswith(">"):

            if current_header is not None:

                records.append({
                    "Header": current_header,
                    "Sequence": "".join(current_sequence)
                })

            current_header = line[1:]
            current_sequence = []

        else:

            current_sequence.append(line)

# final record
if current_header is not None:

    records.append({
        "Header": current_header,
        "Sequence": "".join(current_sequence)
    })


print("\nFASTA records recovered:")
print(len(records))


if len(records) == 0:
    raise ValueError(
        "No FASTA records were found."
    )


# ============================================================
# PARSE THE ACTUAL HEADER FORMAT
#
# Example:
#
# 1679at4891|Zygotorulaspora_mrakii|GCA_030570635.1|protein_length=1279
#
# Field 0 = BUSCO
# Field 1 = Species
# Field 2 = Assembly
# Field 3 = protein length
# ============================================================

parsed = []

for record in records:

    header = record["Header"]

    sequence = (
        record["Sequence"]
        .upper()
        .replace(" ", "")
        .replace("\r", "")
        .replace("\n", "")
        .replace("*", "")
    )

    fields = header.split("|")

    # FIRST FIELD IS THE BUSCO ID
    busco_id = fields[0].strip()

    species = ""
    assembly = ""
    protein_length = None

    if len(fields) >= 2:
        species = fields[1].strip()

    if len(fields) >= 3:
        assembly = fields[2].strip()

    # Extract protein_length
    for field in fields:

        if field.startswith("protein_length="):

            try:
                protein_length = int(
                    field.split("=", 1)[1]
                )
            except:
                protein_length = None

    parsed.append({

        "Original_Header": header,

        "BUSCO_ID": busco_id,

        "Species": species,

        "Assembly_Accession": assembly,

        "Protein_Length_Header": protein_length,

        "Protein_Sequence": sequence

    })


df = pd.DataFrame(parsed)


# ============================================================
# DIAGNOSTIC
# ============================================================

print("\n" + "=" * 80)
print("HEADER PARSING CHECK")
print("=" * 80)

print("\nExample parsed records:")

print(
    df[
        [
            "BUSCO_ID",
            "Species",
            "Assembly_Accession",
            "Protein_Length_Header"
        ]
    ].head(10).to_string(index=False)
)

print("\nBUSCO IDs recovered:")
print(
    df["BUSCO_ID"]
    .nunique()
)


# ============================================================
# FILTER TOP 20
# ============================================================

df = df[
    df["BUSCO_ID"].isin(TOP_SET)
].copy()


print("\nRecords belonging to top-20 BUSCOs:")
print(len(df))


print("\nCandidate BUSCOs recovered:")

for busco in sorted(
    df["BUSCO_ID"].unique(),
    key=lambda x: TOP_BUSCOS.index(x)
):

    print(" ", busco)


# ============================================================
# VALIDATE PROTEIN SEQUENCES
# ============================================================

AA_PATTERN = re.compile(
    r"^[ACDEFGHIKLMNPQRSTVWYBXZJUO]+$"
)

valid_rows = []

for _, row in df.iterrows():

    seq = row["Protein_Sequence"]

    if len(seq) < 20:
        continue

    if AA_PATTERN.fullmatch(seq):

        valid_rows.append(row)


df = pd.DataFrame(valid_rows)


print("\nValid protein records:")
print(len(df))


if len(df) == 0:

    raise ValueError(
        "No valid amino-acid sequences were recovered."
    )


# ============================================================
# HASH SEQUENCES
# ============================================================

df["Sequence_Hash"] = df[
    "Protein_Sequence"
].apply(
    lambda x: hashlib.sha256(
        x.encode("utf-8")
    ).hexdigest()
)


# ============================================================
# COLLAPSE IDENTICAL PROTEINS
# ============================================================

unique = (
    df
    .drop_duplicates(
        subset=["Sequence_Hash"]
    )
    .copy()
    .reset_index(drop=True)
)


unique["Candidate_Protein_ID"] = [
    f"CB420_PROT_{i:05d}"
    for i in range(
        1,
        len(unique) + 1
    )
]


print("\n" + "=" * 80)
print("PROTEIN DEDUPLICATION")
print("=" * 80)

print(
    "Recovered protein records :",
    len(df)
)

print(
    "Unique protein sequences  :",
    len(unique)
)


# ============================================================
# MAP ALL RECORDS TO UNIQUE PROTEIN
# ============================================================

hash_to_id = dict(
    zip(
        unique["Sequence_Hash"],
        unique["Candidate_Protein_ID"]
    )
)

df["Candidate_Protein_ID"] = (
    df["Sequence_Hash"]
    .map(hash_to_id)
)


# ============================================================
# BUSCO SUMMARY
# ============================================================

summary = []

for busco in TOP_BUSCOS:

    sub = unique[
        unique["BUSCO_ID"] == busco
    ]

    summary.append({

        "BUSCO_ID": busco,

        "Unique_Protein_Count":
            int(
                sub["Candidate_Protein_ID"]
                .nunique()
            ),

        "Species_Count":
            int(
                sub["Species"].nunique()
            ),

        "Assembly_Count":
            int(
                sub["Assembly_Accession"]
                .nunique()
            ),

        "Recovered":
            int(len(sub) > 0)

    })


summary_df = pd.DataFrame(summary)


# ============================================================
# WRITE CLEAN FASTA
# ============================================================

clean_fasta = os.path.join(
    OUT_FASTA,
    "Carbon_Breadth_top20_unique_proteins_420.faa"
)


with open(
    clean_fasta,
    "w",
    encoding="utf-8"
) as handle:

    for _, row in unique.iterrows():

        header = (
            f">{row['Candidate_Protein_ID']}"
            f"|BUSCO={row['BUSCO_ID']}"
            f"|Species={row['Species']}"
            f"|Assembly={row['Assembly_Accession']}"
        )

        handle.write(
            header + "\n"
        )

        sequence = row["Protein_Sequence"]

        for i in range(
            0,
            len(sequence),
            80
        ):

            handle.write(
                sequence[i:i + 80] + "\n"
            )


# ============================================================
# FUNCTIONAL ANNOTATION INPUT
# ============================================================

annotation_input = unique[
    [
        "Candidate_Protein_ID",
        "BUSCO_ID",
        "Species",
        "Assembly_Accession",
        "Protein_Sequence"
    ]
].copy()


annotation_path = os.path.join(
    OUT_TABLE,
    "Carbon_Breadth_functional_annotation_input_420.csv"
)


annotation_input.to_csv(
    annotation_path,
    index=False
)


# ============================================================
# PROVENANCE
# ============================================================

provenance_path = os.path.join(
    OUT_TABLE,
    "Carbon_Breadth_protein_provenance_420.csv"
)


df.to_csv(
    provenance_path,
    index=False
)


# ============================================================
# BUSCO SUMMARY
# ============================================================

summary_path = os.path.join(
    OUT_TABLE,
    "Carbon_Breadth_BUSCO_protein_summary_420.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# JSON SUMMARY
# ============================================================

json_summary = {

    "Total_BUSCO_candidates":
        len(TOP_BUSCOS),

    "BUSCO_candidates_with_proteins":
        int(
            summary_df["Recovered"].sum()
        ),

    "Protein_records_recovered":
        int(len(df)),

    "Unique_protein_sequences":
        int(len(unique)),

    "Failed_records":
        0,

    "Input_FASTA":
        FASTA,

    "Clean_FASTA":
        clean_fasta,

    "Functional_annotation_input":
        annotation_path,

    "Next_stage":
        "GO / InterPro / EC / KEGG functional annotation"

}


json_path = os.path.join(
    OUT_TABLE,
    "Carbon_Breadth_protein_preparation_summary_420.json"
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
    OUT_REPORT,
    "Carbon_Breadth_protein_preparation_report_420.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "CARBON BREADTH 420\n"
        "PROTEIN FUNCTIONAL ANNOTATION PREPARATION\n"
    )

    handle.write(
        "=" * 70 + "\n\n"
    )

    handle.write(
        f"BUSCO candidates: {len(TOP_BUSCOS)}\n"
    )

    handle.write(
        f"Protein records recovered: {len(df)}\n"
    )

    handle.write(
        f"Unique protein sequences: {len(unique)}\n"
    )

    handle.write(
        "BUSCOs with recovered proteins: "
        f"{int(summary_df['Recovered'].sum())}\n"
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("PROTEIN PREPARATION COMPLETE")
print("=" * 80)

print(
    "Total BUSCO candidates       :",
    len(TOP_BUSCOS)
)

print(
    "BUSCO candidates recovered   :",
    int(summary_df["Recovered"].sum())
)

print(
    "Protein records recovered    :",
    len(df)
)

print(
    "Unique protein sequences     :",
    len(unique)
)

print(
    "Failed records               :",
    0
)


print("\n" + "=" * 80)
print("OUTPUTS")
print("=" * 80)

print("\nClean protein FASTA:")
print(clean_fasta)

print("\nFunctional annotation input:")
print(annotation_path)

print("\nProtein provenance:")
print(provenance_path)

print("\nBUSCO protein summary:")
print(summary_path)

print("\nJSON summary:")
print(json_path)

print("\nReport:")
print(report_path)


print("\n" + "=" * 80)
print("NEXT SCIENTIFIC STAGE")
print("=" * 80)

print("""
BUSCO candidates
      ↓
recovered proteins                 [DONE]
      ↓
unique protein sequences            [DONE]
      ↓
GO / InterPro / EC / KEGG          [NEXT]
      ↓
pathway grouping
      ↓
integrate with ML importance
      ↓
candidate chassis interpretation
""")