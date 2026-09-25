from pathlib import Path
import pandas as pd
import re
import sys


# ============================================================
# STAGE 4A.1
# PARETO ↔ 435 REPRESENTATIVE RECONCILIATION
#
# PURPOSE
# -------
# Determine why Pareto assemblies are absent from the
# 435-species representative set.
#
# IMPORTANT
# ---------
# This script:
#   - DOES NOT download genomes
#   - DOES NOT delete genomes
#   - DOES NOT modify genomes
#   - DOES NOT change the 435 representative set
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(
    r"C:\Y1000_chassis_project"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage4A1_pareto_representative_reconciliation"
)

REPRESENTATIVE_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage3B1D_canonical_taxonomy"
    / "canonical_species_representatives.csv"
)

PARETO_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage3B2H_pareto_taxonomy_adjudication"
    / "stage3B2H_final_pareto_taxonomy.csv"
)

TAXONOMY_MASTER_FILE = (
    PROJECT_ROOT
    / "results"
    / "stage3B2F_taxonomy_finalization"
    / "stage3B2F_taxonomy_master.csv"
)

GENOME_DIR = (
    PROJECT_ROOT
    / "data"
    / "genomes"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# 2. HELPER FUNCTIONS
# ------------------------------------------------------------

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_accession(value):

    value = clean_text(value)

    match = re.search(
        r"(GC[AF]_\d+\.\d+)",
        value,
        flags=re.IGNORECASE
    )

    if match:
        return match.group(1).upper()

    return ""


def find_column(df, candidates):

    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.strip().lower()

        if key in lookup:
            return lookup[key]

    return None


def normalize_species_name(value):

    value = clean_text(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value


# ------------------------------------------------------------
# 3. START
# ------------------------------------------------------------

print("=" * 80)
print("STAGE 4A.1 — PARETO ↔ 435 REPRESENTATIVE RECONCILIATION")
print("=" * 80)


# ------------------------------------------------------------
# 4. CHECK REQUIRED PATHS
# ------------------------------------------------------------

required_paths = [
    REPRESENTATIVE_FILE,
    PARETO_FILE,
    TAXONOMY_MASTER_FILE,
    GENOME_DIR
]

for path in required_paths:

    if not path.exists():

        print(
            "\nERROR: Required path does not exist:"
        )

        print(path)

        sys.exit(1)


# ------------------------------------------------------------
# 5. LOAD 435 REPRESENTATIVES
# ------------------------------------------------------------

print(
    "\n" + "-" * 80
)

print(
    "LOADING 435 REPRESENTATIVE SET"
)

print(
    "-" * 80
)

representatives = pd.read_csv(
    REPRESENTATIVE_FILE
)

print(
    "Rows:",
    len(representatives)
)


rep_species_col = find_column(
    representatives,
    [
        "Canonical_Species",
        "Species"
    ]
)

rep_accession_col = find_column(
    representatives,
    [
        "Assembly_Accession",
        "Assembly",
        "Accession"
    ]
)


if (
    rep_species_col is None
    or rep_accession_col is None
):

    print(
        "\nERROR: Could not identify representative columns."
    )

    sys.exit(1)


representatives[
    "REP_SPECIES"
] = (
    representatives[
        rep_species_col
    ].map(normalize_species_name)
)

representatives[
    "REP_ACCESSION"
] = (
    representatives[
        rep_accession_col
    ].map(normalize_accession)
)


print(
    "Unique species:",
    representatives[
        "REP_SPECIES"
    ].nunique()
)

print(
    "Unique accessions:",
    representatives[
        "REP_ACCESSION"
    ].nunique()
)


if len(representatives) != 435:

    print(
        "\nERROR: Representative set is not 435 rows."
    )

    sys.exit(1)


# ------------------------------------------------------------
# 6. LOAD PARETO SET
# ------------------------------------------------------------

print(
    "\n" + "-" * 80
)

print(
    "LOADING PARETO SET"
)

print(
    "-" * 80
)

pareto = pd.read_csv(
    PARETO_FILE
)

print(
    "Rows:",
    len(pareto)
)


pareto_accession_col = find_column(
    pareto,
    [
        "Assembly_Accession",
        "Stage3B2F_Assembly_Key",
        "Assembly",
        "Accession"
    ]
)

pareto_name_col = find_column(
    pareto,
    [
        "Final_Canonical_Name",
        "Final_Canonical_Species",
        "Canonical_Species"
    ]
)

pareto_source_col = find_column(
    pareto,
    [
        "Phenotype_Species_Source",
        "Audit_Source_Phenotype_Name"
    ]
)

pareto_decision_col = find_column(
    pareto,
    [
        "Final_Taxonomy_Decision",
        "Audit_Adjudication_Decision"
    ]
)


if pareto_accession_col is None:

    print(
        "\nERROR: Pareto accession column not found."
    )

    sys.exit(1)


pareto[
    "PARETO_ACCESSION"
] = (
    pareto[
        pareto_accession_col
    ].map(normalize_accession)
)


if pareto_name_col:

    pareto[
        "PARETO_CANONICAL_NAME"
    ] = (
        pareto[
            pareto_name_col
        ].map(normalize_species_name)
    )

else:

    pareto[
        "PARETO_CANONICAL_NAME"
    ] = ""


if pareto_source_col:

    pareto[
        "PARETO_SOURCE_NAME"
    ] = (
        pareto[
            pareto_source_col
        ].map(normalize_species_name)
    )

else:

    pareto[
        "PARETO_SOURCE_NAME"
    ] = ""


if pareto_decision_col:

    pareto[
        "PARETO_DECISION"
    ] = (
        pareto[
            pareto_decision_col
        ].map(clean_text)
    )

else:

    pareto[
        "PARETO_DECISION"
    ] = ""


pareto_accessions = {
    x
    for x in pareto[
        "PARETO_ACCESSION"
    ]
    if x
}


print(
    "Unique Pareto accessions:",
    len(pareto_accessions)
)


if len(pareto_accessions) != 11:

    print(
        "\nERROR: Expected exactly 11 Pareto accessions."
    )

    sys.exit(1)


# ------------------------------------------------------------
# 7. FIND MISSING PARETO ACCESSIONS
# ------------------------------------------------------------

representative_accessions = set(
    representatives[
        "REP_ACCESSION"
    ]
)

missing_accessions = (
    pareto_accessions
    - representative_accessions
)

present_accessions = (
    pareto_accessions
    & representative_accessions
)


print(
    "\nPareto accessions already in 435:",
    len(present_accessions)
)

print(
    "Pareto accessions missing from 435:",
    len(missing_accessions)
)


if missing_accessions:

    print(
        "\nMissing Pareto accessions:"
    )

    for accession in sorted(
        missing_accessions
    ):

        print(
            " ",
            accession
        )


# ------------------------------------------------------------
# 8. LOAD 510-ROW TAXONOMY MASTER
# ------------------------------------------------------------

print(
    "\n" + "-" * 80
)

print(
    "LOADING 510-ROW TAXONOMY MASTER"
)

print(
    "-" * 80
)

taxonomy = pd.read_csv(
    TAXONOMY_MASTER_FILE
)

print(
    "Rows:",
    len(taxonomy)
)


tax_accession_col = find_column(
    taxonomy,
    [
        "Assembly_Accession",
        "Stage3B2F_Assembly_Key",
        "Assembly_Key",
        "Accession"
    ]
)

tax_canonical_col = find_column(
    taxonomy,
    [
        "Canonical_Species",
        "Final_Canonical_Name",
        "Canonical_Name"
    ]
)

tax_phenotype_col = find_column(
    taxonomy,
    [
        "Phenotype_Species",
        "Phenotype_Species_Source",
        "Original_Name"
    ]
)

tax_organism_col = find_column(
    taxonomy,
    [
        "NCBI_Organism",
        "Organism_Name",
        "NCBI_Organism_Name"
    ]
)

tax_status_col = find_column(
    taxonomy,
    [
        "Final_Taxonomic_Status",
        "Taxonomic_Status",
        "Stage3B2F_Decision"
    ]
)

tax_mapping_col = find_column(
    taxonomy,
    [
        "Canonical_Mapping_Reason",
        "Mapping_Reason"
    ]
)


if tax_accession_col is None:

    print(
        "\nERROR: Could not identify taxonomy accession column."
    )

    print(
        list(taxonomy.columns)
    )

    sys.exit(1)


taxonomy[
    "TAX_ACCESSION"
] = (
    taxonomy[
        tax_accession_col
    ].map(normalize_accession)
)


if tax_canonical_col:

    taxonomy[
        "TAX_CANONICAL"
    ] = (
        taxonomy[
            tax_canonical_col
        ].map(normalize_species_name)
    )

else:

    taxonomy[
        "TAX_CANONICAL"
    ] = ""


if tax_phenotype_col:

    taxonomy[
        "TAX_PHENOTYPE"
    ] = (
        taxonomy[
            tax_phenotype_col
        ].map(normalize_species_name)
    )

else:

    taxonomy[
        "TAX_PHENOTYPE"
    ] = ""


if tax_organism_col:

    taxonomy[
        "TAX_ORGANISM"
    ] = (
        taxonomy[
            tax_organism_col
        ].map(normalize_species_name)
    )

else:

    taxonomy[
        "TAX_ORGANISM"
    ] = ""


if tax_status_col:

    taxonomy[
        "TAX_STATUS"
    ] = (
        taxonomy[
            tax_status_col
        ].map(clean_text)
    )

else:

    taxonomy[
        "TAX_STATUS"
    ] = ""


if tax_mapping_col:

    taxonomy[
        "TAX_MAPPING_REASON"
    ] = (
        taxonomy[
            tax_mapping_col
        ].map(clean_text)
    )

else:

    taxonomy[
        "TAX_MAPPING_REASON"
    ] = ""


# ------------------------------------------------------------
# 9. EXACT MISSING PARETO RECORDS
# ------------------------------------------------------------

print(
    "\n" + "-" * 80
)

print(
    "RECONCILING MISSING PARETO ACCESSIONS"
)

print(
    "-" * 80
)


missing_tax_rows = taxonomy[
    taxonomy[
        "TAX_ACCESSION"
    ].isin(
        missing_accessions
    )
].copy()


print(
    "Missing Pareto accessions found in 510 taxonomy rows:",
    missing_tax_rows[
        "TAX_ACCESSION"
    ].nunique()
)


# ------------------------------------------------------------
# 10. RECONCILIATION
# ------------------------------------------------------------

reconciliation_rows = []


for _, pareto_row in pareto.iterrows():

    accession = pareto_row[
        "PARETO_ACCESSION"
    ]

    if accession not in missing_accessions:
        continue

    canonical_name = pareto_row[
        "PARETO_CANONICAL_NAME"
    ]

    source_name = pareto_row[
        "PARETO_SOURCE_NAME"
    ]

    decision = pareto_row[
        "PARETO_DECISION"
    ]


    exact_rows = taxonomy[
        taxonomy[
            "TAX_ACCESSION"
        ] == accession
    ].copy()


    if len(exact_rows) == 0:

        reconciliation_rows.append({

            "Pareto_Accession":
                accession,

            "Pareto_Canonical_Name":
                canonical_name,

            "Pareto_Source_Name":
                source_name,

            "Pareto_Decision":
                decision,

            "Exact_Taxonomy_Record_Found":
                False,

            "Exact_Taxonomy_Canonical":
                "",

            "Exact_Taxonomy_Phenotype":
                "",

            "Exact_Taxonomy_Organism":
                "",

            "Exact_Taxonomy_Status":
                "",

            "Exact_Mapping_Reason":
                "",

            "Alternative_Representative_Found":
                False,

            "Alternative_Representative_Accession":
                "",

            "Alternative_Representative_Species":
                "",

            "Alternative_Representative_Reason":
                "NO_EXACT_TAXONOMY_RECORD"

        })

        continue


    exact = exact_rows.iloc[0]

    exact_canonical = exact[
        "TAX_CANONICAL"
    ]


    # Search exact canonical species in 435
    same_species_reps = representatives[
        representatives[
            "REP_SPECIES"
        ].str.lower()
        == exact_canonical.lower()
    ].copy()


    if len(same_species_reps) == 0:

        same_species_reps = representatives[
            representatives[
                "REP_SPECIES"
            ].str.lower()
            == canonical_name.lower()
        ].copy()


    if len(same_species_reps) > 0:

        for _, rep in same_species_reps.iterrows():

            reconciliation_rows.append({

                "Pareto_Accession":
                    accession,

                "Pareto_Canonical_Name":
                    canonical_name,

                "Pareto_Source_Name":
                    source_name,

                "Pareto_Decision":
                    decision,

                "Exact_Taxonomy_Record_Found":
                    True,

                "Exact_Taxonomy_Canonical":
                    exact[
                        "TAX_CANONICAL"
                    ],

                "Exact_Taxonomy_Phenotype":
                    exact[
                        "TAX_PHENOTYPE"
                    ],

                "Exact_Taxonomy_Organism":
                    exact[
                        "TAX_ORGANISM"
                    ],

                "Exact_Taxonomy_Status":
                    exact[
                        "TAX_STATUS"
                    ],

                "Exact_Mapping_Reason":
                    exact[
                        "TAX_MAPPING_REASON"
                    ],

                "Alternative_Representative_Found":
                    True,

                "Alternative_Representative_Accession":
                    rep[
                        "REP_ACCESSION"
                    ],

                "Alternative_Representative_Species":
                    rep[
                        "REP_SPECIES"
                    ],

                "Alternative_Representative_Reason":
                    "SAME_CANONICAL_SPECIES"

            })

    else:

        reconciliation_rows.append({

            "Pareto_Accession":
                accession,

            "Pareto_Canonical_Name":
                canonical_name,

            "Pareto_Source_Name":
                source_name,

            "Pareto_Decision":
                decision,

            "Exact_Taxonomy_Record_Found":
                True,

            "Exact_Taxonomy_Canonical":
                exact[
                    "TAX_CANONICAL"
                ],

            "Exact_Taxonomy_Phenotype":
                exact[
                    "TAX_PHENOTYPE"
                ],

            "Exact_Taxonomy_Organism":
                exact[
                    "TAX_ORGANISM"
                ],

            "Exact_Taxonomy_Status":
                exact[
                    "TAX_STATUS"
                ],

            "Exact_Mapping_Reason":
                exact[
                    "TAX_MAPPING_REASON"
                ],

            "Alternative_Representative_Found":
                False,

            "Alternative_Representative_Accession":
                "",

            "Alternative_Representative_Species":
                "",

            "Alternative_Representative_Reason":
                "NO_SAME_CANONICAL_SPECIES_IN_435"

        })


reconciliation = pd.DataFrame(
    reconciliation_rows
)


# ------------------------------------------------------------
# 11. BROADER SEARCH
# ------------------------------------------------------------

print(
    "\n" + "-" * 80
)

print(
    "BROADER ASSEMBLY SEARCH"
)

print(
    "-" * 80
)


broader_rows = []


for _, rec in reconciliation.iterrows():

    if rec[
        "Alternative_Representative_Found"
    ]:

        continue


    search_names = {
        clean_text(
            rec[
                "Exact_Taxonomy_Canonical"
            ]
        ).lower(),

        clean_text(
            rec[
                "Exact_Taxonomy_Organism"
            ]
        ).lower(),

        clean_text(
            rec[
                "Pareto_Canonical_Name"
            ]
        ).lower()
    }


    search_names = {
        x
        for x in search_names
        if x
    }


    candidate_rows = taxonomy[
        taxonomy[
            "TAX_CANONICAL"
        ].str.lower().isin(
            search_names
        )
    ].copy()


    candidate_rows = candidate_rows[
        candidate_rows[
            "TAX_ACCESSION"
        ]
        != rec[
            "Pareto_Accession"
        ]
    ]


    if len(candidate_rows) > 0:

        for _, candidate in candidate_rows.iterrows():

            broader_rows.append({

                "Pareto_Accession":
                    rec[
                        "Pareto_Accession"
                    ],

                "Pareto_Canonical_Name":
                    rec[
                        "Pareto_Canonical_Name"
                    ],

                "Candidate_Alternative_Accession":
                    candidate[
                        "TAX_ACCESSION"
                    ],

                "Candidate_Canonical_Species":
                    candidate[
                        "TAX_CANONICAL"
                    ],

                "Candidate_Phenotype_Species":
                    candidate[
                        "TAX_PHENOTYPE"
                    ],

                "Candidate_NCBI_Organism":
                    candidate[
                        "TAX_ORGANISM"
                    ],

                "Candidate_Taxonomy_Status":
                    candidate[
                        "TAX_STATUS"
                    ],

                "Candidate_Mapping_Reason":
                    candidate[
                        "TAX_MAPPING_REASON"
                    ],

                "Already_435_Representative":
                    candidate[
                        "TAX_ACCESSION"
                    ]
                    in representative_accessions

            })


# IMPORTANT:
# Explicit columns prevent an empty DataFrame from causing
# KeyError later.

broader_columns = [
    "Pareto_Accession",
    "Pareto_Canonical_Name",
    "Candidate_Alternative_Accession",
    "Candidate_Canonical_Species",
    "Candidate_Phenotype_Species",
    "Candidate_NCBI_Organism",
    "Candidate_Taxonomy_Status",
    "Candidate_Mapping_Reason",
    "Already_435_Representative"
]


broader = pd.DataFrame(
    broader_rows,
    columns=broader_columns
)


print(
    "Alternative assembly records found:",
    len(broader)
)


# ------------------------------------------------------------
# 12. GENOME FILE CHECK
# ------------------------------------------------------------

print(
    "\n" + "-" * 80
)

print(
    "GENOME FILE CHECK"
)

print(
    "-" * 80
)


genome_files = list(
    GENOME_DIR.glob("*.zip")
)


genome_accessions = {
    normalize_accession(
        path.name
    )
    for path in genome_files
}


genome_check_rows = []


for accession in sorted(
    missing_accessions
):

    genome_check_rows.append({

        "Pareto_Accession":
            accession,

        "Genome_ZIP_Present":
            accession in genome_accessions

    })


genome_check = pd.DataFrame(
    genome_check_rows
)


print(
    genome_check.to_string(
        index=False
    )
)


# ------------------------------------------------------------
# 13. FINAL CLASSIFICATION
# ------------------------------------------------------------

classification_rows = []


for accession in sorted(
    missing_accessions
):

    recs = reconciliation[
        reconciliation[
            "Pareto_Accession"
        ] == accession
    ]


    if len(recs) == 0:

        classification = (
            "REVIEW_REQUIRED"
        )

        reason = (
            "No reconciliation record"
        )

        alternative_accession = ""


    else:

        rec = recs.iloc[0]


        if rec[
            "Alternative_Representative_Found"
        ]:

            classification = (
                "REPRESENTED_BY_OTHER_435_ASSEMBLY"
            )

            reason = (
                rec[
                    "Alternative_Representative_Reason"
                ]
            )

            alternative_accession = (
                rec[
                    "Alternative_Representative_Accession"
                ]
            )


        else:

            candidate_rows = broader[
                broader[
                    "Pareto_Accession"
                ] == accession
            ]


            if len(candidate_rows) > 0:

                classification = (
                    "ALTERNATIVE_ASSEMBLY_EXISTS_REQUIRES_REPRESENTATIVE_DECISION"
                )

                reason = (
                    "Same canonical species has "
                    "another assembly in the "
                    "510-row dataset, but it is "
                    "not currently in the 435."
                )

                alternative_accession = "; ".join(
                    candidate_rows[
                        "Candidate_Alternative_Accession"
                    ]
                    .dropna()
                    .astype(str)
                    .unique()
                )


            else:

                classification = (
                    "ADD_PARETO_TO_PHYLOGENY_CANDIDATE_SET"
                )

                reason = (
                    "No alternate representative "
                    "assembly found in the current "
                    "435-species set or broader "
                    "510-row taxonomy set."
                )

                alternative_accession = ""


    classification_rows.append({

        "Pareto_Accession":
            accession,

        "Classification":
            classification,

        "Reason":
            reason,

        "Alternative_Accession":
            alternative_accession

    })


classification = pd.DataFrame(
    classification_rows
)


# ------------------------------------------------------------
# 14. SAVE OUTPUTS
# ------------------------------------------------------------

reconciliation_file = (
    RESULTS_DIR
    / "stage4A1_missing_pareto_reconciliation.csv"
)

reconciliation.to_csv(
    reconciliation_file,
    index=False
)


broader_file = (
    RESULTS_DIR
    / "stage4A1_broader_alternative_assemblies.csv"
)

broader.to_csv(
    broader_file,
    index=False
)


genome_file = (
    RESULTS_DIR
    / "stage4A1_missing_pareto_genome_check.csv"
)

genome_check.to_csv(
    genome_file,
    index=False
)


classification_file = (
    RESULTS_DIR
    / "stage4A1_final_classification.csv"
)

classification.to_csv(
    classification_file,
    index=False
)


# ------------------------------------------------------------
# 15. REPORT
# ------------------------------------------------------------

report_file = (
    RESULTS_DIR
    / "stage4A1_report.txt"
)


with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 4A.1 — PARETO ↔ 435 RECONCILIATION\n"
    )

    f.write(
        "=" * 70 + "\n\n"
    )

    f.write(
        f"435 representative species: "
        f"{len(representatives)}\n"
    )

    f.write(
        f"Pareto candidates: "
        f"{len(pareto_accessions)}\n"
    )

    f.write(
        f"Pareto accessions already in 435: "
        f"{len(present_accessions)}\n"
    )

    f.write(
        f"Pareto accessions missing from 435: "
        f"{len(missing_accessions)}\n\n"
    )

    f.write(
        "FINAL CLASSIFICATION\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for _, row in classification.iterrows():

        f.write(
            f"{row['Pareto_Accession']}\t"
            f"{row['Classification']}\t"
            f"{row['Alternative_Accession']}\n"
        )

    f.write(
        "\nGENOME CHECK\n"
    )

    f.write(
        "-" * 70 + "\n"
    )

    for _, row in genome_check.iterrows():

        f.write(
            f"{row['Pareto_Accession']}\t"
            f"{row['Genome_ZIP_Present']}\n"
        )

    f.write(
        "\nNo genome files were modified.\n"
    )


# ------------------------------------------------------------
# 16. CONSOLE SUMMARY
# ------------------------------------------------------------

print(
    "\n" + "=" * 80
)

print(
    "STAGE 4A.1 COMPLETE"
)

print(
    "=" * 80
)


print(
    "\nMissing Pareto accessions:"
)

for accession in sorted(
    missing_accessions
):

    print(
        " ",
        accession
    )


print(
    "\nClassification:"
)

print(
    classification.to_string(
        index=False
    )
)


print(
    "\nOutputs:"
)

print(
    reconciliation_file
)

print(
    broader_file
)

print(
    genome_file
)

print(
    classification_file
)

print(
    report_file
)


print(
    "\nNO GENOME FILES WERE MODIFIED."
)

print(
    "\nDo NOT change the 435 set yet."
)

print(
    "Review this output before Stage 4B."
)

print(
    "=" * 80
)