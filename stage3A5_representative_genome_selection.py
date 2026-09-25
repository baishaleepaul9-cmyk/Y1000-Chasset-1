import os
import re
import pandas as pd


# ============================================================
# STAGE 3A.5
# REPRESENTATIVE GENOME SELECTION
# ============================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

CANDIDATE_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A4_assembly_candidates",
    "all_species_assembly_candidates.csv"
)

PARETO_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage2B_pareto_analysis",
    "pareto_frontier.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A5_representative_genomes"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# ASSEMBLY LEVEL RANK
# ============================================================

ASSEMBLY_LEVEL_RANK = {
    "complete genome": 4,
    "complete": 4,
    "chromosome": 3,
    "scaffold": 2,
    "contig": 1
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    return str(value).strip()


def normalize_species(value):

    value = clean_text(value)

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.lower()


def assembly_level_score(value):

    value = clean_text(value).lower()

    return ASSEMBLY_LEVEL_RANK.get(
        value,
        0
    )


def is_refseq(accession):

    accession = clean_text(
        accession
    ).upper()

    return accession.startswith(
        "GCF_"
    )


def has_annotation(row):

    gene_count = row.get(
        "Gene_Count",
        ""
    )

    if pd.isna(gene_count):
        return False

    try:

        return float(
            gene_count
        ) > 0

    except:

        return False


def numeric_value(value):

    if pd.isna(value):
        return 0.0

    try:

        return float(value)

    except:

        return 0.0


def date_value(value):

    if pd.isna(value):
        return pd.Timestamp(
            "1900-01-01"
        )

    try:

        parsed = pd.to_datetime(
            value,
            errors="coerce"
        )

        if pd.isna(parsed):
            return pd.Timestamp(
                "1900-01-01"
            )

        return parsed

    except:

        return pd.Timestamp(
            "1900-01-01"
        )


def exact_species_match(
    phenotype_species,
    organism_name
):

    phenotype = normalize_species(
        phenotype_species
    )

    organism = normalize_species(
        organism_name
    )

    if not phenotype or not organism:
        return False

    if organism == phenotype:
        return True

    if organism.startswith(
        phenotype + " "
    ):
        return True

    return False


# ============================================================
# SAFE FILTER FUNCTION
# ============================================================

def retain_best_by_column(
    dataframe,
    column
):

    """
    Keep rows having the best value in a column.

    If the column contains no useful information,
    return the dataframe unchanged.

    This function NEVER returns an empty dataframe
    unless the input dataframe is already empty.
    """

    if dataframe.empty:
        return dataframe

    values = dataframe[
        column
    ]

    if values.isna().all():
        return dataframe

    max_value = values.max()

    filtered = dataframe[
        values == max_value
    ].copy()

    if filtered.empty:
        return dataframe

    return filtered


# ============================================================
# 1. LOAD CANDIDATES
# ============================================================

print("=" * 75)
print(
    "STAGE 3A.5 — REPRESENTATIVE GENOME SELECTION"
)
print("=" * 75)

print(
    "\nLoading assembly candidates..."
)

candidates = pd.read_csv(
    CANDIDATE_FILE
)

print(
    f"Candidate rows: {len(candidates)}"
)


# ============================================================
# 2. LOAD PARETO FRONTIER
# ============================================================

print(
    "\nLoading Pareto frontier..."
)

pareto = pd.read_csv(
    PARETO_FILE
)

pareto_species = set(
    pareto[
        "Species"
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

print(
    f"Pareto species: "
    f"{len(pareto_species)}"
)


# ============================================================
# 3. CLEAN TEXT COLUMNS
# ============================================================

text_columns = [
    "Phenotype_Species",
    "NCBI_Assembly_ID",
    "Assembly_Accession",
    "Organism_Name",
    "Assembly_Name",
    "Assembly_Level",
    "Submitter",
    "BioProject",
    "BioSample",
    "Infraspecies",
    "Release_Date",
    "Candidate_Source"
]

for column in text_columns:

    if column in candidates.columns:

        candidates[column] = (
            candidates[column]
            .fillna("")
            .astype(str)
            .str.strip()
        )


# ============================================================
# 4. REMOVE ROWS WITHOUT ASSEMBLY ACCESSION
# ============================================================

usable = candidates[
    candidates[
        "Assembly_Accession"
    ].str.strip() != ""
].copy()

print(
    f"Usable assembly candidates: "
    f"{len(usable)}"
)


# ============================================================
# 5. CREATE QUALITY FEATURES
# ============================================================

usable[
    "Exact_Species_Match"
] = usable.apply(
    lambda row:
    exact_species_match(
        row[
            "Phenotype_Species"
        ],
        row[
            "Organism_Name"
        ]
    ),
    axis=1
)


usable[
    "Assembly_Level_Score"
] = usable[
    "Assembly_Level"
].apply(
    assembly_level_score
)


usable[
    "Has_Annotation"
] = usable.apply(
    has_annotation,
    axis=1
)


usable[
    "RefSeq"
] = usable[
    "Assembly_Accession"
].apply(
    is_refseq
)


usable[
    "Contig_N50_Numeric"
] = usable[
    "Contig_N50"
].apply(
    numeric_value
)


usable[
    "Genome_Size_Numeric"
] = usable[
    "Genome_Size"
].apply(
    numeric_value
)


usable[
    "Release_Date_Numeric"
] = usable[
    "Release_Date"
].apply(
    date_value
)


# ============================================================
# 6. SELECTION RATIONALE
# ============================================================

def selection_rationale(row):

    reasons = []

    if row[
        "Exact_Species_Match"
    ]:

        reasons.append(
            "exact species match"
        )

    level = clean_text(
        row[
            "Assembly_Level"
        ]
    )

    if level:

        reasons.append(
            f"assembly level={level}"
        )

    if row[
        "Has_Annotation"
    ]:

        reasons.append(
            "annotation available"
        )

    if row[
        "RefSeq"
    ]:

        reasons.append(
            "RefSeq assembly"
        )

    if row[
        "Contig_N50_Numeric"
    ] > 0:

        reasons.append(
            "N50 available"
        )

    return "; ".join(
        reasons
    )


usable[
    "Selection_Rationale"
] = usable.apply(
    selection_rationale,
    axis=1
)


# ============================================================
# 7. SELECT REPRESENTATIVE ASSEMBLY
# ============================================================

print(
    "\nSelecting representative assemblies..."
)

selected_rows = []

selection_log = []

all_species = sorted(
    usable[
        "Phenotype_Species"
    ]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
)


for species in all_species:

    group = usable[
        usable[
            "Phenotype_Species"
        ] == species
    ].copy()


    # --------------------------------------------------------
    # SAFETY CHECK
    # --------------------------------------------------------

    if group.empty:
        continue


    original_count = len(
        group
    )

    working = group.copy()


    # --------------------------------------------------------
    # STEP 1 — EXACT SPECIES MATCH
    # --------------------------------------------------------

    exact = working[
        working[
            "Exact_Species_Match"
        ]
    ].copy()

    if not exact.empty:

        working = exact


    # --------------------------------------------------------
    # STEP 2 — ASSEMBLY LEVEL
    # --------------------------------------------------------

    filtered = retain_best_by_column(
        working,
        "Assembly_Level_Score"
    )

    if not filtered.empty:

        working = filtered


    # --------------------------------------------------------
    # STEP 3 — ANNOTATION
    # --------------------------------------------------------

    annotated = working[
        working[
            "Has_Annotation"
        ]
    ].copy()

    if not annotated.empty:

        working = annotated


    # --------------------------------------------------------
    # STEP 4 — REFSEQ
    # --------------------------------------------------------

    refseq = working[
        working[
            "RefSeq"
        ]
    ].copy()

    if not refseq.empty:

        working = refseq


    # --------------------------------------------------------
    # STEP 5 — CONTIG N50
    # --------------------------------------------------------

    filtered = retain_best_by_column(
        working,
        "Contig_N50_Numeric"
    )

    if not filtered.empty:

        working = filtered


    # --------------------------------------------------------
    # STEP 6 — RELEASE DATE
    # --------------------------------------------------------

    filtered = retain_best_by_column(
        working,
        "Release_Date_Numeric"
    )

    if not filtered.empty:

        working = filtered


    # --------------------------------------------------------
    # STEP 7 — DETERMINISTIC ACCESSION TIE-BREAK
    # --------------------------------------------------------

    working = working.sort_values(
        by=[
            "Assembly_Accession"
        ],
        ascending=True,
        na_position="last"
    ).reset_index(
        drop=True
    )


    # --------------------------------------------------------
    # FINAL SAFETY CHECK
    # --------------------------------------------------------

    if working.empty:

        # This should never happen, but retain
        # the first original candidate if it does.

        working = group.copy()

        working = working.sort_values(
            by=[
                "Assembly_Accession"
            ],
            ascending=True,
            na_position="last"
        ).reset_index(
            drop=True
        )


    if working.empty:

        print(
            f"WARNING: Could not select genome for "
            f"{species}"
        )

        continue


    # --------------------------------------------------------
    # SELECT FIRST REMAINING CANDIDATE
    # --------------------------------------------------------

    selected = working.iloc[
        0
    ].copy()


    # --------------------------------------------------------
    # STORE SELECTED ASSEMBLY
    # --------------------------------------------------------

    selected_rows.append(
        selected.to_dict()
    )


    # --------------------------------------------------------
    # SELECTION LOG
    # --------------------------------------------------------

    selection_log.append(
        {
            "Phenotype_Species": species,

            "Original_Candidate_Count":
                original_count,

            "Final_Tied_Candidate_Count":
                len(working),

            "Selected_Assembly":
                selected[
                    "Assembly_Accession"
                ],

            "Selected_Assembly_Level":
                selected[
                    "Assembly_Level"
                ],

            "Selected_RefSeq":
                selected[
                    "RefSeq"
                ],

            "Selected_Has_Annotation":
                selected[
                    "Has_Annotation"
                ],

            "Selected_Contig_N50":
                selected[
                    "Contig_N50_Numeric"
                ],

            "Selected_Release_Date":
                selected[
                    "Release_Date"
                ],

            "Selection_Rationale":
                selected[
                    "Selection_Rationale"
                ]
        }
    )


# ============================================================
# 8. CREATE REPRESENTATIVE TABLE
# ============================================================

representative = pd.DataFrame(
    selected_rows
)

selection_log_df = pd.DataFrame(
    selection_log
)


# ============================================================
# 9. ADD PARETO STATUS
# ============================================================

representative[
    "Pareto_Optimal"
] = representative[
    "Phenotype_Species"
].isin(
    pareto_species
)

selection_log_df[
    "Pareto_Optimal"
] = selection_log_df[
    "Phenotype_Species"
].isin(
    pareto_species
)


# ============================================================
# 10. ADD SELECTION STATUS
# ============================================================

representative[
    "Selection_Status"
] = "Selected"


# ============================================================
# 11. SAVE REPRESENTATIVE TABLE
# ============================================================

representative_file = os.path.join(
    OUTPUT_DIR,
    "representative_assembly_table.csv"
)

representative.to_csv(
    representative_file,
    index=False
)


# ============================================================
# 12. SAVE SELECTION LOG
# ============================================================

selection_log_file = os.path.join(
    OUTPUT_DIR,
    "assembly_selection_log.csv"
)

selection_log_df.to_csv(
    selection_log_file,
    index=False
)


# ============================================================
# 13. IDENTIFY UNRESOLVED SPECIES
# ============================================================

all_candidate_species = set(
    candidates[
        "Phenotype_Species"
    ]
    .dropna()
    .astype(str)
    .str.strip()
)

selected_species = set(
    representative[
        "Phenotype_Species"
    ]
)

unresolved_species = sorted(
    all_candidate_species
    - selected_species
)


unresolved_rows = []

for species in unresolved_species:

    unresolved_rows.append(
        {
            "Phenotype_Species":
                species,

            "Reason":
                "No usable assembly candidate",

            "Candidate_Count":
                int(
                    len(
                        candidates[
                            candidates[
                                "Phenotype_Species"
                            ] == species
                        ]
                    )
                ),

            "Pareto_Optimal":
                species in pareto_species
        }
    )


unresolved_df = pd.DataFrame(
    unresolved_rows
)


# ============================================================
# 14. SAVE UNRESOLVED TABLE
# ============================================================

unresolved_file = os.path.join(
    OUTPUT_DIR,
    "unresolved_species_review.csv"
)

unresolved_df.to_csv(
    unresolved_file,
    index=False
)


# ============================================================
# 15. CHECK PARETO COVERAGE
# ============================================================

pareto_selected = representative[
    representative[
        "Pareto_Optimal"
    ]
].copy()

pareto_selected_count = len(
    pareto_selected
)

pareto_missing = sorted(
    pareto_species
    - set(
        pareto_selected[
            "Phenotype_Species"
        ]
    )
)


# ============================================================
# 16. PRINT PARETO RESULTS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "PARETO REPRESENTATIVE GENOME CHECK"
)

print(
    "=" * 75
)

print(
    f"\nPareto species: "
    f"{len(pareto_species)}"
)

print(
    f"Pareto species with selected genomes: "
    f"{pareto_selected_count}"
)

print(
    f"Pareto species without selected genomes: "
    f"{len(pareto_missing)}"
)


if len(pareto_selected) > 0:

    print(
        "\nSelected Pareto genomes:"
    )

    print(
        pareto_selected[
            [
                "Phenotype_Species",
                "Assembly_Accession",
                "Assembly_Level",
                "RefSeq",
                "Has_Annotation",
                "Contig_N50",
                "Release_Date"
            ]
        ].to_string(
            index=False
        )
    )


if len(pareto_missing) > 0:

    print(
        "\nPareto species still unresolved:"
    )

    for species in pareto_missing:

        print(
            f"- {species}"
        )


# ============================================================
# 17. OVERALL SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STAGE 3A.5 SUMMARY"
)

print(
    "=" * 75
)

print(
    f"\nPhenotype species: "
    f"{len(all_candidate_species)}"
)

print(
    f"Representative genomes selected: "
    f"{len(representative)}"
)

print(
    f"Species unresolved: "
    f"{len(unresolved_species)}"
)

if len(all_candidate_species) > 0:

    print(
        f"Representative genome coverage: "
        f"{100 * len(representative) / len(all_candidate_species):.2f}%"
    )

if len(pareto_species) > 0:

    print(
        f"Pareto genome coverage: "
        f"{100 * pareto_selected_count / len(pareto_species):.2f}%"
    )


# ============================================================
# 18. ASSEMBLY LEVEL DISTRIBUTION
# ============================================================

print(
    "\nRepresentative assembly levels:"
)

if len(representative) > 0:

    level_counts = (
        representative[
            "Assembly_Level"
        ]
        .replace(
            "",
            "Unknown"
        )
        .value_counts()
    )

    print(
        level_counts.to_string()
    )


# ============================================================
# 19. REFSEQ VS GENBANK
# ============================================================

print(
    "\nRepresentative RefSeq vs GenBank:"
)

if len(representative) > 0:

    refseq_count = int(
        representative[
            "RefSeq"
        ].sum()
    )

else:

    refseq_count = 0


genbank_count = (
    len(representative)
    - refseq_count
)

print(
    f"RefSeq (GCF): "
    f"{refseq_count}"
)

print(
    f"GenBank/other: "
    f"{genbank_count}"
)


# ============================================================
# 20. WRITE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A5_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 3A.5 — REPRESENTATIVE GENOME SELECTION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        "Selection hierarchy:\n"
    )

    f.write(
        "1. Exact species match\n"
    )

    f.write(
        "2. Higher assembly level\n"
    )

    f.write(
        "3. Availability of annotation\n"
    )

    f.write(
        "4. RefSeq assembly when otherwise comparable\n"
    )

    f.write(
        "5. Higher contig N50\n"
    )

    f.write(
        "6. Latest release date\n"
    )

    f.write(
        "7. Assembly accession as deterministic tie-break\n\n"
    )

    f.write(
        f"Phenotype species represented by candidates: "
        f"{len(all_candidate_species)}\n"
    )

    f.write(
        f"Representative genomes selected: "
        f"{len(representative)}\n"
    )

    f.write(
        f"Species unresolved: "
        f"{len(unresolved_species)}\n"
    )

    if len(all_candidate_species) > 0:

        f.write(
            f"Overall representative coverage: "
            f"{100 * len(representative) / len(all_candidate_species):.2f}%\n"
        )

    f.write(
        f"\nPareto species: "
        f"{len(pareto_species)}\n"
    )

    f.write(
        f"Pareto species with representative genome: "
        f"{pareto_selected_count}\n"
    )

    f.write(
        f"Pareto genome coverage: "
        f"{100 * pareto_selected_count / len(pareto_species):.2f}%\n"
    )

    f.write(
        f"\nRefSeq representatives: "
        f"{refseq_count}\n"
    )

    f.write(
        f"GenBank/other representatives: "
        f"{genbank_count}\n"
    )

    f.write(
        "\nUnresolved species:\n"
    )

    if len(unresolved_species) == 0:

        f.write(
            "None\n"
        )

    else:

        for species in unresolved_species:

            f.write(
                f"- {species}\n"
            )

    f.write(
        "\nPareto species without representative genome:\n"
    )

    if len(pareto_missing) == 0:

        f.write(
            "None\n"
        )

    else:

        for species in pareto_missing:

            f.write(
                f"- {species}\n"
            )


# ============================================================
# 21. COMPLETION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STAGE 3A.5 COMPLETED"
)

print(
    "=" * 75
)

print(
    "\nGenerated files:"
)

print(
    representative_file
)

print(
    selection_log_file
)

print(
    unresolved_file
)

print(
    report_file
)

print(
    "\nNO GENOME SEQUENCES WERE DOWNLOADED."
)

print(
    "Representative assemblies were selected "
    "only from the existing candidate metadata."
)