import os
import time
import requests
import pandas as pd


# ============================================================
# STAGE 3A.5B
# VALIDATE SELECTED NCBI ASSEMBLIES
# ============================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

SELECTED_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A5_representative_genomes",
    "representative_assembly_table.csv"
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
    "stage3A5B_metadata_validation"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NCBI E-UTILS
# ============================================================

NCBI_BASE = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
)


def ncbi_esearch(query):

    url = NCBI_BASE + "esearch.fcgi"

    params = {
        "db": "assembly",
        "term": query,
        "retmode": "json",
        "retmax": 20
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    return response.json()["esearchresult"]


def ncbi_esummary(ids):

    if not ids:
        return {}

    url = NCBI_BASE + "esummary.fcgi"

    params = {
        "db": "assembly",
        "id": ",".join(ids),
        "retmode": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    response.raise_for_status()

    return response.json()["result"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean(value):

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def first_value(dictionary, keys):

    if not isinstance(
        dictionary,
        dict
    ):
        return ""

    for key in keys:

        value = dictionary.get(
            key,
            ""
        )

        if value not in [
            "",
            None
        ]:

            return value

    return ""


def flatten(value):

    if value is None:
        return ""

    if isinstance(
        value,
        (str, int, float, bool)
    ):

        return value

    if isinstance(value, list):

        return ";".join(
            str(x)
            for x in value
        )

    if isinstance(value, dict):

        return str(value)

    return str(value)


def is_refseq(accession):

    accession = clean(
        accession
    ).upper()

    return accession.startswith(
        "GCF_"
    )


# ============================================================
# 1. LOAD SELECTED ASSEMBLIES
# ============================================================

print("=" * 75)
print(
    "STAGE 3A.5B — VALIDATE SELECTED NCBI ASSEMBLIES"
)
print("=" * 75)

print(
    "\nLoading selected assembly table..."
)

selected = pd.read_csv(
    SELECTED_FILE
)

print(
    f"Selected assemblies: "
    f"{len(selected)}"
)


# ============================================================
# 2. LOAD PARETO SPECIES
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
# 3. CLEAN ACCESSIONS
# ============================================================

selected[
    "Assembly_Accession"
] = (
    selected[
        "Assembly_Accession"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
)


selected = selected[
    selected[
        "Assembly_Accession"
    ] != ""
].copy()


print(
    f"Assemblies with accession: "
    f"{len(selected)}"
)


# ============================================================
# 4. SEARCH NCBI FOR EACH SELECTED ACCESSION
# ============================================================

print(
    "\nResolving NCBI assembly accessions..."
)

resolved_ids = []

resolution_rows = []


for i, accession in enumerate(
    selected[
        "Assembly_Accession"
    ],
    start=1
):

    print(
        f"[{i}/{len(selected)}] "
        f"{accession}"
    )

    query = (
        f'"{accession}"[Assembly Accession]'
    )

    try:

        result = ncbi_esearch(
            query
        )

        ids = result.get(
            "idlist",
            []
        )

        if ids:

            resolved_id = ids[0]

            resolved_ids.append(
                resolved_id
            )

        else:

            resolved_id = ""

        resolution_rows.append(
            {
                "Assembly_Accession":
                    accession,

                "NCBI_Assembly_ID":
                    resolved_id,

                "Resolved":
                    bool(resolved_id)
            }
        )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )

        resolution_rows.append(
            {
                "Assembly_Accession":
                    accession,

                "NCBI_Assembly_ID":
                    "",

                "Resolved":
                    False
            }
        )

    time.sleep(
        0.25
    )


resolution_df = pd.DataFrame(
    resolution_rows
)


# ============================================================
# 5. SAVE ACCESSION RESOLUTION
# ============================================================

resolution_file = os.path.join(
    OUTPUT_DIR,
    "assembly_accession_resolution.csv"
)

resolution_df.to_csv(
    resolution_file,
    index=False
)


# ============================================================
# 6. RETRIEVE METADATA IN BATCHES
# ============================================================

unique_ids = sorted(
    set(
        x
        for x in resolved_ids
        if x
    )
)

print(
    "\nResolved NCBI assembly records:",
    len(unique_ids)
)


metadata = {}

batch_size = 50


for start in range(
    0,
    len(unique_ids),
    batch_size
):

    batch = unique_ids[
        start:start + batch_size
    ]

    end = min(
        start + batch_size,
        len(unique_ids)
    )

    print(
        f"Metadata batch "
        f"{start + 1}-{end}"
    )

    try:

        result = ncbi_esummary(
            batch
        )

        for key, value in result.items():

            if key == "uids":
                continue

            metadata[
                str(key)
            ] = value

    except Exception as e:

        print(
            "  METADATA ERROR:",
            str(e)
        )

    time.sleep(
        0.4
    )


# ============================================================
# 7. BUILD VALIDATED METADATA TABLE
# ============================================================

validated_rows = []


for _, original in selected.iterrows():

    accession = original[
        "Assembly_Accession"
    ]

    resolution = resolution_df[
        resolution_df[
            "Assembly_Accession"
        ] == accession
    ]

    if len(resolution) > 0:

        ncbi_id = clean(
            resolution.iloc[0][
                "NCBI_Assembly_ID"
            ]
        )

    else:

        ncbi_id = ""


    record = metadata.get(
        ncbi_id,
        {}
    )


    # --------------------------------------------------------
    # CORE FIELDS
    # --------------------------------------------------------

    organism = flatten(
        first_value(
            record,
            [
                "organism"
            ]
        )
    )


    assembly_info = record.get(
        "assemblyinfo",
        {}
    )

    if not isinstance(
        assembly_info,
        dict
    ):

        assembly_info = {}


    assembly_stats = record.get(
        "assemblystats",
        {}
    )

    if not isinstance(
        assembly_stats,
        dict
    ):

        assembly_stats = {}


    annotation_info = record.get(
        "annotationinfo",
        {}
    )

    if not isinstance(
        annotation_info,
        dict
    ):

        annotation_info = {}


    # --------------------------------------------------------
    # ASSEMBLY LEVEL
    # --------------------------------------------------------

    assembly_level = flatten(
        first_value(
            assembly_info,
            [
                "assemblylevel",
                "assemblyLevel",
                "level"
            ]
        )
    )


    # --------------------------------------------------------
    # ASSEMBLY NAME
    # --------------------------------------------------------

    assembly_name = flatten(
        first_value(
            assembly_info,
            [
                "assemblyname",
                "assemblyName",
                "name"
            ]
        )
    )


    # --------------------------------------------------------
    # SUBMITTER
    # --------------------------------------------------------

    submitter = flatten(
        first_value(
            record,
            [
                "submitter"
            ]
        )
    )

    if not submitter:

        submitter = flatten(
            first_value(
                assembly_info,
                [
                    "submitter"
                ]
            )
        )


    # --------------------------------------------------------
    # BIOPROJECT
    # --------------------------------------------------------

    bioproject = flatten(
        first_value(
            record,
            [
                "bioproject",
                "bioprojectaccn",
                "bioprojectAccession"
            ]
        )
    )


    # --------------------------------------------------------
    # BIOSAMPLE
    # --------------------------------------------------------

    biosample = flatten(
        first_value(
            record,
            [
                "biosample",
                "biosampleaccn",
                "biosampleAccession"
            ]
        )
    )


    # --------------------------------------------------------
    # INFRASPECIES
    # --------------------------------------------------------

    infraspecies = flatten(
        first_value(
            record,
            [
                "infraspecies"
            ]
        )
    )

    if not infraspecies:

        infraspecies = flatten(
            first_value(
                assembly_info,
                [
                    "infraspecificname",
                    "infraspecificName"
                ]
            )
        )


    # --------------------------------------------------------
    # RELEASE DATE
    # --------------------------------------------------------

    release_date = flatten(
        first_value(
            assembly_info,
            [
                "releasedate",
                "releaseDate",
                "release_date"
            ]
        )
    )


    # --------------------------------------------------------
    # GENOME SIZE
    # --------------------------------------------------------

    genome_size = flatten(
        first_value(
            assembly_stats,
            [
                "totallength",
                "totalLength",
                "total_sequence_length",
                "totalSequenceLength"
            ]
        )
    )


    # --------------------------------------------------------
    # CONTIG N50
    # --------------------------------------------------------

    contig_n50 = flatten(
        first_value(
            assembly_stats,
            [
                "contign50",
                "contigN50",
                "contig_n50"
            ]
        )
    )


    # --------------------------------------------------------
    # SCAFFOLD N50
    # --------------------------------------------------------

    scaffold_n50 = flatten(
        first_value(
            assembly_stats,
            [
                "scaffoldn50",
                "scaffoldN50",
                "scaffold_n50"
            ]
        )
    )


    # --------------------------------------------------------
    # GENE COUNT
    # --------------------------------------------------------

    gene_count = flatten(
        first_value(
            annotation_info,
            [
                "totalgene",
                "totalGene",
                "total_gene_count",
                "totalGeneCount"
            ]
        )
    )


    # --------------------------------------------------------
    # ANNOTATION STATUS
    # --------------------------------------------------------

    annotation_status = flatten(
        first_value(
            annotation_info,
            [
                "name",
                "provider",
                "release",
                "annotationname",
                "annotationName"
            ]
        )
    )


    # --------------------------------------------------------
    # SOURCE / REFSEQ
    # --------------------------------------------------------

    refseq = is_refseq(
        accession
    )


    # --------------------------------------------------------
    # EXACT SPECIES VALIDATION
    # --------------------------------------------------------

    phenotype_species = clean(
        original[
            "Phenotype_Species"
        ]
    )

    organism_clean = organism.lower()

    phenotype_clean = (
        phenotype_species.lower()
    )

    exact_match = (
        organism_clean
        == phenotype_clean
        or organism_clean.startswith(
            phenotype_clean + " "
        )
    )


    # --------------------------------------------------------
    # ADD ROW
    # --------------------------------------------------------

    validated_rows.append(
        {
            "Phenotype_Species":
                phenotype_species,

            "Assembly_Accession":
                accession,

            "NCBI_Assembly_ID":
                ncbi_id,

            "Organism_Name":
                organism,

            "Exact_Species_Match":
                exact_match,

            "Assembly_Name":
                assembly_name,

            "Assembly_Level":
                assembly_level,

            "RefSeq":
                refseq,

            "Submitter":
                submitter,

            "BioProject":
                bioproject,

            "BioSample":
                biosample,

            "Infraspecies":
                infraspecies,

            "Release_Date":
                release_date,

            "Genome_Size":
                genome_size,

            "Contig_N50":
                contig_n50,

            "Scaffold_N50":
                scaffold_n50,

            "Gene_Count":
                gene_count,

            "Annotation_Status":
                annotation_status,

            "Pareto_Optimal":
                phenotype_species
                in pareto_species
        }
    )


validated = pd.DataFrame(
    validated_rows
)


# ============================================================
# 8. SAVE VALIDATED METADATA
# ============================================================

validated_file = os.path.join(
    OUTPUT_DIR,
    "validated_assembly_metadata.csv"
)

validated.to_csv(
    validated_file,
    index=False
)


# ============================================================
# 9. CHECK RESOLUTION
# ============================================================

validated[
    "Metadata_Resolved"
] = (
    validated[
        "NCBI_Assembly_ID"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .ne("")
)


# ============================================================
# 10. CHECK EXACT SPECIES MATCH
# ============================================================

exact_count = int(
    validated[
        "Exact_Species_Match"
    ].sum()
)

resolved_count = int(
    validated[
        "Metadata_Resolved"
    ].sum()
)


# ============================================================
# 11. PARETO CHECK
# ============================================================

pareto_validated = validated[
    validated[
        "Pareto_Optimal"
    ]
].copy()


pareto_resolved = int(
    pareto_validated[
        "Metadata_Resolved"
    ].sum()
)


pareto_exact = int(
    pareto_validated[
        "Exact_Species_Match"
    ].sum()
)


# ============================================================
# 12. PRINT SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STAGE 3A.5B SUMMARY"
)

print(
    "=" * 75
)

print(
    f"\nSelected assemblies: "
    f"{len(selected)}"
)

print(
    f"NCBI metadata resolved: "
    f"{resolved_count}"
)

print(
    f"Exact species matches: "
    f"{exact_count}"
)

print(
    f"Metadata resolution: "
    f"{100 * resolved_count / len(selected):.2f}%"
)

print(
    f"Exact species-match rate: "
    f"{100 * exact_count / len(selected):.2f}%"
)


print(
    "\nPareto species:"
)

print(
    f"Pareto assemblies: "
    f"{len(pareto_validated)}"
)

print(
    f"Pareto metadata resolved: "
    f"{pareto_resolved}"
)

print(
    f"Pareto exact species matches: "
    f"{pareto_exact}"
)


# ============================================================
# 13. ASSEMBLY LEVEL SUMMARY
# ============================================================

print(
    "\nAssembly levels:"
)

level_counts = (
    validated[
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
# 14. ANNOTATION SUMMARY
# ============================================================

print(
    "\nAnnotation status:"
)

annotation_counts = (
    validated[
        "Annotation_Status"
    ]
    .replace(
        "",
        "Unknown"
    )
    .value_counts()
)

print(
    annotation_counts.head(
        20
    ).to_string()
)


# ============================================================
# 15. UNRESOLVED METADATA
# ============================================================

unresolved = validated[
    ~validated[
        "Metadata_Resolved"
    ]
].copy()


unresolved_file = os.path.join(
    OUTPUT_DIR,
    "assemblies_metadata_unresolved.csv"
)

unresolved.to_csv(
    unresolved_file,
    index=False
)


print(
    f"\nMetadata-unresolved assemblies: "
    f"{len(unresolved)}"
)


# ============================================================
# 16. SPECIES WITH NON-EXACT ORGANISM MATCH
# ============================================================

non_exact = validated[
    ~validated[
        "Exact_Species_Match"
    ]
].copy()


non_exact_file = os.path.join(
    OUTPUT_DIR,
    "assemblies_non_exact_species_match.csv"
)

non_exact.to_csv(
    non_exact_file,
    index=False
)


print(
    f"Non-exact organism matches: "
    f"{len(non_exact)}"
)


# ============================================================
# 17. PARETO VALIDATION TABLE
# ============================================================

pareto_file = os.path.join(
    OUTPUT_DIR,
    "pareto_validated_assemblies.csv"
)

pareto_validated.to_csv(
    pareto_file,
    index=False
)


# ============================================================
# 18. WRITE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A5B_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 3A.5B — ASSEMBLY METADATA VALIDATION\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Selected assemblies: "
        f"{len(selected)}\n"
    )

    f.write(
        f"NCBI metadata resolved: "
        f"{resolved_count}\n"
    )

    f.write(
        f"Exact species matches: "
        f"{exact_count}\n"
    )

    f.write(
        f"Metadata resolution: "
        f"{100 * resolved_count / len(selected):.2f}%\n"
    )

    f.write(
        f"Exact species-match rate: "
        f"{100 * exact_count / len(selected):.2f}%\n"
    )

    f.write(
        f"\nPareto species: "
        f"{len(pareto_validated)}\n"
    )

    f.write(
        f"Pareto metadata resolved: "
        f"{pareto_resolved}\n"
    )

    f.write(
        f"Pareto exact matches: "
        f"{pareto_exact}\n"
    )

    f.write(
        f"\nMetadata-unresolved assemblies: "
        f"{len(unresolved)}\n"
    )

    f.write(
        f"Non-exact organism matches: "
        f"{len(non_exact)}\n"
    )


# ============================================================
# 19. COMPLETION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STAGE 3A.5B COMPLETED"
)

print(
    "=" * 75
)

print(
    "\nGenerated files:"
)

print(
    resolution_file
)

print(
    validated_file
)

print(
    unresolved_file
)

print(
    non_exact_file
)

print(
    pareto_file
)

print(
    report_file
)

print(
    "\nNO GENOME SEQUENCES WERE DOWNLOADED."
)

print(
    "This stage only validates the selected "
    "assembly accessions and their metadata."
)