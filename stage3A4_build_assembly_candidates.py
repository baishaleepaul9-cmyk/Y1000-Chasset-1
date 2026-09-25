import os
import time
import requests
import pandas as pd


# ============================================================
# STAGE 3A.4
# BUILD FINAL NCBI ASSEMBLY CANDIDATE TABLE
# ============================================================

PROJECT_ROOT = r"C:\Y1000_chassis_project"

PHENOTYPE_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage2A_multitrait_profile",
    "species_multitrait_profile.csv"
)

MAPPING_FILE = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A_genome_mapping",
    "species_genome_mapping.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "stage3A4_assembly_candidates"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# NCBI E-UTILS
# ============================================================

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def ncbi_esearch(query):

    url = NCBI_BASE + "esearch.fcgi"

    params = {
        "db": "assembly",
        "term": query,
        "retmode": "json",
        "retmax": 100
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

def safe_get(d, *keys):

    value = d

    for key in keys:

        if not isinstance(value, dict):
            return ""

        value = value.get(
            key,
            ""
        )

    return value


def flatten_value(value):

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

    return str(value)


def extract_accession(record):

    accession = (
        record.get("assemblyaccession")
        or record.get("assembly_accession")
        or record.get("accession")
        or ""
    )

    return accession


def existing_id_in_mapping(
    assembly_id,
    existing_ids
):

    return str(assembly_id) in [
        str(x)
        for x in existing_ids
    ]


# ============================================================
# 1. LOAD PHENOTYPE SPECIES
# ============================================================

print("=" * 75)
print("STAGE 3A.4 — BUILD NCBI ASSEMBLY CANDIDATE TABLE")
print("=" * 75)

print("\nLoading phenotype species...")

phenotype = pd.read_csv(
    PHENOTYPE_FILE
)

species_list = sorted(
    phenotype["Species"]
    .dropna()
    .astype(str)
    .unique()
)

print(
    f"Phenotype species: {len(species_list)}"
)


# ============================================================
# 2. LOAD EXISTING STAGE 3A.1 MAPPING
# ============================================================

print("\nLoading Stage 3A.1 mapping...")

mapping = pd.read_csv(
    MAPPING_FILE
)

print(
    f"Mapping rows: {len(mapping)}"
)


# ============================================================
# 3. IDENTIFY EXISTING ASSEMBLY MATCHES
# ============================================================

mapping_lookup = {}

for _, row in mapping.iterrows():

    species = str(
        row["Species"]
    )

    ids = str(
        row.get(
            "NCBI_Assembly_IDs",
            ""
        )
    )

    if ids.lower() == "nan":
        ids = ""

    id_list = [
        x.strip()
        for x in ids.split(";")
        if x.strip()
    ]

    genome_matched = row.get(
        "Genome_Matched",
        False
    )

    if isinstance(
        genome_matched,
        str
    ):

        genome_matched = (
            genome_matched.lower()
            == "true"
        )

    mapping_lookup[species] = {
        "Genome_Matched": bool(
            genome_matched
        ),
        "Assembly_IDs": id_list
    }


# ============================================================
# 4. BUILD ASSEMBLY CANDIDATE LIST
# ============================================================

print(
    "\nBuilding assembly candidate list..."
)

candidate_ids_by_species = {}

search_records = []

for i, species in enumerate(
    species_list,
    start=1
):

    existing = mapping_lookup.get(
        species,
        {}
    )

    existing_ids = existing.get(
        "Assembly_IDs",
        []
    )


    # --------------------------------------------------------
    # USE EXISTING Y1000+ MATCH
    # --------------------------------------------------------

    if existing_ids:

        candidate_ids_by_species[
            species
        ] = existing_ids

        search_records.append(
            {
                "Species": species,
                "Search_Type": (
                    "Existing_PRJNA736342"
                ),
                "Query": "",
                "Record_Count": len(
                    existing_ids
                ),
                "Assembly_IDs": ";".join(
                    existing_ids
                )
            }
        )

        continue


    # --------------------------------------------------------
    # BROADER NCBI SEARCH
    # --------------------------------------------------------

    query = (
        f'"{species}"[Organism]'
    )

    print(
        f"[{i}/{len(species_list)}] "
        f"Searching: {species}"
    )

    try:

        result = ncbi_esearch(
            query
        )

        ids = result.get(
            "idlist",
            []
        )

        count = int(
            result.get(
                "count",
                0
            )
        )

        candidate_ids_by_species[
            species
        ] = ids

        search_records.append(
            {
                "Species": species,
                "Search_Type": (
                    "Broad_exact_species"
                ),
                "Query": query,
                "Record_Count": count,
                "Assembly_IDs": ";".join(
                    ids
                )
            }
        )

    except Exception as e:

        print(
            "  ERROR:",
            str(e)
        )

        candidate_ids_by_species[
            species
        ] = []

        search_records.append(
            {
                "Species": species,
                "Search_Type": (
                    "Broad_exact_species_ERROR"
                ),
                "Query": query,
                "Record_Count": -1,
                "Assembly_IDs": ""
            }
        )

    time.sleep(
        0.35
    )


# ============================================================
# 5. SAVE SEARCH RESULTS
# ============================================================

search_df = pd.DataFrame(
    search_records
)

search_file = os.path.join(
    OUTPUT_DIR,
    "assembly_search_results.csv"
)

search_df.to_csv(
    search_file,
    index=False
)


# ============================================================
# 6. COLLECT UNIQUE NCBI ASSEMBLY IDS
# ============================================================

all_ids = sorted(
    set(
        assembly_id
        for ids in candidate_ids_by_species.values()
        for assembly_id in ids
    )
)

print(
    "\nUnique NCBI assembly records:",
    len(all_ids)
)


# ============================================================
# 7. RETRIEVE ASSEMBLY METADATA
# ============================================================

print(
    "\nRetrieving NCBI assembly metadata..."
)

metadata_records = {}

batch_size = 100

for start in range(
    0,
    len(all_ids),
    batch_size
):

    batch = all_ids[
        start:start + batch_size
    ]

    end = min(
        start + batch_size,
        len(all_ids)
    )

    print(
        f"Metadata batch "
        f"{start + 1}-{end}"
    )

    try:

        summary = ncbi_esummary(
            batch
        )

        for key, record in summary.items():

            if key == "uids":
                continue

            metadata_records[
                str(key)
            ] = record

    except Exception as e:

        print(
            "Metadata ERROR:",
            str(e)
        )

    time.sleep(
        0.4
    )


# ============================================================
# 8. CONSTRUCT ASSEMBLY CANDIDATE TABLE
# ============================================================

print(
    "\nConstructing candidate table..."
)

candidate_rows = []

for species in species_list:

    ids = candidate_ids_by_species.get(
        species,
        []
    )


    # --------------------------------------------------------
    # NO ASSEMBLY FOUND
    # --------------------------------------------------------

    if not ids:

        candidate_rows.append(
            {
                "Phenotype_Species": species,
                "NCBI_Assembly_ID": "",
                "Assembly_Accession": "",
                "Organism_Name": "",
                "Assembly_Name": "",
                "Assembly_Level": "",
                "Submitter": "",
                "BioProject": "",
                "BioSample": "",
                "Infraspecies": "",
                "Release_Date": "",
                "Gene_Count": "",
                "Genome_Size": "",
                "Contig_N50": "",
                "Candidate_Source": (
                    "No_assembly_found"
                )
            }
        )

        continue


    # --------------------------------------------------------
    # PROCESS EVERY CANDIDATE ASSEMBLY
    # --------------------------------------------------------

    for assembly_id in ids:

        record = metadata_records.get(
            str(assembly_id),
            {}
        )

        accession = extract_accession(
            record
        )

        organism = flatten_value(
            record.get(
                "organism",
                ""
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


        stats = record.get(
            "assemblystats",
            {}
        )

        if not isinstance(
            stats,
            dict
        ):
            stats = {}


        annotation = record.get(
            "annotationinfo",
            {}
        )

        if not isinstance(
            annotation,
            dict
        ):
            annotation = {}


        # ----------------------------------------------------
        # ASSEMBLY NAME
        # ----------------------------------------------------

        assembly_name = (
            assembly_info.get(
                "assemblyname",
                ""
            )
        )

        if not assembly_name:

            assembly_name = (
                assembly_info.get(
                    "assemblyName",
                    ""
                )
            )


        # ----------------------------------------------------
        # ASSEMBLY LEVEL
        # ----------------------------------------------------

        assembly_level = (
            assembly_info.get(
                "assemblylevel",
                ""
            )
        )

        if not assembly_level:

            assembly_level = (
                assembly_info.get(
                    "assemblyLevel",
                    ""
                )
            )


        # ----------------------------------------------------
        # SUBMITTER
        # ----------------------------------------------------

        submitter = flatten_value(
            record.get(
                "submitter",
                ""
            )
        )

        if not submitter:

            submitter = flatten_value(
                assembly_info.get(
                    "submitter",
                    ""
                )
            )


        # ----------------------------------------------------
        # BIOPROJECT
        # ----------------------------------------------------

        bioproject = flatten_value(
            record.get(
                "bioproject",
                ""
            )
        )

        if not bioproject:

            bioproject = flatten_value(
                record.get(
                    "bioprojectaccn",
                    ""
                )
            )


        # ----------------------------------------------------
        # BIOSAMPLE
        # ----------------------------------------------------

        biosample = flatten_value(
            record.get(
                "biosample",
                ""
            )
        )

        if not biosample:

            biosample = flatten_value(
                record.get(
                    "biosampleaccn",
                    ""
                )
            )


        # ----------------------------------------------------
        # INFRASPECIES
        # ----------------------------------------------------

        infraspecies = flatten_value(
            record.get(
                "infraspecies",
                ""
            )
        )

        if not infraspecies:

            infraspecies = flatten_value(
                assembly_info.get(
                    "infraspecificname",
                    ""
                )
            )


        # ----------------------------------------------------
        # RELEASE DATE
        # ----------------------------------------------------

        release_date = (
            assembly_info.get(
                "release_date",
                ""
            )
        )

        if not release_date:

            release_date = (
                assembly_info.get(
                    "releaseDate",
                    ""
                )
            )


        # ----------------------------------------------------
        # GENE COUNT
        # ----------------------------------------------------

        gene_count = (
            annotation.get(
                "total_gene_count",
                ""
            )
        )

        if not gene_count:

            gene_count = (
                annotation.get(
                    "totalGeneCount",
                    ""
                )
            )


        # ----------------------------------------------------
        # GENOME SIZE
        # ----------------------------------------------------

        genome_size = (
            stats.get(
                "total_sequence_length",
                ""
            )
        )

        if not genome_size:

            genome_size = (
                stats.get(
                    "totalSequenceLength",
                    ""
                )
            )


        # ----------------------------------------------------
        # CONTIG N50
        # ----------------------------------------------------

        contig_n50 = (
            stats.get(
                "contig_n50",
                ""
            )
        )

        if not contig_n50:

            contig_n50 = (
                stats.get(
                    "contigN50",
                    ""
                )
            )


        # ----------------------------------------------------
        # DETERMINE SOURCE
        # ----------------------------------------------------

        if existing_id_in_mapping(
            assembly_id,
            mapping_lookup.get(
                species,
                {}
            ).get(
                "Assembly_IDs",
                []
            )
        ):

            source = (
                "Y1000plus_PRJNA736342"
            )

        else:

            source = (
                "Broader_NCBI_search"
            )


        # ----------------------------------------------------
        # ADD ROW
        # ----------------------------------------------------

        candidate_rows.append(
            {
                "Phenotype_Species": species,
                "NCBI_Assembly_ID": assembly_id,
                "Assembly_Accession": accession,
                "Organism_Name": organism,
                "Assembly_Name": assembly_name,
                "Assembly_Level": assembly_level,
                "Submitter": submitter,
                "BioProject": bioproject,
                "BioSample": biosample,
                "Infraspecies": infraspecies,
                "Release_Date": release_date,
                "Gene_Count": gene_count,
                "Genome_Size": genome_size,
                "Contig_N50": contig_n50,
                "Candidate_Source": source
            }
        )


# ============================================================
# 9. CREATE DATAFRAME
# ============================================================

candidate_df = pd.DataFrame(
    candidate_rows
)


# ============================================================
# 10. SAVE COMPLETE CANDIDATE TABLE
# ============================================================

candidate_file = os.path.join(
    OUTPUT_DIR,
    "all_species_assembly_candidates.csv"
)

candidate_df.to_csv(
    candidate_file,
    index=False
)


# ============================================================
# 11. CREATE SPECIES-LEVEL SUMMARY
# ============================================================

def count_valid_accessions(series):

    values = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return int(
        values.ne("").sum()
    )


summary = (
    candidate_df
    .groupby(
        "Phenotype_Species",
        as_index=False
    )
    .agg(
        Candidate_Assembly_Count=(
            "Assembly_Accession",
            count_valid_accessions
        )
    )
)

summary["Has_Assembly"] = (
    summary[
        "Candidate_Assembly_Count"
    ] > 0
)


# ============================================================
# 12. SAVE SPECIES SUMMARY
# ============================================================

summary_file = os.path.join(
    OUTPUT_DIR,
    "species_assembly_candidate_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False
)


# ============================================================
# 13. SPECIES WITHOUT ASSEMBLIES
# ============================================================

unresolved = summary[
    ~summary["Has_Assembly"]
].copy()

unresolved_file = os.path.join(
    OUTPUT_DIR,
    "species_still_unresolved.csv"
)

unresolved.to_csv(
    unresolved_file,
    index=False
)


# ============================================================
# 14. SPECIES WITH MULTIPLE ASSEMBLY CANDIDATES
# ============================================================

multiple = summary[
    summary[
        "Candidate_Assembly_Count"
    ] > 1
].copy()

multiple_file = os.path.join(
    OUTPUT_DIR,
    "species_with_multiple_assembly_candidates.csv"
)

multiple.to_csv(
    multiple_file,
    index=False
)


# ============================================================
# 15. COUNT SOURCES
# ============================================================

y1000_candidates = candidate_df[
    candidate_df[
        "Candidate_Source"
    ] == "Y1000plus_PRJNA736342"
]

broader_candidates = candidate_df[
    candidate_df[
        "Candidate_Source"
    ] == "Broader_NCBI_search"
]

no_assembly = candidate_df[
    candidate_df[
        "Candidate_Source"
    ] == "No_assembly_found"
]


# ============================================================
# 16. SUMMARY STATISTICS
# ============================================================

total_species = len(
    summary
)

species_with_assembly = int(
    summary[
        "Has_Assembly"
    ].sum()
)

species_without_assembly = (
    total_species
    - species_with_assembly
)

species_multiple = len(
    multiple
)

coverage = (
    100
    * species_with_assembly
    / total_species
)


# ============================================================
# 17. PRINT FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("STAGE 3A.4 SUMMARY")
print("=" * 75)

print(
    f"\nPhenotype species: "
    f"{total_species}"
)

print(
    f"Species with >=1 assembly: "
    f"{species_with_assembly}"
)

print(
    f"Species without assembly: "
    f"{species_without_assembly}"
)

print(
    f"Assembly candidate coverage: "
    f"{coverage:.2f}%"
)

print(
    f"Species with multiple candidates: "
    f"{species_multiple}"
)

print(
    f"\nY1000+ candidate records: "
    f"{len(y1000_candidates)}"
)

print(
    f"Broader NCBI candidate records: "
    f"{len(broader_candidates)}"
)

print(
    f"Unresolved candidate rows: "
    f"{len(no_assembly)}"
)


# ============================================================
# 18. PRINT UNRESOLVED SPECIES
# ============================================================

if len(unresolved) > 0:

    print(
        "\n" + "=" * 75
    )

    print(
        "SPECIES STILL WITHOUT ASSEMBLY CANDIDATES"
    )

    print(
        "=" * 75
    )

    for species in unresolved[
        "Phenotype_Species"
    ]:

        print(
            f"- {species}"
        )

else:

    print(
        "\nAll phenotype species have "
        "at least one assembly candidate."
    )


# ============================================================
# 19. PRINT MULTIPLE-CANDIDATE SPECIES
# ============================================================

if len(multiple) > 0:

    print(
        "\n" + "=" * 75
    )

    print(
        "SPECIES WITH MULTIPLE ASSEMBLY CANDIDATES"
    )

    print(
        "=" * 75
    )

    print(
        multiple.to_string(
            index=False
        )
    )


# ============================================================
# 20. WRITE REPORT
# ============================================================

report_file = os.path.join(
    OUTPUT_DIR,
    "stage3A4_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STAGE 3A.4 — NCBI ASSEMBLY CANDIDATE REPORT\n"
    )

    f.write(
        "=" * 75 + "\n\n"
    )

    f.write(
        f"Phenotype species: "
        f"{total_species}\n"
    )

    f.write(
        f"Species with >=1 assembly: "
        f"{species_with_assembly}\n"
    )

    f.write(
        f"Species without assembly: "
        f"{species_without_assembly}\n"
    )

    f.write(
        f"Assembly candidate coverage: "
        f"{coverage:.2f}%\n"
    )

    f.write(
        f"Species with multiple candidates: "
        f"{species_multiple}\n"
    )

    f.write(
        f"\nY1000+ candidate records: "
        f"{len(y1000_candidates)}\n"
    )

    f.write(
        f"Broader NCBI candidate records: "
        f"{len(broader_candidates)}\n"
    )

    f.write(
        "\nSpecies still unresolved:\n"
    )

    if len(unresolved) == 0:

        f.write(
            "None\n"
        )

    else:

        for species in unresolved[
            "Phenotype_Species"
        ]:

            f.write(
                f"- {species}\n"
            )

    f.write(
        "\nSpecies with multiple assembly candidates:\n"
    )

    if len(multiple) == 0:

        f.write(
            "None\n"
        )

    else:

        f.write(
            multiple.to_string(
                index=False
            )
        )

        f.write(
            "\n"
        )


# ============================================================
# 21. COMPLETION
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STAGE 3A.4 COMPLETED"
)

print(
    "=" * 75
)

print(
    "\nGenerated files:"
)

print(
    candidate_file
)

print(
    summary_file
)

print(
    unresolved_file
)

print(
    multiple_file
)

print(
    search_file
)

print(
    report_file
)

print(
    "\nIMPORTANT:"
)

print(
    "No genome sequences were downloaded."
)

print(
    "No representative assembly has been selected yet."
)

print(
    "The candidate table must be inspected before "
    "genome download."
)