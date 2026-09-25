# =============================================================================
# FINAL Y1000+ NETWORK ANALYSIS
# 420 TAXA / PHYLOGENY-AWARE ML
#
# FINAL BIOLOGICAL INTERPRETATION STEP
#
# This script:
#   - uses existing validated ML feature evidence
#   - uses existing BUSCO biological annotation
#   - uses existing normalized pathway mapping
#   - uses existing candidate chassis results
#   - builds BUSCO <-> PATHWAY network
#   - builds BUSCO <-> BIOLOGICAL THEME network
#   - calculates network connectivity
#   - identifies multifunctional / bridging BUSCOs
#   - creates final integrated evidence table
#   - creates publication-ready network figures
#
# NO:
#   - model retraining
#   - permutation analysis
#   - new feature selection
#   - modification of Pareto candidates
#   - modification of existing ML models
# =============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import re

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE = Path(
    r"C:\Y1000_chassis_project\results"
)

ML_BASE = (
    BASE
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
)

BIO_BASE = (
    ML_BASE
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "BUSCO20_annotation"
)

BIO_V3 = (
    BIO_BASE
    / "FINAL_CARBON_BREADTH_BIOLOGICAL_INTERPRETATION_420_V3"
)

CHASSIS_BASE = (
    BASE
    / "stage6_candidate_yeast_chassis_420"
)

CHASSIS_TABLES = CHASSIS_BASE / "tables"

OUT = (
    BASE
    / "stage7_final_network_analysis_420"
)

TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

for directory in [
    OUT,
    TABLES,
    FIGURES,
    REPORTS
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )

# =============================================================================
# INPUT FILES
# =============================================================================

BUSCO_SUMMARY = (
    BIO_V3
    / "tables"
    / "Carbon_Breadth_final_BUSCO_biological_summary_420_v3.csv"
)

PATHWAY_SUMMARY = (
    BIO_V3
    / "tables"
    / "Carbon_Breadth_final_normalized_pathway_summary_420_v3.csv"
)

THEME_SUMMARY = (
    BIO_V3
    / "tables"
    / "Carbon_Breadth_final_biological_theme_summary_420_v3.csv"
)

BUSCO_MATRIX = (
    BIO_V3
    / "tables"
    / "Carbon_Breadth_final_BUSCO_pathway_matrix_420_v3.csv"
)

ML_FEATURES = (
    CHASSIS_TABLES
    / "validated_stable_ML_features_420.csv"
)

BUSCO_EVIDENCE = (
    CHASSIS_TABLES
    / "candidate_chassis_BUSCO_evidence_420.csv"
)

CHASSIS = (
    CHASSIS_TABLES
    / "candidate_yeast_chassis_species_420.csv"
)

PARETO = (
    CHASSIS_TABLES
    / "pareto_optimal_yeast_chassis_candidates_420.csv"
)

# =============================================================================
# HELPERS
# =============================================================================

def banner(text):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"\n{label} not found:\n{path}"
        )


def find_column(df, candidates):
    """
    Find exact or case-insensitive column match.
    """
    for c in candidates:
        if c in df.columns:
            return c

    lower = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for c in candidates:
        if str(c).strip().lower() in lower:
            return lower[
                str(c).strip().lower()
            ]

    return None


def clean_id(value):
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def split_pathways(value):
    """
    Convert a pathway field containing separators such as
    ';', '|', ',' or newline into a list.
    """

    if pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    parts = re.split(
        r";|\||\n",
        text
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


# =============================================================================
# START
# =============================================================================

banner(
    "FINAL Y1000+ NETWORK ANALYSIS\n"
    "420 TAXA / PHYLOGENY-AWARE ML"
)

print(
    """
This is the final biological network-analysis stage.

Existing results will be reused.

NO MODEL RETRAINING
NO PERMUTATION RERUN
NO NEW FEATURE SELECTION
NO CHASSIS MODIFICATION
"""
)

# =============================================================================
# CHECK INPUTS
# =============================================================================

banner("CHECKING INPUT FILES")

input_files = [
    (BUSCO_SUMMARY, "BUSCO biological summary"),
    (PATHWAY_SUMMARY, "Normalized pathway summary"),
    (THEME_SUMMARY, "Biological theme summary"),
    (BUSCO_MATRIX, "BUSCO-pathway matrix"),
    (ML_FEATURES, "Validated stable ML features"),
    (BUSCO_EVIDENCE, "Candidate chassis BUSCO evidence"),
    (CHASSIS, "Candidate yeast chassis"),
    (PARETO, "Pareto-optimal chassis")
]

for path, label in input_files:
    require_file(
        path,
        label
    )
    print(
        f"{label}: FOUND"
    )

# =============================================================================
# LOAD DATA
# =============================================================================

banner("LOADING EXISTING BIOLOGICAL EVIDENCE")

busco_df = pd.read_csv(
    BUSCO_SUMMARY
)

pathway_df = pd.read_csv(
    PATHWAY_SUMMARY
)

theme_df = pd.read_csv(
    THEME_SUMMARY
)

matrix_df = pd.read_csv(
    BUSCO_MATRIX
)

ml_df = pd.read_csv(
    ML_FEATURES
)

evidence_df = pd.read_csv(
    BUSCO_EVIDENCE
)

chassis_df = pd.read_csv(
    CHASSIS
)

pareto_df = pd.read_csv(
    PARETO
)

print(
    "BUSCO summary rows:",
    len(busco_df)
)

print(
    "Normalized pathway rows:",
    len(pathway_df)
)

print(
    "Biological theme rows:",
    len(theme_df)
)

print(
    "BUSCO-pathway matrix rows:",
    len(matrix_df)
)

print(
    "Stable ML features:",
    len(ml_df)
)

print(
    "BUSCO evidence rows:",
    len(evidence_df)
)

print(
    "Candidate chassis:",
    len(chassis_df)
)

print(
    "Pareto candidates:",
    len(pareto_df)
)

# =============================================================================
# IDENTIFY BUSCO COLUMN
# =============================================================================

busco_col = find_column(
    busco_df,
    [
        "BUSCO_ID",
        "BUSCO",
        "Busco_ID",
        "Feature"
    ]
)

if busco_col is None:
    raise ValueError(
        "BUSCO column could not be identified."
    )

print(
    "\nBUSCO identifier:",
    busco_col
)

# =============================================================================
# NORMALIZED PATHWAY COLUMN
# =============================================================================

pathway_name_col = find_column(
    busco_df,
    [
        "Normalized_Pathways",
        "Normalized_Pathway",
        "Pathways",
        "Pathway"
    ]
)

if pathway_name_col is None:
    pathway_name_col = find_column(
        pathway_df,
        [
            "Normalized_Pathway",
            "Pathway",
            "Normalized_Pathways"
        ]
    )

print(
    "Pathway column:",
    pathway_name_col
)

# =============================================================================
# BUILD BUSCO -> PATHWAY EDGES
# =============================================================================

banner(
    "BUILDING BUSCO × PATHWAY NETWORK"
)

edges = []

if (
    pathway_name_col is not None
    and pathway_name_col in busco_df.columns
):

    for _, row in busco_df.iterrows():

        busco = clean_id(
            row[busco_col]
        )

        pathways = split_pathways(
            row[pathway_name_col]
        )

        for pathway in pathways:

            edges.append(
                {
                    "BUSCO_ID": busco,
                    "Normalized_Pathway": pathway
                }
            )

else:

    # Try matrix structure
    matrix_busco_col = find_column(
        matrix_df,
        [
            "BUSCO_ID",
            "BUSCO",
            "Feature"
        ]
    )

    if matrix_busco_col is None:
        raise ValueError(
            "Unable to identify BUSCO column "
            "in pathway matrix."
        )

    for _, row in matrix_df.iterrows():

        busco = clean_id(
            row[matrix_busco_col]
        )

        for col in matrix_df.columns:

            if col == matrix_busco_col:
                continue

            try:
                value = float(
                    row[col]
                )
            except Exception:
                continue

            if value > 0:

                edges.append(
                    {
                        "BUSCO_ID": busco,
                        "Normalized_Pathway": str(
                            col
                        )
                    }
                )

edge_df = pd.DataFrame(
    edges
).drop_duplicates()

print(
    "BUSCO-pathway edges:",
    len(edge_df)
)

print(
    "Unique BUSCOs:",
    edge_df["BUSCO_ID"].nunique()
)

print(
    "Unique pathways:",
    edge_df[
        "Normalized_Pathway"
    ].nunique()
)

edge_df.to_csv(
    TABLES
    / "BUSCO_pathway_network_edges_420.csv",
    index=False
)

# =============================================================================
# BUILD BIPARTITE NETWORK
# =============================================================================

banner(
    "CALCULATING BUSCO-PATHWAY NETWORK TOPOLOGY"
)

G = nx.Graph()

for _, row in edge_df.iterrows():

    busco = row["BUSCO_ID"]
    pathway = row[
        "Normalized_Pathway"
    ]

    busco_node = (
        f"BUSCO::{busco}"
    )

    pathway_node = (
        f"PATHWAY::{pathway}"
    )

    G.add_node(
        busco_node,
        node_type="BUSCO",
        label=busco
    )

    G.add_node(
        pathway_node,
        node_type="PATHWAY",
        label=pathway
    )

    G.add_edge(
        busco_node,
        pathway_node
    )

# =============================================================================
# BUSCO NETWORK CONNECTIVITY
# =============================================================================

busco_nodes = [
    n for n, d in G.nodes(
        data=True
    )
    if d["node_type"] == "BUSCO"
]

pathway_nodes = [
    n for n, d in G.nodes(
        data=True
    )
    if d["node_type"] == "PATHWAY"
]

busco_network = []

for node in busco_nodes:

    data = G.nodes[node]

    degree = G.degree(
        node
    )

    pathways = [
        G.nodes[n]["label"]
        for n in G.neighbors(node)
    ]

    busco_network.append(
        {
            "BUSCO_ID": data["label"],
            "Pathway_Degree": degree,
            "Pathway_Count": len(
                pathways
            ),
            "Connected_Pathways":
                "; ".join(
                    sorted(pathways)
                )
        }
    )

busco_network_df = pd.DataFrame(
    busco_network
)

# =============================================================================
# MAP ML EVIDENCE
# =============================================================================

ml_busco_col = find_column(
    ml_df,
    [
        "BUSCO_ID",
        "BUSCO",
        "Feature"
    ]
)

if ml_busco_col is not None:

    ml_tmp = ml_df.copy()

    ml_tmp[
        "BUSCO_ID"
    ] = ml_tmp[
        ml_busco_col
    ].astype(str)

    # Keep one row per BUSCO
    ml_tmp = (
        ml_tmp
        .drop_duplicates(
            "BUSCO_ID"
        )
    )

    busco_network_df = (
        busco_network_df
        .merge(
            ml_tmp,
            on="BUSCO_ID",
            how="left",
            suffixes=(
                "",
                "_ML"
            )
        )
    )

# =============================================================================
# MAP FUNCTIONAL EVIDENCE
# =============================================================================

busco_evidence_col = find_column(
    evidence_df,
    [
        "BUSCO_ID",
        "BUSCO",
        "Feature"
    ]
)

if busco_evidence_col is not None:

    evidence_tmp = (
        evidence_df
        .copy()
    )

    evidence_tmp[
        "BUSCO_ID"
    ] = evidence_tmp[
        busco_evidence_col
    ].astype(str)

    evidence_tmp = (
        evidence_tmp
        .drop_duplicates(
            "BUSCO_ID"
        )
    )

    busco_network_df = (
        busco_network_df
        .merge(
            evidence_tmp,
            on="BUSCO_ID",
            how="left",
            suffixes=(
                "",
                "_Evidence"
            )
        )
    )

# =============================================================================
# SAVE BUSCO NETWORK CENTRALITY
# =============================================================================

busco_network_df = (
    busco_network_df
    .sort_values(
        [
            "Pathway_Degree",
            "Pathway_Count"
        ],
        ascending=False
    )
)

busco_network_df.to_csv(
    TABLES
    / "BUSCO_network_connectivity_420.csv",
    index=False
)

print(
    "\nTop multifunctional BUSCOs:"
)

print(
    busco_network_df[
        [
            "BUSCO_ID",
            "Pathway_Degree",
            "Pathway_Count",
            "Connected_Pathways"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)

# =============================================================================
# PATHWAY CONNECTIVITY
# =============================================================================

pathway_network = []

for node in pathway_nodes:

    data = G.nodes[node]

    buscos = [
        G.nodes[n]["label"]
        for n in G.neighbors(node)
    ]

    pathway_network.append(
        {
            "Normalized_Pathway":
                data["label"],
            "BUSCO_Degree":
                G.degree(node),
            "BUSCO_Count":
                len(buscos),
            "Connected_BUSCOs":
                "; ".join(
                    sorted(buscos)
                )
        }
    )

pathway_network_df = pd.DataFrame(
    pathway_network
)

pathway_network_df = (
    pathway_network_df
    .sort_values(
        [
            "BUSCO_Degree",
            "BUSCO_Count"
        ],
        ascending=False
    )
)

pathway_network_df.to_csv(
    TABLES
    / "pathway_network_connectivity_420.csv",
    index=False
)

# =============================================================================
# BIOLOGICAL THEME NETWORK
# =============================================================================

banner(
    "BUILDING BUSCO × BIOLOGICAL THEME NETWORK"
)

theme_pathway_col = find_column(
    pathway_df,
    [
        "Normalized_Pathway",
        "Pathway",
        "Normalized_Pathways"
    ]
)

theme_col = find_column(
    pathway_df,
    [
        "Biological_Theme",
        "Theme",
        "Broad_Biological_Theme"
    ]
)

# If pathway summary does not contain theme,
# attempt to use the theme summary.

if (
    theme_pathway_col is not None
    and theme_col is not None
):

    pathway_theme = (
        pathway_df[
            [
                theme_pathway_col,
                theme_col
            ]
        ]
        .dropna()
        .drop_duplicates()
    )

else:

    theme_pathway_col = find_column(
        theme_df,
        [
            "Normalized_Pathway",
            "Pathway"
        ]
    )

    theme_col = find_column(
        theme_df,
        [
            "Biological_Theme",
            "Theme",
            "Broad_Biological_Theme"
        ]
    )

    if (
        theme_pathway_col is not None
        and theme_col is not None
    ):

        pathway_theme = (
            theme_df[
                [
                    theme_pathway_col,
                    theme_col
                ]
            ]
            .dropna()
            .drop_duplicates()
        )

    else:

        pathway_theme = pd.DataFrame(
            columns=[
                "Normalized_Pathway",
                "Biological_Theme"
            ]
        )

pathway_theme.columns = [
    "Normalized_Pathway",
    "Biological_Theme"
]

theme_edges = (
    edge_df
    .merge(
        pathway_theme,
        on="Normalized_Pathway",
        how="left"
    )
    .dropna(
        subset=[
            "Biological_Theme"
        ]
    )
    .drop_duplicates()
)

print(
    "BUSCO-theme edges:",
    len(theme_edges)
)

theme_edges.to_csv(
    TABLES
    / "BUSCO_biological_theme_network_edges_420.csv",
    index=False
)

# =============================================================================
# THEME CONNECTIVITY
# =============================================================================

theme_network = (
    theme_edges
    .groupby(
        "Biological_Theme"
    )
    .agg(
        BUSCO_Count=(
            "BUSCO_ID",
            "nunique"
        ),
        Pathway_Count=(
            "Normalized_Pathway",
            "nunique"
        ),
        BUSCOs=(
            "BUSCO_ID",
            lambda x:
            "; ".join(
                sorted(
                    set(x)
                )
            )
        ),
        Pathways=(
            "Normalized_Pathway",
            lambda x:
            "; ".join(
                sorted(
                    set(x)
                )
            )
        )
    )
    .reset_index()
)

theme_network = (
    theme_network
    .sort_values(
        [
            "BUSCO_Count",
            "Pathway_Count"
        ],
        ascending=False
    )
)

theme_network.to_csv(
    TABLES
    / "biological_theme_network_connectivity_420.csv",
    index=False
)

print(
    "\nBiological theme connectivity:"
)

print(
    theme_network.to_string(
        index=False
    )
)

# =============================================================================
# FINAL INTEGRATED BUSCO EVIDENCE
# =============================================================================

banner(
    "BUILDING FINAL INTEGRATED BUSCO NETWORK EVIDENCE"
)

final_busco = busco_network_df.copy()

# Number of pathways
final_busco[
    "Network_Pathway_Degree"
] = final_busco[
    "Pathway_Degree"
]

# Determine biological themes
busco_theme = (
    theme_edges
    .groupby(
        "BUSCO_ID"
    )[
        "Biological_Theme"
    ]
    .agg(
        lambda x:
        "; ".join(
            sorted(
                set(x)
            )
        )
    )
    .reset_index()
)

final_busco = (
    final_busco
    .merge(
        busco_theme,
        on="BUSCO_ID",
        how="left"
    )
)

# =============================================================================
# EVIDENCE FLAGS
# =============================================================================

final_busco[
    "Pathway_Supported"
] = (
    final_busco[
        "Pathway_Count"
    ] > 0
)

final_busco[
    "Network_Connected"
] = (
    final_busco[
        "Pathway_Degree"
    ] > 1
)

# ML evidence detection
if ml_busco_col is not None:

    final_busco[
        "ML_Supported"
    ] = (
        final_busco[
            ml_busco_col
        ]
        .notna()
    )

else:

    final_busco[
        "ML_Supported"
    ] = False

# =============================================================================
# NETWORK EVIDENCE CLASS
# =============================================================================

def evidence_class(row):

    ml = bool(
        row["ML_Supported"]
    )

    pathway = bool(
        row["Pathway_Supported"]
    )

    network = bool(
        row["Network_Connected"]
    )

    if ml and pathway and network:
        return "ML + Functional + Network"

    if ml and pathway:
        return "ML + Functional"

    if pathway and network:
        return "Functional + Network"

    if ml:
        return "ML-supported"

    if pathway:
        return "Functionally supported"

    return "Network-only"


final_busco[
    "Integrated_Evidence_Class"
] = final_busco.apply(
    evidence_class,
    axis=1
)

final_busco = (
    final_busco
    .sort_values(
        [
            "ML_Supported",
            "Pathway_Degree",
            "Pathway_Count"
        ],
        ascending=False
    )
)

final_busco.to_csv(
    TABLES
    / "final_integrated_BUSCO_network_evidence_420.csv",
    index=False
)

# =============================================================================
# PARETO / CHASSIS SUMMARY
# =============================================================================

banner(
    "LINKING NETWORK EVIDENCE TO CANDIDATE CHASSIS"
)

species_col = find_column(
    chassis_df,
    [
        "Species",
        "species",
        "Taxon"
    ]
)

if species_col is not None:

    chassis_summary = chassis_df.copy()

else:

    chassis_summary = pd.DataFrame()

pareto_species_col = find_column(
    pareto_df,
    [
        "Species",
        "species",
        "Taxon"
    ]
)

if pareto_species_col is not None:

    pareto_species = set(
        pareto_df[
            pareto_species_col
        ]
        .astype(str)
        .str.strip()
    )

else:

    pareto_species = set()

if species_col is not None:

    chassis_summary[
        "Pareto_Supported"
    ] = (
        chassis_summary[
            species_col
        ]
        .astype(str)
        .str.strip()
        .isin(
            pareto_species
        )
    )

chassis_summary.to_csv(
    TABLES
    / "final_candidate_chassis_network_context_420.csv",
    index=False
)

# =============================================================================
# NETWORK FIGURE
# =============================================================================

banner(
    "GENERATING NETWORK FIGURES"
)

# Limit plotting to the existing biological network.
# No new biological filtering is performed.

plot_graph = G.copy()

# Layout
if len(plot_graph.nodes) > 0:

    positions = nx.spring_layout(
        plot_graph,
        seed=420,
        k=1.2
    )

    plt.figure(
        figsize=(18, 14)
    )

    busco_plot_nodes = [
        n for n in plot_graph.nodes
        if plot_graph.nodes[n][
            "node_type"
        ] == "BUSCO"
    ]

    pathway_plot_nodes = [
        n for n in plot_graph.nodes
        if plot_graph.nodes[n][
            "node_type"
        ] == "PATHWAY"
    ]

    nx.draw_networkx_edges(
        plot_graph,
        positions,
        alpha=0.35,
        width=1
    )

    nx.draw_networkx_nodes(
        plot_graph,
        positions,
        nodelist=busco_plot_nodes,
        node_size=650,
        node_shape="o"
    )

    nx.draw_networkx_nodes(
        plot_graph,
        positions,
        nodelist=pathway_plot_nodes,
        node_size=1000,
        node_shape="s"
    )

    labels = {
        n: plot_graph.nodes[n]["label"]
        for n in plot_graph.nodes
    }

    nx.draw_networkx_labels(
        plot_graph,
        positions,
        labels=labels,
        font_size=7
    )

    plt.title(
        "Y1000+ BUSCO–Pathway Functional Network"
    )

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "Y1000_BUSCO_pathway_network_420.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

# =============================================================================
# TOP BUSCO CONNECTIVITY FIGURE
# =============================================================================

top_buscos = (
    busco_network_df
    .head(20)
)

if len(top_buscos) > 0:

    plt.figure(
        figsize=(12, 8)
    )

    labels = (
        top_buscos[
            "BUSCO_ID"
        ]
        .astype(str)
    )

    values = (
        top_buscos[
            "Pathway_Degree"
        ]
        .astype(float)
    )

    plt.barh(
        labels[::-1],
        values[::-1]
    )

    plt.xlabel(
        "Number of Connected Normalized Pathways"
    )

    plt.ylabel(
        "BUSCO"
    )

    plt.title(
        "Top BUSCOs by Functional Network Connectivity"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "top_BUSCO_network_connectivity_420.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

# =============================================================================
# THEME FIGURE
# =============================================================================

if len(theme_network) > 0:

    plt.figure(
        figsize=(12, 8)
    )

    labels = (
        theme_network[
            "Biological_Theme"
        ]
        .astype(str)
    )

    values = (
        theme_network[
            "BUSCO_Count"
        ]
        .astype(float)
    )

    plt.barh(
        labels[::-1],
        values[::-1]
    )

    plt.xlabel(
        "Number of Associated BUSCOs"
    )

    plt.ylabel(
        "Biological Theme"
    )

    plt.title(
        "Biological Theme Network Representation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES
        / "biological_theme_network_420.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

# =============================================================================
# REPORT
# =============================================================================

banner(
    "GENERATING FINAL NETWORK REPORT"
)

report = []

report.append(
    "FINAL Y1000+ NETWORK ANALYSIS"
)

report.append(
    "=" * 80
)

report.append(
    "Dataset: Y1000+ / 420 taxa"
)

report.append(
    "Purpose: final biological network interpretation"
)

report.append("")

report.append(
    f"BUSCOs represented in network: "
    f"{edge_df['BUSCO_ID'].nunique()}"
)

report.append(
    f"Normalized pathways represented: "
    f"{edge_df['Normalized_Pathway'].nunique()}"
)

report.append(
    f"BUSCO-pathway edges: "
    f"{len(edge_df)}"
)

report.append(
    f"Biological themes represented: "
    f"{theme_edges['Biological_Theme'].nunique()}"
)

report.append("")

report.append(
    "TOP FUNCTIONALLY CONNECTED BUSCOs"
)

report.append(
    "-" * 80
)

for _, row in (
    busco_network_df
    .head(20)
    .iterrows()
):

    report.append(
        f"{row['BUSCO_ID']} | "
        f"Pathways={row['Pathway_Count']} | "
        f"{row['Connected_Pathways']}"
    )

report.append("")

report.append(
    "BIOLOGICAL THEMES"
)

report.append(
    "-" * 80
)

for _, row in (
    theme_network
    .iterrows()
):

    report.append(
        f"{row['Biological_Theme']} | "
        f"BUSCOs={row['BUSCO_Count']} | "
        f"Pathways={row['Pathway_Count']}"
    )

report.append("")

report.append(
    "INTERPRETATION"
)

report.append(
    "-" * 80
)

report.append(
    "The network analysis integrates previously "
    "identified genomic features with normalized "
    "functional pathways and broad biological themes."
)

report.append(
    "BUSCOs connected to multiple pathways represent "
    "functionally multifunctional nodes within the "
    "existing annotation framework."
)

report.append(
    "The network analysis does not constitute an "
    "additional predictive model and does not alter "
    "the Pareto-derived candidate chassis set."
)

report.append(
    "Candidate chassis should therefore be interpreted "
    "using the existing phenotype/Pareto evidence together "
    "with genomic and functional evidence."
)

report.append("")

report.append(
    "FINAL PIPELINE STATUS"
)

report.append(
    "-" * 80
)

report.append(
    "Phenotype characterization: COMPLETE"
)

report.append(
    "Pareto analysis: COMPLETE"
)

report.append(
    "420-taxon genomic ML matrix: COMPLETE"
)

report.append(
    "Phylogeny-aware CV: COMPLETE"
)

report.append(
    "Permutation/null validation: COMPLETE"
)

report.append(
    "BUSCO feature interpretation: COMPLETE"
)

report.append(
    "KEGG/GO/pathway interpretation: COMPLETE"
)

report.append(
    "Candidate yeast chassis identification: COMPLETE"
)

report.append(
    "Network analysis: COMPLETE"
)

report.append("")

report.append(
    "Recommended next stage: manuscript preparation "
    "and final figure/table assembly."
)

report_file = (
    REPORTS
    / "final_y1000_network_analysis_420_report.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )

# =============================================================================
# FINAL OUTPUT
# =============================================================================

banner(
    "FINAL NETWORK ANALYSIS COMPLETE"
)

print(
    "\nTABLES:"
)

for file in sorted(
    TABLES.glob("*.csv")
):
    print(file)

print(
    "\nFIGURES:"
)

for file in sorted(
    FIGURES.glob("*.png")
):
    print(file)

print(
    "\nREPORT:"
)

print(
    report_file
)

print(
    """
================================================================================
PROJECT ANALYSIS STATUS
================================================================================

The network analysis has been completed using existing evidence.

No ML models were retrained.
No permutation tests were rerun.
No feature set was changed.
No Pareto candidates were changed.

This is the final computational biological interpretation stage.
"""
)