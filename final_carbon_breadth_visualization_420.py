from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CARBON BREADTH — FINAL VISUALIZATION & INTERPRETATION
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

BASE = (
    PROJECT
    / "results"
    / "stage5_phylogeny_ml_dataset"
    / "phylogeny_aware_ml"
    / "phylogeny_cv"
    / "model_training"
    / "feature_interpretation_420"
    / "functional_annotation_420"
    / "BUSCO20_annotation"
)

FINAL_DIR = BASE / "FINAL_CARBON_BREADTH_ANNOTATION_420"

FINAL_TABLE = (
    FINAL_DIR
    / "Carbon_Breadth_top20_FINAL_integrated_annotation_420.csv"
)

PATHWAY_TABLE = (
    FINAL_DIR
    / "Carbon_Breadth_top20_FINAL_pathway_annotation_420.csv"
)

PATHWAY_SUMMARY = (
    FINAL_DIR
    / "Carbon_Breadth_FINAL_pathway_summary_420.csv"
)

OUT = BASE / "FINAL_CARBON_BREADTH_INTERPRETATION_420"

TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

for directory in [TABLES, FIGURES, REPORTS]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def clean(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in ["nan", "none", "null"]:
        return ""
    return x


def split_items(value):
    value = clean(value)

    if not value:
        return []

    return [
        x.strip()
        for x in value.split(";")
        if x.strip()
    ]


def count_items(value):
    return len(split_items(value))


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 80)
print("CARBON BREADTH — FINAL VISUALIZATION & INTERPRETATION")
print("=" * 80)

if not FINAL_TABLE.exists():
    raise FileNotFoundError(
        f"Final integrated annotation not found:\n{FINAL_TABLE}"
    )

if not PATHWAY_TABLE.exists():
    raise FileNotFoundError(
        f"Pathway annotation not found:\n{PATHWAY_TABLE}"
    )

if not PATHWAY_SUMMARY.exists():
    raise FileNotFoundError(
        f"Pathway summary not found:\n{PATHWAY_SUMMARY}"
    )


final = pd.read_csv(FINAL_TABLE)
pathway = pd.read_csv(PATHWAY_TABLE)
pathway_summary = pd.read_csv(PATHWAY_SUMMARY)

print("\nFinal annotation shape:", final.shape)
print("Pathway annotation shape:", pathway.shape)
print("Pathway summary shape:", pathway_summary.shape)


# ============================================================
# 1. ANNOTATION COVERAGE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ANNOTATION COVERAGE")
print("=" * 80)

total = len(final)

coverage = []

if "UniProt_Count_Final" in final.columns:
    coverage.append({
        "Annotation": "UniProt",
        "BUSCO_Count": int(
            (final["UniProt_Count_Final"] > 0).sum()
        )
    })

if "GO_Count_Final" in final.columns:
    coverage.append({
        "Annotation": "GO",
        "BUSCO_Count": int(
            (final["GO_Count_Final"] > 0).sum()
        )
    })

if "KEGG_Annotation_Count_Final" in final.columns:
    coverage.append({
        "Annotation": "KEGG",
        "BUSCO_Count": int(
            (final["KEGG_Annotation_Count_Final"] > 0).sum()
        )
    })

if "EC_Count_Final" in final.columns:
    coverage.append({
        "Annotation": "EC",
        "BUSCO_Count": int(
            (final["EC_Count_Final"] > 0).sum()
        )
    })

if "KEGG_Pathway_Count" in final.columns:
    coverage.append({
        "Annotation": "KEGG pathway",
        "BUSCO_Count": int(
            (final["KEGG_Pathway_Count"] > 0).sum()
        )
    })

coverage_df = pd.DataFrame(coverage)

coverage_df["Total_Candidates"] = total

coverage_df["Percentage"] = (
    coverage_df["BUSCO_Count"] /
    total * 100
)

coverage_df.to_csv(
    TABLES / "Carbon_Breadth_annotation_coverage_420.csv",
    index=False
)

print(coverage_df.to_string(index=False))


# ============================================================
# 2. ANNOTATION COVERAGE FIGURE
# ============================================================

plt.figure(figsize=(9, 6))

plt.bar(
    coverage_df["Annotation"],
    coverage_df["BUSCO_Count"]
)

plt.ylabel("Number of BUSCO candidates")
plt.xlabel("Annotation type")
plt.title(
    "Carbon Breadth — Functional Annotation Coverage"
)

plt.ylim(0, total + 2)

for i, value in enumerate(
    coverage_df["BUSCO_Count"]
):
    plt.text(
        i,
        value + 0.2,
        str(value),
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    FIGURES / "Carbon_Breadth_annotation_coverage_420.png",
    dpi=300
)

plt.close()


# ============================================================
# 3. BUSCO PATHWAY TABLE
# ============================================================

print("\n" + "=" * 80)
print("BUSCO → PATHWAY SUMMARY")
print("=" * 80)

busco_pathway = final.copy()

pathway_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "KEGG_Gene_Count",
    "KEGG_Pathway_Count",
    "KEGG_Pathway_Names",
    "Pathway_Mapped"
]

pathway_columns = [
    c for c in pathway_columns
    if c in busco_pathway.columns
]

busco_pathway = busco_pathway[pathway_columns]

busco_pathway.to_csv(
    TABLES / "Carbon_Breadth_BUSCO_pathway_summary_final_420.csv",
    index=False
)

print(busco_pathway.to_string(index=False))


# ============================================================
# 4. PATHWAY FREQUENCY
# ============================================================

print("\n" + "=" * 80)
print("PATHWAY FREQUENCY")
print("=" * 80)

pathway_counts = {}

for _, row in final.iterrows():

    busco = clean(
        row.get("BUSCO_ID", "")
    )

    pathway_ids = split_items(
        row.get("KEGG_Pathway_IDs", "")
    )

    pathway_names = split_items(
        row.get("KEGG_Pathway_Names", "")
    )

    for i, pid in enumerate(pathway_ids):

        if i < len(pathway_names):
            pname = pathway_names[i]
        else:
            pname = ""

        key = (
            pid,
            pname
        )

        if key not in pathway_counts:
            pathway_counts[key] = {
                "Pathway_ID": pid,
                "Pathway_Name": pname,
                "BUSCOs": set()
            }

        pathway_counts[key]["BUSCOs"].add(
            busco
        )


pathway_frequency = []

for key, item in pathway_counts.items():

    pathway_frequency.append({
        "Pathway_ID": item["Pathway_ID"],
        "Pathway_Name": item["Pathway_Name"],
        "BUSCO_Count": len(item["BUSCOs"]),
        "BUSCOs": "; ".join(
            sorted(item["BUSCOs"])
        )
    })


pathway_frequency = pd.DataFrame(
    pathway_frequency
)

if len(pathway_frequency) > 0:

    pathway_frequency = pathway_frequency.sort_values(
        "BUSCO_Count",
        ascending=False
    )

pathway_frequency.to_csv(
    TABLES / "Carbon_Breadth_pathway_frequency_420.csv",
    index=False
)

print(
    f"\nUnique pathways represented: "
    f"{len(pathway_frequency)}"
)


# ============================================================
# 5. PATHWAY FREQUENCY FIGURE
# ============================================================

if len(pathway_frequency) > 0:

    plot_df = pathway_frequency.head(20).copy()

    labels = []

    for _, row in plot_df.iterrows():

        name = clean(row["Pathway_Name"])

        if not name:
            name = clean(row["Pathway_ID"])

        if len(name) > 55:
            name = name[:52] + "..."

        labels.append(name)

    plt.figure(
        figsize=(12, 8)
    )

    plt.barh(
        labels[::-1],
        plot_df["BUSCO_Count"].values[::-1]
    )

    plt.xlabel(
        "Number of Carbon Breadth BUSCO candidates"
    )

    plt.ylabel(
        "KEGG pathway"
    )

    plt.title(
        "KEGG Pathways Represented by Carbon Breadth Candidates"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES / "Carbon_Breadth_KEGG_pathway_frequency_420.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 6. FUNCTIONAL DESCRIPTION TABLE
# ============================================================

functional_columns = [
    "BUSCO_ID",
    "Functional_Description",
    "UniProt",
    "Protein",
    "Gene",
    "GO",
    "KEGG",
    "EC",
    "KEGG_Gene_IDs",
    "KEGG_Pathway_IDs",
    "KEGG_Pathway_Names",
    "Pathway_Mapped",
    "Annotation_Status"
]

functional_columns = [
    c for c in functional_columns
    if c in final.columns
]

functional_table = final[
    functional_columns
].copy()

functional_table.to_csv(
    TABLES
    / "Carbon_Breadth_final_functional_candidate_table_420.csv",
    index=False
)


# ============================================================
# 7. GO TERM SUMMARY
# ============================================================

go_terms = {}

if "GO" in final.columns:

    for _, row in final.iterrows():

        busco = clean(
            row["BUSCO_ID"]
        )

        terms = split_items(
            row["GO"]
        )

        for term in terms:

            if term not in go_terms:
                go_terms[term] = set()

            go_terms[term].add(
                busco
            )


go_summary = []

for term, buscos in go_terms.items():

    go_summary.append({
        "GO_ID": term,
        "BUSCO_Count": len(buscos),
        "BUSCOs": "; ".join(
            sorted(buscos)
        )
    })


go_summary = pd.DataFrame(
    go_summary
)

if len(go_summary) > 0:

    go_summary = go_summary.sort_values(
        "BUSCO_Count",
        ascending=False
    )

go_summary.to_csv(
    TABLES / "Carbon_Breadth_GO_term_summary_420.csv",
    index=False
)


# ============================================================
# 8. EC SUMMARY
# ============================================================

ec_terms = {}

if "EC" in final.columns:

    for _, row in final.iterrows():

        busco = clean(
            row["BUSCO_ID"]
        )

        terms = split_items(
            row["EC"]
        )

        for term in terms:

            if term not in ec_terms:
                ec_terms[term] = set()

            ec_terms[term].add(
                busco
            )


ec_summary = []

for term, buscos in ec_terms.items():

    ec_summary.append({
        "EC_Number": term,
        "BUSCO_Count": len(buscos),
        "BUSCOs": "; ".join(
            sorted(buscos)
        )
    })


ec_summary = pd.DataFrame(
    ec_summary
)

ec_summary.to_csv(
    TABLES / "Carbon_Breadth_EC_summary_420.csv",
    index=False
)


# ============================================================
# 9. FINAL REPORT
# ============================================================

report_file = (
    REPORTS
    / "Carbon_Breadth_final_biological_interpretation_420.txt"
)

with open(
    report_file,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 80 + "\n"
    )

    report.write(
        "CARBON BREADTH — FINAL BIOLOGICAL INTERPRETATION\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "CANDIDATE SET\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Total candidate BUSCOs: {total}\n\n"
    )

    report.write(
        "ANNOTATION COVERAGE\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    for _, row in coverage_df.iterrows():

        report.write(
            f"{row['Annotation']}: "
            f"{row['BUSCO_Count']}/{total} "
            f"({row['Percentage']:.1f}%)\n"
        )

    report.write("\n")

    report.write(
        "PATHWAY RESULTS\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"BUSCOs mapped to pathways: "
        f"{int((final['KEGG_Pathway_Count'] > 0).sum())}\n"
    )

    report.write(
        f"Unique KEGG pathways: "
        f"{len(pathway_frequency)}\n"
    )

    report.write("\n")

    report.write(
        "CANDIDATE-LEVEL FUNCTIONAL INTERPRETATION\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    for _, row in final.iterrows():

        busco = clean(
            row.get("BUSCO_ID", "")
        )

        function = clean(
            row.get(
                "Functional_Description",
                ""
            )
        )

        kegg_count = row.get(
            "KEGG_Gene_Count",
            0
        )

        pathway_count = row.get(
            "KEGG_Pathway_Count",
            0
        )

        status = clean(
            row.get(
                "Annotation_Status",
                ""
            )
        )

        report.write(
            f"\n{busco}\n"
        )

        report.write(
            f"  Functional description: {function}\n"
        )

        report.write(
            f"  KEGG gene count: {kegg_count}\n"
        )

        report.write(
            f"  KEGG pathway count: {pathway_count}\n"
        )

        report.write(
            f"  Annotation status: {status}\n"
        )

    report.write("\n\n")

    report.write(
        "INTERPRETATION NOTE\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        "The candidate BUSCOs were selected from the Carbon Breadth "
        "predictive feature analysis. Functional descriptions and "
        "database annotations provide biological context for these "
        "predictive candidates. KEGG pathway mapping was available "
        "for a subset of candidates and should be interpreted as "
        "pathway association rather than direct evidence that a "
        "candidate causes the phenotype.\n"
    )

    report.write("\n")

    report.write(
        "FILES GENERATED\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Annotation coverage:\n"
        f"{TABLES / 'Carbon_Breadth_annotation_coverage_420.csv'}\n\n"
    )

    report.write(
        f"BUSCO pathway summary:\n"
        f"{TABLES / 'Carbon_Breadth_BUSCO_pathway_summary_final_420.csv'}\n\n"
    )

    report.write(
        f"Pathway frequency:\n"
        f"{TABLES / 'Carbon_Breadth_pathway_frequency_420.csv'}\n\n"
    )

    report.write(
        f"Final functional table:\n"
        f"{TABLES / 'Carbon_Breadth_final_functional_candidate_table_420.csv'}\n\n"
    )

    report.write(
        f"GO summary:\n"
        f"{TABLES / 'Carbon_Breadth_GO_term_summary_420.csv'}\n\n"
    )

    report.write(
        f"EC summary:\n"
        f"{TABLES / 'Carbon_Breadth_EC_summary_420.csv'}\n\n"
    )

    report.write(
        f"Report:\n{report_file}\n"
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("FINAL CARBON BREADTH INTERPRETATION COMPLETE")
print("=" * 80)

print(f"\nCandidates: {total}")

print(
    f"Pathway-mapped candidates: "
    f"{int((final['KEGG_Pathway_Count'] > 0).sum())}"
)

print(
    f"Unique pathways: "
    f"{len(pathway_frequency)}"
)

print("\nOUTPUT DIRECTORY:")
print(OUT)

print("\nTABLES:")
for f in TABLES.glob("*.csv"):
    print(f"  {f}")

print("\nFIGURES:")
for f in FIGURES.glob("*.png"):
    print(f"  {f}")

print("\nREPORT:")
print(report_file)

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)