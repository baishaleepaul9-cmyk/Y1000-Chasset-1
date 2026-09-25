from pathlib import Path
import pandas as pd
import os

# ============================================================
# PARETO GENOME FILE CHECK
# ============================================================

PROJECT = Path(r"C:\Y1000_chassis_project")

BASE = (
    PROJECT
    / "results"
    / "stage4B_phylogeny"
)

MANIFEST = BASE / "stage4B_genome_fasta_manifest.csv"


# ------------------------------------------------------------
# Pareto candidates from Stage 4A.1
# ------------------------------------------------------------

pareto = {
    "GCA_003705225.1": "Ambrosiozyma vanderkliftii",
    "GCA_003123585.1": "Barnettozyma californica",
    "GCA_003709245.3": "Cyberlindnera saturnus",
    "GCA_030558845.1": "Kodamaea laetipori",
    "GCA_030563145.1":
        "Schwanniomyces polymorphus var. africanus",
    "GCA_030463025.1":
        "Schwanniomyces polymorphus var. polymorphus",
    "GCA_030583345.1":
        "Schwanniomyces pseudopolymorphus",
    "GCA_030583405.1":
        "Sugiyamaella americana",
    "GCA_030579815.1":
        "Sugiyamaella smithiae",
    "GCA_030558095.1":
        "Teunomyces funiuensis",
    "GCA_030564625.1":
        "Zygoascus hellenicus",
}


# ------------------------------------------------------------
# Read manifest
# ------------------------------------------------------------

print("=" * 80)
print("PARETO GENOME FILE CHECK")
print("=" * 80)

if not MANIFEST.exists():
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST}"
    )

manifest = pd.read_csv(MANIFEST)

manifest["Assembly_Accession"] = (
    manifest["Assembly_Accession"]
    .astype(str)
    .str.strip()
)

manifest["Species"] = (
    manifest["Species"]
    .astype(str)
    .str.strip()
)


# ------------------------------------------------------------
# Check every Pareto accession
# ------------------------------------------------------------

results = []

for accession, pareto_name in pareto.items():

    rows = manifest[
        manifest["Assembly_Accession"] == accession
    ]

    if len(rows) == 0:

        results.append({
            "Assembly_Accession": accession,
            "Pareto_Name": pareto_name,
            "In_436_Manifest": False,
            "Genome_FASTA": "",
            "File_Exists": False,
            "Status": "NOT_IN_436_REPRESENTATIVE_SET"
        })

        continue


    row = rows.iloc[0]

    fasta_value = str(
        row["Genome_FASTA"]
    ).strip()

    fasta_path = Path(fasta_value)

    # Try direct Windows path
    possible_paths = [
        fasta_path
    ]

    # If path is relative
    if not fasta_path.is_absolute():
        possible_paths.extend([
            PROJECT / fasta_path,
            BASE / fasta_path
        ])

    # Convert WSL /mnt/c/... path to Windows
    if fasta_value.startswith("/mnt/c/"):
        windows_path = (
            Path("C:/")
            / fasta_value[len("/mnt/c/"):]
        )
        possible_paths.append(windows_path)

    existing = None

    for p in possible_paths:
        try:
            if p.exists() and p.is_file():
                existing = p
                break
        except Exception:
            pass


    if existing:

        status = "FILE_PRESENT"

    else:

        status = "MANIFEST_ENTRY_BUT_FILE_NOT_FOUND"


    results.append({
        "Assembly_Accession": accession,
        "Pareto_Name": pareto_name,
        "In_436_Manifest": True,
        "Genome_FASTA": fasta_value,
        "File_Exists": existing is not None,
        "Resolved_Path": (
            str(existing)
            if existing
            else ""
        ),
        "Status": status
    })


# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("RESULTS")
print("=" * 80)

for _, row in df.iterrows():

    symbol = (
        "✓"
        if row["File_Exists"]
        else "!"
    )

    print(
        f"{symbol} "
        f"{row['Assembly_Accession']:18s}  "
        f"{row['Pareto_Name']}"
    )

    print(
        f"    Manifest: {row['In_436_Manifest']}"
    )

    print(
        f"    Status:   {row['Status']}"
    )

    if row["Resolved_Path"]:
        print(
            f"    FASTA:    {row['Resolved_Path']}"
        )


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

out = (
    BASE
    / "pareto_genome_file_check.csv"
)

df.to_csv(
    out,
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print(
    "Pareto candidates checked:",
    len(df)
)

print(
    "Files physically present:",
    df["File_Exists"].sum()
)

print(
    "Directly represented in 436:",
    df["In_436_Manifest"].sum()
)

print(
    "Not directly represented:",
    (~df["In_436_Manifest"]).sum()
)

print(
    f"\nQC written to:\n{out}"
)