from pathlib import Path
import pandas as pd
import os

ROOT = Path(r"C:\Y1000_chassis_project")

BUSCO_ROOT = (
    ROOT
    / "results"
    / "stage4B_phylogeny"
    / "busco"
)

print("=" * 80)
print("BUSCO DIRECTORY DIAGNOSTIC")
print("=" * 80)

# ---------------------------------------------------------
# 1. Find one full_table.tsv
# ---------------------------------------------------------

full_tables = list(BUSCO_ROOT.rglob("full_table.tsv"))

print(f"\nFull tables found: {len(full_tables)}")

if not full_tables:
    raise FileNotFoundError("No full_table.tsv files found.")

example = full_tables[0]

print("\nExample full_table.tsv:")
print(example)

# ---------------------------------------------------------
# 2. Read the file WITHOUT assuming a fixed number of columns
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("RAW BUSCO TABLE")
print("=" * 80)

with open(example, "r", errors="replace") as f:
    lines = f.readlines()

for i, line in enumerate(lines[:20], start=1):
    print(f"{i}: {line.rstrip()}")

# ---------------------------------------------------------
# 3. Find the first candidate BUSCO
# ---------------------------------------------------------

candidate_ids = {
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
    "34052at4891",
}

print("\n" + "=" * 80)
print("SEARCHING FOR CANDIDATE BUSCO")
print("=" * 80)

candidate_line = None

for line in lines:
    if any(line.startswith(busco + "\t") for busco in candidate_ids):
        candidate_line = line.rstrip()
        break

if candidate_line:
    print("\nCandidate line found:")
    print(candidate_line)

    fields = candidate_line.split("\t")

    print("\nNumber of fields:", len(fields))

    print("\nFields:")
    for i, value in enumerate(fields):
        print(f"[{i}] = {value}")

else:
    print("No candidate BUSCO found in this example table.")

# ---------------------------------------------------------
# 4. Inspect the COMPLETE BUSCO directory
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("BUSCO DIRECTORY CONTENTS")
print("=" * 80)

busco_dir = example.parent

print("\nDirectory:")
print(busco_dir)

for item in sorted(busco_dir.iterdir()):
    if item.is_file():
        print("FILE :", item.name)
    else:
        print("DIR  :", item.name)

# ---------------------------------------------------------
# 5. Search ONLY this assembly directory for annotation files
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("ANNOTATION FILES IN THIS BUSCO DIRECTORY / PARENT")
print("=" * 80)

search_root = busco_dir.parent.parent

print("\nSearch root:")
print(search_root)

extensions = {
    ".gff",
    ".gff3",
    ".gtf",
    ".gb",
    ".gbff",
    ".faa",
    ".fasta",
    ".fa",
    ".fna",
}

found = []

for p in search_root.rglob("*"):
    if p.is_file() and p.suffix.lower() in extensions:
        found.append(p)

print(f"\nRelevant files found: {len(found)}")

for p in found[:100]:
    print(p)

if len(found) > 100:
    print(f"... and {len(found) - 100} more")

print("\n" + "=" * 80)
print("DIAGNOSTIC COMPLETE")
print("=" * 80)