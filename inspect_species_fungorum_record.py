import requests
from bs4 import BeautifulSoup
import re
import html
import unicodedata

BASE = "https://www.speciesfungorum.org"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    )
}


def clean(text):

    if text is None:
        return ""

    text = html.unescape(str(text))

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


session = requests.Session()
session.headers.update(HEADERS)


# ================================================================
# Test exact record:
# Ambrosiozyma vanderkliftii
# RecordID = 804712
# ================================================================

record_url = (
    BASE +
    "/names/NamesRecord.asp?RecordID=804712"
)

print("=" * 70)
print("SPECIES FUNGORUM RECORD INSPECTION")
print("=" * 70)

print()
print("URL:")
print(record_url)

response = session.get(
    record_url,
    timeout=30
)

print()
print(
    "Status:",
    response.status_code
)

print(
    "Final URL:",
    response.url
)

print(
    "HTML length:",
    len(response.text)
)


# Save raw record HTML

output = (
    r"C:\Y1000_chassis_project\results"
    r"\stage3B1G_species_fungorum_validation"
    r"\Ambrosiozyma_vanderkliftii_record.html"
)

with open(
    output,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        response.text
    )

print()
print("Saved raw record:")
print(output)


# ================================================================
# PARSE PAGE
# ================================================================

soup = BeautifulSoup(
    response.text,
    "html.parser"
)

print()
print("TITLE:")

if soup.title:

    print(
        clean(
            soup.title.get_text(
                " ",
                strip=True
            )
        )
    )

else:

    print("NO TITLE")


# ================================================================
# PAGE TEXT
# ================================================================

text = soup.get_text(
    "\n",
    strip=True
)

text = clean(
    text
)

print()
print("=" * 70)
print("PAGE TEXT")
print("=" * 70)

print(
    text[:10000]
)


# ================================================================
# TABLE / DIV / DL INSPECTION
# ================================================================

print()
print("=" * 70)
print("HTML STRUCTURE")
print("=" * 70)

print(
    "Tables:",
    len(
        soup.find_all("table")
    )
)

print(
    "Definition lists:",
    len(
        soup.find_all("dl")
    )
)

print(
    "Divs:",
    len(
        soup.find_all("div")
    )
)

print(
    "Paragraphs:",
    len(
        soup.find_all("p")
    )
)


# ================================================================
# FIND LABELS
# ================================================================

labels = [
    "Name",
    "Taxon status",
    "Current name",
    "Parent taxon",
    "Synonym",
    "Accepted"
]

print()
print("=" * 70)
print("LABEL ELEMENTS")
print("=" * 70)

for label in labels:

    matches = soup.find_all(
        string=re.compile(
            rf"^{re.escape(label)}$",
            re.IGNORECASE
        )
    )

    print()
    print(
        f"LABEL: {label}"
    )

    print(
        f"Matches: {len(matches)}"
    )

    for m in matches[:10]:

        print(
            "Parent:"
        )

        print(
            m.parent.prettify()[:3000]
        )


print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)