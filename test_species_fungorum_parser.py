import requests
from bs4 import BeautifulSoup
import re
import html
import unicodedata


URL = "https://www.speciesfungorum.org/Names/Names.asp"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    )
}


def normalize_text(value):

    if value is None:
        return ""

    value = html.unescape(str(value))

    value = unicodedata.normalize(
        "NFKC",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def normalize_binomial(name):

    name = normalize_text(name)

    parts = name.split()

    if len(parts) < 2:
        return ""

    return f"{parts[0]} {parts[1]}"


def inspect_genus(genus):

    print("=" * 70)
    print(f"TESTING GENUS: {genus}")
    print("=" * 70)

    response = requests.get(
        URL,
        params={"strGenus": genus},
        headers=HEADERS,
        timeout=30
    )

    print(
        "Status:",
        response.status_code
    )

    print(
        "URL:",
        response.url
    )

    soup = BeautifulSoup(
        response.text,
        "html.parser"
    )

    # ------------------------------------------------------------
    # Find the result heading
    # ------------------------------------------------------------

    page_text = soup.get_text(
        " ",
        strip=True
    )

    match = re.search(
        r"(\d+)\s+results",
        page_text,
        re.IGNORECASE
    )

    if match:

        print(
            "Species Fungorum result count:",
            match.group(1)
        )

    else:

        print(
            "Could not detect result count."
        )

    # ------------------------------------------------------------
    # Inspect links containing species names
    # ------------------------------------------------------------

    species_links = []

    for a in soup.find_all(
        "a",
        href=True
    ):

        text = normalize_text(
            a.get_text(
                " ",
                strip=True
            )
        )

        # A species name begins with capitalized genus
        # followed by lowercase epithet.

        if re.match(
            r"^[A-Z][A-Za-z-]+\s+[a-z][A-Za-z0-9-]+",
            text
        ):

            species_links.append(
                {
                    "text": text,
                    "href": a.get("href")
                }
            )

    print()
    print(
        "Potential species links:",
        len(species_links)
    )

    for i, item in enumerate(
        species_links,
        start=1
    ):

        print(
            f"{i:02d}.",
            item["text"],
            "->",
            item["href"]
        )

    # ------------------------------------------------------------
    # Inspect all text-containing elements around a target
    # ------------------------------------------------------------

    target = "Ambrosiozyma vanderkliftii"

    print()
    print(
        "Searching for:",
        target
    )

    found = False

    for element in soup.find_all(
        string=re.compile(
            "Ambrosiozyma vanderkliftii",
            re.IGNORECASE
        )
    ):

        found = True

        print()
        print(
            "FOUND ELEMENT:"
        )

        print(
            element.parent.prettify()[:5000]
        )

        print()
        print(
            "PARENT TEXT:"
        )

        print(
            element.parent.get_text(
                " | ",
                strip=True
            )
        )

    if not found:

        print(
            "Target species was not found."
        )


# ================================================================
# TEST 1
# ================================================================

inspect_genus(
    "Ambrosiozyma"
)

print()
print()
print(
    "TEST COMPLETE"
)