import requests
from bs4 import BeautifulSoup

url = "https://www.speciesfungorum.org/Names/Names.asp"

params = {
    "strGenus": "Ambrosiozyma"
}

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/153.0 Safari/537.36"
    )
}

print("Requesting Species Fungorum...")

response = requests.get(
    url,
    params=params,
    headers=headers,
    timeout=30
)

print("Status:", response.status_code)
print("Final URL:", response.url)
print("HTML length:", len(response.text))

# Save the EXACT HTML returned by the website
output = r"C:\Y1000_chassis_project\results\stage3B1G_species_fungorum_validation\ambrosiozyma_raw.html"

with open(
    output,
    "w",
    encoding="utf-8"
) as f:
    f.write(response.text)

print()
print("Saved raw HTML:")
print(output)

# Basic inspection
soup = BeautifulSoup(
    response.text,
    "html.parser"
)

print()
print("TITLE:")
print(
    soup.title.get_text(
        " ",
        strip=True
    )
    if soup.title
    else "NO TITLE"
)

print()
print("TABLE COUNT:")
print(
    len(
        soup.find_all("table")
    )
)

print()
print("LINK COUNT:")
print(
    len(
        soup.find_all("a")
    )
)

print()
print("TEXT SAMPLE:")
print(
    soup.get_text(
        " ",
        strip=True
    )[:5000]
)

print()
print("DONE")