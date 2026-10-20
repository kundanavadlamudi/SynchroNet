import re
import pandas as pd
from pathlib import Path


# Project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Input files
RAW_DIR = BASE_DIR / "data" / "raw"

KEYWORD_FILES = [
    RAW_DIR / "Hashtags .txt",
    RAW_DIR / "hashtags2.txt"
]

# Output file
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cleaned_keywords.csv"


def load_keywords(file_path):
    """Read a text file and extract keywords and hashtags."""

    keywords = []

    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return keywords

    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as file:
        content = file.read()

    # Extract hashtags, phrases, and individual terms
    hashtags = re.findall(r"#\w+", content, flags=re.UNICODE)

    # Split text into lines and clean each line
    for line in content.splitlines():
        line = line.strip()

        if not line:
            continue

        # Remove common query operators and punctuation
        line = re.sub(r"\bOR\b", " ", line, flags=re.IGNORECASE)
        line = re.sub(r"[()\"']", " ", line)

        # Split comma-separated terms
        terms = line.split(",")

        for term in terms:
            term = term.strip()

            if term:
                keywords.append(term)

    keywords.extend(hashtags)

    return keywords


def main():
    """Load, clean, deduplicate, and save keywords."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_keywords = []

    for file_path in KEYWORD_FILES:
        keywords = load_keywords(file_path)

        print(f"{file_path.name}: {len(keywords)} extracted entries")

        all_keywords.extend(keywords)

    # Normalize whitespace and remove empty entries
    cleaned_keywords = [
        re.sub(r"\s+", " ", keyword).strip()
        for keyword in all_keywords
        if keyword.strip()
    ]

    # Remove duplicates without losing the original order
    unique_keywords = list(
        {keyword.casefold(): keyword for keyword in reversed(cleaned_keywords)}.values()
    )
    unique_keywords.reverse()

    # Save the cleaned keywords
    df = pd.DataFrame({"keyword": unique_keywords})

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nKeyword processing completed!")
    print(f"Total extracted entries: {len(all_keywords)}")
    print(f"Unique keywords: {len(unique_keywords)}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()