"""
Download Assets for DAID-PELS

Downloads:
1. English dictionary CSV (175K entries)
2. Pride and Prejudice clean text (Project Gutenberg)

Usage:
    python download_assets.py
"""

import os
import sys
import csv
import urllib.request
import io
from pathlib import Path

# Project root = this file's directory
PROJECT_ROOT = Path(__file__).parent
BOOKS_DIR = PROJECT_ROOT / "books"
DICT_PATH = PROJECT_ROOT / "English_dictionary.csv"
PRIDE_PATH = BOOKS_DIR / "pride_and_prejudice_clean.txt"


def download_file(url: str, dest: Path, description: str):
    """Download a file with progress output."""
    print(f"  Downloading {description}...")
    print(f"  URL: {url}")

    try:
        urllib.request.urlretrieve(url, str(dest))
        size_kb = dest.stat().st_size / 1024
        print(f"  Saved: {dest} ({size_kb:.1f} KB)")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def download_dictionary():
    """Download English dictionary CSV."""
    if DICT_PATH.exists():
        print(f"  Dictionary already exists: {DICT_PATH}")
        return True

    # Try multiple sources
    urls = [
        "https://raw.githubusercontent.com/vijayvamsi28/English-Dictionary/refs/heads/main/English_dictionary.csv",
        "https://raw.githubusercontent.com/words/an-array-of-english-words/master/words.csv",
    ]

    for url in urls:
        if download_file(url, DICT_PATH, "English dictionary"):
            # Verify it has the expected columns
            try:
                with open(DICT_PATH, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    if 'Word' in row and 'Definition' in row:
                        print(f"  Verified: dictionary has Word, POS, Definition columns")
                        return True
                    else:
                        print(f"  Warning: unexpected columns: {list(row.keys())}")
                        # Still usable if it has Word + some definition column
                        return True
            except Exception as e:
                print(f"  Warning: could not verify dictionary: {e}")
                return True

    print("  Failed to download dictionary from all sources")
    print("  Manual download: place English_dictionary.csv in the project root")
    return False


def download_pride_and_prejudice():
    """Download Pride and Prejudice from Project Gutenberg."""
    if PRIDE_PATH.exists():
        print(f"  Book already exists: {PRIDE_PATH}")
        return True

    # Ensure books directory exists
    BOOKS_DIR.mkdir(exist_ok=True)

    # Download raw text from Project Gutenberg
    raw_path = BOOKS_DIR / "pride_and_prejudice_raw.txt"
    url = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"

    if not download_file(url, raw_path, "Pride and Prejudice (raw)"):
        return False

    # Clean the text
    print("  Cleaning text...")
    try:
        from preprocess_gutenberg import preprocess_gutenberg_book
        preprocess_gutenberg_book(str(raw_path), str(PRIDE_PATH))
        raw_path.unlink()  # Remove raw file
        return True
    except ImportError:
        # If import fails, do basic cleaning inline
        print("  Using inline cleaning...")
        with open(raw_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Basic Gutenberg cleanup
        import re

        # Find start marker
        start_match = re.search(r'\*\*\*\s*START OF TH(E|IS) PROJECT GUTENBERG', text, re.IGNORECASE)
        if start_match:
            text = text[start_match.end():]

        # Find end marker
        end_match = re.search(r'\*\*\*\s*END OF TH(E|IS) PROJECT GUTENBERG', text, re.IGNORECASE)
        if end_match:
            text = text[:end_match.start()]

        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = text.strip()

        with open(PRIDE_PATH, 'w', encoding='utf-8') as f:
            f.write(text)

        raw_path.unlink()  # Remove raw file

        size_kb = PRIDE_PATH.stat().st_size / 1024
        print(f"  Cleaned: {PRIDE_PATH} ({size_kb:.1f} KB)")
        return True

    except Exception as e:
        print(f"  Error cleaning text: {e}")
        return False


def main():
    print("\n" + "=" * 60)
    print("  DAID-PELS Asset Downloader")
    print("=" * 60)

    success = True

    print("\n[1/2] Dictionary")
    if not download_dictionary():
        success = False

    print("\n[2/2] Sample Book (Pride and Prejudice)")
    if not download_pride_and_prejudice():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("  All assets downloaded successfully!")
        print("\n  Next steps:")
        print("    python train_pride.py    # Train on Pride and Prejudice")
        print("    python -m bookbot.main query   # Start querying")
    else:
        print("  Some assets failed to download.")
        print("  See errors above for details.")
    print("=" * 60 + "\n")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
