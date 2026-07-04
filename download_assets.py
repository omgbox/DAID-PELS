"""
Download Assets for DAID-PELS

Downloads:
1. English dictionary CSV (175K entries)
2. 466K word list (370K unique words for validation)
3. Old English dictionary (42K entries)
4. Pride and Prejudice clean text (Project Gutenberg)

Then merges dictionaries into combined_english_dictionary.csv

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
WORD_LIST_PATH = PROJECT_ROOT / "words_466k.txt"
OLD_ENGLISH_DICT_PATH = PROJECT_ROOT / "old_english_dictionary.csv"
COMBINED_DICT_PATH = PROJECT_ROOT / "combined_english_dictionary.csv"
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


def download_word_list():
    """Download 466K word list from dwyl/english-words."""
    if WORD_LIST_PATH.exists():
        print(f"  Word list already exists: {WORD_LIST_PATH}")
        return True

    url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    if download_file(url, WORD_LIST_PATH, "466K word list"):
        # Verify
        with open(WORD_LIST_PATH, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f if line.strip()]
        print(f"  Verified: {len(words):,} words")
        return True
    return False


def download_old_english_dictionary():
    """Download Old English dictionary from fhardison/old-english-dict."""
    if OLD_ENGLISH_DICT_PATH.exists():
        print(f"  Old English dictionary already exists: {OLD_ENGLISH_DICT_PATH}")
        return True

    # Download raw JSON first
    import json
    import re

    json_path = PROJECT_ROOT / "oe_raw.json"
    url = "https://raw.githubusercontent.com/fhardison/old-english-dict/main/data/oe.json"

    if not download_file(url, json_path, "Old English dictionary (raw JSON)"):
        return False

    # Parse and convert to CSV
    print("  Parsing Old English dictionary...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        keys = [k for k in data.keys() if not k.startswith('##')]

        def clean_html(text):
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()

        entries = {}
        for k in keys:
            val = data[k]
            if isinstance(val, str):
                headword = k.split(' (')[0].strip()
                definition = clean_html(val)
                key = headword.lower()
                if key not in entries or len(definition) > len(entries[key]):
                    entries[key] = definition

        with open(OLD_ENGLISH_DICT_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['headword', 'definition'])
            for hw, defn in sorted(entries.items()):
                writer.writerow([hw, defn])

        json_path.unlink()  # Remove raw JSON
        print(f"  Saved: {OLD_ENGLISH_DICT_PATH} ({len(entries):,} entries)")
        return True

    except Exception as e:
        print(f"  Error parsing Old English dictionary: {e}")
        return False


def merge_dictionaries():
    """Merge all dictionaries into combined_english_dictionary.csv."""
    if COMBINED_DICT_PATH.exists():
        print(f"  Combined dictionary already exists: {COMBINED_DICT_PATH}")
        return True

    print("  Merging dictionaries...")
    entries = {}

    # 1. Load current English dictionary (has definitions)
    if DICT_PATH.exists():
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                word = row.get('Word', '').strip()
                pos = row.get('POS', '').strip().strip('"')
                definition = row.get('Definition', '').strip().strip('"')
                if word and definition:
                    key = word.lower()
                    if key not in entries or len(definition) > len(entries[key].get('definition', '')):
                        entries[key] = {
                            'word': word, 'pos': pos,
                            'definition': definition, 'language': 'english'
                        }
        print(f"    English (with definitions): {len(entries):,}")

    # 2. Load 466K word list (word validation)
    if WORD_LIST_PATH.exists():
        added = 0
        with open(WORD_LIST_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip()
                if word and word.lower() not in entries:
                    entries[word.lower()] = {
                        'word': word, 'pos': '', 'definition': '',
                        'language': 'english'
                    }
                    added += 1
        print(f"    Added {added:,} new words from word list")

    # 3. Load Old English dictionary
    oe_count = 0
    if OLD_ENGLISH_DICT_PATH.exists():
        with open(OLD_ENGLISH_DICT_PATH, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                headword = row.get('headword', '').strip()
                definition = row.get('definition', '').strip()
                if headword:
                    entries[f"oe_{headword.lower()}"] = {
                        'word': headword, 'pos': 'OE',
                        'definition': definition[:500] if definition else '',
                        'language': 'old_english'
                    }
                    oe_count += 1
        print(f"    Old English: {oe_count:,}")

    # Save
    with open(COMBINED_DICT_PATH, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'pos', 'definition', 'language'])
        writer.writeheader()
        for key in sorted(entries.keys()):
            writer.writerow(entries[key])

    print(f"  Saved combined dictionary: {len(entries):,} total entries")
    return True


def main():
    print("\n" + "=" * 60)
    print("  DAID-PELS Asset Downloader")
    print("=" * 60)

    success = True

    print("\n[1/5] English Dictionary")
    if not download_dictionary():
        success = False

    print("\n[2/5] 466K Word List")
    if not download_word_list():
        success = False

    print("\n[3/5] Old English Dictionary")
    if not download_old_english_dictionary():
        success = False

    print("\n[4/5] Merge Dictionaries")
    if not merge_dictionaries():
        success = False

    print("\n[5/5] Sample Book (Pride and Prejudice)")
    if not download_pride_and_prejudice():
        success = False

    print("\n" + "=" * 60)
    if success:
        print("  All assets downloaded successfully!")
        print("\n  Dictionary stats:")
        print("    - English (with definitions): ~147K words")
        print("    - English (word list only): ~302K words")
        print("    - Old English: ~42K words")
        print("    - Total: ~491K entries")
        print("\n  Next steps:")
        print("    python train_all_books.py    # Train on all books")
        print("    python -m bookbot.main query # Start querying")
    else:
        print("  Some assets failed to download.")
        print("  See errors above for details.")
    print("=" * 60 + "\n")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
