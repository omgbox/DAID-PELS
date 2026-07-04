"""
Merge English dictionaries:
1. English_dictionary.csv (175K entries with definitions)
2. words_466k.txt (370K word list for validation)
3. old_english_dictionary.csv (42K Old English entries)

Output: combined_english_dictionary.csv
"""
import csv
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

def merge_dictionaries():
    entries = {}
    
    # 1. Load current English dictionary (has definitions)
    print("Loading English_dictionary.csv...")
    with open('English_dictionary.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row.get('Word', '').strip()
            pos = row.get('POS', '').strip().strip('"')
            definition = row.get('Definition', '').strip().strip('"')
            if word and definition:
                key = word.lower()
                if key not in entries or len(definition) > len(entries[key].get('definition', '')):
                    entries[key] = {
                        'word': word,
                        'pos': pos,
                        'definition': definition,
                        'language': 'english'
                    }
    print(f"  Loaded {len(entries):,} English entries with definitions")
    
    # 2. Load 466K word list (word validation - no definitions, just words)
    print("Loading words_466k.txt...")
    word_list_count = 0
    with open('words_466k.txt', 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip()
            if word and word.lower() not in entries:
                entries[word.lower()] = {
                    'word': word,
                    'pos': '',
                    'definition': '',
                    'language': 'english'
                }
                word_list_count += 1
    print(f"  Added {word_list_count:,} new words from word list")
    print(f"  Total unique words: {len(entries):,}")
    
    # 3. Load Old English dictionary
    print("Loading old_english_dictionary.csv...")
    oe_count = 0
    with open('old_english_dictionary.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            headword = row.get('headword', '').strip()
            definition = row.get('definition', '').strip()
            if headword:
                key = f"oe_{headword.lower()}"
                entries[key] = {
                    'word': headword,
                    'pos': 'OE',
                    'definition': definition[:500] if definition else '',
                    'language': 'old_english'
                }
                oe_count += 1
    print(f"  Loaded {oe_count:,} Old English entries")
    
    # 4. Save merged dictionary
    output_path = 'combined_english_dictionary.csv'
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['word', 'pos', 'definition', 'language'])
        writer.writeheader()
        for key in sorted(entries.keys()):
            writer.writerow(entries[key])
    
    print(f"\nSaved merged dictionary to {output_path}")
    print(f"  English (with definitions): {sum(1 for v in entries.values() if v['language'] == 'english' and v['definition']):,}")
    print(f"  English (word list only): {sum(1 for v in entries.values() if v['language'] == 'english' and not v['definition']):,}")
    print(f"  Old English: {oe_count:,}")
    print(f"  Total entries: {len(entries):,}")

if __name__ == '__main__':
    merge_dictionaries()
