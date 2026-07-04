import csv
import sys
from nltk.corpus import wordnet

print("Starting dictionary generation...", flush=True)

entries = set()
for syn in wordnet.all_synsets():
    pos = syn.pos()
    pos_map = {'n': 'noun', 'v': 'verb', 'a': 'adjective', 'r': 'adverb'}
    pos_full = pos_map.get(pos, 'noun')
    for lemma in syn.lemmas():
        word = lemma.name().replace('_', ' ')
        definition = syn.definition()
        entries.add((word, pos_full, definition))

print(f"Found {len(entries)} unique entries", flush=True)

with open('English_dictionary.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Word', 'POS', 'Definition'])
    for word, pos, defn in sorted(entries):
        writer.writerow([word, pos, defn])

print("Done! Dictionary saved to English_dictionary.csv", flush=True)
