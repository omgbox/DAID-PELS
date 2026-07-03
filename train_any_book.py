"""Train on any book using the fast pipeline."""
import sys
import time
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bookbot.config import DATABASE_PATH
from bookbot.database.db_manager import DBManager
from bookbot.pipeline_context import PipelineContext
from bookbot.core.tokenizer import Tokenizer, normalize_text
from bookbot.core.pos_tagger import POSTagger
from bookbot.core.ner_extractor import NERExtractor
from bookbot.core.svo_extractor import SVOExtractor
from bookbot.core.entity_graph import EntityGraph
from bookbot.core.coreference import Coreference
from bookbot.core.topic_modeler import TopicModeler
import csv

BOOK = sys.argv[1] if len(sys.argv) > 1 else 'books/alice_clean.txt'

print(f"\n{'='*60}")
print(f"  TRAINING on: {BOOK}")
print(f"{'='*60}")
start = time.time()

# Delete DB
import os
if os.path.exists(DATABASE_PATH):
    os.remove(DATABASE_PATH)

db = DBManager(DATABASE_PATH)
db.connect()
db.initialize_schema()

# Load dictionary
print("[1/9] Loading dictionary...")
t = time.time()
with open('English_dictionary.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    defs = [{'word': r.get('Word',''), 'pos_canonical': r.get('POS','').strip('"'),
             'pos_original': r.get('POS','').strip('"'), 'definition': r.get('Definition','').strip('"'),
             'word_lower': r.get('Word','').lower(), 'word_length': len(r.get('Word','')),
             'def_word_count': len(r.get('Definition','').split()), 'entry_index': i}
            for i, r in enumerate(reader) if r.get('Word') and r.get('Definition')]
db.insert_many('definitions', defs)
print(f"  {len(defs):,} entries in {time.time()-t:.1f}s")

# Load book
print("[2/9] Loading book...")
with open(BOOK, 'r', encoding='utf-8') as f:
    book_text = f.read()
print(f"  {len(book_text):,} chars")

context = PipelineContext()
context.raw_text = normalize_text(book_text)
# Collapse all newlines so pysbd can sentence-split by linguistic cues
cleaned = re.sub(r'\s+', ' ', context.raw_text).strip()
context.normalized_text = cleaned

# Tokenize
print("[3/9] Tokenizing...")
t = time.time()
tok = Tokenizer()
context.update(tok.process(context))
print(f"  {len(context.sentences)} sentences in {time.time()-t:.1f}s")

# POS
print("[4/9] POS tagging...")
t = time.time()
pos = POSTagger()
context.update(pos.process(context))
print(f"  Done in {time.time()-t:.1f}s")

# NER
print("[5/9] NER...")
t = time.time()
ner = NERExtractor()
context.update(ner.process(context))
print(f"  {len(context.entities)} entities in {time.time()-t:.1f}s")

# SVO
print("[6/9] SVO extraction...")
t = time.time()
long = [s for s in context.sentences if len(s.get('tokens',[])) >= 5]
orig = context.sentences
context.sentences = long
svo = SVOExtractor()
context.update(svo.process(context))
context.sentences = orig
print(f"  {len(context.svo_triples)} triples from {len(long)} sents in {time.time()-t:.1f}s")

# Entity Graph
print("[7/9] Entity graph...")
t = time.time()
graph = EntityGraph()
context.update(graph.process(context))
print(f"  {len(context.knowledge_edges)} edges in {time.time()-t:.1f}s")

# Coreference
print("[8/9] Coreference...")
t = time.time()
coref = Coreference()
context.update(coref.process(context))
print(f"  {len(context.coreferences)} chains in {time.time()-t:.1f}s")

# Topics (auto-sampled inside TopicModeler)
print("[9/9] Topics...")
t = time.time()
topics = TopicModeler()
context.update(topics.process(context))
print(f"  {len(context.topics)} topics in {time.time()-t:.1f}s")

# Save everything
print("\nSaving to database...")
for sent in context.sentences:
    db.insert('sentences', {
        'sentence_id': sent.get('sentence_id'),
        'chapter_id': sent.get('chapter_id'),
        'paragraph_id': sent.get('paragraph_id'),
        'position_in_para': sent.get('position_in_para'),
        'raw_text': sent.get('text', ''),
        'normalized_text': sent.get('normalized_text', ''),
        'token_count': len(sent.get('tokens', [])),
        'word_count': len([t for t in sent.get('tokens', []) if not t.get('is_punctuation')]),
    })
    for token in sent.get('tokens', []):
        db.insert('sentence_tokens', {
            'sentence_id': sent.get('sentence_id'),
            'position': token.get('position', 0),
            'token': token.get('token', ''),
            'token_lower': token.get('token', '').lower(),
            'pos_tag': token.get('pos_tag', ''),
            'is_punctuation': 1 if token.get('is_punctuation') else 0,
            'is_stopword': 1 if token.get('is_stopword') else 0,
        })

for e in context.entities:
    db.insert('entities', {
        'entity_id': e.get('entity_id'), 'canonical_name': e.get('canonical_name'),
        'entity_type': e.get('entity_type'), 'frequency': e.get('frequency', 1),
        'centrality': e.get('centrality', 0.0),
    })

for t in context.svo_triples:
    db.insert('svo_triples', {
        'subject': t.get('subject',''), 'verb': t.get('verb',''),
        'object': t.get('object',''), 'sentence_id': t.get('sentence_id',0),
        'confidence': t.get('confidence', 0.5), 'passive': 1 if t.get('passive') else 0,
    })

edge_rows = []
for e in context.knowledge_edges:
    edge_rows.append({
        'source_type': e.get('source_type','entity'), 'source_id': e.get('source_id',''),
        'target_type': e.get('target_type','entity'), 'target_id': e.get('target_id',''),
        'edge_type': e.get('edge_type','co_occurrence'), 'weight': e.get('weight', 1.0),
    })
db.insert_many('knowledge_edges', edge_rows)

for c in context.coreferences:
    db.insert('coreference_chains', {
        'representative': c.get('antecedent',''), 'mention_count': 1,
    })

for tp in context.topics:
    db.insert('topics', {
        'label': tp.get('label',''), 'top_terms': tp.get('top_terms',''),
        'sentence_count': tp.get('sentence_count', 0),
    })

stats = db.get_stats()
elapsed = time.time() - start
print(f"\n{'='*60}")
print(f"  TRAINING COMPLETE in {elapsed:.1f}s")
print(f"{'='*60}")
for table, count in stats.items():
    if count > 0: print(f"    {table}: {count:,}")
print(f"{'='*60}")
db.disconnect()
