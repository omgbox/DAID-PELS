"""
BookBot Fast Training Script for Pride and Prejudice

Usage:
    cd bookbot
    python train_pride.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path so we can import bookbot as a package
sys.path.insert(0, str(Path(__file__).parent.parent))

from bookbot.config import BOOK_PATH, DICTIONARY_PATH, DATABASE_PATH
from bookbot.database.db_manager import DBManager
from bookbot.pipeline_context import PipelineContext
from bookbot.core.tokenizer import Tokenizer
from bookbot.core.pos_tagger import POSTagger
from bookbot.core.ner_extractor import NERExtractor


def print_progress(step: str, current: int = None, total: int = None):
    """Print progress with optional progress bar."""
    if current is not None and total is not None:
        percent = current / total * 100
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f'\r  {step} [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)
    else:
        print(f'  {step}')


def fast_train():
    """Fast training with streaming and progress output."""
    print("\n" + "=" * 60)
    print("  BOOKBOT TRAINING - Pride and Prejudice")
    print("=" * 60)
    start_time = time.time()

    # Delete existing database to avoid UNIQUE constraint errors
    import os
    if os.path.exists(DATABASE_PATH):
        os.remove(DATABASE_PATH)
        print("  Deleted existing database")

    # Initialize database
    print("\n[1/5] Initializing database...")
    db = DBManager(DATABASE_PATH)
    db.connect()
    db.initialize_schema()

    # Load dictionary using streaming
    print("[2/5] Loading dictionary (streaming)...")
    dict_start = time.time()
    import csv
    with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        definitions = []
        for i, row in enumerate(reader):
            word = row.get('Word', '')
            pos = row.get('POS', '').strip('"')
            definition = row.get('Definition', '').strip('"')
            if word and definition:
                definitions.append({
                    'word': word,
                    'pos_canonical': pos,
                    'pos_original': pos,
                    'definition': definition,
                    'word_lower': word.lower(),
                    'word_length': len(word),
                    'def_word_count': len(definition.split()),
                    'entry_index': i,
                })
    
    # Insert definitions in batches
    batch_size = 50000
    for i in range(0, len(definitions), batch_size):
        batch = definitions[i:i + batch_size]
        db.insert_many('definitions', batch)
    print(f"\n  Loaded {len(definitions):,} entries in {time.time()-dict_start:.1f}s")

    # Load book
    print("[3/5] Loading Pride and Prejudice...")
    book_start = time.time()
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
        book_text = f.read()
    print(f"  Loaded {len(book_text):,} chars in {time.time()-book_start:.1f}s")

    # Create context
    context = PipelineContext()
    context.raw_text = book_text
    
    # Clean text: normalize whitespace but keep paragraph structure
    import re
    cleaned = book_text
    # Collapse runs of blank lines to a single double-newline (paragraph break)
    cleaned = re.sub(r'\n[ \t]*\n[ \t]*\n+', '\n\n', cleaned)
    # Collapse multiple spaces within a line
    cleaned = re.sub(r'[^\S\n]+', ' ', cleaned)
    # Collapse multiple newlines (but keep paragraph breaks as single \n)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    cleaned = cleaned.strip()
    context.normalized_text = cleaned

    # Tokenize
    print("[4/5] Tokenizing...")
    tok_start = time.time()
    tokenizer = Tokenizer()
    result = tokenizer.process(context)
    context.update(result)
    sentences = context.sentences
    print(f"  Tokenized {len(sentences)} sentences in {time.time()-tok_start:.1f}s")

    # POS Tag + NER
    print("[5/5] POS tagging + NER...")
    tag_start = time.time()
    pos_tagger = POSTagger()
    result = pos_tagger.process(context)
    context.update(result)
    print(f"  POS tagged in {time.time()-tag_start:.1f}s")

    ner_start = time.time()
    ner = NERExtractor()
    result = ner.process(context)
    context.update(result)
    print(f"  NER: {len(context.entities)} entities in {time.time()-ner_start:.1f}s")

    # Save to database
    print("\nSaving to database...")
    save_start = time.time()

    # Save sentences and tokens in batches
    batch_size = 500
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        sent_rows = []
        token_rows = []
        for sent in batch:
            sent_rows.append({
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
                token_rows.append({
                    'sentence_id': sent.get('sentence_id'),
                    'position': token.get('position', 0),
                    'token': token.get('token', ''),
                    'token_lower': token.get('token', '').lower(),
                    'pos_tag': token.get('pos_tag', ''),
                    'is_punctuation': 1 if token.get('is_punctuation') else 0,
                    'is_stopword': 1 if token.get('is_stopword') else 0,
                })
        db.insert_many('sentences', sent_rows)
        db.insert_many('sentence_tokens', token_rows)
        print_progress("Saving sentences", min(i+batch_size, len(sentences)), len(sentences))

    # Save entities
    print()
    for entity in context.entities:
        db.insert('entities', {
            'entity_id': entity.get('entity_id'),
            'canonical_name': entity.get('canonical_name'),
            'entity_type': entity.get('entity_type'),
            'frequency': entity.get('frequency', 1),
            'centrality': entity.get('centrality', 0.0),
        })
    print(f"  Saved {len(context.entities)} entities")

    # Print final statistics
    stats = db.get_stats()
    total_time = time.time() - start_time

    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Total time: {total_time:.1f}s")
    print("\n  Database Statistics:")
    for table, count in stats.items():
        if count > 0:
            print(f"    {table}: {count:,}")
    print("=" * 60)

    db.disconnect()


if __name__ == '__main__':
    fast_train()
