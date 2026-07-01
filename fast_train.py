"""
BookBot Fast Training Script
Optimized for speed with streaming and progress output.
"""

import sys
import time
sys.path.insert(0, r'C:\projects')

from bookbot.config import BOOK_PATH, DICTIONARY_PATH, DATABASE_PATH
from bookbot.database.db_manager import DBManager
from bookbot.stream_processor import stream_dictionary, load_dictionary_streaming
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
    print("  BOOKBOT FAST TRAINING")
    print("=" * 60)
    start_time = time.time()

    # Initialize database
    print("\n[1/5] Initializing database...")
    db = DBManager(DATABASE_PATH)
    db.connect()
    db.initialize_schema()

    # Load dictionary using streaming
    print("[2/5] Loading dictionary (streaming)...")
    dict_start = time.time()
    total_entries = load_dictionary_streaming(DICTIONARY_PATH, db, batch_size=50000)
    print(f"\n  Loaded {total_entries:,} entries in {time.time()-dict_start:.1f}s")

    # Load book
    print("[3/5] Loading book...")
    book_start = time.time()
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
        book_text = f.read()
    print(f"  Loaded {len(book_text):,} chars in {time.time()-book_start:.1f}s")

    # Create context
    context = PipelineContext()
    context.raw_text = book_text
    context.normalized_text = book_text

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

    # Save sentences in batches
    batch_size = 500
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i+batch_size]
        for sent in batch:
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
