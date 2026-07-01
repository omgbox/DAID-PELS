"""
BookBot Fast Training - Part 2
Skip dictionary loading, just do tokenization/NER.
"""

import sys
import time
sys.path.insert(0, r'C:\projects')

from bookbot.config import BOOK_PATH, DATABASE_PATH
from bookbot.database.db_manager import DBManager
from bookbot.pipeline_context import PipelineContext
from bookbot.core.tokenizer import Tokenizer
from bookbot.core.pos_tagger import POSTagger
from bookbot.core.ner_extractor import NERExtractor


def fast_train_part2():
    """Fast training part 2 - tokenization and NER."""
    print("\n" + "=" * 60)
    print("  BOOKBOT FAST TRAINING - PART 2")
    print("=" * 60)
    start_time = time.time()

    # Initialize database
    print("\n[1/4] Connecting to database...")
    db = DBManager(DATABASE_PATH)
    db.connect()

    # Load book
    print("[2/4] Loading book...")
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
        book_text = f.read()
    print(f"  Loaded {len(book_text):,} chars")

    # Create context
    context = PipelineContext()
    context.raw_text = book_text
    context.normalized_text = book_text

    # Tokenize
    print("[3/4] Tokenizing...")
    tok_start = time.time()
    tokenizer = Tokenizer()
    result = tokenizer.process(context)
    context.update(result)
    sentences = context.sentences
    print(f"  Tokenized {len(sentences)} sentences in {time.time()-tok_start:.1f}s")

    # POS Tag + NER
    print("[4/4] POS tagging + NER...")
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

    # Save sentences
    for i, sent in enumerate(sentences):
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
        if (i + 1) % 100 == 0:
            print(f'\r  Saved {i+1}/{len(sentences)} sentences', end='', flush=True)

    print(f'\n  Saved {len(sentences)} sentences in {time.time()-save_start:.1f}s')

    # Save entities
    for entity in context.entities:
        db.insert('entities', {
            'entity_id': entity.get('entity_id'),
            'canonical_name': entity.get('canonical_name'),
            'entity_type': entity.get('entity_type'),
            'frequency': entity.get('frequency', 1),
            'centrality': entity.get('centrality', 0.0),
        })
    print(f'  Saved {len(context.entities)} entities')

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
    fast_train_part2()
