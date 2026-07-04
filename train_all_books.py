"""
Train BookBot on ALL downloaded books.
Combines all books into one training corpus for comprehensive knowledge.
"""

import sys
import os
import time
import glob
import re
import csv

sys.path.insert(0, r'C:\projects')

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


def print_progress(step, current=None, total=None):
    """Print progress with optional progress bar."""
    if current is not None and total is not None:
        percent = current / total * 100
        bar_len = 30
        filled = int(bar_len * current / total)
        bar = '#' * filled + '-' * (bar_len - filled)
        print(f'\r  {step} [{bar}] {percent:.1f}% ({current}/{total})', end='', flush=True)
    else:
        print(f'  {step}')


def train_on_book(book_path, db, pipeline_modules, book_num, total_books):
    """Train on a single book."""
    book_name = os.path.basename(book_path)
    print(f'\n[{book_num}/{total_books}] {book_name}')
    print('-' * 60)

    # Load book
    print('  Loading book...', end=' ', flush=True)
    with open(book_path, 'r', encoding='utf-8') as f:
        book_text = f.read()
    print(f'{len(book_text):,} chars')

    # Normalize text
    context = PipelineContext()
    context.raw_text = normalize_text(book_text)

    # Clean text
    cleaned = re.sub(r'\s+', ' ', context.raw_text).strip()
    context.normalized_text = cleaned

    # Tokenize
    print('  Tokenizing...', end=' ', flush=True)
    tokenizer = pipeline_modules['tokenizer']
    result = tokenizer.process(context)
    context.update(result)
    sentences = context.sentences
    print(f'{len(sentences)} sentences')

    # POS Tag - with progress
    print('  POS tagging...', end=' ', flush=True)
    pos_tagger = pipeline_modules['pos_tagger']
    result = pos_tagger.process(context)
    context.update(result)
    print(f'done ({len(sentences)} tagged)')

    # NER - with progress
    print('  NER...', end=' ', flush=True)
    ner = pipeline_modules['ner_extractor']
    result = ner.process(context)
    context.update(result)
    print(f'{len(context.entities)} entities')

    # SVO Extraction - with progress
    print('  SVO extraction...', end=' ', flush=True)
    svo = pipeline_modules['svo_extractor']
    long_sents = [s for s in context.sentences if len(s.get('tokens', [])) >= 5]
    original_sents = context.sentences
    context.sentences = long_sents
    result = svo.process(context)
    context.update(result)
    context.sentences = original_sents
    print(f'{len(context.svo_triples)} triples')

    # Entity Graph - with progress
    print('  Entity graph...', end=' ', flush=True)
    graph = pipeline_modules['entity_graph']
    result = graph.process(context)
    context.update(result)
    print(f'{len(context.knowledge_edges)} edges')

    # Coreference - with progress
    print('  Coreference...', end=' ', flush=True)
    coref = pipeline_modules['coreference']
    result = coref.process(context)
    context.update(result)
    print(f'{len(context.coreferences)} chains')

    # Topic Modeling
    print('  Topics...', end=' ', flush=True)
    topics = pipeline_modules['topic_modeler']
    result = topics.process(context)
    context.update(result)
    print(f'{len(context.topics)} topics')

    # Save to database
    print('  Saving to database...')
    save_start = time.time()

    # Save sentences and tokens in batches with progress bar
    batch_size = 500
    total_batches = (len(sentences) + batch_size - 1) // batch_size
    for batch_num, i in enumerate(range(0, len(sentences), batch_size), 1):
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
        
        # Progress bar
        progress = batch_num / total_batches * 100
        bar_len = 30
        filled = int(bar_len * batch_num / total_batches)
        bar = '#' * filled + '-' * (bar_len - filled)
        elapsed = time.time() - save_start
        eta = elapsed / batch_num * (total_batches - batch_num) if batch_num > 0 else 0
        sys.stdout.write(f'\r    Saving: [{bar}] {progress:.0f}% | {elapsed:.0f}s | ETA: {eta:.0f}s')
        sys.stdout.flush()
    print()

    # Save entities
    print(f'    Saving {len(context.entities)} entities...', end=' ', flush=True)
    for entity in context.entities:
        # Use INSERT OR IGNORE to avoid duplicates, let SQLite auto-generate entity_id
        db.execute(
            "INSERT OR IGNORE INTO entities (canonical_name, entity_type, frequency, centrality) "
            "VALUES (?, ?, ?, ?)",
            (entity.get('canonical_name'), entity.get('entity_type'),
             entity.get('frequency', 1), entity.get('centrality', 0.0))
        )
    print('done')

    # Save SVO triples
    if context.svo_triples:
        print(f'    Saving {len(context.svo_triples)} SVO triples...', end=' ', flush=True)
        svo_rows = []
        for triple in context.svo_triples:
            svo_rows.append({
                'subject': triple.get('subject', ''),
                'verb': triple.get('verb', ''),
                'object': triple.get('object', ''),
                'sentence_id': triple.get('sentence_id', 0),
                'confidence': triple.get('confidence', 0.5),
                'passive': 1 if triple.get('passive') else 0,
            })
        db.insert_many('svo_triples', svo_rows)
        print('done')

    # Save knowledge edges
    if context.knowledge_edges:
        print(f'    Saving {len(context.knowledge_edges)} knowledge edges...', end=' ', flush=True)
        edge_rows = []
        for edge in context.knowledge_edges:
            edge_rows.append({
                'source_type': edge.get('source_type', 'entity'),
                'source_id': edge.get('source_id', ''),
                'target_type': edge.get('target_type', 'entity'),
                'target_id': edge.get('target_id', ''),
                'edge_type': edge.get('edge_type', 'co_occurrence'),
                'weight': edge.get('weight', 1.0),
            })
        db.insert_many('knowledge_edges', edge_rows)
        print('done')

    # Save coreference chains
    if context.coreferences:
        print(f'    Saving {len(context.coreferences)} coreference chains...', end=' ', flush=True)
        chain_rows = []
        for chain in context.coreferences:
            chain_rows.append({
                'representative': chain.get('antecedent', ''),
                'mention_count': 1,
            })
        db.insert_many('coreference_chains', chain_rows)
        print('done')

    # Save topics
    if context.topics:
        print(f'    Saving {len(context.topics)} topics...', end=' ', flush=True)
        topic_rows = []
        for topic in context.topics:
            topic_rows.append({
                'label': topic.get('label', ''),
                'top_terms': topic.get('top_terms', ''),
                'sentence_count': topic.get('sentence_count', 0),
            })
        db.insert_many('topics', topic_rows)
        print('done')

    save_time = time.time() - save_start
    print(f'    Save complete in {save_time:.1f}s')

    return {
        'book': book_name,
        'chars': len(book_text),
        'sentences': len(sentences),
        'entities': len(context.entities),
        'svo_triples': len(context.svo_triples),
        'edges': len(context.knowledge_edges),
    }


def main():
    print('=' * 60)
    print('  BOOKBOT - TRAIN ON ALL BOOKS')
    print('=' * 60)

    # Find all books
    books_dir = 'books'
    book_files = sorted(glob.glob(os.path.join(books_dir, '*.txt')))

    # Filter out already-trained book
    book_files = [f for f in book_files if 'pride_and_prejudice_clean' not in f]

    print(f'\nFound {len(book_files)} books to train on')

    # Database path
    db_path = DATABASE_PATH

    # Delete existing database (with retry)
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print('Deleted existing database')
        except PermissionError:
            # Try alternative: use new database name
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            db_path = db_path.replace('.db', f'_{timestamp}.db')
            print(f'Could not delete existing database, using: {db_path}')

    # Initialize database
    print('\nInitializing database...')
    db = DBManager(db_path)
    db.connect()
    db.initialize_schema()

    # Initialize pipeline modules
    print('Initializing NLP pipeline...')
    config = {}
    modules = {
        'tokenizer': Tokenizer(config),
        'pos_tagger': POSTagger(config),
        'ner_extractor': NERExtractor(config),
        'svo_extractor': SVOExtractor(config),
        'entity_graph': EntityGraph(config),
        'coreference': Coreference(config),
        'topic_modeler': TopicModeler(config),
    }

    # Load dictionary
    print('Loading dictionary...')
    dict_path = 'combined_english_dictionary.csv'
    with open(dict_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        definitions = []
        for i, row in enumerate(reader):
            word = row.get('word', '')
            pos = row.get('pos', '').strip('"')
            definition = row.get('definition', '').strip('"')
            language = row.get('language', 'english')
            if word and definition and language == 'english':
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

    # Insert definitions
    batch_size = 50000
    for i in range(0, len(definitions), batch_size):
        batch = definitions[i:i + batch_size]
        db.insert_many('definitions', batch)
    print(f'  Loaded {len(definitions):,} dictionary entries')

    # Train on each book
    total_start = time.time()
    stats = []

    for i, book_path in enumerate(book_files, 1):
        try:
            stat = train_on_book(book_path, db, modules, i, len(book_files))
            stats.append(stat)
        except Exception as e:
            print(f'\n  ERROR: {e}')
            continue

    # Print summary
    total_time = time.time() - total_start
    print('\n' + '=' * 60)
    print('  TRAINING COMPLETE')
    print('=' * 60)
    print(f'\n  Total time: {total_time:.0f}s ({total_time/60:.1f} minutes)')
    print(f'  Books trained: {len(stats)}')
    print()

    # Print database stats
    db_stats = db.get_stats()
    print('  Database Statistics:')
    for table, count in db_stats.items():
        if count > 0:
            print(f'    {table}: {count:,}')

    print()

    # Print per-book stats
    print('  Per-Book Statistics:')
    print(f'  {"Book":<40} {"Sentences":>10} {"Entities":>10} {"SVO":>10}')
    print('  ' + '-' * 70)
    for stat in stats:
        book_name = stat['book'][:40]
        print(f'  {book_name:<40} {stat["sentences"]:>10,} {stat["entities"]:>10,} {stat["svo_triples"]:>10,}')

    db.disconnect()


if __name__ == '__main__':
    main()
