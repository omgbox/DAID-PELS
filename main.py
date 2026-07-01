"""
BookBot Main Entry Point
Command-line interface for training and querying with live visualization.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

from .config import BOOK_PATH, DICTIONARY_PATH, DATABASE_PATH
from .logging_config import setup_logging
from .database.db_manager import DBManager
from .pipeline import Pipeline
from .pipeline_context import PipelineContext
from .training.visualizer import TrainingVisualizer

# Core modules
from .core.ocr_normalizer import OCRNormalizer
from .core.tokenizer import Tokenizer
from .core.pos_tagger import POSTagger
from .core.pos_guesser import POSGuesser
from .core.definition_linker import DefinitionLinker
from .core.ner_extractor import NERExtractor
from .core.svo_extractor import SVOExtractor
from .core.coreference import Coreference
from .core.entity_graph import EntityGraph
from .core.idiom_detector import IdiomDetector
from .core.metaphor_detector import MetaphorDetector
from .core.temporal_reasoner import TemporalReasoner
from .core.topic_modeler import TopicModeler
from .core.idf_builder import IDFBuilder

# Training modules
from .training.convergence_tracker import ConvergenceTracker

# Query modules
from .query.query_classifier import QueryClassifier
from .query.bm25_engine import BM25Engine
from .query.trilateral_bm25 import TrilateralBM25Engine
from .query.answer_engine import AnswerEngine
from .query.confidence_scorer import ConfidenceScorer
from .query.response_formatter import ResponseFormatter
from .query.conversation_context import ConversationContext
from .query.conversation_memory import ConversationMemory
from .query.contextual_rewriter import ContextualQueryRewriter


def create_pipeline(config: dict = None) -> Pipeline:
    """
    Create and configure the BookBot pipeline.

    Args:
        config: Configuration dict

    Returns:
        Configured Pipeline
    """
    pipeline = Pipeline(config)

    # Register core modules
    pipeline.register_module('ocr_normalizer', OCRNormalizer(config))
    pipeline.register_module('tokenizer', Tokenizer(config))
    pipeline.register_module('pos_tagger', POSTagger(config))
    pipeline.register_module('pos_guesser', POSGuesser(config))
    pipeline.register_module('definition_linker', DefinitionLinker(config))
    pipeline.register_module('idf_builder', IDFBuilder(config))
    pipeline.register_module('ner_extractor', NERExtractor(config))
    pipeline.register_module('svo_extractor', SVOExtractor(config))
    pipeline.register_module('coreference', Coreference(config))
    pipeline.register_module('entity_graph', EntityGraph(config))
    pipeline.register_module('idiom_detector', IdiomDetector(config))
    pipeline.register_module('metaphor_detector', MetaphorDetector(config))
    pipeline.register_module('temporal_reasoner', TemporalReasoner(config))
    pipeline.register_module('topic_modeler', TopicModeler(config))

    # Register training modules
    pipeline.register_module('convergence_tracker', ConvergenceTracker(config))

    # Register query modules
    pipeline.register_module('query_classifier', QueryClassifier(config))
    pipeline.register_module('contextual_rewriter', ContextualQueryRewriter(config))
    pipeline.register_module('trilateral_bm25', TrilateralBM25Engine(config))
    pipeline.register_module('bm25_engine', BM25Engine(config))
    pipeline.register_module('answer_engine', AnswerEngine(config))
    pipeline.register_module('confidence_scorer', ConfidenceScorer(config))
    pipeline.register_module('response_formatter', ResponseFormatter(config))

    return pipeline


def train(book_path: str, dict_path: str, db_path: str, max_passes: int = None):
    """
    Train BookBot on a book with live visualization.

    Args:
        book_path: Path to book file
        dict_path: Path to dictionary file
        db_path: Path to database file
        max_passes: Maximum training passes
    """
    logger = setup_logging('bookbot.train')
    viz = TrainingVisualizer()

    # Initialize
    viz.start(max_passes or 3)
    viz.update_step("Initializing...")

    # Delete existing database to avoid UNIQUE constraint errors
    import os
    if os.path.exists(db_path):
        os.remove(db_path)
        viz.update_step("Deleted existing database")

    # Initialize database
    viz.update_step("Connecting to database...")
    db = DBManager(db_path)
    db.connect()
    db.initialize_schema()

    # Load dictionary
    viz.update_step("Loading dictionary...")
    import csv
    with open(dict_path, 'r', encoding='utf-8') as f:
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

    viz.update_stats({'Dictionary entries': len(definitions)})
    viz.update_step(f"Loaded {len(definitions):,} dictionary entries")

    # Insert definitions in batches
    viz.update_step("Inserting definitions into database...")
    batch_size = 10000
    for i in range(0, len(definitions), batch_size):
        batch = definitions[i:i + batch_size]
        db.insert_many('definitions', batch)
        progress = min(1.0, (i + batch_size) / len(definitions))
        viz.update_step(f"Inserting definitions: {i + len(batch)}/{len(definitions)}", progress)

    # Load book
    viz.update_step("Loading book...")
    with open(book_path, 'r', encoding='utf-8') as f:
        book_text = f.read()

    word_count = len(book_text.split())
    line_count = book_text.count('\n')
    viz.update_stats({
        'Book characters': len(book_text),
        'Book words': word_count,
        'Book lines': line_count,
    })
    viz.update_step(f"Loaded {word_count:,} words from book")

    # Create pipeline
    config = {}
    if max_passes:
        config['convergence'] = {'max_passes': max_passes}

    pipeline = create_pipeline(config)

    # Set database on all modules
    for module_name, module in pipeline.modules.items():
        module.db = db

    # Create context
    context = PipelineContext()
    context.raw_text = book_text

    # Run training passes
    total_passes = max_passes or 3
    for pass_num in range(1, total_passes + 1):
        pass_start = time.time()
        viz.start_pass(pass_num, f"Training Pass {pass_num}")

        # Pass 0: OCR Normalization
        viz.update_step("OCR Normalization...")
        ocr = pipeline.modules.get('ocr_normalizer')
        if ocr:
            result = ocr.process(context)
            context.update(result)
        viz.update_stats({'OCR corrections': ocr.correction_count if ocr else 0})

        # Pass 1: Tokenization
        viz.update_step("Tokenizing sentences...")
        tokenizer = pipeline.modules.get('tokenizer')
        if tokenizer:
            result = tokenizer.process(context)
            context.update(result)
        viz.update_stats({'Sentences': len(context.sentences)})

        # Pass 1: POS Tagging
        viz.update_step("POS tagging...")
        pos_tagger = pipeline.modules.get('pos_tagger')
        if pos_tagger:
            result = pos_tagger.process(context)
            context.update(result)

        # Pass 1: IDF Building
        viz.update_step("Computing IDF scores...")
        idf = pipeline.modules.get('idf_builder')
        if idf:
            result = idf.process(context)
            context.update(result)
        viz.update_stats({'Unique words': len(context.vocabulary) if context.vocabulary else 0})

        # Pass 2: NER
        viz.update_step("Extracting named entities...")
        ner = pipeline.modules.get('ner_extractor')
        if ner:
            result = ner.process(context)
            context.update(result)
        viz.update_stats({'Entities': len(context.entities)})

        # Pass 2: SVO Extraction
        viz.update_step("Extracting SVO triples...")
        svo = pipeline.modules.get('svo_extractor')
        if svo:
            result = svo.process(context)
            context.update(result)
        viz.update_stats({'SVO triples': len(context.svo_triples)})

        # Pass 2: Entity Graph
        viz.update_step("Building entity graph...")
        graph = pipeline.modules.get('entity_graph')
        if graph:
            result = graph.process(context)
            context.update(result)
        viz.update_stats({'Knowledge edges': len(context.knowledge_edges)})

        # Pass 3: Coreference
        viz.update_step("Resolving coreferences...")
        coref = pipeline.modules.get('coreference')
        if coref:
            result = coref.process(context)
            context.update(result)
        viz.update_stats({'Coreferences': len(context.coreferences)})

        # Pass 3: Topic Modeling
        viz.update_step("Building topic clusters...")
        topic = pipeline.modules.get('topic_modeler')
        if topic:
            result = topic.process(context)
            context.update(result)
        viz.update_stats({'Topics': len(context.topics)})

        # End pass
        pass_duration = time.time() - pass_start
        viz.end_pass(pass_num, pass_duration)
        viz.print_stats()

        # Check convergence
        if pass_num > 1:
            conv = pipeline.modules.get('convergence_tracker')
            if conv:
                result = conv.process(context)
                if result.get('converged'):
                    viz.update_step("CONVERGED - No new relationships found")
                    break

    # Save to database
    viz.update_step("Saving to database...")

    # Save sentences
    if context.sentences:
        viz.update_step(f"Saving {len(context.sentences)} sentences...")
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
            # Save individual tokens
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

    # Save entities
    if context.entities:
        viz.update_step(f"Saving {len(context.entities)} entities...")
        for entity in context.entities:
            db.insert('entities', {
                'entity_id': entity.get('entity_id'),
                'canonical_name': entity.get('canonical_name'),
                'entity_type': entity.get('entity_type'),
                'frequency': entity.get('frequency', 1),
                'centrality': entity.get('centrality', 0.0),
            })

    # Save SVO triples
    if context.svo_triples:
        viz.update_step(f"Saving {len(context.svo_triples)} SVO triples...")
        for triple in context.svo_triples:
            db.insert('svo_triples', {
                'subject': triple.get('subject'),
                'verb': triple.get('verb'),
                'object': triple.get('object'),
                'sentence_id': triple.get('sentence_id'),
                'confidence': triple.get('confidence', 0.5),
            })

    # Print final statistics
    stats = db.get_stats()
    viz.update_step("Training complete!")
    print("\nDatabase Statistics:")
    for table, count in stats.items():
        if count > 0:
            print(f"  {table}: {count:,}")

    db.disconnect()
    viz.end()


def query(db_path: str, single_query: str = None):
    """
    Query BookBot with conversation memory.

    Args:
        db_path: Path to database file
        single_query: Single query to answer (None for interactive mode)
    """
    logger = setup_logging('bookbot.query')

    # Initialize database
    db = DBManager(db_path)
    db.connect()

    # Create pipeline
    pipeline = create_pipeline()

    # Set database on all modules
    for module_name, module in pipeline.modules.items():
        module.db = db

    # Load context from database
    context = PipelineContext()

    # Load sentences
    sentences = db.execute("SELECT sentence_id, chapter_id, paragraph_id, raw_text FROM sentences")
    context.sentences = [
        {
            'sentence_id': row[0],
            'chapter_id': row[1],
            'paragraph_id': row[2],
            'text': row[3],
            'tokens': [],
        }
        for row in sentences
    ]

    # Load tokens for each sentence
    for sent in context.sentences:
        token_rows = db.execute(
            "SELECT position, token, token_lower, pos_tag, is_punctuation, is_stopword "
            "FROM sentence_tokens WHERE sentence_id = ? ORDER BY position",
            (sent['sentence_id'],)
        )
        sent['tokens'] = [
            {
                'position': r[0],
                'token': r[1],
                'token_lower': r[2],
                'pos_tag': r[3],
                'is_punctuation': bool(r[4]),
                'is_stopword': bool(r[5]),
            }
            for r in token_rows
        ]

    # Initialize conversation memory
    conversation_memory = ConversationMemory(max_history=10)

    print("\n" + "=" * 60)
    print("  BOOKBOT - Pride and Prejudice")
    print("=" * 60)
    print(f"  Loaded {len(context.sentences):,} sentences")
    print("  Type your questions, or 'quit' to exit.")
    print("  Try: 'Who is Elizabeth?', 'Tell me more about Darcy'")
    print("=" * 60)

    if single_query:
        # Single query mode
        result = pipeline.run_query(single_query, context, conversation_memory)
        _print_response(result)
    else:
        # Interactive mode
        print("\n")

        while True:
            try:
                user_input = input("> ").strip()
                if user_input.lower() in ('quit', 'exit', 'q'):
                    print("\nGoodbye!")
                    break
                if not user_input:
                    continue

                result = pipeline.run_query(user_input, context, conversation_memory)
                _print_response(result)
                print()

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"Error: {e}")

    db.disconnect()


def _print_response(result: dict):
    """Print formatted response."""
    print("\n" + result.get('answer', 'No answer found.'))
    
    if result.get('sources'):
        print("\n[SOURCES]")
        for source in result['sources'][:3]:
            print(f"  - {source}")
    
    if result.get('suggested_followups'):
        print("\n[SUGGESTIONS]")
        for followup in result['suggested_followups'][:3]:
            print(f"  - {followup}")


def stats(db_path: str):
    """
    Show database statistics.

    Args:
        db_path: Path to database file
    """
    db = DBManager(db_path)
    db.connect()

    stats = db.get_stats()
    print("\n" + "=" * 40)
    print("  DATABASE STATISTICS")
    print("=" * 40)
    for table, count in stats.items():
        if count > 0:
            print(f"  {table}: {count:,}")
    print("=" * 40)

    db.disconnect()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='BookBot - Book-trained chatbot')
    subparsers = parser.add_subparsers(dest='command', help='Command')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train on a book')
    train_parser.add_argument('--book', default=BOOK_PATH, help='Path to book file')
    train_parser.add_argument('--dict', default=DICTIONARY_PATH, help='Path to dictionary file')
    train_parser.add_argument('--db', default=DATABASE_PATH, help='Path to database file')
    train_parser.add_argument('--passes', type=int, default=3, help='Maximum training passes')

    # Query command
    query_parser = subparsers.add_parser('query', help='Query BookBot')
    query_parser.add_argument('--db', default=DATABASE_PATH, help='Path to database file')
    query_parser.add_argument('--single', default=None, help='Single query to answer')

    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    stats_parser.add_argument('--db', default=DATABASE_PATH, help='Path to database file')

    args = parser.parse_args()

    if args.command == 'train':
        train(args.book, args.dict, args.db, args.passes)
    elif args.command == 'query':
        query(args.db, args.single)
    elif args.command == 'stats':
        stats(args.db)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
