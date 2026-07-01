"""
BookBot Stream Processor
Line-by-line streaming for memory-efficient processing.
"""

import csv
import logging
from typing import Dict, List, Generator, Optional

logger = logging.getLogger('bookbot.stream_processor')


def stream_dictionary(dict_path: str, batch_size: int = 1000) -> Generator[List[Dict], None, None]:
    """
    Stream dictionary file line by line.

    Args:
        dict_path: Path to dictionary CSV file
        batch_size: Number of entries to yield at a time

    Yields:
        Batches of dictionary entries
    """
    batch = []
    with open(dict_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            word = row.get('Word', '')
            pos = row.get('POS', '').strip('"')
            definition = row.get('Definition', '').strip('"')

            if word and definition:
                batch.append({
                    'word': word,
                    'pos_canonical': pos,
                    'pos_original': pos,
                    'definition': definition,
                    'word_lower': word.lower(),
                    'word_length': len(word),
                    'def_word_count': len(definition.split()),
                    'entry_index': i,
                })

            if len(batch) >= batch_size:
                yield batch
                batch = []

    # Yield remaining entries
    if batch:
        yield batch


def stream_book_lines(book_path: str) -> Generator[str, None, None]:
    """
    Stream book file line by line.

    Args:
        book_path: Path to book text file

    Yields:
        Lines from the book
    """
    with open(book_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield line.rstrip('\n')


def stream_book_chunks(book_path: str, chunk_size: int = 10000) -> Generator[str, None, None]:
    """
    Stream book file in chunks.

    Args:
        book_path: Path to book text file
        chunk_size: Number of characters per chunk

    Yields:
        Text chunks
    """
    with open(book_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def load_dictionary_streaming(dict_path: str, db_manager, batch_size: int = 10000) -> int:
    """
    Load dictionary into database using streaming.

    Args:
        dict_path: Path to dictionary CSV file
        db_manager: Database manager instance
        batch_size: Number of entries per batch insert

    Returns:
        Total number of entries loaded
    """
    total = 0
    for batch in stream_dictionary(dict_path, batch_size):
        db_manager.insert_many('definitions', batch)
        total += len(batch)
        logger.info(f"Loaded {total:,} dictionary entries...")

    return total


def load_book_streaming(book_path: str) -> str:
    """
    Load book into memory (for small books) or return path for streaming.

    Args:
        book_path: Path to book text file

    Returns:
        Book text content
    """
    with open(book_path, 'r', encoding='utf-8') as f:
        return f.read()


def process_sentences_streaming(sentences: List[Dict], batch_size: int = 100,
                                 processor=None) -> List[Dict]:
    """
    Process sentences in batches.

    Args:
        sentences: List of sentence dicts
        batch_size: Number of sentences per batch
        processor: Processing function to apply

    Returns:
        Processed sentences
    """
    if processor is None:
        return sentences

    processed = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        result = processor(batch)
        processed.extend(result)
        logger.info(f"Processed {min(i + batch_size, len(sentences))}/{len(sentences)} sentences")

    return processed
