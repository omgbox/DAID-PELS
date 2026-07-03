"""
BookBot IDF Builder
Computes IDF scores from combined corpus.
"""

import logging
import math
from typing import Dict, List
from collections import defaultdict

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.idf_builder')


class IDFBuilder(BaseModule):
    """IDF computation module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and compute IDF scores.

        Args:
            input_data: PipelineContext or dict with 'sentences' key

        Returns:
            Dict with 'vocabulary' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Compute document frequency
        doc_freq = defaultdict(int)
        word_freq = defaultdict(int)
        total_words = 0

        for sent in sentences:
            words_in_sent = set()
            for token in sent.get('tokens', []):
                if not token.get('is_punctuation'):
                    word = token.get('token_lower', '')
                    words_in_sent.add(word)
                    word_freq[word] += 1
                    total_words += 1

            for word in words_in_sent:
                doc_freq[word] += 1

        # Compute IDF
        num_sentences = len(sentences)
        vocabulary = {}

        for word in doc_freq:
            idf = math.log((num_sentences + 1) / (doc_freq[word] + 1)) + 1
            vocabulary[word] = {
                'word': word,
                'frequency': word_freq[word],
                'document_freq': doc_freq[word],
                'idf': idf,
            }

        # Store in database
        if self.db:
            self._store_vocabulary(vocabulary)

        self._initialized = True
        self.logger.info(f"Computed IDF for {len(vocabulary)} words")

        return {'vocabulary': vocabulary}

    def _store_vocabulary(self, vocabulary: Dict[str, Dict]):
        """
        Store vocabulary in database.

        Args:
            vocabulary: Dict mapping words to vocabulary data
        """
        try:
            for word, data in vocabulary.items():
                self.db.insert('vocabulary', {
                    'word': word,
                    'frequency': data['frequency'],
                    'document_freq': data['document_freq'],
                    'idf': data['idf'],
                })
        except Exception as e:
            self.logger.warning(f"Could not store vocabulary: {e}")
