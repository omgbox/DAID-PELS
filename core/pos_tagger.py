"""
BookBot POS Tagger
Part-of-speech tagging using NLTK (optimized for speed).
"""

import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.pos_tagger')

# Try to import NLTK
try:
    import nltk
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


class POSTagger(BaseModule):
    """Part-of-speech tagging module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.tagger = None
        self._cache = {}  # Cache POS tags for known words

    def process(self, input_data) -> dict:
        """
        Process input data and return POS-tagged sentences.

        Args:
            input_data: PipelineContext or dict with 'sentences' key

        Returns:
            Dict with 'sentences' key (updated with POS tags)
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Initialize tagger
        if NLTK_AVAILABLE:
            self._initialize_tagger()

        # Collect all words for batch tagging
        all_words = []
        word_map = []  # (sent_idx, token_idx)

        for sent_idx, sent in enumerate(sentences):
            tokens = sent.get('tokens', [])
            for token_idx, token in enumerate(tokens):
                if not token.get('is_punctuation', False):
                    word = token['token']
                    # Check cache first
                    if word in self._cache:
                        token['pos_tag'] = self._cache[word]
                    else:
                        all_words.append(word)
                        word_map.append((sent_idx, token_idx))

        # Batch tag all uncached words
        if all_words and NLTK_AVAILABLE:
            try:
                tagged = nltk.pos_tag(all_words)
                # Store in cache and map back to tokens
                for i, (word, tag) in enumerate(tagged):
                    self._cache[word] = tag
                    sent_idx, token_idx = word_map[i]
                    sentences[sent_idx]['tokens'][token_idx]['pos_tag'] = tag
            except Exception as e:
                self.logger.warning(f"POS tagging failed: {e}")
                # Default to NN
                for sent_idx, token_idx in word_map:
                    sentences[sent_idx]['tokens'][token_idx]['pos_tag'] = 'NN'

        # Set punctuation tags
        for sent in sentences:
            for token in sent.get('tokens', []):
                if token.get('is_punctuation', False):
                    token['pos_tag'] = '.'
                elif 'pos_tag' not in token:
                    token['pos_tag'] = 'NN'

            # Update pos_tags string
            sent['pos_tags'] = ' '.join(t.get('pos_tag', '') for t in sent.get('tokens', []))

        self._initialized = True
        self.logger.info(f"POS tagged {len(sentences)} sentences ({len(self._cache)} cached words)")

        return {'sentences': sentences}

    def _initialize_tagger(self):
        """Initialize NLTK POS tagger."""
        try:
            # Try to load the tagger
            model = self.get_config('pos_tagger_model', 'averaged_perceptron_tagger')
            nltk.data.find(f'taggers/{model}')
        except LookupError:
            # Download if not available
            try:
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('punkt', quiet=True)
            except Exception as e:
                self.logger.warning(f"Could not download NLTK data: {e}")

    def get_pos_distribution(self, sentences: List[Dict]) -> Dict[str, Dict[str, int]]:
        """
        Get POS distribution for each word.

        Args:
            sentences: List of sentence dicts

        Returns:
            Dict mapping words to POS distributions
        """
        distributions = {}

        for sent in sentences:
            for token in sent.get('tokens', []):
                word = token.get('token_lower', '')
                pos = token.get('pos_tag', 'NN')

                if word not in distributions:
                    distributions[word] = {}

                distributions[word][pos] = distributions[word].get(pos, 0) + 1

        return distributions
