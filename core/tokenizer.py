"""
BookBot Tokenizer
Sentence and word tokenization using pysbd.
"""

import re
import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.tokenizer')

# Try to import pysbd
try:
    import pysbd
    PYSBD_AVAILABLE = True
except ImportError:
    PYSBD_AVAILABLE = False


class Tokenizer(BaseModule):
    """Sentence and word tokenization module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.segmenter = None

    def process(self, input_data) -> dict:
        """
        Process input data and return tokenized sentences.

        Args:
            input_data: PipelineContext or dict with 'normalized_text' key

        Returns:
            Dict with 'sentences' key
        """
        if isinstance(input_data, PipelineContext):
            text = input_data.normalized_text
        else:
            text = input_data.get('normalized_text', '')

        # Initialize segmenter
        if PYSBD_AVAILABLE:
            self.segmenter = pysbd.Segmenter(language="en", clean=False)

        # Tokenize sentences
        sentences = self._tokenize_sentences(text)

        # Tokenize words in each sentence
        for sent in sentences:
            sent['tokens'] = self._tokenize_words(sent['text'])

        self._initialized = True
        self.logger.info(f"Tokenized {len(sentences)} sentences")

        return {'sentences': sentences}

    def _tokenize_sentences(self, text: str) -> List[Dict]:
        """
        Tokenize text into sentences.

        Args:
            text: Input text

        Returns:
            List of sentence dicts
        """
        sentences = []

        if PYSBD_AVAILABLE and self.segmenter:
            # Use pysbd for sentence segmentation
            raw_sentences = self.segmenter.segment(text)
        else:
            # Fallback: regex-based sentence splitting
            raw_sentences = re.split(r'(?<=[.!?])\s+', text)

        # Build sentence objects
        current_pos = 0
        for i, sent_text in enumerate(raw_sentences):
            sent_text = sent_text.strip()
            if not sent_text:
                continue

            # Find position in original text
            start_pos = text.find(sent_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(sent_text)

            # Detect chapter boundaries
            chapter_id = self._detect_chapter(sent_text, i)

            sentences.append({
                'sentence_id': i,
                'text': sent_text,
                'normalized_text': sent_text,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'chapter_id': chapter_id,
                'paragraph_id': self._detect_paragraph(sent_text, i),
                'position_in_para': 0,  # Will be updated later
                'token_count': 0,  # Will be updated later
                'word_count': 0,  # Will be updated later
            })

            current_pos = end_pos

        return sentences

    def _tokenize_words(self, text: str) -> List[Dict]:
        """
        Tokenize sentence into words.

        Args:
            text: Input text

        Returns:
            List of token dicts
        """
        tokens = []
        # Simple word tokenization
        word_pattern = re.compile(r'\b\w+\b|[^\w\s]')
        matches = word_pattern.finditer(text)

        for i, match in enumerate(matches):
            token = match.group()
            is_punctuation = not token.isalnum()

            tokens.append({
                'position': i,
                'token': token,
                'token_lower': token.lower(),
                'is_punctuation': is_punctuation,
                'is_stopword': False,  # Will be updated later
            })

        return tokens

    def _detect_chapter(self, text: str, sentence_idx: int) -> int:
        """
        Detect chapter number from text.

        Args:
            text: Sentence text
            sentence_idx: Sentence index

        Returns:
            Chapter ID (0-based)
        """
        # Check for chapter markers
        chapter_patterns = [
            r'^CHAPTER\s+(\w+)',
            r'^(\w+)\s*$',  # Roman numerals
            r'^(\d+)\s*$',
        ]

        for pattern in chapter_patterns:
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                chapter_str = match.group(1)
                # Try to convert Roman numerals
                try:
                    return self._roman_to_int(chapter_str) - 1
                except:
                    try:
                        return int(chapter_str) - 1
                    except:
                        pass

        # Default: estimate chapter from position
        return sentence_idx // 100  # Rough estimate

    def _detect_paragraph(self, text: str, sentence_idx: int) -> int:
        """
        Detect paragraph number.

        Args:
            text: Sentence text
            sentence_idx: Sentence index

        Returns:
            Paragraph ID (0-based)
        """
        # Simple heuristic: new paragraph after blank lines
        return sentence_idx // 3  # Rough estimate

    def _roman_to_int(self, roman: str) -> int:
        """
        Convert Roman numeral to integer.

        Args:
            roman: Roman numeral string

        Returns:
            Integer value
        """
        roman = roman.upper()
        values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        result = 0
        prev_value = 0

        for c in reversed(roman):
            value = values.get(c, 0)
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value

        return result
