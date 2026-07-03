"""
BookBot Tokenizer
Sentence and word tokenization.
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

# Abbreviations that should not trigger sentence splits
_ABBREVIATIONS = {
    'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'rev', 'hon', 'st', 'lt',
    'capt', 'col', 'gen', 'sgt', 'cpl', 'gov', 'sr', 'jr', 'esq',
    'dept', 'ave', 'blvd', 'rd', 'sq', 'pl', 'mt', 'ft',
    'vs', 'etc', 'inc', 'corp', 'co', 'ltd', 'est',
    'vol', 'no', 'pp', 'ch', 'ex', 'ref', 'art', 'sec', 'fig',
    'al', 'viz', 'ibid', 'op', 'cit',
    'de', 'la', 'le', 'du', 'ca',
    'approx', 'min', 'max', 'temp', 'obs', 'info',
    'ed', 'tr', 'trans',
    # Roman numeral-like (common in chapter headings)
    'i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x',
}

# Compiled patterns (class-level for reuse)
_RE_COLLAPSE = re.compile(r'\s+')
_RE_WORD = re.compile(r'\b\w+\b|[^\w\s]')
# Basic sentence split by punctuation + space + capital
_RE_SENTENCE_BREAK = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')

# Smart-quote & dash normalization mapping
_SMART_QUOTES = {
    '\u2018': "'", '\u2019': "'", '\u201a': "'", '\u201b': "'",
    '\u201c': '"', '\u201d': '"', '\u201e': '"', '\u201f': '"',
    '\u2039': '<', '\u203a': '>',
}
_SMART_DASHES = {
    '\u2013': '-', '\u2014': '-', '\u2015': '-',
}
_EXTRA_UNICODE_PUNCT = str.maketrans({
    '\u2026': '...', '\u2022': '-', '\u00a0': ' ',
    '\u00ab': '"', '\u00bb': '"',
})

# Contraction mapping (specific irregulars first, then general patterns)
# Order matters: specific before general
_CONTRACTION_SPECIFIC = {
    "can't": "cannot",
    "won't": "will not", "shan't": "shall not",
    "ain't": "is not", "let's": "let us",
    "o'clock": "o'clock",
}
# Words where 's means "is" (NOT possessive)
_CONTRACTION_S_WHITELIST = {
    'it', 'that', 'there', 'here', 'who', 'what', 'where', 'how',
    'when', 'why', 'he', 'she', 'everybody', 'everyone', 'everything',
    'nobody', 'no one', 'someone', 'somebody', 'something',
}
# Pronouns where 'd means "would"
_CONTRACTION_D_WHITELIST = {
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'who', 'that', 'there',
}


def normalize_text(text: str) -> str:
    """Normalize smart quotes, dashes, and contractions to ASCII-safe text.
    Must be called before sentence splitting and tokenization.
    """
    if not text:
        return text
    # Smart quotes -> ASCII
    for orig, ascii_ in _SMART_QUOTES.items():
        text = text.replace(orig, ascii_)
    # Dashes -> hyphen
    for orig, ascii_ in _SMART_DASHES.items():
        text = text.replace(orig, ascii_)
    # Other unicode punctuation
    text = text.translate(_EXTRA_UNICODE_PUNCT)
    # Gutenberg-ism: _italic_ -> italic (remove underscore markup)
    text = re.sub(r'_(.+?)_', r'\1', text)
    # Expansion must happen after smart-quote -> ASCII apostrophe
    # 1. Specific irregular contractions (case-insensitive)
    for orig, expanded in _CONTRACTION_SPECIFIC.items():
        text = re.sub(re.escape(orig), expanded, text, flags=re.IGNORECASE)
    # 2. General 'll -> will
    text = re.sub(r"(\w+)'ll", r'\1 will', text, flags=re.IGNORECASE)
    # 3. General 've -> have
    text = re.sub(r"(\w+)'ve", r'\1 have', text, flags=re.IGNORECASE)
    # 4. General 're -> are
    text = re.sub(r"(\w+)'re", r'\1 are', text, flags=re.IGNORECASE)
    # 5. General 'm -> am
    text = re.sub(r"(\w+)'m", r'\1 am', text, flags=re.IGNORECASE)
    # 6. General n't -> not (after irregulars handled above)
    text = re.sub(r"(\w+)n't", r'\1 not', text, flags=re.IGNORECASE)
    # 7. 's -> is only for whitelisted pronoun words
    for w in sorted(_CONTRACTION_S_WHITELIST, key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(w) + r"'s\b", w + ' is', text, flags=re.IGNORECASE)
    # 8. 'd -> would for whitelisted pronoun words
    for w in sorted(_CONTRACTION_D_WHITELIST, key=len, reverse=True):
        text = re.sub(r'\b' + re.escape(w) + r"'d\b", w + ' would', text, flags=re.IGNORECASE)
    return text


class Tokenizer(BaseModule):
    """Sentence and word tokenization module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.segmenter = None

    def process(self, input_data) -> dict:
        if isinstance(input_data, PipelineContext):
            text = input_data.normalized_text
        else:
            text = input_data.get('normalized_text', '')

        # Tokenize sentences
        sentences = self._tokenize_sentences(text)

        # Tokenize words (fast regex on each sentence)
        for sent in sentences:
            sent['tokens'] = self._tokenize_words(sent['text'])

        self._initialized = True
        self.logger.info(f"Tokenized {len(sentences)} sentences")
        return {'sentences': sentences}

    def _preprocess(self, text: str) -> str:
        """Normalize encoding, expand contractions, collapse whitespace."""
        if not text:
            return text
        text = normalize_text(text)
        text = text.replace('\r\n', '\n')
        text = text.replace('\n', ' ')
        return _RE_COLLAPSE.sub(' ', text).strip()

    def _tokenize_sentences(self, text: str) -> List[Dict]:
        """Fast O(n) sentence splitting with abbreviation handling."""
        text = self._preprocess(text)
        if not text:
            return []

        # Protect abbreviation periods so they don't trigger splits
        # Replace "Abbrev." with "Abbrev<PRD>" temporarily
        word_boundary = r'(?<=\b)'
        abbrev_pattern = r'(' + '|'.join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + r')\.'
        protected = re.sub(abbrev_pattern, r'\1<PRD>', text, flags=re.IGNORECASE)
        # Also protect single-capital-letter initials (e.g., "J. Smith")
        protected = re.sub(r'\b([A-Z])\.', r'\1<PRD>', protected)

        # Split on sentence boundaries
        raw = _RE_SENTENCE_BREAK.split(protected)
        raw = [s.strip() for s in raw if s.strip()]

        # Restore periods and build sentence objects
        sentences = []
        current_pos = 0
        for i, chunk in enumerate(raw):
            sent_text = chunk.replace('<PRD>', '.')
            # Re-collapse any whitespace from the split
            start_pos = text.find(sent_text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            end_pos = start_pos + len(sent_text)

            sentences.append({
                'sentence_id': i,
                'text': sent_text,
                'normalized_text': sent_text,
                'start_pos': start_pos,
                'end_pos': end_pos,
                'chapter_id': self._detect_chapter(sent_text, i),
                'paragraph_id': i // 3,
                'position_in_para': 0,
                'token_count': 0,
                'word_count': 0,
            })
            current_pos = end_pos

        return sentences

    def _tokenize_words(self, text: str) -> List[Dict]:
        """Fast regex word tokenization."""
        tokens = []
        for i, match in enumerate(_RE_WORD.finditer(text)):
            token = match.group()
            tokens.append({
                'position': i,
                'token': token,
                'token_lower': token.lower(),
                'is_punctuation': not token.isalnum(),
                'is_stopword': False,
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
