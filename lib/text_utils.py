"""
Text Utilities Library
Text normalization and tokenization utilities.
"""

import re
from typing import List


def normalize_spaces(text: str) -> str:
    """
    Collapse multiple spaces to single space.

    Args:
        text: Input text

    Returns:
        Text with normalized spaces
    """
    return re.sub(r'  +', ' ', text)


def remove_punctuation(text: str) -> str:
    """
    Remove punctuation from text.

    Args:
        text: Input text

    Returns:
        Text without punctuation
    """
    return re.sub(r'[^\w\s]', '', text)


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Input text

    Returns:
        List of words
    """
    return re.findall(r'\b\w+\b', text)


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences (simple regex-based).

    Args:
        text: Input text

    Returns:
        List of sentences
    """
    # Split on sentence-ending punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def lowercase(text: str) -> str:
    """
    Convert text to lowercase.

    Args:
        text: Input text

    Returns:
        Lowercase text
    """
    return text.lower()


def strip_whitespace(text: str) -> str:
    """
    Strip leading and trailing whitespace.

    Args:
        text: Input text

    Returns:
        Stripped text
    """
    return text.strip()


def normalize_unicode(text: str) -> str:
    """
    Normalize Unicode characters.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    import unicodedata
    return unicodedata.normalize('NFC', text)


def fix_hyphenation(text: str) -> str:
    """
    Fix line-break hyphenation.

    Args:
        text: Input text

    Returns:
        Text with fixed hyphenation
    """
    # Join hyphenated words at line breaks
    return re.sub(r'-\n\s*', '', text)


def remove_page_numbers(text: str) -> str:
    """
    Remove standalone page numbers.

    Args:
        text: Input text

    Returns:
        Text without page numbers
    """
    # Remove lines that are just numbers
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        if not re.match(r'^\d+\s*$', line.strip()):
            cleaned.append(line)
    return '\n'.join(cleaned)


def is_stopword(word: str, stopwords: set = None) -> bool:
    """
    Check if a word is a stopword.

    Args:
        word: Input word
        stopwords: Set of stopwords (uses default if None)

    Returns:
        True if word is a stopword
    """
    if stopwords is None:
        # Default English stopwords
        stopwords = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'shall', 'can',
            'of', 'in', 'to', 'for', 'with', 'on', 'at', 'from', 'by',
            'about', 'as', 'into', 'through', 'during', 'before', 'after',
            'above', 'below', 'between', 'out', 'off', 'over', 'under',
            'again', 'further', 'then', 'once', 'here', 'there', 'when',
            'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
            'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
            'and', 'but', 'or', 'if', 'while', 'this', 'that', 'these',
            'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he',
            'him', 'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their',
        }
    return word.lower() in stopwords
