"""
Gematria Library
Numerical encoding of text using various gematria systems.
"""

from typing import Dict, List


def gematria_simple(word: str) -> int:
    """
    Compute simple gematria value for a word.
    a=1, b=2, c=3, ..., z=26

    Args:
        word: Input word

    Returns:
        Gematria value
    """
    return sum(ord(c) - ord('a') + 1 for c in word.lower() if c.isalpha())


def gematria_reduced(word: str) -> int:
    """
    Compute reduced gematria value for a word.
    Sum digits until single digit.

    Args:
        word: Input word

    Returns:
        Reduced gematria value (1-9)
    """
    value = gematria_simple(word)
    while value > 9:
        value = sum(int(d) for d in str(value))
    return value


def gematria_ordinal(word: str) -> int:
    """
    Compute ordinal gematria value for a word.
    Position in alphabet: a=1, b=2, ..., z=26

    Args:
        word: Input word

    Returns:
        Ordinal gematria value
    """
    return gematria_simple(word)


def gematria_reverse(word: str) -> int:
    """
    Compute reverse gematria value for a word.
    z=1, y=2, ..., a=26

    Args:
        word: Input word

    Returns:
        Reverse gematria value
    """
    return sum(27 - (ord(c) - ord('a')) for c in word.lower() if c.isalpha())


def build_gematria_index(words: List[str]) -> Dict[int, List[str]]:
    """
    Build index mapping gematria values to words.

    Args:
        words: List of words

    Returns:
        Dictionary mapping gematria values to word lists
    """
    index = {}
    for word in words:
        value = gematria_simple(word)
        if value not in index:
            index[value] = []
        index[value].append(word)
    return index


def find_gematria_matches(word: str, index: Dict[int, List[str]]) -> List[str]:
    """
    Find words with the same gematria value.

    Args:
        word: Input word
        index: Gematria index

    Returns:
        List of words with matching gematria value
    """
    value = gematria_simple(word)
    return index.get(value, [])
