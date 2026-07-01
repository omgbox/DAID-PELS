"""
Anagram Library
Anagram detection and grouping.
"""

from typing import Dict, List, Tuple
from collections import defaultdict


def anagram_signature(word: str) -> str:
    """
    Compute anagram signature for a word.
    Two words are anagrams if they have the same signature.

    Args:
        word: Input word

    Returns:
        Anagram signature (sorted letters)
    """
    return ''.join(sorted(word.lower()))


def are_anagrams(word1: str, word2: str) -> bool:
    """
    Check if two words are anagrams.

    Args:
        word1: First word
        word2: Second word

    Returns:
        True if words are anagrams
    """
    return anagram_signature(word1) == anagram_signature(word2)


def find_anagrams(word: str, dictionary: List[str]) -> List[str]:
    """
    Find all anagrams of a word in a dictionary.

    Args:
        word: Input word
        dictionary: List of dictionary words

    Returns:
        List of anagram words
    """
    sig = anagram_signature(word)
    return [w for w in dictionary if anagram_signature(w) == sig]


def build_anagram_index(words: List[str]) -> Dict[str, List[str]]:
    """
    Build index mapping anagram signatures to word groups.

    Args:
        words: List of words

    Returns:
        Dictionary mapping signatures to word lists
    """
    index = defaultdict(list)
    for word in words:
        sig = anagram_signature(word)
        index[sig].append(word)
    return dict(index)


def find_anagram_groups(index: Dict[str, List[str]],
                        min_group_size: int = 2) -> List[List[str]]:
    """
    Find all anagram groups in an index.

    Args:
        index: Anagram index
        min_group_size: Minimum group size to include

    Returns:
        List of anagram groups
    """
    return [group for group in index.values() if len(group) >= min_group_size]


def anagram_signature_hash(word: str) -> int:
    """
    Compute hash-based anagram signature using prime numbers.
    Two words are anagrams if they have the same hash.

    Args:
        word: Input word

    Returns:
        Hash value
    """
    primes = {
        'a': 2, 'b': 3, 'c': 5, 'd': 7, 'e': 11, 'f': 13,
        'g': 17, 'h': 19, 'i': 23, 'j': 29, 'k': 31, 'l': 37,
        'm': 41, 'n': 43, 'o': 47, 'p': 53, 'q': 59, 'r': 61,
        's': 67, 't': 71, 'u': 73, 'v': 79, 'w': 83, 'x': 89,
        'y': 97, 'z': 101,
    }
    result = 1
    for c in word.lower():
        if c in primes:
            result *= primes[c]
    return result
