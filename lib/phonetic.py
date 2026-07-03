"""
Phonetic Library
Soundex, Metaphone, and Double Metaphone algorithms.
"""

from typing import Optional


def soundex(word: str) -> str:
    """
    Compute Soundex code for a word.

    Soundex encodes names by sound as pronounced in English.
    Result: first letter + 3 digits (padded with zeros).

    Args:
        word: Input word

    Returns:
        4-character Soundex code
    """
    if not word:
        return "0000"

    word = word.upper()

    # Keep first letter
    first = word[0]

    # Map consonants to digits
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2',
        'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6',
    }

    # Map remaining characters
    digits = first
    for c in word[1:]:
        digit = mapping.get(c, '')
        # Skip if same as last digit
        if digit and digit != digits[-1]:
            digits += digit

    # Remove vowels and special characters (already handled by mapping)
    # Keep only first letter + digits
    result = first
    for c in digits[1:]:
        if c.isdigit():
            result += c

    # Pad or truncate to 4 characters
    result = (result + "000")[:4]

    return result


def metaphone(word: str) -> str:
    """
    Compute Metaphone code for a word.

    Metaphone is an improvement over Soundex that uses English
    spelling/pronunciation knowledge.

    Args:
        word: Input word

    Returns:
        Metaphone code
    """
    if not word:
        return ""

    word = word.lower()

    # Drop initial silent letters
    if word.startswith('kn'):
        word = word[1:]
    elif word.startswith('gn'):
        word = word[1:]
    elif word.startswith('wr'):
        word = word[1:]
    elif word.startswith('pn'):
        word = word[1:]

    # Metaphone encoding rules
    result = []
    i = 0
    length = len(word)

    while i < length:
        c = word[i]

        # Skip vowels after first character
        if i > 0 and c in 'aeiou':
            i += 1
            continue

        # C
        if c == 'c':
            if i + 1 < length and word[i+1] in 'eiy':
                result.append('s')
            else:
                result.append('k')
        # G
        elif c == 'g':
            if i + 1 < length and word[i+1] in 'eiy':
                result.append('j')
            else:
                result.append('k')
        # PH -> F
        elif c == 'p' and i + 1 < length and word[i+1] == 'h':
            result.append('f')
            i += 1
        # TH -> 0 (theta)
        elif c == 't' and i + 1 < length and word[i+1] == 'h':
            result.append('0')
            i += 1
        # X -> KS
        elif c == 'x':
            result.append('ks')
        # SH -> X
        elif c == 's' and i + 1 < length and word[i+1] == 'h':
            result.append('x')
            i += 1
        # CH -> X
        elif c == 'c' and i + 1 < length and word[i+1] == 'h':
            result.append('x')
            i += 1
        # GH -> K (unless at end)
        elif c == 'g' and i + 1 < length and word[i+1] == 'h':
            if i + 2 < length:
                result.append('k')
            i += 1
        # Silent H after vowel
        elif c == 'h' and i > 0 and word[i-1] in 'aeiou':
            if i + 1 >= length or word[i+1] not in 'aeiou':
                i += 1
                continue
        # Default
        else:
            result.append(c)

        i += 1

    return ''.join(result)


def double_metaphone(word: str) -> tuple:
    """
    Compute Double Metaphone codes for a word.

    Double Metaphone returns both primary and secondary codes,
    handling ambiguous pronunciations.

    Args:
        word: Input word

    Returns:
        Tuple of (primary_code, secondary_code)
    """
    # Simplified Double Metaphone - returns primary only for now
    primary = metaphone(word)
    return (primary, primary)


def phonetic_match(word1: str, word2: str) -> bool:
    """
    Check if two words sound alike using Metaphone.

    Args:
        word1: First word
        word2: Second word

    Returns:
        True if words have the same Metaphone code
    """
    return metaphone(word1) == metaphone(word2)
