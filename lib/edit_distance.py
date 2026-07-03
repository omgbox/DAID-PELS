"""
Edit Distance Library
Levenshtein, Damerau-Levenshtein, Jaro-Winkler distance algorithms.
"""

from typing import List, Tuple


def levenshtein(s1: str, s2: str) -> int:
    """
    Compute Levenshtein edit distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def damerau_levenshtein(s1: str, s2: str) -> int:
    """
    Compute Damerau-Levenshtein distance between two strings.
    Includes transpositions as a single edit operation.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance including transpositions
    """
    len1 = len(s1)
    len2 = len(s2)

    # Create matrix
    d = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        d[i][0] = i
    for j in range(len2 + 1):
        d[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            d[i][j] = min(
                d[i-1][j] + 1,      # deletion
                d[i][j-1] + 1,      # insertion
                d[i-1][j-1] + cost   # substitution
            )
            # Transposition
            if (i > 1 and j > 1 and
                s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]):
                d[i][j] = min(d[i][j], d[i-2][j-2] + cost)

    return d[len1][len2]


def jaro_winkler(s1: str, s2: str, winklerize: bool = True) -> float:
    """
    Compute Jaro-Winkler similarity between two strings.

    Args:
        s1: First string
        s2: Second string
        winklerize: Apply Winkler modification (boost common prefix)

    Returns:
        Similarity score (0.0 to 1.0, higher = more similar)
    """
    if s1 == s2:
        return 1.0

    len1 = len(s1)
    len2 = len(s2)

    if len1 == 0 or len2 == 0:
        return 0.0

    # Maximum matching distance
    max_dist = max(len1, len2) // 2 - 1
    if max_dist < 0:
        max_dist = 0

    # Match arrays
    s1_matches = [False] * len1
    s2_matches = [False] * len2

    matches = 0
    transpositions = 0

    # Find matches
    for i in range(len1):
        start = max(0, i - max_dist)
        end = min(i + max_dist + 1, len2)
        for j in range(start, end):
            if s2_matches[j] or s1[i] != s2[j]:
                continue
            s1_matches[i] = True
            s2_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    # Count transpositions
    k = 0
    for i in range(len1):
        if not s1_matches[i]:
            continue
        while not s2_matches[k]:
            k += 1
        if s1[i] != s2[k]:
            transpositions += 1
        k += 1

    # Jaro similarity
    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3

    if not winklerize:
        return jaro

    # Winkler modification
    prefix = 0
    for i in range(min(4, len1, len2)):
        if s1[i] == s2[i]:
            prefix += 1
        else:
            break

    return jaro + prefix * 0.1 * (1 - jaro)


def find_closest_word(word: str, dictionary: List[str],
                      max_distance: int = 2) -> List[Tuple[str, int]]:
    """
    Find closest words in dictionary by edit distance.

    Args:
        word: Input word
        dictionary: List of dictionary words
        max_distance: Maximum edit distance to consider

    Returns:
        List of (word, distance) tuples, sorted by distance
    """
    word_lower = word.lower()
    candidates = []

    for dict_word in dictionary:
        dist = levenshtein(word_lower, dict_word.lower())
        if dist <= max_distance:
            candidates.append((dict_word, dist))

    candidates.sort(key=lambda x: x[1])
    return candidates
