"""
FST Engine - Generic Finite State Transducer library for BookBot.

This module provides two implementations:
1. PyniniFST - Uses Pynini (OpenFst backend) for C++ speed
2. CustomFST - Pure Python fallback if Pynini not available

Usage:
    try:
        from lib.fst_engine import PyniniFST, PyniniOCRCorrector
    except ImportError:
        from lib.fst_engine import CustomFST, CustomOCRCorrector
"""

import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# Try to import Pynini
try:
    import pynini
    PYNINI_AVAILABLE = True
    logger.info("Pynini available - using OpenFst backend")
except ImportError:
    PYNINI_AVAILABLE = False
    logger.warning("Pynini not available - using custom FST implementation")


# =============================================================================
# PYNINI IMPLEMENTATION
# =============================================================================

class PyniniFST:
    """Pynini-based FST wrapper for string transduction."""

    def __init__(self):
        self.sigma = self._build_sigma()
        self.sigma_star = self.sigma.closure()
        self.fst = None

    def _build_sigma(self) -> 'pynini.Fst':
        """Build the character alphabet."""
        chars = [chr(i) for i in range(32, 127)]  # printable ASCII
        return pynini.string_map(chars)

    def build_from_pairs(self, pairs: List[Tuple[str, str, float]]):
        """
        Build FST from input-output pairs with weights.

        Args:
            pairs: List of (input, output, weight) tuples
        """
        fsts = []
        for inp, out, weight in pairs:
            fsts.append(pynini.cross(inp, out, weight))
        self.fst = pynini.union(*fsts)

    def build_from_dictionary(self, words: List[str]):
        """
        Build dictionary FSA from word list.

        Args:
            words: List of words
        """
        self.fst = pynini.string_map(words)

    def compose(self, other: 'PyniniFST') -> 'PyniniFST':
        """
        Compose this FST with another.

        Args:
            other: Another FST to compose with

        Returns:
            New composed FST
        """
        result = PyniniFST()
        result.fst = pynini.compose(self.fst, other.fst)
        return result

    def apply(self, input_str: str) -> str:
        """
        Apply FST to input string.

        Args:
            input_str: Input string

        Returns:
            Output string (best path)
        """
        try:
            lattice = input_str @ self.fst
            best = pynini.shortestpath(lattice)
            return best.string()
        except pynini.FstOpError:
            return input_str

    def apply_n_best(self, input_str: str, n: int = 5) -> List[Tuple[str, float]]:
        """
        Apply FST and return n-best results.

        Args:
            input_str: Input string
            n: Number of results

        Returns:
            List of (output, weight) tuples
        """
        try:
            lattice = input_str @ self.fst
            best = pynini.shortestpath(lattice, nshortest=n, unique=True)
            results = []
            for path in pynini.paths(best):
                results.append((path.ostring, float(path.weight)))
            return results
        except pynini.FstOpError:
            return [(input_str, float('inf'))]

    def optimize(self):
        """Optimize the FST (determinize + minimize)."""
        if self.fst:
            self.fst = pynini.optimize(self.fst)

    def save(self, path: str):
        """Save FST to file."""
        if self.fst:
            self.fst.write(path)

    def load(self, path: str):
        """Load FST from file."""
        self.fst = pynini.Fst.read(path)


class PyniniOCRCorrector:
    """Pynini-based OCR error corrector."""

    def __init__(self, dictionary_path: str, confusion_matrix: Optional[Dict] = None):
        """
        Initialize OCR corrector.

        Args:
            dictionary_path: Path to dictionary file
            confusion_matrix: OCR confusion matrix (optional)
        """
        self.dictionary = self._load_dictionary(dictionary_path)
        self.error_model = self._build_error_model(confusion_matrix or {})
        self.corrections = self.error_model @ self.dictionary

    def _load_dictionary(self, path: str) -> 'pynini.Fst':
        """Load dictionary as deterministic FSA."""
        words = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word and len(word) > 1:
                    words.append(word)
        return pynini.string_map(words)

    def _build_error_model(self, confusion_matrix: Dict) -> 'pynini.Fst':
        """Build OCR error transducer from confusion matrix."""
        all_chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        all_chars += [chr(i) for i in range(ord('0'), ord('9') + 1)]
        all_chars += [' ', '.', ',', '-', "'"]

        fsts = []

        # Identity arcs (correct recognition, cost 0)
        fsts.append(pynini.string_map([(c, c) for c in all_chars]))

        # Confusion-based substitutions
        for (observed, correct), weight in confusion_matrix.items():
            fsts.append(pynini.cross(observed, correct, weight))

        # Generic character substitutions (higher cost)
        for c1 in all_chars[:26]:  # letters only
            for c2 in all_chars[:26]:
                if c1 != c2:
                    fsts.append(pynini.cross(c1, c2, weight=3.0))

        # Insertions and deletions
        for c in all_chars:
            fsts.append(pynini.cross("", c, weight=2.0))   # insertion
            fsts.append(pynini.cross(c, "", weight=2.0))   # deletion

        single_edit = pynini.union(*fsts)
        return pynini.compose(single_edit, single_edit)  # up to 2 edits

    def correct_word(self, word: str, n_best: int = 5) -> List[Tuple[str, float]]:
        """
        Find the best corrections for an OCR'd word.

        Args:
            word: OCR'd word
            n_best: Number of corrections to return

        Returns:
            List of (correction, weight) tuples
        """
        word_lower = word.lower()

        # If the word is already in the dictionary, return it
        try:
            if (word_lower @ self.dictionary).num_states() > 0:
                return [(word, 0.0)]
        except pynini.FstOpError:
            pass

        try:
            # Generate candidates through error model
            lattice = word_lower @ self.corrections
            lattice.project("output")
            lattice.optimize()

            # Get n-best corrections
            best = pynini.shortestpath(lattice, nshortest=n_best, unique=True)
            results = []
            for path in pynini.paths(best):
                results.append((path.ostring, float(path.weight)))
            return results
        except pynini.FstOpError:
            return [(word, float('inf'))]

    def correct_text(self, text: str) -> str:
        """
        Correct OCR errors in a full text.

        Args:
            text: OCR'd text

        Returns:
            Corrected text
        """
        words = text.split()
        corrected = []
        for word in words:
            # Preserve leading/trailing punctuation
            prefix = ""
            suffix = ""
            clean = word
            while clean and not clean[0].isalnum():
                prefix += clean[0]
                clean = clean[1:]
            while clean and not clean[-1].isalnum():
                suffix = clean[-1] + suffix
                clean = clean[:-1]

            if clean:
                candidates = self.correct_word(clean)
                best_word = candidates[0][0] if candidates else clean
                corrected.append(prefix + best_word + suffix)
            else:
                corrected.append(word)

        return ' '.join(corrected)


class PyniniTextNormalizer:
    """Pynini-based text normalizer."""

    def __init__(self):
        self.sigma_star = self._build_sigma_star()
        self._build_rules()

    def _build_sigma_star(self) -> 'pynini.Fst':
        """Build closure over all printable ASCII + common Unicode."""
        chars = [chr(i) for i in range(1, 127)]
        chars += ['\u2013', '\u2014', '\u2018', '\u2019', '\u201c', '\u201d',
                  '\u00ad', '\u2026', '\u00a0']
        return pynini.string_map(chars).closure()

    def _build_rules(self):
        """Build all normalization rule transducers."""

        # Whitespace normalization
        multi_space = pynini.accep(" ").closure(2)  # 2+ spaces
        self.space_rule = pynini.cdrewrite(
            pynini.cross(multi_space, pynini.accep(" ")),
            "", "", self.sigma_star
        )

        # Unicode normalization
        self.unicode_rule = pynini.cdrewrite(
            pynini.union(
                pynini.cross("\u2018", "'"),   # left single quote
                pynini.cross("\u2019", "'"),   # right single quote
                pynini.cross("\u201c", '"'),   # left double quote
                pynini.cross("\u201d", '"'),   # right double quote
                pynini.cross("\u2013", '-'),   # en dash
                pynini.cross("\u2014", '--'),  # em dash
                pynini.cross("\u2026", '...'), # ellipsis
                pynini.cross("\u00a0", ' '),   # non-breaking space
                pynini.cross("\u00ad", ''),    # soft hyphen
            ),
            "", "", self.sigma_star
        )

        # Soft hyphen removal (line-break artifacts)
        self.soft_hyphen_rule = pynini.cdrewrite(
            pynini.cross("-\n", ""),
            "", "", self.sigma_star
        )

    def normalize(self, text: str) -> str:
        """
        Apply the full normalization cascade.

        Args:
            text: Input text

        Returns:
            Normalized text
        """
        result = text

        # Step 1: Soft hyphen removal
        try:
            result = (result @ self.soft_hyphen_rule).string()
        except:
            pass

        # Step 2: Unicode normalization
        try:
            result = (result @ self.unicode_rule).string()
        except:
            pass

        # Step 3: Whitespace normalization
        try:
            result = (result @ self.space_rule).string()
        except:
            pass

        return result


# =============================================================================
# CUSTOM IMPLEMENTATION (Fallback)
# =============================================================================

class CustomFST:
    """Custom FST implementation (fallback if Pynini not available)."""

    def __init__(self):
        self.states = set()
        self.transitions = {}  # (state, input) -> [(next_state, output, weight)]
        self.start_state = None
        self.accept_states = set()

    def build_from_pairs(self, pairs: List[Tuple[str, str, float]]):
        """Build FST from input-output pairs."""
        # Simple implementation - just store pairs for lookup
        self.pairs = pairs

    def build_from_dictionary(self, words: List[str]):
        """Build dictionary from word list."""
        self.words = set(w.lower() for w in words)

    def apply(self, input_str: str) -> str:
        """Apply FST to input (simple edit distance fallback)."""
        if hasattr(self, 'words'):
            # Dictionary lookup
            if input_str.lower() in self.words:
                return input_str
            # Find closest word by edit distance
            best_word = input_str
            best_distance = float('inf')
            for word in self.words:
                dist = self._edit_distance(input_str.lower(), word)
                if dist < best_distance:
                    best_distance = dist
                    best_word = word
            return best_word
        return input_str

    def apply_n_best(self, input_str: str, n: int = 5) -> List[Tuple[str, float]]:
        """Apply FST and return n-best results."""
        if hasattr(self, 'words'):
            candidates = []
            for word in self.words:
                dist = self._edit_distance(input_str.lower(), word)
                candidates.append((word, dist))
            candidates.sort(key=lambda x: x[1])
            return candidates[:n]
        return [(input_str, float('inf'))]

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
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

    def save(self, path: str):
        """Save FST to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path: str):
        """Load FST from file."""
        import pickle
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))


class CustomOCRCorrector:
    """Custom OCR corrector (fallback if Pynini not available)."""

    def __init__(self, dictionary_path: str, confusion_matrix: Optional[Dict] = None):
        """Initialize OCR corrector."""
        self.words = self._load_dictionary(dictionary_path)
        self.confusion_matrix = confusion_matrix or {}

    def _load_dictionary(self, path: str) -> set:
        """Load dictionary as set."""
        words = set()
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    words.add(word)
        return words

    def correct_word(self, word: str, n_best: int = 5) -> List[Tuple[str, float]]:
        """Find the best corrections for an OCR'd word."""
        word_lower = word.lower()
        if word_lower in self.words:
            return [(word, 0.0)]

        # Find closest words by edit distance
        candidates = []
        for dict_word in self.words:
            dist = self._edit_distance(word_lower, dict_word)
            candidates.append((dict_word, dist))
        candidates.sort(key=lambda x: x[1])
        return candidates[:n_best]

    def correct_text(self, text: str) -> str:
        """Correct OCR errors in a full text."""
        words = text.split()
        corrected = []
        for word in words:
            candidates = self.correct_word(word)
            best_word = candidates[0][0] if candidates else word
            corrected.append(best_word)
        return ' '.join(corrected)

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
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


class CustomTextNormalizer:
    """Custom text normalizer (fallback if Pynini not available)."""

    def __init__(self):
        pass

    def normalize(self, text: str) -> str:
        """Apply text normalization."""
        import re

        # Soft hyphen removal
        text = text.replace('-\n', '')

        # Unicode normalization
        text = text.replace('\u2018', "'")   # left single quote
        text = text.replace('\u2019', "'")   # right single quote
        text = text.replace('\u201c', '"')   # left double quote
        text = text.replace('\u201d', '"')   # right double quote
        text = text.replace('\u2013', '-')   # en dash
        text = text.replace('\u2014', '--')  # em dash
        text = text.replace('\u2026', '...') # ellipsis
        text = text.replace('\u00a0', ' ')   # non-breaking space
        text = text.replace('\u00ad', '')    # soft hyphen

        # Whitespace normalization
        text = re.sub(r'  +', ' ', text)

        return text
