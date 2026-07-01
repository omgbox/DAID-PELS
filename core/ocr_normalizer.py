"""
BookBot OCR Normalizer
OCR artifact correction using Pynini (OpenFst backend).
Falls back to custom implementation if Pynini not available.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.ocr_normalizer')

# Try to import Pynini (disabled for now due to issues)
PYNINI_AVAILABLE = False


class OCRNormalizer(BaseModule):
    """OCR artifact correction module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.dictionary = set()
        self.confusion_matrix = {}
        self.correction_count = 0

    def process(self, input_data) -> dict:
        """
        Process input data and return OCR-normalized text.

        Args:
            input_data: PipelineContext or dict with 'raw_text' key

        Returns:
            Dict with 'normalized_text' key
        """
        if isinstance(input_data, PipelineContext):
            text = input_data.raw_text
        else:
            text = input_data.get('raw_text', '')

        # Load dictionary
        if self.db:
            self._load_dictionary()

        # Process text
        normalized = self._normalize_text(text)

        self._initialized = True
        self.logger.info(f"OCR normalization complete. Applied {self.correction_count} corrections.")

        return {'normalized_text': normalized}

    def _load_dictionary(self):
        """Load dictionary words from database."""
        if not self.db:
            return

        try:
            results = self.db.execute(
                "SELECT DISTINCT word_lower FROM definitions"
            )
            self.dictionary = {row[0] for row in results}
            self.logger.info(f"Loaded {len(self.dictionary)} dictionary words")
        except Exception as e:
            self.logger.warning(f"Could not load dictionary: {e}")

    def _normalize_text(self, text: str) -> str:
        """
        Normalize OCR text.

        Args:
            text: Raw OCR text

        Returns:
            Normalized text
        """
        # Step 1: Remove metadata
        if self.get_config('remove_metadata', True):
            text = self._remove_metadata(text)

        # Step 2: Collapse multiple spaces
        if self.get_config('collapse_spaces', True):
            text = self._collapse_spaces(text)

        # Step 3: Fix hyphenation
        if self.get_config('hyphenation_repair', True):
            text = self._fix_hyphenation(text)

        # Step 4: Correct OCR errors
        if self.get_config('use_fst', True) and PYNINI_AVAILABLE:
            text = self._correct_with_pynini(text)
        else:
            text = self._correct_with_edit_distance(text)

        # Step 5: Remove page numbers
        if self.get_config('remove_page_numbers', True):
            text = self._remove_page_numbers(text)

        return text

    def _remove_metadata(self, text: str) -> str:
        """Remove library metadata from text."""
        lines = text.split('\n')
        metadata_lines = self.get_config('metadata_lines_start', 100)
        trailing_lines = self.get_config('metadata_lines_end', 40)

        # Remove first N lines (library metadata)
        if len(lines) > metadata_lines:
            lines = lines[metadata_lines:]

        # Remove last N lines (trailing metadata)
        if len(lines) > trailing_lines:
            lines = lines[:-trailing_lines]

        return '\n'.join(lines)

    def _collapse_spaces(self, text: str) -> str:
        """Collapse multiple spaces to single space."""
        return re.sub(r'  +', ' ', text)

    def _fix_hyphenation(self, text: str) -> str:
        """Fix line-break hyphenation."""
        # Join hyphenated words at line breaks
        return re.sub(r'-\n\s*', '', text)

    def _remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers."""
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            if not re.match(r'^\d+\s*$', line.strip()):
                cleaned.append(line)
        return '\n'.join(cleaned)

    def _correct_with_pynini(self, text: str) -> str:
        """
        Correct OCR errors using Pynini FST.

        Args:
            text: Input text

        Returns:
            Corrected text
        """
        # Build OCR correction FST
        confusion_pairs = self._get_confusion_pairs()
        dictionary_fsa = pynini.string_map(list(self.dictionary))

        # Build error model
        all_chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        all_chars += [chr(i) for i in range(ord('0'), ord('9') + 1)]
        all_chars += [' ', '.', ',', '-', "'"]

        fsts = []

        # Identity arcs (correct recognition, cost 0)
        fsts.append(pynini.string_map([(c, c) for c in all_chars]))

        # Confusion-based substitutions
        for (observed, correct), weight in confusion_pairs:
            fsts.append(pynini.cross(observed, correct))

        # Generic character substitutions (higher cost)
        for c1 in all_chars[:26]:
            for c2 in all_chars[:26]:
                if c1 != c2:
                    fsts.append(pynini.cross(c1, c2))

        # Insertions and deletions
        for c in all_chars:
            fsts.append(pynini.cross('', c))   # insertion
            fsts.append(pynini.cross(c, ''))   # deletion

        single_edit = pynini.union(*fsts)

        # Compose with dictionary
        corrections = single_edit @ dictionary_fsa

        # Apply to each word
        words = text.split()
        corrected_words = []
        for word in words:
            # Preserve leading/trailing punctuation
            prefix = ''
            suffix = ''
            clean = word
            while clean and not clean[0].isalnum():
                prefix += clean[0]
                clean = clean[1:]
            while clean and not clean[-1].isalnum():
                suffix = clean[-1] + suffix
                clean = clean[:-1]

            if clean and clean.lower() not in self.dictionary:
                try:
                    lattice = clean.lower() @ corrections
                    best = pynini.shortestpath(lattice)
                    corrected = best.string()
                    if corrected != clean.lower():
                        self.correction_count += 1
                        self._log_correction(clean, corrected)
                    corrected_words.append(prefix + corrected + suffix)
                except:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

    def _correct_with_edit_distance(self, text: str) -> str:
        """
        Correct OCR errors using edit distance (fallback).

        Args:
            text: Input text

        Returns:
            Corrected text
        """
        words = text.split()
        corrected_words = []
        min_word_length = self.get_config('min_word_length', 2)

        # Limit dictionary size for performance
        dict_list = list(self.dictionary)[:10000]  # Use top 10K words

        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if len(clean) >= min_word_length and clean.lower() not in self.dictionary:
                # Find closest word by edit distance
                best_word = clean
                best_distance = float('inf')
                for dict_word in dict_list:
                    if abs(len(dict_word) - len(clean)) > 1:
                        continue
                    dist = self._edit_distance(clean.lower(), dict_word)
                    if dist < best_distance:
                        best_distance = dist
                        best_word = dict_word

                if best_distance <= 1:
                    self.correction_count += 1
                    self._log_correction(clean, best_word)
                    corrected_words.append(best_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        return ' '.join(corrected_words)

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

    def _get_confusion_pairs(self) -> List[Tuple[Tuple[str, str], float]]:
        """
        Get OCR confusion pairs from config.

        Returns:
            List of ((observed, correct), weight) tuples
        """
        # Default confusion matrix
        default_confusions = {
            ('rn', 'm'): 0.85,
            ('cl', 'd'): 0.75,
            ('li', 'h'): 0.70,
            ('vv', 'w'): 0.65,
            ('ii', 'n'): 0.60,
            ('O', '0'): 0.55,
            ('l', '1'): 0.50,
            ('S', '5'): 0.45,
        }

        confusions = self.get_config('confusion_matrix', 'auto')
        if confusions == 'auto':
            return [(k, v) for k, v in default_confusions.items()]
        elif isinstance(confusions, dict):
            return [(k, v) for k, v in confusions.items()]
        return []

    def _log_correction(self, original: str, corrected: str):
        """Log OCR correction to database."""
        if self.db:
            try:
                self.db.insert('ocr_corrections', {
                    'original_text': original,
                    'corrected_text': corrected,
                    'rule_applied': 'fst_correction',
                    'confidence': self.get_config('correction_confidence', 0.9),
                })
            except Exception as e:
                self.logger.warning(f"Could not log correction: {e}")
