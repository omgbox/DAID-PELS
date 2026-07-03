"""
BookBot Definition Linker
Maps book words to dictionary entries.
"""

import logging
from typing import Dict, List, Optional, Tuple

from .base_module import BaseModule
from ..pipeline_context import PipelineContext
from ..lib.edit_distance import levenshtein

logger = logging.getLogger('bookbot.core.definition_linker')


class DefinitionLinker(BaseModule):
    """Definition linking module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.definitions = {}
        self.word_forms = {}

    def process(self, input_data) -> dict:
        """
        Process input data and link words to definitions.

        Args:
            input_data: PipelineContext or dict with 'sentences' key

        Returns:
            Dict with 'sentences' key (updated with definition links)
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Load definitions
        self._load_definitions()

        # Link words to definitions
        link_count = 0
        for sent in sentences:
            for token in sent.get('tokens', []):
                if not token.get('is_punctuation', False):
                    link = self._find_definition(token)
                    if link:
                        token['definition_id'] = link[0]
                        token['def_link_confidence'] = link[1]
                        link_count += 1

        self._initialized = True
        self.logger.info(f"Linked {link_count} words to definitions")

        return {'sentences': sentences}

    def _load_definitions(self):
        """Load definitions from database."""
        if not self.db:
            return

        try:
            results = self.db.execute(
                "SELECT definition_id, word_lower, pos_canonical, definition FROM definitions"
            )
            for def_id, word_lower, pos, definition in results:
                if word_lower not in self.definitions:
                    self.definitions[word_lower] = []
                self.definitions[word_lower].append({
                    'id': def_id,
                    'pos': pos,
                    'definition': definition,
                })

            self.logger.info(f"Loaded {len(self.definitions)} definition entries")
        except Exception as e:
            self.logger.warning(f"Could not load definitions: {e}")

    def _find_definition(self, token: Dict) -> Optional[Tuple[int, float]]:
        """
        Find definition for a token.

        Args:
            token: Token dict

        Returns:
            Tuple of (definition_id, confidence) or None
        """
        word = token.get('token_lower', '')
        if not word:
            return None

        # Exact match
        if word in self.definitions:
            defs = self.definitions[word]
            if len(defs) == 1:
                return (defs[0]['id'], 1.0)
            else:
                # Multiple definitions - use POS to disambiguate
                pos = token.get('pos_tag', 'NN')
                best = self._select_by_pos(defs, pos)
                if best:
                    return (best['id'], 0.9)

        # Try morphological variants
        morph = self._try_morphological(word)
        if morph and morph in self.definitions:
            defs = self.definitions[morph]
            if len(defs) == 1:
                return (defs[0]['id'], 0.8)
            else:
                pos = token.get('pos_tag', 'NN')
                best = self._select_by_pos(defs, pos)
                if best:
                    return (best['id'], 0.7)

        # Fuzzy match
        if self.get_config('skip_pos_disambiguation', True):
            fuzzy = self._fuzzy_match(word)
            if fuzzy:
                return (fuzzy[0], 0.6)

        return None

    def _select_by_pos(self, definitions: List[Dict], pos: str) -> Optional[Dict]:
        """
        Select definition by POS tag.

        Args:
            definitions: List of definition dicts
            pos: POS tag to match

        Returns:
            Best matching definition or None
        """
        # Map POS tags to canonical forms
        pos_map = {
            'NN': 'n.', 'NNS': 'n.', 'NNP': 'n.', 'NNPS': 'n.',
            'VB': 'v. t.', 'VBD': 'v.', 'VBG': 'v.', 'VBN': 'v.',
            'VBP': 'v.', 'VBZ': 'v.',
            'JJ': 'a.', 'JJR': 'a.', 'JJS': 'a.',
            'RB': 'adv.', 'RBR': 'adv.', 'RBS': 'adv.',
            'IN': 'prep.',
            'CC': 'conj.',
            'UH': 'interj.',
        }

        canonical_pos = pos_map.get(pos, '')

        # Find matching definition
        for defn in definitions:
            if defn.get('pos') == canonical_pos:
                return defn

        # Return first definition if no match
        return definitions[0] if definitions else None

    def _try_morphological(self, word: str) -> Optional[str]:
        """
        Try morphological variants of a word.

        Args:
            word: Input word

        Returns:
            Base form or None
        """
        # Try removing common suffixes
        suffixes = ['ing', 'ed', 's', 'es', 'ly', 'er', 'est', 'tion', 'ment']

        for suffix in suffixes:
            if word.endswith(suffix):
                base = word[:-len(suffix)]
                if base in self.definitions:
                    return base

        # Try with 'e' added
        if word + 'e' in self.definitions:
            return word + 'e'

        # Try doubling last consonant
        if len(word) > 2 and word[-1] == word[-2]:
            base = word[:-1]
            if base in self.definitions:
                return base

        return None

    def _fuzzy_match(self, word: str) -> Optional[Tuple[int, float]]:
        """
        Find closest match by edit distance.

        Args:
            word: Input word

        Returns:
            Tuple of (definition_id, confidence) or None
        """
        best_word = None
        best_distance = float('inf')
        max_distance = self.get_config('max_definition_match_candidates', 5)

        for dict_word in self.definitions.keys():
            if abs(len(dict_word) - len(word)) > max_distance:
                continue
            dist = levenshtein(word, dict_word)
            if dist < best_distance and dist <= max_distance:
                best_distance = dist
                best_word = dict_word

        if best_word and best_distance <= max_distance:
            defs = self.definitions[best_word]
            if defs:
                return (defs[0]['id'], 0.6 - (best_distance * 0.1))

        return None
