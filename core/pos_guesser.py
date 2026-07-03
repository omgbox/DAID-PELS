"""
BookBot POS Guesser
POS guessing from dictionary definition patterns + distributional evidence.
"""

import re
import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.pos_guesser')


class POSGuesser(BaseModule):
    """POS guessing module for dictionary entries with empty POS fields."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.phase = 0  # 0 = not started, 1 = definition-based, 2 = distributional

    def process(self, input_data) -> dict:
        """
        Process input data and guess POS for empty entries.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'pos_guesses' key
        """
        if isinstance(input_data, PipelineContext):
            context = input_data
        else:
            context = None

        if self.phase == 0:
            # Phase 1: Definition-based rules
            guesses = self._phase1_definition_based(context)
            self.phase = 1
            self.logger.info(f"Phase 1 (definition-based): {len(guesses)} POS guesses")
        elif self.phase == 1:
            # Phase 2: Distributional evidence
            guesses = self._phase2_distributional(context)
            self.phase = 2
            self.logger.info(f"Phase 2 (distributional): {len(guesses)} POS updates")
        else:
            guesses = {}

        self._initialized = True

        return {'pos_guesses': guesses}

    def advance_phase(self):
        """Advance to next phase."""
        self.phase += 1

    def _phase1_definition_based(self, context: PipelineContext = None) -> Dict[str, str]:
        """
        Phase 1: Guess POS from definition patterns.

        Returns:
            Dict mapping words to guessed POS
        """
        guesses = {}

        if not self.db:
            return guesses

        # Get entries with empty POS
        try:
            results = self.db.execute(
                "SELECT definition_id, word, definition FROM definitions WHERE pos_canonical IS NULL OR pos_canonical = ''"
            )
        except Exception as e:
            self.logger.warning(f"Could not query definitions: {e}")
            return guesses

        for def_id, word, definition in results:
            pos = self._guess_from_definition(word, definition)
            if pos:
                guesses[word] = pos
                # Update database
                try:
                    self.db.update('definitions',
                                   {'pos_canonical': pos, 'pos_source': 'guessed'},
                                   'definition_id = ?', (def_id,))
                except Exception as e:
                    self.logger.warning(f"Could not update definition: {e}")

        return guesses

    def _phase2_distributional(self, context: PipelineContext = None) -> Dict[str, str]:
        """
        Phase 2: Update POS guesses using distributional evidence from book.

        Returns:
            Dict mapping words to updated POS
        """
        updates = {}

        if not context or not self.db:
            return updates

        # Get POS distribution from book
        pos_dist = self._get_pos_distribution(context)

        # Get entries that were guessed in Phase 1
        try:
            results = self.db.execute(
                "SELECT word, pos_canonical FROM definitions WHERE pos_source = 'guessed'"
            )
        except Exception as e:
            self.logger.warning(f"Could not query guessed entries: {e}")
            return updates

        for word, guessed_pos in results:
            if word in pos_dist:
                # Use distributional evidence
                dist_pos = max(pos_dist[word], key=pos_dist[word].get)
                dist_count = pos_dist[word][dist_pos]

                if dist_count >= 5:
                    # High confidence from distribution
                    updates[word] = dist_pos
                    # Update database
                    try:
                        self.db.update('definitions',
                                       {'pos_canonical': dist_pos, 'pos_source': 'distributional'},
                                       'word_lower = ?', (word.lower(),))
                    except Exception as e:
                        self.logger.warning(f"Could not update definition: {e}")

        return updates

    def _guess_from_definition(self, word: str, definition: str) -> Optional[str]:
        """
        Guess POS from definition text.

        Args:
            word: Word to guess POS for
            definition: Definition text

        Returns:
            Guessed POS or None
        """
        if not definition:
            return self.get_config('pos_guesser_default', 'NN')

        definition_lower = definition.lower().strip()

        # Rule 1: Definition starts with "To " -> VB
        if definition_lower.startswith('to '):
            return 'VB'

        # Rule 2: Definition starts with "One who" or "A person who" -> NN
        if definition_lower.startswith('one who') or definition_lower.startswith('a person who'):
            return 'NN'

        # Rule 3: Definition starts with "The " + noun phrase -> NN
        if definition_lower.startswith('the '):
            return 'NN'

        # Rule 4: Definition starts with "Having" or "Being" -> JJ
        if definition_lower.startswith('having') or definition_lower.startswith('being'):
            return 'JJ'

        # Rule 5: Definition contains "pertaining to" or "relating to" -> JJ
        if 'pertaining to' in definition_lower or 'relating to' in definition_lower:
            return 'JJ'

        # Rule 6: Definition starts with "In " or "On " or "At " -> IN
        if definition_lower.startswith(('in ', 'on ', 'at ')):
            return 'IN'

        # Rule 7: Multi-word entry -> NN
        if ' ' in word:
            return 'NN'

        # Rule 8: Suffix rules
        suffix_pos = self._guess_from_suffix(word)
        if suffix_pos:
            return suffix_pos

        # Default: NN
        return self.get_config('pos_guesser_default', 'NN')

    def _guess_from_suffix(self, word: str) -> Optional[str]:
        """
        Guess POS from word suffix.

        Args:
            word: Word to guess POS for

        Returns:
            Guessed POS or None
        """
        word_lower = word.lower()

        # Noun suffixes
        noun_suffixes = ['-ness', '-tion', '-ment', '-ity', '-ence', '-ance',
                         '-er', '-or', '-ist', '-ism', '-dom', '-ship']
        for suffix in noun_suffixes:
            if word_lower.endswith(suffix):
                return 'NN'

        # Adverb suffixes
        if word_lower.endswith('ly'):
            return 'RB'

        # Adjective suffixes
        adj_suffixes = ['-ful', '-ous', '-ive', '-able', '-ible', '-al',
                        '-less', '-ish', '-like']
        for suffix in adj_suffixes:
            if word_lower.endswith(suffix):
                return 'JJ'

        # Verb suffixes
        verb_suffixes = ['-ize', '-ify', '-ate']
        for suffix in verb_suffixes:
            if word_lower.endswith(suffix):
                return 'VB'

        return None

    def _get_pos_distribution(self, context: PipelineContext) -> Dict[str, Dict[str, int]]:
        """
        Get POS distribution from book sentences.

        Args:
            context: Pipeline context

        Returns:
            Dict mapping words to POS distributions
        """
        distributions = {}

        for sent in context.sentences:
            for token in sent.get('tokens', []):
                word = token.get('token_lower', '')
                pos = token.get('pos_tag', 'NN')

                if word not in distributions:
                    distributions[word] = {}

                distributions[word][pos] = distributions[word].get(pos, 0) + 1

        return distributions
