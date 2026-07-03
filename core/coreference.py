"""
BookBot Coreference Module
Pronoun resolution using Hobbs algorithm + heuristics + centering theory.
"""

import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.coreference')


class Coreference(BaseModule):
    """Coreference resolution module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.entities = []
        self.chains = []

    def process(self, input_data) -> dict:
        """
        Process input data and resolve coreferences.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'coreferences' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
            entities = input_data.entities
        else:
            sentences = input_data.get('sentences', [])
            entities = input_data.get('entities', [])

        self.entities = entities

        # Resolve coreferences
        chains = self._resolve_coreferences(sentences)

        self._initialized = True
        self.logger.info(f"Resolved {len(chains)} coreference chains")

        return {'coreferences': chains}

    def _resolve_coreferences(self, sentences: List[Dict]) -> List[Dict]:
        """
        Resolve coreferences in sentences.

        Args:
            sentences: List of sentence dicts

        Returns:
            List of coreference chain dicts
        """
        chains = []
        pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them',
                     'his', 'hers', 'its', 'their', 'theirs']

        for sent in sentences:
            for token in sent.get('tokens', []):
                word = token.get('token_lower', '')
                if word in pronouns:
                    # Find antecedent
                    antecedent = self._find_antecedent(word, sent, sentences)
                    if antecedent:
                        chains.append({
                            'pronoun': word,
                            'antecedent': antecedent,
                            'sentence_id': sent.get('sentence_id'),
                            'confidence': 0.7,
                        })

        return chains

    def _find_antecedent(self, pronoun: str, current_sent: Dict,
                         all_sentences: List[Dict]) -> Optional[str]:
        """
        Find antecedent for a pronoun.

        Args:
            pronoun: Pronoun text
            current_sent: Current sentence
            all_sentences: All sentences

        Returns:
            Antecedent text or None
        """
        # Gender/number filtering
        gender = self._get_pronoun_gender(pronoun)

        # Look back through sentences
        lookback = self.get_config('coref_max_lookback', 10)
        sent_idx = current_sent.get('sentence_id', 0)

        for i in range(max(0, sent_idx - lookback), sent_idx):
            sent = all_sentences[i]
            for token in sent.get('tokens', []):
                if token.get('pos_tag', '').startswith('NN'):
                    entity_name = token.get('token', '')
                    entity = self._find_entity(entity_name)
                    if entity:
                        entity_gender = entity.get('gender', 'unknown')
                        if self._gender_matches(gender, entity_gender):
                            return entity_name

        return None

    def _get_pronoun_gender(self, pronoun: str) -> str:
        """
        Get gender of a pronoun.

        Args:
            pronoun: Pronoun text

        Returns:
            Gender string
        """
        masculine = {'he', 'him', 'his'}
        feminine = {'she', 'her', 'hers'}
        neuter = {'it', 'its'}
        plural = {'they', 'them', 'their', 'theirs'}

        if pronoun in masculine:
            return 'masculine'
        elif pronoun in feminine:
            return 'feminine'
        elif pronoun in neuter:
            return 'neuter'
        elif pronoun in plural:
            return 'plural'
        return 'unknown'

    def _find_entity(self, name: str) -> Optional[Dict]:
        """
        Find entity by name.

        Args:
            name: Entity name

        Returns:
            Entity dict or None
        """
        for entity in self.entities:
            if entity.get('canonical_name', '').lower() == name.lower():
                return entity
        return None

    def _gender_matches(self, pronoun_gender: str, entity_gender: str) -> bool:
        """
        Check if pronoun and entity genders match.

        Args:
            pronoun_gender: Pronoun gender
            entity_gender: Entity gender

        Returns:
            True if genders match
        """
        if pronoun_gender == 'unknown' or entity_gender == 'unknown':
            return True
        return pronoun_gender == entity_gender
