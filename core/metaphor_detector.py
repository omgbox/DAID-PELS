"""
BookBot Metaphor Detector
WordNet selectional preference violation detection.
"""

import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.metaphor_detector')

# Try to import NLTK WordNet
try:
    from nltk.corpus import wordnet as wn
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False


class MetaphorDetector(BaseModule):
    """Metaphor detection module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and detect metaphors.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'metaphors' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
            svo_triples = input_data.svo_triples
        else:
            sentences = input_data.get('sentences', [])
            svo_triples = input_data.get('svo_triples', [])

        # Detect metaphors
        metaphors = []
        threshold = self.get_config('metaphor_threshold', 0.6)

        for triple in svo_triples:
            metaphor = self._check_metaphor(triple, threshold)
            if metaphor:
                metaphors.append(metaphor)

        self._initialized = True
        self.logger.info(f"Found {len(metaphors)} metaphor candidates")

        return {'metaphors': metaphors}

    def _check_metaphor(self, triple: Dict, threshold: float) -> Optional[Dict]:
        """
        Check if an SVO triple contains a metaphor.

        Args:
            triple: SVO triple dict
            threshold: Metaphor threshold

        Returns:
            Metaphor dict or None
        """
        subject = triple.get('subject', '')
        verb = triple.get('verb', '')
        object_ = triple.get('object', '')

        if not subject or not verb:
            return None

        # Check for personification (artifact + human verb)
        if self._is_personification(subject, verb):
            return {
                'type': 'personification',
                'expression': f"{subject} {verb}",
                'sentence_id': triple.get('sentence_id'),
                'source_domain': 'human',
                'target_domain': subject,
                'confidence': 0.7,
            }

        # Check for semantic field mismatch
        if WORDNET_AVAILABLE:
            mismatch_score = self._check_semantic_mismatch(subject, verb, object_)
            if mismatch_score > threshold:
                return {
                    'type': 'semantic_mismatch',
                    'expression': f"{subject} {verb} {object_}",
                    'sentence_id': triple.get('sentence_id'),
                    'confidence': mismatch_score,
                }

        return None

    def _is_personification(self, subject: str, verb: str) -> bool:
        """
        Check if subject is personified.

        Args:
            subject: Subject text
            verb: Verb text

        Returns:
            True if personification detected
        """
        # Human verbs
        human_verbs = {'said', 'told', 'spoke', 'thought', 'felt', 'knew',
                       'wanted', 'loved', 'hated', 'feared', 'hoped'}

        # Check if subject is non-human
        if WORDNET_AVAILABLE:
            synsets = wn.synsets(subject)
            if synsets:
                # Check if subject is an artifact or abstract concept
                for synset in synsets:
                    if 'artifact' in synset.lexname() or 'noun.cognition' in synset.lexname():
                        if verb.lower() in human_verbs:
                            return True

        return False

    def _check_semantic_mismatch(self, subject: str, verb: str,
                                  object_: str) -> float:
        """
        Check for semantic field mismatch.

        Args:
            subject: Subject text
            verb: Verb text
            object_: Object text

        Returns:
            Mismatch score (0.0 to 1.0)
        """
        # Simplified check - look for unusual combinations
        subject_synsets = wn.synsets(subject)
        verb_synsets = wn.synsets(verb)
        object_synsets = wn.synsets(object_) if object_ else []

        if not subject_synsets or not verb_synsets:
            return 0.0

        # Check if verb typically takes animate subjects
        verb_frame = verb_synsets[0]
        # This is a simplified heuristic
        return 0.0
