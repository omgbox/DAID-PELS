"""
BookBot Idiom Detector
Hash-table lookup for idiom detection.
"""

import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.idiom_detector')


class IdiomDetector(BaseModule):
    """Idiom detection module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.idioms = {}

    def process(self, input_data) -> dict:
        """
        Process input data and detect idioms.

        Args:
            input_data: PipelineContext or dict with 'sentences' key

        Returns:
            Dict with 'idioms' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Load idioms
        self._load_idioms()

        # Detect idioms
        idiom_instances = []
        for sent in sentences:
            instances = self._detect_idioms(sent)
            idiom_instances.extend(instances)

        self._initialized = True
        self.logger.info(f"Found {len(idiom_instances)} idiom instances")

        return {'idioms': idiom_instances}

    def _load_idioms(self):
        """Load idioms from database."""
        if not self.db:
            return

        try:
            results = self.db.execute(
                "SELECT idiom_id, idiom_text, meaning FROM idiom_lexicon"
            )
            for idiom_id, text, meaning in results:
                self.idioms[text.lower()] = {
                    'id': idiom_id,
                    'meaning': meaning,
                }

            self.logger.info(f"Loaded {len(self.idioms)} idioms")
        except Exception as e:
            self.logger.warning(f"Could not load idioms: {e}")

    def _detect_idioms(self, sent: Dict) -> List[Dict]:
        """
        Detect idioms in a sentence.

        Args:
            sent: Sentence dict

        Returns:
            List of idiom instance dicts
        """
        instances = []
        tokens = sent.get('tokens', [])
        text = sent.get('text', '').lower()

        # Sliding window approach
        for idiom_text, idiom_data in self.idioms.items():
            words = idiom_text.split()
            window_size = len(words)

            for i in range(len(tokens) - window_size + 1):
                window_tokens = tokens[i:i + window_size]
                window_text = ' '.join(t.get('token_lower', '') for t in window_tokens)

                if window_text == idiom_text:
                    instances.append({
                        'idiom_id': idiom_data['id'],
                        'idiom_text': idiom_text,
                        'meaning': idiom_data['meaning'],
                        'sentence_id': sent.get('sentence_id'),
                        'token_start': i,
                        'token_end': i + window_size - 1,
                        'confidence': 1.0,
                    })

        return instances
