"""
BookBot Temporal Reasoner
Temporal expression extraction and ordering.
"""

import re
import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.temporal_reasoner')


class TemporalReasoner(BaseModule):
    """Temporal reasoning module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and extract temporal information.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'temporal_events' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Extract temporal events
        events = []
        for sent in sentences:
            sent_events = self._extract_temporal_expressions(sent)
            events.extend(sent_events)

        self._initialized = True
        self.logger.info(f"Found {len(events)} temporal events")

        return {'temporal_events': events}

    def _extract_temporal_expressions(self, sent: Dict) -> List[Dict]:
        """
        Extract temporal expressions from a sentence.

        Args:
            sent: Sentence dict

        Returns:
            List of temporal event dicts
        """
        events = []
        text = sent.get('text', '')

        # Date patterns
        date_patterns = [
            (r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', 'absolute'),
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', 'absolute'),
            (r'\b\d{4}\b', 'absolute'),
        ]

        for pattern, time_type in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append({
                    'sentence_id': sent.get('sentence_id'),
                    'raw_expression': match.group(),
                    'normalized_time': match.group(),
                    'time_type': time_type,
                })

        # Relative time patterns
        relative_patterns = [
            (r'\b(before|after|when|while|during|until|since)\b', 'relative'),
            (r'\b(the next|the previous|the following)\s+\w+\b', 'relative'),
            (r'\b\d+\s+(hours?|days?|weeks?|months?|years?)\s+(later|before|after)\b', 'relative'),
        ]

        for pattern, time_type in relative_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                events.append({
                    'sentence_id': sent.get('sentence_id'),
                    'raw_expression': match.group(),
                    'normalized_time': match.group(),
                    'time_type': time_type,
                })

        return events
