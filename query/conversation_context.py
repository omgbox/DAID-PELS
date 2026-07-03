"""
BookBot Conversation Context
Multi-turn conversation state management.
"""

import logging
from typing import Dict, List, Optional
from collections import deque

from ..core.base_module import BaseModule

logger = logging.getLogger('bookbot.query.conversation_context')


class ConversationContext(BaseModule):
    """Conversation context module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.history = deque(maxlen=self.config.get('conversation_history_size', 5))
        self.current_entities = []

    def process(self, input_data) -> dict:
        """
        Process input data and update conversation context.

        Args:
            input_data: Dict with 'query', 'response' keys

        Returns:
            Dict with 'context' key
        """
        query = input_data.get('query', '')
        response = input_data.get('response', {})

        # Add to history
        self.history.append({
            'query': query,
            'response': response.get('answer', ''),
            'entities': self._extract_entities(query),
        })

        # Update current entities
        self.current_entities = self._extract_entities(query)

        self._initialized = True

        return {'context': self.get_context()}

    def get_context(self) -> Dict:
        """
        Get current conversation context.

        Returns:
            Context dict
        """
        return {
            'history': list(self.history),
            'current_entities': self.current_entities,
            'turn_count': len(self.history),
        }

    def _extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text.

        Args:
            text: Input text

        Returns:
            List of entity strings
        """
        # Simple: extract capitalized words
        entities = []
        for word in text.split():
            if word[0].isupper() and len(word) > 2:
                entities.append(word)
        return entities
