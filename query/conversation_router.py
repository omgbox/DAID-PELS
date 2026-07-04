"""
Conversation Router
Routes user intents to appropriate handlers (book QA, conversation, personal, knowledge).
"""

import logging
from typing import Dict

from ..core.base_module import BaseModule

logger = logging.getLogger('bookbot.query.conversation_router')


class ConversationRouter(BaseModule):
    """Routes queries to appropriate handlers based on intent."""

    # Intent → route mapping
    ROUTES = {
        # Conversational intents
        'GREETING': 'conversational',
        'FAREWELL': 'conversational',
        'HELP': 'conversational',
        'EMOTIONAL': 'conversational',
        'OPINION': 'conversational',
        'CORRECTION': 'conversational',
        'ACKNOWLEDGMENT': 'conversational',
        'STATEMENT': 'conversational',

        # Personal statement intents
        'PERSONAL_STATEMENT': 'personal',

        # General knowledge intents
        'GENERAL_KNOWN': 'knowledge',

        # Book QA intents (existing)
        'DEFINITIONAL': 'book',
        'FACTUAL': 'book',
        'CAUSAL': 'book',
        'TEMPORAL': 'book',
        'COMPARATIVE': 'book',
        'SUMMARIZATION': 'book',
        'EXPLANATORY': 'book',
        'LISTING': 'book',
    }

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and determine route.

        Args:
            input_data: Dict with 'intent' and 'query' keys

        Returns:
            Dict with 'route' key
        """
        intent = input_data.get('intent', 'EXPLANATORY')
        query = input_data.get('query', '')

        route = self.route(intent, query)

        return {'route': route}

    def route(self, intent: str, query: str = '') -> str:
        """
        Determine the route for a given intent.

        Args:
            intent: Classified intent
            query: Original query text (for fallback)

        Returns:
            Route string: 'conversational', 'personal', 'knowledge', or 'book'
        """
        route = self.ROUTES.get(intent, 'book')

        # Override: if query mentions a known book entity, route to book
        # (e.g., "What do you think about Elizabeth?" should be book, not opinion)
        if route == 'opinion' and self._mentions_book_entity(query):
            route = 'book'

        logger.debug(f"Intent: {intent} → Route: {route}")
        return route

    def _mentions_book_entity(self, query: str) -> bool:
        """Check if query mentions a known book entity."""
        if not self.db_manager:
            return False

        try:
            # Check if any capitalized word is a known entity
            words = query.split()
            for word in words:
                cleaned = word.strip('.,;:!?()[]"\'')
                if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                    result = self.db_manager.execute(
                        "SELECT 1 FROM entities WHERE canonical_name LIKE ? LIMIT 1",
                        (f'%{cleaned}%',)
                    )
                    if result:
                        return True
        except Exception:
            pass

        return False
