"""
BookBot Query Classifier
Rule-based + keyword intent classification.
"""

import re
import logging
from typing import Dict, List, Optional

from ..core.base_module import BaseModule

logger = logging.getLogger('bookbot.query.query_classifier')


class QueryClassifier(BaseModule):
    """Query intent classification module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and classify query intent.

        Args:
            input_data: Dict with 'query' key

        Returns:
            Dict with 'intent' and 'query_terms' keys
        """
        query = input_data.get('query', '')
        intent = self._classify_intent(query)
        query_terms = self._extract_query_terms(query)

        self._initialized = True

        return {
            'intent': intent,
            'query_terms': query_terms,
        }

    def _classify_intent(self, query: str) -> str:
        """
        Classify query intent.

        Args:
            query: Query text

        Returns:
            Intent string
        """
        query_lower = query.lower().strip()

        # Definitional
        if re.search(r'\b(what does|define|meaning of|what is|what are)\b', query_lower):
            return 'DEFINITIONAL'

        # Who/What is X? (definitional - describes an entity)
        if re.search(r'\bwho (is|was|are|were)\b', query_lower):
            return 'DEFINITIONAL'

        # Factual (who, what, when, where, how many)
        if re.search(r'\b(who|when|where|how many|how much)\b', query_lower):
            return 'FACTUAL'

        # What questions (could be definitional or factual)
        if re.search(r'\bwhat\b', query_lower):
            # Check if it's asking about a specific entity
            if re.search(r'\bwhat (happened|did|was|were|is|are)\b', query_lower):
                return 'FACTUAL'
            return 'DEFINITIONAL'

        # Causal
        if re.search(r'\b(why|what caused|reason|because|what made)\b', query_lower):
            return 'CAUSAL'

        # Temporal
        if re.search(r'\b(when|before|after|during|timeline|what time)\b', query_lower):
            return 'TEMPORAL'

        # Comparative
        if re.search(r'\b(difference|compare|vs|versus|better|worse)\b', query_lower):
            return 'COMPARATIVE'

        # Summarization
        if re.search(r'\b(summarize|summary|overview|what about|tell me about)\b', query_lower):
            return 'SUMMARIZATION'

        # Listing
        if re.search(r'\b(list|all|every|enumerate|name)\b', query_lower):
            return 'LISTING'

        # Explanatory (default)
        return 'EXPLANATORY'

    def _extract_query_terms(self, query: str) -> List[str]:
        """
        Extract important query terms.

        Args:
            query: Query text

        Returns:
            List of query terms
        """
        # Remove common stop words
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
            'with', 'on', 'at', 'from', 'by', 'about', 'as', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'between', 'out', 'off',
            'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there',
            'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'or',
            'if', 'while', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
            'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'it', 'its',
            'they', 'them', 'their', 'what', 'which', 'who', 'whom', 'whose',
        }

        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        query_terms = [word for word in words if word not in stop_words and len(word) > 2]

        return query_terms
