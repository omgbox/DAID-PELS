"""
BookBot Query Classifier
DYNAMIC intent classification — no hard-coded topics.
"""

import re
import logging
from typing import Dict, List, Optional

from ..core.base_module import BaseModule

logger = logging.getLogger('bookbot.query.query_classifier')


class QueryClassifier(BaseModule):
    """Query intent classification module."""

    # Book-specific entities
    BOOK_ENTITIES = {
        'elizabeth', 'darcy', 'jane', 'bingley', 'wickham', 'lydia',
        'mary', 'kitty', 'bennet', 'collins', 'longbourn', 'netherfield',
        'pemberley', 'meryton', 'hertfordshire', 'derbyshire',
        'pride', 'prejudice', 'austen',
    }

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
        DYNAMIC intent classification.
        If it looks like a question and isn't about a book character,
        treat it as general knowledge.
        """
        query_lower = query.lower().strip()

        # === Conversational intents (check first) ===

        # Greeting
        if re.search(r'^(hi|hello|hey|good morning|good afternoon|good evening|howdy|greetings|what\'s up|sup|yo)\b', query_lower):
            return 'GREETING'

        # Farewell
        if re.search(r'^(bye|goodbye|see you|good night|take care|later|farewell|cheers|cya)\b', query_lower):
            return 'FAREWELL'

        # Help request
        if re.search(r'^(help|can you help|assist me|i need help|could you help|what can you do)\b', query_lower):
            return 'HELP'

        # Emotional expressions
        if re.search(r"i('m|\s+am)\s+(feeling|so\s+|very\s+)?\s*(sad|happy|excited|angry|frustrated|anxious|worried|great|terrible|amazing|awful|depressed|overwhelmed|grateful|thankful|stressed|tired|exhausted|energetic|motivated|inspired)", query_lower):
            return 'EMOTIONAL'

        # Personal statements: preferences
        if re.search(r'\b(i\s+(like|love|enjoy|hate|dislike|prefer|adore|can\'t stand|really like|absolutely love|am into|am fond of))\b', query_lower):
            return 'PERSONAL_STATEMENT'

        # Personal statements: facts about self
        if re.search(r'\b(i\s+am|i\'m|i\s+work|i\s+live|my\s+name\s+is|i\s+have|i\s+was\s+born|i\s+come\s+from|i\s+study|i\s+go\s+to)\b', query_lower):
            return 'PERSONAL_STATEMENT'

        # Opinion requests
        if re.search(r'what do you think|do you (like|prefer|enjoy|believe|agree)|your opinion|what\'s your (view|take|stance)|how do you feel about', query_lower):
            return 'OPINION'

        # Negative feedback / corrections
        if re.search(r'^(no|nope|wrong|incorrect|thats not|that\'s not|not right|not correct|you\'re wrong|you are wrong|i meant|i mean|actually|correction)', query_lower):
            return 'CORRECTION'

        # Acknowledgments / confirmations
        if re.search(r'^(yes|yeah|yep|correct|right|exactly|true|sure|ok|okay|got it|understood|thanks|thank you)', query_lower):
            return 'ACKNOWLEDGMENT'

        # Statements about facts (not questions)
        # "X is Y", "X was Y", "X has Y"
        if re.search(r'^[a-z]+\s+(is|was|are|were|has|have|had|can|will|would|could|should)\s+', query_lower):
            # Check if it's about a book entity
            if self._is_book_query(query_lower):
                return 'FACTUAL'
            # Otherwise treat as a statement (conversational)
            return 'STATEMENT'

        # === Book-specific queries (in the text) ===
        if self._is_book_query(query_lower):
            return 'FACTUAL'

        # === Explicit "in the text/book" queries ===
        if re.search(r'\b(in the (text|book|story|novel|chapter|passage|excerpt))\b', query_lower):
            return 'FACTUAL'

        # === DYNAMIC: If it looks like a question about the world ===
        if self._is_world_question(query_lower):
            return 'GENERAL_KNOWN'

        # === Book-specific intents ===

        # Summarization (only for book-related)
        if re.search(r'\b(summarize|summary|overview)\b', query_lower):
            return 'SUMMARIZATION'

        # Tell me about (could be book or general)
        if re.search(r'\btell me about\b', query_lower):
            if self._is_book_query(query_lower):
                return 'SUMMARIZATION'
            return 'GENERAL_KNOWN'

        # Listing
        if re.search(r'\b(list|all|every|enumerate|name)\b', query_lower):
            return 'LISTING'

        # Explanatory (default)
        return 'EXPLANATORY'

    def _is_world_question(self, query: str) -> bool:
        """
        Dynamically detect if input is a question about the real world.
        No hard-coded topics — just structural detection.
        """
        # Must look like a question
        if not self._looks_like_question(query):
            return False

        # Exclude book queries
        if self._is_book_query(query):
            return False

        return True

    def _looks_like_question(self, query: str) -> bool:
        """Check if text looks like a question."""
        # Starts with question word
        question_starters = [
            'what', 'who', 'when', 'where', 'why', 'how', 'which',
            'can', 'does', 'do', 'did', 'is', 'are', 'was', 'were',
            'tell me', 'explain', 'describe',
        ]

        for starter in question_starters:
            if query.startswith(starter):
                return True

        # Ends with question mark
        if query.rstrip().endswith('?'):
            return True

        return False

    def _is_book_query(self, query: str) -> bool:
        """Check if query is specifically about the book."""
        # Check hardcoded book entities
        for entity in self.BOOK_ENTITIES:
            if entity in query:
                return True

        # Check database for known entities
        if hasattr(self, 'db_manager') and self.db_manager:
            try:
                words = query.split()
                for word in words:
                    cleaned = word.strip('.,;:!?()[]"\'')
                    if cleaned and len(cleaned) > 2:
                        result = self.db_manager.execute(
                            "SELECT 1 FROM entities WHERE canonical_name LIKE ? LIMIT 1",
                            (f'%{cleaned}%',)
                        )
                        if result:
                            return True
            except Exception:
                pass

        return False

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
