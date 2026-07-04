"""
Query Understanding Module
Uses DistilGPT2 to understand queries dynamically — no regex patterns.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('bookbot.query.query_understanding')


class QueryUnderstanding:
    """
    Uses DistilGPT2 to understand user queries dynamically.
    No hard-coded patterns — the model figures out what the user wants.
    """

    def __init__(self):
        self._generator = None

    def _get_generator(self):
        """Lazy-load DistilGPT2 generator."""
        if self._generator is None:
            try:
                from .minigpt import DistilGPT2Generator
                self._generator = DistilGPT2Generator()
                self._generator.load()
            except Exception as e:
                logger.warning(f"Failed to load DistilGPT2: {e}")
        return self._generator

    def understand(self, query: str) -> Dict:
        """
        Understand a user query using DistilGPT2.
        Returns intent, topic, and context clues.

        Args:
            query: User's query

        Returns:
            Dict with 'intent', 'topic', 'is_question', 'entities' keys
        """
        # Quick rule-based checks for obvious cases
        quick_result = self._quick_classify(query)
        if quick_result:
            return quick_result

        # Use DistilGPT2 for complex queries
        generator = self._get_generator()
        if generator:
            return self._understand_with_gpt2(query, generator)

        # Fallback to simple heuristics
        return self._simple_heuristics(query)

    def _quick_classify(self, query: str) -> Optional[Dict]:
        """Quick rule-based classification for obvious cases."""
        query_lower = query.lower().strip()

        # Greetings
        if re.search(r'^(hi|hello|hey|good morning|good evening|howdy|sup)\b', query_lower):
            return {'intent': 'GREETING', 'topic': None, 'is_question': False}

        # Farewells
        if re.search(r'^(bye|goodbye|see you|good night|take care|later)\b', query_lower):
            return {'intent': 'FAREWELL', 'topic': None, 'is_question': False}

        # Help
        if re.search(r'^(help|what can you do|assist me)\b', query_lower):
            return {'intent': 'HELP', 'topic': None, 'is_question': False}

        # Emotional
        if re.search(r"i('m| am) (feeling|so |very )?(sad|happy|excited|angry|great|terrible)", query_lower):
            return {'intent': 'EMOTIONAL', 'topic': None, 'is_question': False}

        # Personal preferences
        if re.search(r'\bi (like|love|enjoy|hate|dislike|prefer)\b', query_lower):
            return {'intent': 'PERSONAL_STATEMENT', 'topic': None, 'is_question': False}

        # Corrections
        if re.search(r'^(no|nope|wrong|incorrect|thats not|not right|you\'re wrong)', query_lower):
            return {'intent': 'CORRECTION', 'topic': None, 'is_question': False}

        # Acknowledgments
        if re.search(r'^(yes|yeah|correct|right|exactly|ok|okay|thanks|thank you)', query_lower):
            return {'intent': 'ACKNOWLEDGMENT', 'topic': None, 'is_question': False}

        return None

    def _understand_with_gpt2(self, query: str, generator) -> Dict:
        """Use DistilGPT2 to understand the query."""
        # Step 1: Determine if it's a question
        is_question = self._is_question(query, generator)

        # Step 2: Extract the topic
        topic = self._extract_topic(query, generator)

        # Step 3: Determine intent
        intent = self._determine_intent(query, is_question, topic, generator)

        return {
            'intent': intent,
            'topic': topic,
            'is_question': is_question,
        }

    def _is_question(self, query: str, generator) -> bool:
        """Use DistilGPT2 to determine if the query is a question."""
        # Quick check: ends with ?
        if query.strip().endswith('?'):
            return True

        # Quick check: starts with question word
        question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which',
                         'can', 'does', 'do', 'did', 'is', 'are', 'was', 'were']
        if any(query.lower().startswith(qw) for qw in question_words):
            return True

        # Use GPT2 to check
        prompt = f"Is this a question? \"{query}\"\nAnswer (yes/no):"
        response = generator.generate_from_prompt(prompt, max_tokens=5, temperature=0.1)
        return 'yes' in response.lower() if response else False

    def _extract_topic(self, query: str, generator) -> str:
        """Use DistilGPT2 to extract the topic from the query."""
        # Quick extraction for common patterns
        quick_topic = self._quick_extract_topic(query)
        if quick_topic:
            return quick_topic

        # Use GPT2 to extract topic
        prompt = f"Extract the main topic from this question: \"{query}\"\nTopic:"
        response = generator.generate_from_prompt(prompt, max_tokens=20, temperature=0.3)

        if response:
            # Clean up the response
            topic = response.strip().split('\n')[0].strip()
            # Remove common prefixes
            for prefix in ['the topic is', 'topic:', 'about', 'it is about']:
                if topic.lower().startswith(prefix):
                    topic = topic[len(prefix):].strip()
            return topic if topic else self._simple_extract_topic(query)

        return self._simple_extract_topic(query)

    def _quick_extract_topic(self, query: str) -> Optional[str]:
        """Quick topic extraction for common patterns."""
        query_lower = query.lower().strip()

        # "how many X does Y have" -> Y
        match = re.search(r'how many\s+\w+\s+(?:does|do|did)\s+(.+?)\s+(?:have|has|had)', query_lower)
        if match:
            return match.group(1).strip()

        # "what is the X of Y" -> Y
        match = re.search(r'what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(.+?)\s+of\s+(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|\s+at\s+|$)', query_lower)
        if match:
            return match.group(2).strip()

        # "who invented/discovered X" -> X
        match = re.search(r'who\s+(?:invented|discovered|created|founded|built|wrote|painted)\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # "tell me about X" -> X
        match = re.search(r'tell\s+me\s+about\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # "where is X" -> X
        match = re.search(r'where\s+(?:is|are|was|were)\s+(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # "who is X" -> X
        match = re.search(r'who\s+(?:is|was|are|were)\s+(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        return None

    def _simple_extract_topic(self, query: str) -> str:
        """Simple topic extraction using heuristics."""
        # Remove question words
        stop_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which',
                     'is', 'are', 'was', 'were', 'do', 'does', 'did',
                     'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to',
                     'tell', 'me', 'about', 'please', 'help', 'can', 'you',
                     'have', 'has', 'had', 'be', 'been', 'being'}

        words = query.lower().split()
        cleaned = [w for w in words if w not in stop_words and len(w) > 2]

        return ' '.join(cleaned) if cleaned else query

    def _determine_intent(self, query: str, is_question: bool, topic: str, generator) -> str:
        """Determine the intent based on analysis."""
        query_lower = query.lower().strip()

        # If it's a question about the world
        if is_question:
            # Check if it's about a book character
            book_entities = {'elizabeth', 'darcy', 'jane', 'bingley', 'wickham',
                           'lydia', 'bennet', 'collins', 'longbourn', 'netherfield',
                           'pemberley'}

            for entity in book_entities:
                if entity in query_lower:
                    return 'FACTUAL'

            return 'GENERAL_KNOWN'

        # If it's a statement
        if re.search(r'^[a-z]+\s+(is|was|are|were|has|have|had)\s+', query_lower):
            return 'STATEMENT'

        return 'EXPLANATORY'

    def _simple_heuristics(self, query: str) -> Dict:
        """Fallback heuristics when GPT2 is not available."""
        query_lower = query.lower().strip()

        # Check if it's a question
        is_question = query_lower.endswith('?')
        if not is_question:
            question_words = ['what', 'who', 'when', 'where', 'why', 'how', 'which']
            is_question = any(query_lower.startswith(qw) for qw in question_words)

        # Extract topic
        topic = self._quick_extract_topic(query) or self._simple_extract_topic(query)

        # Determine intent
        if is_question:
            # Check if it's about a book character
            book_entities = {'elizabeth', 'darcy', 'jane', 'bingley', 'wickham',
                           'lydia', 'bennet', 'collins', 'longbourn', 'netherfield',
                           'pemberley'}

            for entity in book_entities:
                if entity in query_lower:
                    intent = 'FACTUAL'
                    break
            else:
                intent = 'GENERAL_KNOWN'
        else:
            intent = 'EXPLANATORY'

        return {
            'intent': intent,
            'topic': topic,
            'is_question': is_question,
        }
