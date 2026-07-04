"""
Conversational Pipeline — Like a real AI assistant
Question → Understand → Retrieve → Generate → Answer
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.conversational_pipeline')


class ConversationalPipeline:
    """
    A simple, conversational pipeline — like ChatGPT/Claude.
    No rigid patterns, no separate handlers — just understand and respond.
    """

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._generator = None
        self._wiki = None
        self._conversation_history = []

    def _get_generator(self):
        """Lazy-load DistilGPT2."""
        if self._generator is None:
            try:
                from .minigpt import DistilGPT2Generator
                self._generator = DistilGPT2Generator()
                self._generator.load()
                logger.info("Loaded DistilGPT2 for conversation")
            except Exception as e:
                logger.warning(f"DistilGPT2 not available: {e}")
        return self._generator

    def _get_wiki(self):
        """Lazy-load Wikipedia API."""
        if self._wiki is None:
            try:
                import wikipediaapi
                self._wiki = wikipediaapi.Wikipedia(
                    user_agent='BookBot/1.0',
                    language='en'
                )
                logger.info("Wikipedia API available")
            except Exception as e:
                logger.warning(f"Wikipedia not available: {e}")
        return self._wiki

    def chat(self, query: str) -> str:
        """
        Process a user message and return a natural response.
        Like a real AI assistant — no rigid patterns.

        Args:
            query: User's message

        Returns:
            Natural language response
        """
        # Step 1: Quick responses for obvious cases
        quick = self._quick_response(query)
        if quick:
            self._add_to_history(query, quick)
            return quick

        # Step 2: Extract what the user is asking about
        topic = self._extract_topic(query)

        # Step 3: Search Wikipedia for facts
        facts = self._search_wikipedia(topic) if topic else None

        # Step 4: Generate a natural response
        if facts:
            response = self._generate_response(query, facts)
        else:
            response = self._generate_conversational_response(query)

        # Step 5: Add to conversation history
        self._add_to_history(query, response)

        return response

    def _quick_response(self, query: str) -> Optional[str]:
        """Quick responses for simple cases — no need for Wikipedia."""
        query_lower = query.lower().strip()

        # Greetings
        if re.search(r'^(hi|hello|hey|good morning|good evening|howdy|sup)\b', query_lower):
            import random
            return random.choice([
                "Hey! How can I help you?",
                "Hello! What would you like to know?",
                "Hi there! Ask me anything.",
            ])

        # Farewells
        if re.search(r'^(bye|goodbye|see you|good night|take care)\b', query_lower):
            import random
            return random.choice([
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care! Feel free to come back anytime.",
            ])

        # Thanks
        if re.search(r'^(thanks|thank you|thx)\b', query_lower):
            import random
            return random.choice([
                "You're welcome!",
                "Happy to help!",
                "Anytime! Let me know if you have more questions.",
            ])

        # Yes/No/OK
        if re.search(r'^(yes|yeah|yep|no|nope|ok|okay|sure)\b', query_lower):
            return "Got it! What else would you like to know?"

        return None

    def _extract_topic(self, query: str) -> Optional[str]:
        """Extract what the user is asking about — flexible, no patterns."""
        query_lower = query.lower().strip()

        # Remove question marks
        query_lower = query_lower.rstrip('?').strip()

        # Common question patterns
        patterns = [
            # "how many X does Y have" -> Y
            (r'how many\s+\w+\s+(?:does|do|did)\s+(.+?)\s+(?:have|has|had)', 1),
            # "what is the X of Y" -> Y
            (r'what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(.+?)\s+of\s+(.+)', 2),
            # "who invented/discovered X" -> X
            (r'who\s+(?:invented|discovered|created|founded|built|wrote|painted)\s+(?:the\s+)?(.+)', 1),
            # "when was X invented" -> X
            (r'when\s+(?:was|did|were)\s+(.+?)\s+(?:invented|discovered|created|happened)', 1),
            # "where is X" -> X
            (r'where\s+(?:is|are|was|were)\s+(.+)', 1),
            # "who is X" -> X
            (r'who\s+(?:is|was|are|were)\s+(.+)', 1),
            # "what is X" -> X
            (r'what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(.+)', 1),
            # "tell me about X" -> X
            (r'tell\s+me\s+about\s+(?:the\s+)?(.+)', 1),
            # "explain X" -> X
            (r'explain\s+(?:the\s+)?(.+)', 1),
        ]

        for pattern, group in patterns:
            match = re.search(pattern, query_lower)
            if match:
                topic = match.group(group).strip()
                # Clean up the topic
                topic = self._clean_topic(topic)
                if topic and len(topic) > 1:
                    return topic

        # Fallback: use the whole query as topic
        return self._clean_topic(query_lower)

    def _clean_topic(self, topic: str) -> str:
        """Clean up a topic string."""
        # Remove common suffixes
        for suffix in [' in the book', ' in the story', ' in the text',
                      ' from the book', ' from the story', ' please',
                      ' for me', ' now', ' today']:
            if topic.endswith(suffix):
                topic = topic[:-len(suffix)]

        # Remove leading articles
        if topic.startswith('the '):
            topic = topic[4:]

        return topic.strip()

    def _search_wikipedia(self, topic: str) -> Optional[str]:
        """Search Wikipedia for facts about a topic."""
        wiki = self._get_wiki()
        if not wiki:
            return None

        # Map common terms
        term_map = {
            'bike': 'bicycle',
            'car': 'automobile',
            'phone': 'telephone',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'vr': 'virtual reality',
        }

        # Try the topic directly
        search_terms = [topic]

        # Add mapped term if available
        if topic.lower() in term_map:
            search_terms.append(term_map[topic.lower()])

        # Try each search term
        for term in search_terms:
            page = wiki.page(term)
            if page.exists():
                # Get first 2-3 sentences
                summary = page.summary
                sentences = summary.split('. ')
                return '. '.join(sentences[:3]) + '.'

        return None

    def _generate_response(self, query: str, facts: str) -> str:
        """Generate a natural response using DistilGPT2 and Wikipedia facts."""
        generator = self._get_generator()

        if generator:
            # Use DistilGPT2 to rewrite the facts as a natural answer
            prompt = f"User asked: {query}\nFacts: {facts}\nNatural answer:"
            response = generator.generate_from_prompt(
                prompt,
                max_tokens=100,
                temperature=0.7
            )

            if response and len(response) > 30:
                # Clean up the response
                response = self._clean_response(response, prompt)
                if response:
                    return response

        # Fallback: return the facts directly
        return facts

    def _generate_conversational_response(self, query: str) -> str:
        """Generate a conversational response when no facts are available."""
        generator = self._get_generator()

        if generator:
            # Use DistilGPT2 to generate a conversational response
            prompt = f"User said: {query}\nResponse:"
            response = generator.generate_from_prompt(
                prompt,
                max_tokens=80,
                temperature=0.8
            )

            if response and len(response) > 20:
                response = self._clean_response(response, prompt)
                if response:
                    return response

        # Fallback responses
        import random
        return random.choice([
            "I'm not sure about that. Could you tell me more?",
            "That's interesting! What else would you like to know?",
            "I don't have specific information about that. Can you ask in a different way?",
            "Hmm, I'm not sure I understand. Could you rephrase that?",
        ])

    def _clean_response(self, response: str, prompt: str) -> Optional[str]:
        """Clean up a generated response."""
        # Remove the prompt if it's repeated
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        # Take only the first few sentences
        sentences = response.split('. ')
        cleaned = '. '.join(sentences[:3])

        # Ensure it ends with punctuation
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '.'

        # Remove any weird artifacts
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Collapse whitespace
        cleaned = re.sub(r'^[^a-zA-Z]*', '', cleaned)  # Remove leading non-alpha

        return cleaned if len(cleaned) > 20 else None

    def _add_to_history(self, query: str, response: str):
        """Add to conversation history for context."""
        self._conversation_history.append({
            'query': query,
            'response': response,
        })

        # Keep only last 10 turns
        if len(self._conversation_history) > 10:
            self._conversation_history = self._conversation_history[-10:]

    def get_history(self) -> List[Dict]:
        """Get conversation history."""
        return self._conversation_history
