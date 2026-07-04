"""
Response Personalizer
Post-processes responses to incorporate user preferences and conversation history.
"""

import re
import logging
import random
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.query.response_personalizer')


class ResponsePersonalizer:
    """Personalizes responses based on user profile and conversation history."""

    def __init__(self):
        pass

    def personalize(self, response: str, intent: str, user_profile=None,
                   conversation_memory=None) -> str:
        """
        Personalize a response based on user context.

        Args:
            response: Original response text
            intent: Query intent
            user_profile: UserProfile instance
            conversation_memory: ConversationMemory instance

        Returns:
            Personalized response text
        """
        if not user_profile and not conversation_memory:
            return response

        # Get user context
        user_name = None
        preferences = []
        recent_topics = []

        if user_profile:
            user_name = user_profile.get_user_name()
            preferences = user_profile.get_preferences()

        if conversation_memory:
            if not user_name:
                user_name = conversation_memory.get_user_name()
            recent_topics = conversation_memory.topic_tracker.current_keywords[:3]

        # Apply personalization strategies
        personalized = response

        # Strategy 1: Add user name for greetings
        if intent == 'GREETING' and user_name:
            personalized = self._add_name_to_greeting(personalized, user_name)

        # Strategy 2: Reference preferences for relevant topics
        if preferences and intent in ('GENERAL_KNOWN', 'OPINION', 'DEFINITIONAL'):
            personalized = self._reference_preferences(personalized, preferences)

        # Strategy 3: Add context from recent conversation
        if recent_topics and intent in ('GENERAL_KNOWN', 'FACTUAL'):
            personalized = self._add_conversation_context(personalized, recent_topics)

        return personalized

    def _add_name_to_greeting(self, response: str, name: str) -> str:
        """Add user name to greeting response."""
        # Check if name is already in response
        if name.lower() in response.lower():
            return response

        # Add name to greeting
        greeting_patterns = [
            (r'^(Hello|Hi|Hey)', f'\\1, {name}'),
            (r'^(Good morning|Good afternoon|Good evening)', f'\\1, {name}'),
        ]

        for pattern, replacement in greeting_patterns:
            if re.match(pattern, response, re.IGNORECASE):
                return re.sub(pattern, replacement, response, count=1, flags=re.IGNORECASE)

        return response

    def _reference_preferences(self, response: str, preferences: List[Dict]) -> str:
        """Reference user preferences when relevant."""
        # Check if any preference is relevant to the response
        for pref in preferences[:3]:  # Check top 3 preferences
            value = pref.get('value', '').lower()
            sentiment = pref.get('sentiment', 'positive')

            # Check if preference topic appears in response
            if value and value in response.lower():
                # Add a personal reference
                if sentiment == 'positive':
                    references = [
                        f" Since you enjoy {value}, you might find this interesting.",
                        f" As someone who likes {value}, you may appreciate this.",
                        f" Given your interest in {value}, this is worth noting.",
                    ]
                else:
                    references = [
                        f" I recall you mentioned {value} isn't your thing.",
                        f" Since you're not fond of {value}, you might want to skip this.",
                    ]

                # Only add reference sometimes (30% chance)
                if random.random() < 0.3:
                    response += random.choice(references)
                break

        return response

    def _add_conversation_context(self, response: str, recent_topics: List[str]) -> str:
        """Add context from recent conversation topics."""
        # Only add context if response is about a different topic
        response_lower = response.lower()
        for topic in recent_topics:
            if topic.lower() in response_lower:
                # Topic already mentioned, no need to add context
                return response

        # Add a subtle connection to recent topics
        if recent_topics and random.random() < 0.2:  # 20% chance
            topic = recent_topics[0]
            connections = [
                f" By the way, we were just discussing {topic}.",
                f" Speaking of which, related to our earlier topic about {topic}...",
            ]
            response += random.choice(connections)

        return response

    def format_for_display(self, response: str, intent: str, confidence: float,
                          sources: List[str] = None) -> str:
        """
        Format response for display with metadata.

        Args:
            response: Response text
            intent: Query intent
            confidence: Confidence score
            sources: List of source references

        Returns:
            Formatted response string
        """
        parts = [response]

        # Add sources if available
        if sources and intent in ('GENERAL_KNOWN', 'FACTUAL', 'DEFINITIONAL'):
            source_str = ', '.join(sources[:3])
            parts.append(f"\n[Sources: {source_str}]")

        # Add confidence indicator for low-confidence responses
        if confidence < 0.3:
            parts.append("\n[Note: I'm not very confident about this answer.]")

        return ''.join(parts)
