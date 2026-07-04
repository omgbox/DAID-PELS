"""
Personal Statement Handler
Handles personal statements like "I like gardening" and stores user preferences.
"""

import re
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger('bookbot.query.personal_statement_handler')


class PersonalStatementHandler:
    """Handles personal statements and stores user preferences."""

    # Pattern: (regex, sentiment, category)
    PREFERENCE_PATTERNS = [
        # Positive preferences
        (r'i (like|love|enjoy|adore|am into|am fond of|really like|absolutely love) (.+)', 'positive', 'interest'),
        (r'i (like|love|enjoy) (.+) (a lot|very much|so much|a bunch)', 'positive', 'interest'),

        # Negative preferences
        (r'i (hate|dislike|can\'t stand|detest|don\'t like|do not like|loathe) (.+)', 'negative', 'interest'),
        (r'i (hate|dislike) (.+) (a lot|very much|so much)', 'negative', 'interest'),

        # Preferences (X over Y)
        (r'i (prefer) (.+) (over|to|rather than) (.+)', 'preference', 'preference'),

        # Activities
        (r'i (play|do|practice|participate in) (.+)', 'positive', 'activity'),
        (r'i (used to play|used to do|used to practice) (.+)', 'neutral', 'activity'),

        # Possessions
        (r'i (have|own|got) (.+)', 'neutral', 'possession'),
    ]

    # Pattern: (regex, fact_type)
    FACT_PATTERNS = [
        (r'(my name is|i\'m called|call me|people call me) (.+)', 'name'),
        (r'i (am|i\'m) (\d+) (years old|year old)', 'age'),
        (r'i (live in|reside in|am from|come from|am based in) (.+)', 'location'),
        (r'i (work at|work for|am employed at|am employed by) (.+)', 'workplace'),
        (r'i (work as|am a|am an|my job is) (.+)', 'occupation'),
        (r'i (study|am studying|major in|am majoring in) (.+)', 'education'),
        (r'i (go to|attend) (.+)', 'education'),
        (r'(i have|i\'ve got) (\d+) (kids|children|cats|dogs|pets|brothers|sisters)', 'family'),
    ]

    def __init__(self, user_profile=None):
        self.user_profile = user_profile

    def process(self, query: str) -> Dict:
        """
        Process a personal statement.

        Args:
            query: User's statement

        Returns:
            Dict with 'response', 'stored', 'category', 'value' keys
        """
        query_lower = query.lower().strip()

        # Try preference patterns
        for pattern, sentiment, category in self.PREFERENCE_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return self._handle_preference(match, sentiment, category, query)

        # Try fact patterns
        for pattern, fact_type in self.FACT_PATTERNS:
            match = re.search(pattern, query_lower)
            if match:
                return self._handle_fact(match, fact_type, query)

        # Generic personal statement
        return self._handle_generic(query)

    def _handle_preference(self, match, sentiment: str, category: str, query: str) -> Dict:
        """Handle a preference statement."""
        # Extract the preference value
        if sentiment == 'preference':
            # "I prefer X over Y"
            value = match.group(2).strip('.,!?')
            alternative = match.group(4).strip('.,!?')
            response = f"I'll remember that you prefer {value} over {alternative}."
            stored_value = f"{value} over {alternative}"
        else:
            # "I like X" or "I hate X"
            value = match.group(2).strip('.,!?')
            if sentiment == 'positive':
                response = self._positive_response(value)
            else:
                response = self._negative_response(value)
            stored_value = value

        # Store in user profile
        if self.user_profile:
            self.user_profile.store_preference(category, stored_value, sentiment)

        return {
            'response': response,
            'stored': True,
            'category': category,
            'value': stored_value,
            'sentiment': sentiment,
        }

    def _handle_fact(self, match, fact_type: str, query: str) -> Dict:
        """Handle a personal fact statement."""
        if fact_type == 'name':
            value = match.group(2).strip('.,!?')
            response = f"Nice to meet you, {value}! I'll remember that."
        elif fact_type == 'age':
            age = match.group(2)
            response = f"Got it, you're {age} years old."
        elif fact_type == 'location':
            value = match.group(2).strip('.,!?')
            response = f"I'll remember that you're from {value}."
        elif fact_type == 'workplace':
            value = match.group(2).strip('.,!?')
            response = f"Noted — you work at {value}."
        elif fact_type == 'occupation':
            value = match.group(2).strip('.,!?')
            response = f"So you're a {value}. That's interesting!"
        elif fact_type == 'education':
            value = match.group(2).strip('.,!?')
            response = f"You study {value}. What do you enjoy most about it?"
        elif fact_type == 'family':
            count = match.group(2)
            item = match.group(3)
            response = f"Noted — you have {count} {item}."
        else:
            value = match.group(0).strip('.,!?')
            response = f"I'll remember that."

        # Store in user profile
        if self.user_profile:
            self.user_profile.store_fact(fact_type, value if 'value' in dir() else match.group(0))

        return {
            'response': response,
            'stored': True,
            'category': fact_type,
            'value': value if 'value' in dir() else match.group(0),
        }

    def _handle_generic(self, query: str) -> Dict:
        """Handle a generic personal statement that doesn't match patterns."""
        return {
            'response': "That's interesting! Tell me more about that.",
            'stored': False,
            'category': 'general',
            'value': query,
        }

    def _positive_response(self, value: str) -> str:
        """Generate a response for positive preferences."""
        import random
        responses = [
            f"That's wonderful! I'll remember that you enjoy {value}.",
            f"Great to know you like {value}! What specifically interests you about it?",
            f"Noted — you're into {value}. How did you get started with that?",
            f"I'll keep that in mind. {value.title()} is a great interest to have!",
        ]
        return random.choice(responses)

    def _negative_response(self, value: str) -> str:
        """Generate a response for negative preferences."""
        import random
        responses = [
            f"I understand you don't care for {value}. I'll remember that.",
            f"Noted — {value} isn't your thing. That's fair enough.",
            f"I'll keep that in mind. Everyone has their preferences.",
        ]
        return random.choice(responses)
