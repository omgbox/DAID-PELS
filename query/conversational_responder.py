"""
Conversational Responder
Handles greetings, farewells, emotional expressions, and general chitchat.
"""

import random
import logging
from typing import Dict, Optional

logger = logging.getLogger('bookbot.query.conversational_responder')


class ConversationalResponder:
    """Generates responses for conversational intents."""

    # Response templates by intent
    RESPONSES = {
        'GREETING': [
            "Hello! How can I help you today?",
            "Hi there! What would you like to talk about?",
            "Hey! I'm here to chat. What's on your mind?",
            "Hello! Feel free to ask me anything or just chat.",
            "Hi! Great to see you. What shall we discuss?",
            "Hey there! I'm ready for a conversation. What's up?",
        ],
        'FAREWELL': [
            "Goodbye! It was nice chatting with you.",
            "See you later! Feel free to come back anytime.",
            "Take care! I enjoyed our conversation.",
            "Bye! Have a great day!",
            "Farewell! I'll be here whenever you want to chat.",
            "See you! Thanks for the conversation!",
        ],
        'HELP': [
            "I can help you with several things:\n"
            "- Have a conversation about any topic\n"
            "- Answer questions about books (if loaded)\n"
            "- Remember your preferences and interests\n"
            "- Discuss topics you're interested in\n"
            "Just type naturally and I'll do my best!",
            "Here's what I can do:\n"
            "- Chat about anything on your mind\n"
            "- Answer questions using knowledge sources\n"
            "- Remember things you tell me about yourself\n"
            "- Help with general knowledge questions\n"
            "What would you like to try?",
        ],
    }

    # Emotional response templates
    EMOTIONAL_RESPONSES = {
        'positive': [
            "That's wonderful to hear! What's making you feel {emotion}?",
            "I'm glad you're feeling {emotion}! Tell me more about it.",
            "That's great! {emotion_cap} is a beautiful feeling. What brought it on?",
        ],
        'negative': [
            "I'm sorry to hear you're feeling {emotion}. Is there anything I can help with?",
            "That sounds tough. Would you like to talk about what's making you feel {emotion}?",
            "I understand. Feeling {emotion} is never easy. I'm here if you want to chat about it.",
        ],
        'neutral': [
            "I hear you. Tell me more about how you're feeling.",
            "Thanks for sharing that. How can I help?",
        ],
    }

    # Emotion classification
    POSITIVE_EMOTIONS = {'happy', 'excited', 'great', 'amazing', 'grateful', 'thankful',
                         'energetic', 'motivated', 'inspired', 'wonderful', 'fantastic',
                         'thrilled', 'delighted', 'content', 'peaceful', 'joyful'}
    NEGATIVE_EMOTIONS = {'sad', 'angry', 'frustrated', 'anxious', 'worried', 'terrible',
                         'awful', 'depressed', 'overwhelmed', 'stressed', 'tired',
                         'exhausted', 'miserable', 'lonely', 'confused', 'disappointed'}

    def __init__(self):
        pass

    def process(self, intent: str, query: str, context: Dict = None) -> Dict:
        """
        Generate a conversational response.

        Args:
            intent: Classified intent
            query: User's query
            context: Optional context (user profile, conversation history)

        Returns:
            Dict with 'response' key
        """
        if intent == 'GREETING':
            response = self._handle_greeting(query, context)
        elif intent == 'FAREWELL':
            response = self._handle_farewell(query, context)
        elif intent == 'HELP':
            response = self._handle_help(query, context)
        elif intent == 'EMOTIONAL':
            response = self._handle_emotional(query, context)
        elif intent == 'OPINION':
            response = self._handle_opinion(query, context)
        elif intent == 'CORRECTION':
            response = self._handle_correction(query, context)
        elif intent == 'ACKNOWLEDGMENT':
            response = self._handle_acknowledgment(query, context)
        elif intent == 'STATEMENT':
            response = self._handle_statement(query, context)
        else:
            response = self._handle_generic(query, context)

        return {'response': response}

    def _handle_greeting(self, query: str, context: Dict = None) -> str:
        """Handle a greeting."""
        # Check if user has a name stored
        if context and context.get('user_name'):
            name = context['user_name']
            return f"Hello, {name}! How can I help you today?"

        return random.choice(self.RESPONSES['GREETING'])

    def _handle_farewell(self, query: str, context: Dict = None) -> str:
        """Handle a farewell."""
        return random.choice(self.RESPONSES['FAREWELL'])

    def _handle_help(self, query: str, context: Dict = None) -> str:
        """Handle a help request."""
        return random.choice(self.RESPONSES['HELP'])

    def _handle_emotional(self, query: str, context: Dict = None) -> str:
        """Handle an emotional expression."""
        import re

        # Extract the emotion
        emotion_match = re.search(
            r"i('m|\s+am)\s+(feeling|so\s+|very\s+)?\s*(sad|happy|excited|angry|frustrated|anxious|worried|great|terrible|amazing|awful|depressed|overwhelmed|grateful|thankful|stressed|tired|exhausted|energetic|motivated|inspired)",
            query.lower()
        )

        if emotion_match:
            emotion = emotion_match.group(3)
            if emotion in self.POSITIVE_EMOTIONS:
                template = random.choice(self.EMOTIONAL_RESPONSES['positive'])
                return template.format(emotion=emotion, emotion_cap=emotion.capitalize())
            elif emotion in self.NEGATIVE_EMOTIONS:
                template = random.choice(self.EMOTIONAL_RESPONSES['negative'])
                return template.format(emotion=emotion, emotion_cap=emotion.capitalize())

        return random.choice(self.EMOTIONAL_RESPONSES['neutral'])

    def _handle_opinion(self, query: str, context: Dict = None) -> str:
        """Handle an opinion request."""
        import re

        # Extract the topic
        topic_match = re.search(r'what do you think about (.+)', query.lower())
        if not topic_match:
            topic_match = re.search(r'do you (like|prefer|enjoy|believe|agree) (.+)', query.lower())
            if topic_match:
                topic = topic_match.group(2).strip('.,!?')
            else:
                topic = None
        else:
            topic = topic_match.group(1).strip('.,!?')

        if topic:
            # Check if we have stored knowledge about the topic
            if context and context.get('learned_knowledge'):
                for fact in context['learned_knowledge']:
                    if topic.lower() in fact.get('topic', '').lower():
                        return f"Based on what I know, {fact['fact']}"

            # Check if we have user preferences about the topic
            if context and context.get('user_preferences'):
                for pref in context['user_preferences']:
                    if topic.lower() in pref.get('value', '').lower():
                        if pref.get('sentiment') == 'positive':
                            return f"I remember you mentioned you enjoy {topic}. That's a great interest!"
                        elif pref.get('sentiment') == 'negative':
                            return f"I recall you don't care for {topic}. What would you like to discuss instead?"

            # Try to generate a response using DistilGPT2
            try:
                from .minigpt import DistilGPT2Generator
                generator = DistilGPT2Generator()
                if generator.load():
                    prompt = f"I think {topic} is"
                    response = generator.generate_from_prompt(prompt, max_tokens=50, temperature=0.7)
                    if response and len(response) > 20:
                        return f"I think {response}"
            except Exception:
                pass

            # Generic opinion response
            responses = [
                f"That's an interesting topic — {topic}. I don't have a strong opinion on it, but I'd be happy to discuss it with you. What do you think?",
                f"I think {topic} is worth exploring. What's your take on it?",
                f"I find {topic} fascinating to think about. What aspects interest you most?",
            ]
            return random.choice(responses)

        return "That's a good question. What specifically would you like to know my thoughts on?"

    def _handle_correction(self, query: str, context: Dict = None) -> str:
        """Handle a correction or negative feedback."""
        import re

        # Check if user is providing a correction
        correction_match = re.search(r'(?:i meant|i mean|actually|correction|the correct|what i meant)', query.lower())
        if correction_match:
            return "I understand. Could you clarify what you meant?"

        # Generic acknowledgment of correction
        responses = [
            "I apologize for the confusion. Could you tell me more about what you were looking for?",
            "I see. Let me try to understand better. What specifically were you asking about?",
            "Thanks for the correction. Could you provide more details?",
            "I understand. What would you like to know instead?",
        ]
        return random.choice(responses)

    def _handle_acknowledgment(self, query: str, context: Dict = None) -> str:
        """Handle an acknowledgment or confirmation."""
        responses = [
            "Great! Is there anything else you'd like to know?",
            "Understood. What else can I help you with?",
            "Got it. Feel free to ask me anything else.",
            "Thanks! Let me know if you have more questions.",
        ]
        return random.choice(responses)

    def _handle_statement(self, query: str, context: Dict = None) -> str:
        """Handle a factual statement (not a question)."""
        import re

        # Check if user is providing information about themselves
        personal_match = re.search(r'(?:i am|i\'m|i work|i live|my name is|i have|i was born)', query.lower())
        if personal_match:
            return "Thanks for sharing that! Is there anything you'd like to know?"

        # Check if user is stating a fact about the world
        fact_match = re.search(r'^(\w+)\s+(is|was|are|were|has|have|had)\s+', query.lower())
        if fact_match:
            subject = fact_match.group(1)
            return f"Interesting! Tell me more about {subject}."

        # Generic response to statements
        responses = [
            "That's interesting! Tell me more.",
            "Thanks for sharing. What else would you like to discuss?",
            "I see. Is there something specific you'd like to know?",
            "Interesting point! What would you like to explore next?",
        ]
        return random.choice(responses)

    def _handle_generic(self, query: str, context: Dict = None) -> str:
        """Handle a generic conversational input."""
        responses = [
            "That's interesting! Tell me more about that.",
            "I'd love to hear more. What else can you share?",
            "That's a good point. What made you think of that?",
            "Interesting! How do you feel about that?",
        ]
        return random.choice(responses)
