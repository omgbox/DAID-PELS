"""
BookBot Conversational AI — Built like a real assistant
Simple pipeline: Understand → Retrieve → Generate → Respond
"""

import re
import logging
import random
import os
from typing import Dict, List, Optional

# Suppress verbose logging during model loading
logging.getLogger('bookbot.query.minigpt').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)

# Suppress HuggingFace progress bars
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

logger = logging.getLogger('bookbot.conversational_ai')


class ConversationalAI:
    """
    A conversational AI that works like ChatGPT/Claude.
    No rigid patterns, no separate handlers — just natural conversation.
    """

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._generator = None
        self._rewriter = None
        self._wiki = None
        self._history = []
        self._user_context = {}  # Remember user info

    def _get_generator(self):
        """Lazy-load DistilGPT2 with visual progress bar."""
        if self._generator is None:
            import sys
            import time

            # Progress bar for model loading
            print()  # New line before progress
            bar_len = 30
            for i in range(5):
                progress = (i + 1) * 20
                filled = int(bar_len * progress / 100)
                bar = '#' * filled + '-' * (bar_len - filled)
                labels = ['Downloading', 'Loading tokenizer', 'Loading weights', 'Initializing', 'Ready']
                sys.stdout.write(f'\r  [{bar}] {progress}% {labels[i]}')
                sys.stdout.flush()
                time.sleep(0.2)

            try:
                from .minigpt import DistilGPT2Generator
                self._generator = DistilGPT2Generator()
                self._generator.load()

                # Final progress
                bar = '#' * bar_len
                sys.stdout.write(f'\r  [{bar}] 100% AI model ready!   \n')
                sys.stdout.flush()

            except Exception as e:
                sys.stdout.write(f'\r  AI model error: {str(e)[:50]}   \n')
                sys.stdout.flush()
                logger.debug(f"DistilGPT2 not available: {e}")
        return self._generator

    def _get_wiki(self):
        """Lazy-load Wikipedia with visual progress bar."""
        if self._wiki is None:
            import sys
            import time

            # Progress bar for Wikipedia connection
            print()  # New line before progress
            bar_len = 30
            for i in range(4):
                progress = (i + 1) * 25
                filled = int(bar_len * progress / 100)
                bar = '#' * filled + '-' * (bar_len - filled)
                labels = ['Connecting', 'Authenticating', 'Loading API', 'Ready']
                sys.stdout.write(f'\r  [{bar}] {progress}% {labels[i]}')
                sys.stdout.flush()
                time.sleep(0.15)

            try:
                import wikipediaapi
                self._wiki = wikipediaapi.Wikipedia(user_agent='BookBot/1.0', language='en')

                # Final progress
                bar = '#' * bar_len
                sys.stdout.write(f'\r  [{bar}] 100% Wikipedia ready!   \n')
                sys.stdout.flush()

            except Exception as e:
                sys.stdout.write(f'\r  Wikipedia error: {str(e)[:50]}   \n')
                sys.stdout.flush()
                logger.debug(f"Wikipedia not available: {e}")
        return self._wiki

    def _get_rewriter(self):
        """Lazy-load T5 Paraphrase rewriter with visual progress bar."""
        if self._rewriter is None:
            import sys
            import time

            # Progress bar for model loading
            print()  # New line before progress
            bar_len = 30
            for i in range(5):
                progress = (i + 1) * 20
                filled = int(bar_len * progress / 100)
                bar = '#' * filled + '-' * (bar_len - filled)
                labels = ['Downloading', 'Loading tokenizer', 'Loading weights', 'Initializing', 'Ready']
                sys.stdout.write(f'\r  [{bar}] {progress}% {labels[i]}')
                sys.stdout.flush()
                time.sleep(0.2)

            try:
                from .minigpt import T5Rewriter
                self._rewriter = T5Rewriter()
                self._rewriter.load()

                # Final progress
                bar = '#' * bar_len
                sys.stdout.write(f'\r  [{bar}] 100% Rewriter ready!   \n')
                sys.stdout.flush()

            except Exception as e:
                sys.stdout.write(f'\r  Rewriter error: {str(e)[:50]}   \n')
                sys.stdout.flush()
                logger.debug(f"T5 Paraphrase not available: {e}")
        return self._rewriter

    def chat(self, message: str) -> str:
        """
        Process a message and return a natural response.
        Like talking to a real person.
        """
        message = message.strip()
        if not message:
            return "I'm here! What would you like to talk about?"

        # Step 1: Handle simple cases instantly
        quick = self._instant_response(message)
        if quick:
            self._remember(message, quick)
            return quick

        # Step 2: Understand what the user wants
        intent = self._understand(message)

        # Step 3: Get information based on intent type
        facts = None
        if intent['type'] == 'book':
            # Search book database
            facts = self._search_book(intent['topic'])
        elif intent['needs_info']:
            # Search Wikipedia
            facts = self._get_facts(intent['topic'])

        # Step 4: Generate a natural response
        response = self._respond(message, intent, facts)

        # Step 5: Remember this conversation
        self._remember(message, response)

        return response

    def _instant_response(self, message: str) -> Optional[str]:
        """Instant responses for simple social cues."""
        m = message.lower().strip()

        # Greetings
        if re.match(r'^(hi|hello|hey|howdy|sup|yo|greetings)\b', m):
            return random.choice([
                "Hey! What's on your mind?",
                "Hello! Ask me anything.",
                "Hi there! How can I help?",
            ])

        # Farewells
        if re.match(r'^(bye|goodbye|see ya|later|take care|good night)\b', m):
            return random.choice([
                "Goodbye! Have a great day!",
                "See you later!",
                "Take care! Come back anytime.",
            ])

        # Thanks
        if re.match(r'^(thanks|thank you|thx|cheers)\b', m):
            return random.choice([
                "You're welcome!",
                "Happy to help!",
                "Anytime!",
            ])

        # Simple confirmations/denials
        if re.match(r'^(yes|yeah|yep|no|nope|ok|okay|sure|right|correct)\b', m):
            return "Got it! What else?"

        return None

    def _understand(self, message: str) -> Dict:
        """
        Understand what the user wants.
        Returns intent type and extracted topic.
        """
        m = message.lower().strip()

        # Check if it's a question
        is_question = (
            m.endswith('?') or
            any(m.startswith(w) for w in ['what', 'who', 'when', 'where', 'why', 'how', 'which', 'can', 'does', 'is', 'are', 'was', 'were', 'tell me', 'explain'])
        )

        # Check if it's a personal statement
        is_personal = any(m.startswith(w) for w in ['i am', "i'm", 'i like', 'i love', 'i hate', 'i work', 'i live', 'my name'])

        # Check if it's an emotional expression
        is_emotional = bool(re.search(r"i('m| am) (feeling|so |very )?(sad|happy|excited|angry|great|terrible|tired|stressed)", m))

        # Check if it's about a book character
        book_chars = {'elizabeth', 'darcy', 'jane', 'bingley', 'wickham', 'lydia', 'bennet', 'collins'}
        is_book = any(char in m for char in book_chars)

        # Extract topic
        topic = self._extract_topic(message)

        # Determine what we need
        if is_personal:
            return {'type': 'personal', 'topic': topic, 'needs_info': False}
        elif is_emotional:
            return {'type': 'emotional', 'topic': topic, 'needs_info': False}
        elif is_book:
            return {'type': 'book', 'topic': topic, 'needs_info': True}
        elif is_question:
            return {'type': 'question', 'topic': topic, 'needs_info': True}
        else:
            return {'type': 'statement', 'topic': topic, 'needs_info': False}

    def _respond(self, message: str, intent: Dict, facts: Optional[str]) -> str:
        """Generate a natural response."""
        gen = self._get_generator()

        if intent['type'] == 'personal':
            return self._respond_personal(message)

        if intent['type'] == 'emotional':
            return self._respond_emotional(message)

        if intent['type'] == 'book':
            if facts:
                return facts
            return f"I don't have specific information about {intent['topic']} in my knowledge base."

        if facts:
            return facts

        if gen:
            return self._generate_conversational(message, gen)

        return self._fallback_response(message)

    def _respond_book(self, message: str, topic: str) -> str:
        """Respond to book-related queries."""
        # Search the book database
        if self.db:
            try:
                # Search for sentences mentioning the entity
                rows = self.db.execute(
                    "SELECT raw_text FROM sentences "
                    "WHERE raw_text LIKE ? AND LENGTH(raw_text) > 20 "
                    "AND LENGTH(raw_text) < 300 "
                    "ORDER BY LENGTH(raw_text) DESC LIMIT 5",
                    (f'%{topic}%',)
                )
                if rows:
                    sentences = [r[0] for r in rows]
                    return ' '.join(sentences[:3])
            except Exception:
                pass

        return f"I don't have specific information about {topic} in my knowledge base."

    def _extract_topic(self, message: str) -> str:
        """Extract the main topic from a message."""
        m = message.lower().strip().rstrip('?')

        # Common term mappings (expand abbreviations)
        expand_map = {
            'ufo': 'unidentified flying object',
            'ufos': 'unidentified flying object',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'vr': 'virtual reality',
            'ar': 'augmented reality',
        }

        # Pattern: "how many X does Y have" -> Y
        match = re.search(r'how many\s+\w+\s+(?:does|do|did)\s+(.+?)\s+(?:have|has|had)', m)
        if match:
            return self._expand_topic(match.group(1).strip(), expand_map)

        # Pattern: "what is the X of Y" -> Y
        match = re.search(r'what\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)\s+of\s+(.+)', m)
        if match:
            return self._expand_topic(match.group(2).strip(), expand_map)

        # Pattern: "who/what/where/when is X" -> X
        match = re.search(r'(?:who|what|where|when|why|how)\s+(?:is|are|was|were|do|does|did)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(.+)', m)
        if match:
            topic = match.group(1).strip()
            # Remove trailing words like "reported", "invented", etc.
            topic = re.sub(r'\s+(reported|invented|discovered|created|founded|built|written|painted|happened|occurred)$', '', topic)
            # Remove ordinal words like "first", "last", "next", etc.
            topic = re.sub(r'^(first|last|next|biggest|smallest|oldest|newest|most|best|worst|greatest|famous|important)\s+', '', topic)
            return self._expand_topic(topic, expand_map)

        # Pattern: "tell me about X" -> X
        match = re.search(r'tell\s+me\s+about\s+(?:the\s+)?(.+)', m)
        if match:
            return self._expand_topic(match.group(1).strip(), expand_map)

        # Pattern: "explain X" -> X
        match = re.search(r'explain\s+(?:the\s+)?(.+)', m)
        if match:
            return self._expand_topic(match.group(1).strip(), expand_map)

        # Fallback: remove question words and use the rest
        stop = {'what', 'who', 'when', 'where', 'why', 'how', 'which',
                'is', 'are', 'was', 'were', 'do', 'does', 'did',
                'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to',
                'tell', 'me', 'about', 'can', 'you', 'please',
                'first', 'last', 'next', 'biggest', 'smallest', 'oldest',
                'newest', 'most', 'best', 'worst', 'greatest', 'famous',
                'important', 'reported', 'invented', 'discovered', 'created',
                'founded', 'built', 'written', 'painted', 'happened', 'occurred'}
        words = m.split()
        topic_words = [w for w in words if w not in stop and len(w) > 2]
        topic = ' '.join(topic_words) if topic_words else m

        return self._expand_topic(topic, expand_map)

    def _expand_topic(self, topic: str, expand_map: dict) -> str:
        """Expand abbreviations in topic."""
        for key, val in expand_map.items():
            if key in topic.split():
                topic = topic.replace(key, val)
        return topic

    def _get_facts(self, topic: str) -> Optional[str]:
        """Get facts about a topic from Wikipedia."""
        wiki = self._get_wiki()
        if not wiki:
            return None

        # Common term mappings (ambiguous terms → correct Wikipedia page)
        mappings = {
            'bike': 'bicycle',
            'car': 'automobile',
            'phone': 'telephone',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'python': 'python (programming language)',
            'javascript': 'javascript',
            'java': 'java (programming language)',
            'rust': 'rust (programming language)',
            'go': 'go (programming language)',
            'swift': 'swift (programming language)',
            'kotlin': 'kotlin',
            'typescript': 'typescript',
            'ruby': 'ruby (programming language)',
            'php': 'php',
            'sql': 'sql',
            'html': 'html',
            'css': 'css',
            'react': 'react (javascript library)',
            'angular': 'angular (web framework)',
            'vue': 'vue.js',
            'node': 'node.js',
            'docker': 'docker (software)',
            'kubernetes': 'kubernetes',
            'linux': 'linux',
            'windows': 'microsoft windows',
            'macos': 'macos',
            'android': 'android (operating system)',
            'ios': 'ios',
            'blockchain': 'blockchain',
            'bitcoin': 'bitcoin',
            'ethereum': 'ethereum',
            'machine learning': 'machine learning',
            'deep learning': 'deep learning',
            'neural network': 'artificial neural network',
            'quantum computing': 'quantum computing',
            'ufo': 'unidentified flying object',
            'ufos': 'unidentified flying object',
        }

        # Try mapped term first if available
        if topic.lower() in mappings:
            try:
                page = wiki.page(mappings[topic.lower()])
                if page.exists() and 'may refer to' not in page.summary[:100]:
                    sentences = page.summary.split('. ')
                    return '. '.join(sentences[:3]) + '.'
            except Exception:
                pass

        # Try direct search
        try:
            page = wiki.page(topic)
            if page.exists():
                # Check if it's a disambiguation page
                if 'may refer to' in page.summary[:100]:
                    # Try to find the most common meaning
                    # For now, just return the first option
                    pass
                else:
                    sentences = page.summary.split('. ')
                    return '. '.join(sentences[:3]) + '.'
        except Exception:
            pass

        return None

    def _respond(self, message: str, intent: Dict, facts: Optional[str]) -> str:
        """Generate a natural response."""
        gen = self._get_generator()

        if intent['type'] == 'personal':
            return self._respond_personal(message)

        if intent['type'] == 'emotional':
            return self._respond_emotional(message)

        if facts:
            if gen:
                return self._generate_with_facts(message, facts, gen)
            return facts

        if gen:
            return self._generate_conversational(message, gen)

        return self._fallback_response(message)

    def _respond_personal(self, message: str) -> str:
        """Respond to personal statements."""
        m = message.lower()

        if 'my name is' in m:
            name = message.split('my name is')[-1].strip().rstrip('.')
            self._user_context['name'] = name
            return f"Nice to meet you, {name}! What would you like to know?"

        if any(w in m for w in ['i like', 'i love', 'i enjoy']):
            thing = re.search(r'i (?:like|love|enjoy)\s+(.+)', m)
            if thing:
                return f"That's great! {thing.group(1).strip().title()} is interesting. Tell me more about it!"

        if any(w in m for w in ['i hate', 'i dislike']):
            thing = re.search(r'i (?:hate|dislike)\s+(.+)', m)
            if thing:
                return f"I understand. Not everyone enjoys {thing.group(1).strip()}. What do you prefer instead?"

        return "Thanks for sharing! What else would you like to talk about?"

    def _respond_emotional(self, message: str) -> str:
        """Respond to emotional expressions."""
        m = message.lower()

        if any(w in m for w in ['happy', 'great', 'excited', 'amazing']):
            return "That's wonderful to hear! What's making you feel good?"

        if any(w in m for w in ['sad', 'tired', 'stressed', 'terrible', 'angry']):
            return "I'm sorry to hear that. Is there anything I can help with?"

        return "Thanks for sharing how you feel. What would you like to do?"

    def _search_book(self, topic: str) -> Optional[str]:
        """Search the book database for information about a topic."""
        if not self.db:
            return None

        try:
            # Search for sentences mentioning the entity
            rows = self.db.execute(
                "SELECT raw_text FROM sentences "
                "WHERE raw_text LIKE ? AND LENGTH(raw_text) > 20 "
                "AND LENGTH(raw_text) < 300 "
                "ORDER BY LENGTH(raw_text) DESC LIMIT 5",
                (f'%{topic}%',)
            )
            if rows:
                sentences = [r[0] for r in rows]
                return ' '.join(sentences[:3])
        except Exception as e:
            logger.debug(f"Book search failed: {e}")

        return None

    def _generate_with_facts(self, query: str, facts: str, gen) -> str:
        """Generate a response using facts from Wikipedia."""
        rewriter = self._get_rewriter()

        if rewriter:
            # Use T5 Paraphrase to rewrite the facts as a natural answer
            response = rewriter.rewrite_for_chat(facts, context=query)

            if response and len(response) > 30:
                return response

        # Fallback: return the facts directly
        return facts

    def _generate_conversational(self, message: str, gen) -> str:
        """Generate a conversational response."""
        # Build context from recent history
        context = ""
        if self._history:
            last = self._history[-1]
            context = f"Previous: {last['query']}\n"

        prompt = f"{context}User: {message}\nAssistant:"
        response = gen.generate_from_prompt(prompt, max_tokens=80, temperature=0.8)

        if response and len(response) > 20:
            cleaned = self._clean(response, prompt)
            if cleaned:
                return cleaned

        return self._fallback_response(message)

    def _fallback_response(self, message: str) -> str:
        """Fallback response when nothing else works."""
        return random.choice([
            "I'm not sure about that. Could you tell me more?",
            "Interesting! What else would you like to know?",
            "I don't have information about that. Can you ask differently?",
            "Hmm, I'm not sure I understand. Could you rephrase?",
        ])

    def _clean(self, response: str, prompt: str) -> Optional[str]:
        """Clean up a generated response."""
        # Remove prompt if repeated
        if response.startswith(prompt):
            response = response[len(prompt):].strip()

        # Take first few sentences
        sentences = response.split('. ')
        cleaned = '. '.join(sentences[:3])

        # Ensure ends with punctuation
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '.'

        # Clean whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        return cleaned if len(cleaned) > 20 else None

    def _remember(self, query: str, response: str):
        """Remember conversation for context."""
        self._history.append({'query': query, 'response': response})
        if len(self._history) > 10:
            self._history = self._history[-10:]
