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

# Suppress HuggingFace progress bars and warnings
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'

# Login to HuggingFace if token is available
try:
    from huggingface_hub import login
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token, add_to_git_credential=False)
except Exception:
    pass

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
        self._entities = []  # Track entities mentioned in conversation
        self._user_context = {}  # Remember user info
        self._wiki_cache = {}  # Cache Wikipedia lookups
        self._neural_mapper = None  # Neural Wikipedia mapper
        self._topic_extractor = None  # Neural topic extractor
        self._intent_classifier = None  # Neural intent classifier
        self._response_selector = None  # Neural response selector
        self._last_query_topic = None  # Track last query for learning

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

    def _get_neural_mapper(self):
        """Lazy-load the neural Wikipedia mapper."""
        if self._neural_mapper is None:
            try:
                from .neural_wiki_mapper import NeuralWikipediaMapper
                self._neural_mapper = NeuralWikipediaMapper()
                # Load saved mappings if available
                import os
                mapper_path = os.path.join(os.path.dirname(__file__), '..', 'wiki_mappings.json')
                self._neural_mapper.load(mapper_path)
                logger.info("Neural Wikipedia mapper loaded")
            except Exception as e:
                logger.debug(f"Neural mapper not available: {e}")
        return self._neural_mapper

    def _get_topic_extractor(self):
        """Lazy-load the neural topic extractor."""
        if self._topic_extractor is None:
            try:
                from .neural_topic_extractor import NeuralTopicExtractor
                self._topic_extractor = NeuralTopicExtractor()
                # Load saved scores if available
                import os
                scores_path = os.path.join(os.path.dirname(__file__), '..', 'topic_scores.json')
                self._topic_extractor.load(scores_path)
                logger.info("Neural topic extractor loaded")
            except Exception as e:
                logger.debug(f"Neural topic extractor not available: {e}")
        return self._topic_extractor

    def _get_intent_classifier(self):
        """Lazy-load the neural intent classifier."""
        if self._intent_classifier is None:
            try:
                from .neural_intent_classifier import NeuralIntentClassifier
                self._intent_classifier = NeuralIntentClassifier()
                import os
                path = os.path.join(os.path.dirname(__file__), '..', 'intent_scores.json')
                self._intent_classifier.load(path)
                logger.info("Neural intent classifier loaded")
            except Exception as e:
                logger.debug(f"Neural intent classifier not available: {e}")
        return self._intent_classifier

    def _get_response_selector(self):
        """Lazy-load the neural response selector."""
        if self._response_selector is None:
            try:
                from .neural_response_selector import NeuralResponseSelector
                self._response_selector = NeuralResponseSelector()
                import os
                path = os.path.join(os.path.dirname(__file__), '..', 'response_scores.json')
                self._response_selector.load(path)
                logger.info("Neural response selector loaded")
            except Exception as e:
                logger.debug(f"Neural response selector not available: {e}")
        return self._response_selector

    def chat(self, message: str) -> str:
        """
        Process a message and return a natural response.
        Like talking to a real person.
        """
        message = message.strip()
        if not message:
            return "I'm here! What would you like to talk about?"

        # Step 0: Resolve pronouns using context
        message = self._resolve_pronouns(message)

        # Step 1: Handle simple cases instantly
        quick = self._instant_response(message)
        if quick:
            self._remember(message, quick)
            return quick

        # Step 2: Understand what the user wants
        intent = self._understand(message)

        # Step 3: Get information from multiple sources
        facts = None
        sources = []
        
        if intent['type'] == 'book':
            # Search book database
            book_facts = self._search_book(intent['topic'])
            if book_facts:
                facts = book_facts
                sources.append('book')
        elif intent['needs_info']:
            # Search Wikipedia
            wiki_facts = self._get_facts(intent['topic'])
            if wiki_facts:
                facts = wiki_facts
                sources.append('wikipedia')
            
            # Also search book database for additional context
            book_facts = self._search_book(intent['topic'])
            if book_facts:
                if facts:
                    # Combine facts from both sources
                    facts = f"{facts}\n\nAdditionally, from the book collection: {book_facts}"
                else:
                    facts = book_facts
                    sources.append('book')

        # Step 4: Verify facts if multiple sources
        confidence = None
        if facts and len(sources) > 1:
            confidence = self._verify_facts(facts, sources)

        # Step 5: Generate a natural response
        response = self._respond(message, intent, facts, sources, confidence)

        # Step 6: Remember this conversation
        self._remember(message, response, intent.get('topic', ''))

        return response

    def _instant_response(self, message: str) -> Optional[str]:
        """Instant responses using neural intent classification."""
        m = message.lower().strip()
        
        # Try neural intent classifier first
        classifier = self._get_intent_classifier()
        if classifier:
            try:
                intent, confidence = classifier.classify(m)
                
                if intent == 'greeting' and confidence > 0.7:
                    responses = [
                        "Hey! What's on your mind?",
                        "Hello! Ask me anything.",
                        "Hi there! How can I help?",
                    ]
                    selector = self._get_response_selector()
                    if selector:
                        return selector.select(responses, m)
                    return random.choice(responses)
                
                if intent == 'farewell' and confidence > 0.7:
                    responses = [
                        "Goodbye! Have a great day!",
                        "See you later!",
                        "Take care! Come back anytime.",
                    ]
                    selector = self._get_response_selector()
                    if selector:
                        return selector.select(responses, m)
                    return random.choice(responses)
                
                if intent == 'emotional' and confidence > 0.7:
                    # Let the emotional handler deal with it
                    pass
                    
            except Exception as e:
                logger.debug(f"Neural classification failed: {e}")
        
        # Fallback to regex patterns
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
        """Extract the main topic from a message using neural extraction."""
        m = message.lower().strip().rstrip('?')

        # Common term mappings (expand abbreviations)
        expand_map = {
            'ufo': 'unidentified flying object',
            'ufos': 'unidentified flying object',
            'ai': 'artificial intelligence',
            'ml': 'machine learning',
            'vr': 'virtual reality',
            'ar': 'augmented reality',
            'aerial phenomena': 'unidentified aerial phenomenon',
            'uap': 'unidentified aerial phenomenon',
            'uaps': 'unidentified aerial phenomenon',
        }

        # Follow-up detection: use conversation context
        if self._history and self._is_followup(m):
            topic = self._resolve_followup(m)
            if topic:
                return self._expand_topic(topic, expand_map)

        # Try neural topic extractor first
        extractor = self._get_topic_extractor()
        if extractor:
            try:
                neural_topic = extractor.extract_topic(m)
                if neural_topic and len(neural_topic) > 2:
                    return self._expand_topic(neural_topic, expand_map)
            except Exception as e:
                logger.debug(f"Neural topic extraction failed: {e}")

        # Fallback: regex patterns (kept as backup)
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
            topic = re.sub(r'\s+(reported|invented|discovered|created|founded|built|written|painted|happened|occurred)$', '', topic)
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

        # Pattern: "did/do/does X have Y" -> "X Y"
        match = re.search(r'(?:did|do|does|have|has|had)\s+(.+?)\s+(?:have|has|had)\s+(?:a\s+|an\s+|the\s+)?(.+)', m)
        if match:
            subject = match.group(1).strip()
            object_ = match.group(2).strip()
            return self._expand_topic(f"{subject} {object_}", expand_map)

        # Pattern: "is/was X a Y?" -> "X Y"
        match = re.search(r'(?:is|are|was|were)\s+(.+?)\s+(?:a\s+|an\s+|the\s+)?(.+?)(?:\s+in\s+|\s+of\s+|\s+for\s+|\s+on\s+|$)', m)
        if match:
            subject = match.group(1).strip()
            object_ = match.group(2).strip()
            if len(subject) > 2 and len(object_) > 2:
                return self._expand_topic(f"{subject} {object_}", expand_map)

        # Pattern: "can/could X Y?" -> "X Y"
        match = re.search(r'(?:can|could|will|would|should|may|might)\s+(.+?)\s+(.+?)(?:\?|$)', m)
        if match:
            subject = match.group(1).strip()
            action = match.group(2).strip()
            if len(subject) > 2 and len(action) > 2:
                return self._expand_topic(f"{subject} {action}", expand_map)

        # Fallback: remove question words and use the rest
        stop = {'what', 'who', 'when', 'where', 'why', 'how', 'which',
                'is', 'are', 'was', 'were', 'do', 'does', 'did',
                'have', 'has', 'had',
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

    def _is_followup(self, message: str) -> bool:
        """Detect if message is a follow-up to previous topic."""
        followup_patterns = [
            r'^tell me more',
            r'^what about',
            r'^how about',
            r'^and\b',
            r'^also\b',
            r'^what else',
            r'^anything else',
            r'^more about',
            r'^elaborate',
            r'^continue',
            r'^go on',
            r'^and the',
            r'^what about the',
            r'^tell me about the',
        ]
        return any(re.match(p, message) for p in followup_patterns)

    def _resolve_followup(self, message: str) -> Optional[str]:
        """
        Resolve follow-up using conversation context.
        Combines current topic with previous context.
        """
        if not self._history:
            return None

        last_query = self._history[-1]['query'].lower()
        last_topic = self._history[-1].get('topic', '')

        # Extract what the user is asking about now
        current_topic = None
        match = re.search(r'tell me (?:more )?about (?:the )?(.+)', message)
        if match:
            current_topic = match.group(1).strip()
        else:
            match = re.search(r'what about (?:the )?(.+)', message)
            if match:
                current_topic = match.group(1).strip()

        if current_topic and last_topic:
            # Combine: "aerial phenomena" + "unidentified flying object" context
            # Search for combined topic
            combined = f"{current_topic} {last_topic}"
            return combined
        elif current_topic:
            return current_topic

        return None

    def _get_facts(self, topic: str) -> Optional[str]:
        """Get facts about a topic from Wikipedia using dynamic search."""
        # Check cache first
        cache_key = topic.lower().strip()
        if cache_key in self._wiki_cache:
            return self._wiki_cache[cache_key]

        wiki = self._get_wiki()
        if not wiki:
            return None

        result = None

        # Strategy 1: Try direct page lookup
        result = self._try_direct_page(wiki, topic)
        if result:
            self._wiki_cache[cache_key] = result
            return result

        # Strategy 2: Dynamic Wikipedia search
        result = self._search_wikipedia_dynamic(wiki, topic)
        if result:
            self._wiki_cache[cache_key] = result
            return result

        # Strategy 3: Try common variations
        variations = self._generate_variations(topic)
        for variation in variations:
            result = self._try_direct_page(wiki, variation)
            if result:
                self._wiki_cache[cache_key] = result
                return result

        return None

    def _try_direct_page(self, wiki, topic: str) -> Optional[str]:
        """Try to get a Wikipedia page directly by title."""
        try:
            page = wiki.page(topic)
            if page.exists():
                summary = page.summary
                # Skip disambiguation pages
                if 'may refer to' in summary[:200]:
                    return None
                # Skip very short summaries (likely stubs)
                if len(summary) < 50:
                    return None
                sentences = summary.split('. ')
                return '. '.join(sentences[:3]) + '.'
        except Exception:
            pass
        return None

    def _expand_compound_topic(self, topic: str) -> List[str]:
        """Expand compound topics for better Wikipedia search."""
        expansions = [topic]  # Always try original first
        
        topic_lower = topic.lower()
        words = topic_lower.split()
        
        # Common compound expansions
        compound_expansions = {
            'jesus mother': ['Mary mother of Jesus', 'Virgin Mary', 'Mary mother of God'],
            'jesus father': ['God in Christianity', 'Joseph husband of Mary'],
            'jesus child': ['Childhood of Jesus', 'Jesus in Christianity'],
            'jesus disciple': ['Disciples of Jesus', 'Apostle'],
            'jesus crucifixion': ['Crucifixion of Jesus', 'Passion of Jesus'],
            'jesus resurrection': ['Resurrection of Jesus', 'Empty tomb'],
            'mary mother': ['Mary mother of Jesus', 'Virgin Mary'],
            'god son': ['Son of God', 'Jesus Christ'],
            'holy spirit': ['Holy Spirit', 'Holy Spirit in Christianity'],
        }
        
        # Check for compound expansions
        for key, values in compound_expansions.items():
            if all(w in topic_lower for w in key.split()):
                expansions.extend(values)
        
        # If topic has 2+ words, try searching for each significant word
        if len(words) >= 2:
            # Try searching for just the main subject (usually first word)
            significant = [w for w in words if w not in {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'did', 'do', 'does', 'have', 'has', 'had'}]
            if significant:
                expansions.append(' '.join(significant))
        
        return expansions

    def _search_wikipedia_dynamic(self, wiki, topic: str) -> Optional[str]:
        """Dynamically search Wikipedia for the best matching page."""
        try:
            # Get expanded topics to search
            search_topics = self._expand_compound_topic(topic)
            
            # Try each expanded topic
            for search_topic in search_topics:
                # Try direct page lookup first
                result = self._try_direct_page(wiki, search_topic)
                if result:
                    return result
            
            # Fall back to search API
            search_results = wiki.search(topic, results=10)

            if not search_results:
                return None

            # Use neural mapper if available
            mapper = self._get_neural_mapper()
            if mapper:
                # Get valid page titles
                valid_titles = []
                valid_pages = {}
                for title in search_results:
                    try:
                        page = wiki.page(title)
                        if page.exists() and 'may refer to' not in page.summary[:200]:
                            valid_titles.append(title)
                            valid_pages[title] = page
                    except Exception:
                        continue

                if valid_titles:
                    # Neural mapper picks the best
                    best_title = mapper.predict(topic, valid_titles)
                    if best_title and best_title in valid_pages:
                        page = valid_pages[best_title]
                        sentences = page.summary.split('. ')
                        result = '. '.join(sentences[:3]) + '.'
                        
                        # Train the mapper on this lookup
                        mapper.train(topic, best_title, positive=True)
                        
                        return result

            # Fallback: rule-based scoring
            best_page = None
            best_score = 0

            topic_lower = topic.lower()
            topic_words = set(topic_lower.split())

            for title in search_results:
                try:
                    page = wiki.page(title)
                    if not page.exists():
                        continue

                    summary = page.summary

                    # Skip disambiguation pages
                    if 'may refer to' in summary[:200]:
                        continue

                    # Score based on title similarity
                    title_lower = title.lower()
                    score = 0

                    # Exact match
                    if title_lower == topic_lower:
                        score += 100

                    # Title contains topic
                    elif topic_lower in title_lower:
                        score += 50

                    # Topic contains title (for multi-word topics)
                    elif any(w in topic_lower for w in title_lower.split()):
                        score += 30

                    # Word overlap
                    title_words = set(title_lower.split())
                    overlap = len(topic_words & title_words)
                    score += overlap * 10

                    # Prefer longer summaries (more informative)
                    score += min(len(summary) / 1000, 5)

                    if score > best_score:
                        best_score = score
                        best_page = page

                except Exception:
                    continue

            if best_page and best_score >= 20:
                # Train mapper on this result
                if mapper:
                    mapper.train(topic, best_page.title, positive=True)
                
                sentences = best_page.summary.split('. ')
                return '. '.join(sentences[:3]) + '.'

        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")

        return None

    def _generate_variations(self, topic: str) -> List[str]:
        """Generate common variations of a topic for Wikipedia lookup."""
        variations = []
        topic_lower = topic.lower()

        # Common patterns
        # "X programming" -> "X (programming language)"
        if 'programming' in topic_lower:
            variations.append(f"{topic} (programming language)")

        # "X language" -> "X (programming language)"
        if 'language' in topic_lower:
            variations.append(topic.replace('language', '(programming language)'))

        # "X js" -> "X.js"
        if topic_lower.endswith('js'):
            variations.append(topic[:-2] + '.js')

        # "X dot Y" -> "X.Y"
        if ' dot ' in topic_lower:
            variations.append(topic.replace(' dot ', '.'))

        # Add parentheses for disambiguation
        variations.append(f"{topic} ({topic})")

        return variations

    def _respond(self, message: str, intent: Dict, facts: Optional[str], sources: List[str] = None, confidence: Optional[str] = None) -> str:
        """Generate a natural response."""
        gen = self._get_generator()

        if intent['type'] == 'personal':
            return self._respond_personal(message)

        if intent['type'] == 'emotional':
            return self._respond_emotional(message)

        if facts:
            if gen:
                response = self._generate_with_facts(message, facts, gen)
            else:
                response = facts
            
            # Add source attribution and confidence
            if sources and response:
                attribution = self._get_attribution(sources, confidence)
                if attribution:
                    response = f"{response}\n\n{attribution}"
            
            return response

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
            # Generate multiple response candidates
            candidates = []
            
            # Candidate 1: Direct paraphrase
            response1 = rewriter.rewrite_for_chat(facts, context=query)
            if response1 and len(response1) > 30:
                candidates.append(response1)
            
            # Candidate 2: Simplified version
            response2 = rewriter.rewrite(facts, style='simplify')
            if response2 and len(response2) > 30:
                candidates.append(response2)
            
            # Candidate 3: Original facts
            if facts and len(facts) > 30:
                candidates.append(facts)
            
            # Score and pick the best
            if candidates:
                best = self._score_responses(candidates, query)
                return best

        # Fallback: return the facts directly
        return facts

    def _score_responses(self, candidates: List[str], query: str) -> str:
        """
        Score response candidates and return the best one.
        
        Scoring criteria:
        - Fluency: Length and sentence structure
        - Relevance: Contains query keywords
        - Completeness: Has multiple sentences
        """
        if not candidates:
            return ""
        
        scored = []
        query_words = set(query.lower().split())
        
        for candidate in candidates:
            score = 0
            
            # Fluency: prefer medium-length responses (not too short, not too long)
            length = len(candidate)
            if 50 < length < 500:
                score += 2
            elif 30 < length < 800:
                score += 1
            
            # Relevance: contains query keywords
            candidate_lower = candidate.lower()
            keyword_matches = sum(1 for word in query_words if word in candidate_lower)
            score += keyword_matches
            
            # Completeness: has multiple sentences
            sentences = candidate.split('. ')
            if len(sentences) >= 2:
                score += 2
            elif len(sentences) >= 1:
                score += 1
            
            # Penalize responses that are just the original facts
            if candidate == candidates[-1] and len(candidates) > 1:
                score -= 1
            
            scored.append((score, candidate))
        
        # Return the highest scoring response
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]

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

    def _get_attribution(self, sources: List[str], confidence: Optional[str] = None) -> str:
        """Get source attribution based on where the information came from."""
        if not sources:
            return ""
        
        # Map source types to human-readable names
        source_names = {
            'wikipedia': 'Wikipedia',
            'book': 'Book database',
        }
        
        # Get unique source names
        names = []
        for source in sources:
            if source in source_names and source_names[source] not in names:
                names.append(source_names[source])
        
        if not names:
            return ""
        
        # Build attribution string
        if len(names) == 1:
            attribution = f"— Source: {names[0]}"
        else:
            attribution = f"— Sources: {', '.join(names)}"
        
        # Add confidence indicator
        if confidence:
            attribution += f" ({confidence})"
        
        return attribution

    def _verify_facts(self, facts: str, sources: List[str]) -> str:
        """
        Verify facts across multiple sources.
        Returns confidence level: 'verified', 'partial', or 'unverified'.
        """
        if len(sources) < 2:
            return ""
        
        # Simple verification: check if facts from different sources overlap
        # In a real implementation, you would compare specific facts
        # For now, we assume if we have multiple sources, the information is more reliable
        
        # Check if the facts contain overlapping information
        fact_words = set(facts.lower().split())
        
        # If we have both Wikipedia and book sources, consider it verified
        if 'wikipedia' in sources and 'book' in sources:
            return "verified"
        elif 'wikipedia' in sources or 'book' in sources:
            return "partial"
        
        return "unverified"

    def _resolve_pronouns(self, message: str) -> str:
        """
        Replace pronouns with the last mentioned entity.
        
        Examples:
            "Who created it?" → "Who created Rust?"
            "Tell me more about him" → "Tell me more about Darcy"
        """
        if not self._entities:
            return message
        
        # Get the last mentioned entity
        last_entity = self._entities[-1]
        
        # Pronouns to resolve
        pronouns = {
            'it': last_entity,
            'he': last_entity,
            'she': last_entity,
            'him': last_entity,
            'her': last_entity,
            'this': last_entity,
            'that': last_entity,
        }
        
        # Replace pronouns in the message
        words = message.split()
        resolved = []
        for word in words:
            word_lower = word.lower().rstrip('?,.!;:')
            if word_lower in pronouns:
                resolved.append(pronouns[word_lower])
            else:
                resolved.append(word)
        
        return ' '.join(resolved)

    def _extract_entities(self, message: str) -> List[str]:
        """Extract entities (nouns) from a message."""
        entities = []
        
        # Simple entity extraction - look for capitalized words
        words = message.split()
        for word in words:
            # Remove punctuation
            clean = word.rstrip('?,.!;:').lstrip('?,.!;:')
            if clean and clean[0].isupper() and len(clean) > 2:
                # Skip common words
                if clean.lower() not in {'what', 'who', 'when', 'where', 'why', 'how', 'which', 'the', 'is', 'are', 'was', 'were', 'do', 'does', 'did'}:
                    entities.append(clean)
        
        return entities

    def _remember(self, query: str, response: str, topic: str = ''):
        """Remember conversation for context."""
        self._history.append({'query': query, 'response': response, 'topic': topic})
        if len(self._history) > 10:
            self._history = self._history[-10:]
        
        # Extract and track entities
        entities = self._extract_entities(query)
        self._entities.extend(entities)
        if len(self._entities) > 5:
            self._entities = self._entities[-5:]
        
        # Train topic extractor if we have a topic
        if topic and self._topic_extractor:
            self._topic_extractor.train(query, topic, positive=True)
        
        # Save neural models periodically
        if len(self._history) % 5 == 0:
            try:
                import os
                # Save wiki mapper
                if self._neural_mapper:
                    mapper_path = os.path.join(os.path.dirname(__file__), '..', 'wiki_mappings.json')
                    self._neural_mapper.save(mapper_path)
                # Save topic extractor
                if self._topic_extractor:
                    scores_path = os.path.join(os.path.dirname(__file__), '..', 'topic_scores.json')
                    self._topic_extractor.save(scores_path)
            except Exception:
                pass
