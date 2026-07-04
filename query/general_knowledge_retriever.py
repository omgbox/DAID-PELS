"""
General Knowledge Retriever
DYNAMIC topic extraction — works with any question.
"""

import re
import logging
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.query.general_knowledge_retriever')


class GeneralKnowledgeRetriever:
    """Retrieves general knowledge from multiple sources."""

    # Common term mappings to Wikipedia titles
    TERM_MAP = {
        'bike': 'bicycle',
        'car': 'automobile',
        'phone': 'telephone',
        'computer': 'computer',
        'internet': 'internet',
        'tv': 'television',
        'television': 'television',
        'airplane': 'airplane',
        'plane': 'airplane',
        'aircraft': 'airplane',
        'light bulb': 'incandescent light bulb',
        'lightbulb': 'incandescent light bulb',
        'ai': 'artificial intelligence',
        'ml': 'machine learning',
        'vr': 'virtual reality',
        'ar': 'augmented reality',
        # Old English / Anglo-Saxon
        'old english': 'Old English',
        'anglo-saxon': 'Old English',
        'anglo saxon': 'Old English',
        'beowulf': 'Beowulf',
        'anglo': 'Anglo-Saxons',
        'saxon': 'Saxons',
        'wergild': 'Wergild',
        'wyrd': 'Wyrd',
        'thane': 'Thane',
        'ealdorman': 'Ealdorman',
        'witan': 'Witenagemot',
        'fyrd': 'Fyrd',
        'burh': 'Burh',
        'scip': 'Scip',
        'hid': 'Hide (unit)',
        # Programming
        'rust': 'Rust (programming language)',
        'python': 'Python (programming language)',
        'java': 'Java (programming language)',
        'javascript': 'JavaScript',
        'typescript': 'TypeScript',
        'go': 'Go (programming language)',
        'golang': 'Go (programming language)',
        'c++': 'C++',
        'c#': 'C Sharp (programming language)',
        'swift': 'Swift (programming language)',
        'kotlin': 'Kotlin',
        'ruby': 'Ruby (programming language)',
        'php': 'PHP',
        'scala': 'Scala (programming language)',
        'haskell': 'Haskell',
        'elixir': 'Elixir (programming language)',
        'erlang': 'Erlang',
        # Science
        'dna': 'DNA',
        'rna': 'RNA',
        'atp': 'Adenosine triphosphate',
        'cpu': 'Central processing unit',
        'gpu': 'Graphics processing unit',
        'ram': 'Random-access memory',
        'ssd': 'Solid-state drive',
        'hdd': 'Hard disk drive',
        'usb': 'Universal Serial Bus',
        'wifi': 'Wi-Fi',
        'bluetooth': 'Bluetooth',
        'led': 'Light-emitting diode',
        'lcd': 'Liquid crystal display',
        'oled': 'Organic light-emitting diode',
        'laser': 'Laser',
        'radar': 'Radar',
        'sonar': 'Sonar',
        # Geography
        'usa': 'United States',
        'uk': 'United Kingdom',
        'eu': 'European Union',
        'un': 'United Nations',
        'nato': 'NATO',
        'who': 'World Health Organization',
        'nasa': 'NASA',
        'fbi': 'Federal Bureau of Investigation',
        'cia': 'Central Intelligence Agency',
        'nsa': 'National Security Agency',
    }

    def __init__(self, db_manager=None):
        self.db = db_manager
        self._wikipedia_available = None
        self._cache = {}

    def _check_wikipedia(self) -> bool:
        """Check if wikipedia-api is available."""
        if self._wikipedia_available is None:
            try:
                import wikipediaapi
                self._wikipedia_available = True
                logger.info("Wikipedia API available")
            except ImportError:
                self._wikipedia_available = False
                logger.info("Wikipedia API not available")
        return self._wikipedia_available

    def retrieve(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Retrieve knowledge about a topic from multiple sources.
        DYNAMICALLY extracts topic from any question.
        """
        results = []

        # Check cache
        cache_key = query.lower().strip()
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Source 1: Wikipedia
        wiki_results = self._search_wikipedia(query)
        results.extend(wiki_results)

        # Source 2: Local knowledge base
        local_results = self._search_local_kb(query)
        results.extend(local_results)

        # Sort and limit
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        results = results[:max_results]

        # Cache
        self._cache[cache_key] = results

        return results

    def _search_wikipedia(self, query: str) -> List[Dict]:
        """Search Wikipedia — dynamically extracts topic."""
        if not self._check_wikipedia():
            return []

        try:
            import wikipediaapi

            # Extract topic dynamically
            topic = self._extract_topic(query)

            # Initialize Wikipedia API
            wiki = wikipediaapi.Wikipedia(
                user_agent='BookBot/1.0 (https://github.com/omgbox/DAID-PELS)',
                language='en'
            )

            # Try multiple search strategies
            strategies = [
                topic,                          # Direct topic
                self._simplify_topic(topic),    # Simplified
                self._map_term(topic),          # Mapped term
            ]

            # Also try the full query with question words removed
            query_topic = self._extract_noun_phrase(query)
            if query_topic and query_topic != topic:
                strategies.append(query_topic)

            for search_term in strategies:
                if not search_term:
                    continue

                page = wiki.page(search_term)
                if page.exists():
                    summary = page.summary
                    sentences = summary.split('. ')
                    text = '. '.join(sentences[:3]) + '.'

                    return [{
                        'text': text,
                        'title': page.title,
                        'source': 'wikipedia',
                        'confidence': 0.8,
                        'url': page.fullurl,
                    }]

            return []

        except Exception as e:
            logger.debug(f"Wikipedia search failed: {e}")
            return []

    def _extract_topic(self, query: str) -> str:
        """
        DYNAMICALLY extract the main topic from any question.
        No hard-coded patterns — uses linguistic heuristics.
        """
        query_lower = query.lower().strip()

        # Remove question marks
        query_lower = query_lower.rstrip('?').strip()

        # Strategy 1: "how many X does Y have" -> "Y"
        match = re.search(r'how many\s+\w+\s+(?:does|do|did)\s+(.+?)\s+(?:have|has|had)', query_lower)
        if match:
            return match.group(1).strip()

        # Strategy 2: "what is the X of Y" -> "Y"
        match = re.search(r'what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(.+?)\s+of\s+(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|\s+at\s+|$)', query_lower)
        if match:
            # Return the more specific part (usually after "of")
            part1 = match.group(1).strip()
            part2 = match.group(2).strip()
            # If part2 is a country/city/entity, use it
            if len(part2) > len(part1):
                return part2
            return part2

        # Strategy 3: "who invented/discovered/created X" -> "X"
        match = re.search(r'who\s+(?:invented|discovered|created|founded|built|wrote|painted|composed|designed|developed|made)\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # Strategy 4: "when was X invented" -> "X"
        match = re.search(r'when\s+(?:was|did|were)\s+(.+?)\s+(?:invented|discovered|created|founded|built|written|painted|composed|designed|developed|happen|occur)', query_lower)
        if match:
            return match.group(1).strip()

        # Strategy 5: "tell me about X" -> "X"
        match = re.search(r'tell\s+me\s+about\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # Strategy 6: "what is X" -> "X"
        match = re.search(r'what\s+(?:is|are|was|were)\s+(?:the\s+)?(?:a\s+)?(?:an\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|\s+at\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # Strategy 7: "who is X" -> "X"
        match = re.search(r'who\s+(?:is|was|are|were)\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return match.group(1).strip()

        # Strategy 8: "which X is the Y" -> extract noun phrase
        match = re.search(r'which\s+(.+?)\s+(?:is|are|was|were)\s+(?:the\s+)?(.+?)(?:\s+in\s+|\s+for\s+|\s+on\s+|$)', query_lower)
        if match:
            return f"{match.group(1).strip()} {match.group(2).strip()}"

        # Fallback: extract noun phrase
        return self._extract_noun_phrase(query_lower)

    def _extract_noun_phrase(self, query: str) -> str:
        """Extract the main noun phrase from a query."""
        # Remove common question words and verbs
        stop_words = {
            'what', 'who', 'when', 'where', 'why', 'how', 'which',
            'is', 'are', 'was', 'were', 'do', 'does', 'did',
            'the', 'a', 'an', 'of', 'in', 'for', 'and', 'or', 'to',
            'have', 'has', 'had', 'be', 'been', 'being',
            'tell', 'me', 'about', 'please', 'help',
            'many', 'much', 'big', 'small', 'tall', 'old', 'long',
            'wide', 'deep', 'high', 'fast', 'slow', 'hot', 'cold',
        }

        words = query.lower().split()
        cleaned = [w for w in words if w not in stop_words and len(w) > 2]

        if cleaned:
            return ' '.join(cleaned)

        return query

    def _simplify_topic(self, topic: str) -> str:
        """Simplify topic for better Wikipedia matching."""
        # Remove leading "the"
        if topic.startswith('the '):
            topic = topic[4:]

        # Remove trailing prepositions
        for suffix in [' of', ' in', ' for', ' about', ' on', ' at']:
            if topic.endswith(suffix):
                topic = topic[:-len(suffix)]

        return topic.strip()

    def _map_term(self, term: str) -> str:
        """Map common terms to Wikipedia titles."""
        term_lower = term.lower().strip()
        return self.TERM_MAP.get(term_lower, term)

    def _search_local_kb(self, query: str) -> List[Dict]:
        """Search the local knowledge base."""
        if not self.db:
            return []

        try:
            rows = self.db.execute(
                "SELECT topic, fact, confidence FROM learned_knowledge "
                "WHERE topic LIKE ? OR fact LIKE ? "
                "ORDER BY confidence DESC LIMIT 3",
                (f'%{query}%', f'%{query}%')
            )

            return [
                {
                    'text': r[1],
                    'title': r[0],
                    'source': 'learned',
                    'confidence': r[2] * 0.9,
                }
                for r in rows
            ]

        except Exception as e:
            logger.debug(f"Local KB search failed: {e}")
            return []

    def store_knowledge(self, topic: str, fact: str, source: str = 'conversation',
                       confidence: float = 0.5):
        """Store a fact in the knowledge base."""
        if not self.db:
            return

        try:
            from datetime import datetime
            timestamp = datetime.now().isoformat()

            self.db.execute(
                "INSERT INTO learned_knowledge (topic, fact, source, confidence, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (topic, fact, source, confidence, timestamp)
            )

            self._cache.clear()
            logger.debug(f"Stored knowledge: {topic}")

        except Exception as e:
            logger.warning(f"Failed to store knowledge: {e}")

    def clear_cache(self):
        """Clear the in-memory cache."""
        self._cache.clear()
