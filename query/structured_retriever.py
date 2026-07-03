"""
Structured Retriever
Query SVO triples, entity graph, coreference chains, and dictionary at query time.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.query.structured_retriever')

# Stopwords to filter from entity results
STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'over',
    'and', 'or', 'but', 'not', 'so', 'if', 'then', 'than', 'too',
    'very', 'just', 'about', 'more', 'some', 'any', 'all', 'each',
    'both', 'few', 'many', 'much', 'own', 'same', 'other',
    # Sentence-start adverbs that are NOT entities
    'again', 'also', 'thus', 'hence', 'else', 'even', 'still', 'yet',
    'now', 'here', 'there', 'then', 'though', 'although', 'however',
    'nevertheless', 'nonetheless', 'meanwhile', 'furthermore', 'moreover',
    'indeed', 'perhaps', 'maybe', 'almost', 'quite', 'rather', 'well',
    'accordingly', 'afterwards', 'almost', 'already', 'always', 'anyway',
    'besides', 'consequently', 'elsewhere', 'everywhere', 'finally',
    'further', 'henceforth', 'hereafter', 'hereby', 'herein', 'lastly',
    'likewise', 'moreover', 'never', 'next', 'nonetheless', 'nowadays',
    'otherwise', 'overall', 'presently', 'previously', 'regardless',
    'similarly', 'subsequently', 'surely', 'therefore', 'thus', 'together',
    'usually', 'whenever', 'whereas', 'whereby', 'wherein',
    # Interjections and sentence-start discourse markers
    'ah', 'oh', 'alas', 'well', 'why', 'indeed', 'aye', 'nay',
    'dear', 'look', 'listen', 'hark', 'behold', 'lo',
    # Short common words that appear capitalized at sentence starts
    'yes', 'no', 'so', 'now', 'then', 'here', 'there',
}


class StructuredRetriever:
    """Query structured knowledge tables at query time."""

    def __init__(self, db):
        self.db = db
        self._definitions = None

    def _load_definitions(self):
        """Lazy-load definitions from DB."""
        if self._definitions is not None:
            return
        try:
            rows = self.db.execute(
                "SELECT word_lower, pos_canonical, definition FROM definitions LIMIT 200000"
            )
            self._definitions = {}
            for word_lower, pos, definition in (rows or []):
                if word_lower not in self._definitions:
                    self._definitions[word_lower] = []
                self._definitions[word_lower].append({
                    'pos': pos, 'definition': definition,
                })
        except Exception:
            self._definitions = {}

    def find_definition(self, word: str) -> str:
        """Find dictionary definition for a word."""
        self._load_definitions()
        defs = self._definitions.get(word.lower(), [])
        if defs:
            return defs[0]['definition']
        return ''

    def find_definitions_for_words(self, words: List[str]) -> Dict[str, str]:
        """Find definitions for a list of words."""
        self._load_definitions()
        result = {}
        for word in words:
            defs = self._definitions.get(word.lower(), [])
            if defs:
                result[word] = defs[0]['definition']
        return result

    def find_relationships(self, entity_a: str, entity_b: str = '') -> List[Dict]:
        """Find SVO triples connecting two entities."""
        if entity_b:
            sql = """
                SELECT s.subject, s.verb, s.object, s.confidence, s.sentence_id,
                       t.raw_text
                FROM svo_triples s
                LEFT JOIN sentences t ON s.sentence_id = t.sentence_id
                WHERE (LOWER(s.subject) LIKE ? AND LOWER(s.object) LIKE ?)
                   OR (LOWER(s.subject) LIKE ? AND LOWER(s.object) LIKE ?)
                ORDER BY s.confidence DESC
                LIMIT 15
            """
            a, b = f'%{entity_a.lower()}%', f'%{entity_b.lower()}%'
            return self.db.execute(sql, (a, b, b, a)) or []
        else:
            return self.find_entity_actions(entity_a)

    def find_entity_actions(self, entity: str) -> List[Dict]:
        """Find what an entity does (subject of SVO triples)."""
        sql = """
            SELECT s.subject, s.verb, s.object, s.confidence, s.sentence_id,
                   t.raw_text
            FROM svo_triples s
            LEFT JOIN sentences t ON s.sentence_id = t.sentence_id
            WHERE LOWER(s.subject) LIKE ?
            ORDER BY s.confidence DESC
            LIMIT 15
        """
        return self.db.execute(sql, (f'%{entity.lower()}%',)) or []

    def find_entity_experience(self, entity: str) -> List[Dict]:
        """Find what happens to an entity (object of SVO triples)."""
        sql = """
            SELECT s.subject, s.verb, s.object, s.confidence, s.sentence_id,
                   t.raw_text
            FROM svo_triples s
            LEFT JOIN sentences t ON s.sentence_id = t.sentence_id
            WHERE LOWER(s.object) LIKE ?
            ORDER BY s.confidence DESC
            LIMIT 15
        """
        return self.db.execute(sql, (f'%{entity.lower()}%',)) or []

    def find_related_entities(self, entity: str) -> List[Dict]:
        """Find entities connected via knowledge graph."""
        sql = """
            SELECT target_id AS related, edge_type, weight
            FROM knowledge_edges
            WHERE LOWER(source_id) LIKE ?
            UNION
            SELECT source_id AS related, edge_type, weight
            FROM knowledge_edges
            WHERE LOWER(target_id) LIKE ?
            ORDER BY weight DESC
            LIMIT 30
        """
        e = f'%{entity.lower()}%'
        rows = self.db.execute(sql, (e, e)) or []
        # Filter out stopwords
        filtered = []
        for row in rows:
            name = row[0] if isinstance(row, (list, tuple)) else row.get('related', '')
            if name and name.lower() not in STOPWORDS and len(name) > 1:
                filtered.append(row)
        return filtered[:20]

    def get_coreference_chain(self, entity: str) -> List[Dict]:
        """Find all pronouns/names referring to this entity."""
        sql = """
            SELECT representative, mention_count
            FROM coreference_chains
            WHERE LOWER(representative) LIKE ?
            LIMIT 10
        """
        return self.db.execute(sql, (f'%{entity.lower()}%',)) or []

    def find_entity_by_name(self, name: str) -> Optional[Dict]:
        """Find entity in entities table."""
        sql = "SELECT * FROM entities WHERE LOWER(canonical_name) LIKE ? LIMIT 1"
        rows = self.db.execute(sql, (f'%{name.lower()}%',))
        if rows:
            row = rows[0]
            return {
                'entity_id': row[0], 'canonical_name': row[1],
                'entity_type': row[2], 'frequency': row[3],
                'centrality': row[4],
            }
        return None

    def find_shared_connections(self, entity_a: str, entity_b: str) -> List[str]:
        """Find entities connected to both A and B."""
        related_a = self.find_related_entities(entity_a)
        related_b = self.find_related_entities(entity_b)
        def _get_name(r):
            return r[0] if isinstance(r, (list, tuple)) else r.get('related', '')
        names_a = {_get_name(r) for r in related_a}
        names_b = {_get_name(r) for r in related_b}
        return list(names_a & names_b)

    def get_entity_context(self, entity: str) -> Dict:
        """Get full structured context for an entity."""
        return {
            'entity': self.find_entity_by_name(entity),
            'actions': self.find_entity_actions(entity),
            'experience': self.find_entity_experience(entity),
            'related': self.find_related_entities(entity),
            'coreferences': self.get_coreference_chain(entity),
        }

    def find_entity_descriptions(self, entity: str) -> List[str]:
        """Find sentences that describe an entity (with be-verbs, adjectives)."""
        # Priority 1: "Entity is/was a X" — most descriptive, shortest first
        sql_p1 = """
            SELECT raw_text FROM sentences
            WHERE raw_text LIKE ?
              AND LENGTH(raw_text) > 20 AND LENGTH(raw_text) < 150
              AND (raw_text LIKE ? OR raw_text LIKE ? OR raw_text LIKE ? OR raw_text LIKE ?)
            ORDER BY LENGTH(raw_text) ASC
            LIMIT 5
        """
        rows = self.db.execute(sql_p1, (
            f'{entity} %',
            f'{entity} is a %', f'{entity} was a %',
            f'{entity} is an %', f'{entity} was an %',
        )) or []
        results = [r[0] for r in rows if r[0]]

        # Priority 2: Sentences starting with entity name (entity is the topic)
        if len(results) < 3:
            sql_p2 = """
                SELECT raw_text FROM sentences
                WHERE raw_text LIKE ? AND LENGTH(raw_text) > 20 AND LENGTH(raw_text) < 150
                ORDER BY LENGTH(raw_text) ASC
                LIMIT 5
            """
            rows2 = self.db.execute(sql_p2, (f'{entity} %',)) or []
            for r in rows2:
                if r[0] and r[0] not in results:
                    results.append(r[0])

        # Priority 3: "Entity has/was the" patterns
        if len(results) < 3:
            sql_p3 = """
                SELECT raw_text FROM sentences
                WHERE raw_text LIKE ?
                  AND LENGTH(raw_text) > 20 AND LENGTH(raw_text) < 150
                  AND (raw_text LIKE ? OR raw_text LIKE ? OR raw_text LIKE ?)
                ORDER BY LENGTH(raw_text) ASC
                LIMIT 5
            """
            rows3 = self.db.execute(sql_p3, (
                f'%{entity}%',
                f'%{entity} has %', f'%{entity} had %',
                f'%{entity} was the %',
            )) or []
            for r in rows3:
                if r[0] and r[0] not in results:
                    results.append(r[0])

        return results[:10]

    def find_entity_attributes(self, entity: str) -> List[str]:
        """Extract attributes about an entity from descriptive sentences."""
        attributes = []
        seen = set()

        descs = self.find_entity_descriptions(entity)
        for desc in descs[:8]:
            lower = desc.lower()
            entity_lower = entity.lower()
            # Only match patterns where entity is the subject
            for pattern in [f'{entity_lower} is a ', f'{entity_lower} was a ',
                           f'{entity_lower} is an ', f'{entity_lower} was an ',
                           f'{entity_lower} is ', f'{entity_lower} was ',
                           f'{entity_lower} has ', f'{entity_lower} had ']:
                idx = lower.find(pattern)
                if idx >= 0:
                    rest = desc[idx + len(pattern):].strip()
                    # Cut at next sentence boundary or conjunction
                    for ch in ['.', ',', ';', '—', '--', ':', 'and', 'but', 'who']:
                        cut = rest.lower().find(ch if isinstance(ch, str) else ch)
                        if 0 < cut < 80:
                            rest = rest[:cut]
                    rest = rest.strip().rstrip('.,;:!?')
                    if rest and 3 < len(rest) < 80:
                        # Skip if it just restates the entity name
                        if entity.lower() not in rest.lower() and rest not in seen:
                            seen.add(rest)
                            attributes.append(rest)
                            break

        return attributes[:5]

    def find_entity_definition(self, entity: str) -> str:
        """Find dictionary definition, or derive from descriptive context."""
        # Try the entity name directly
        defn = self.find_definition(entity)
        if defn:
            return defn

        # For proper nouns, derive from descriptive sentences
        descs = self.find_entity_descriptions(entity)
        for desc in descs[:15]:
            lower = desc.lower()
            entity_lower = entity.lower()
            # Look for "entity is a/an X" or "entity was a/an X"
            for pattern in [f'{entity_lower} is a ', f'{entity_lower} was a ',
                           f'{entity_lower} is an ', f'{entity_lower} was an ']:
                idx = lower.find(pattern)
                if idx >= 0:
                    rest = desc[idx + len(pattern):].strip()
                    # Cut at punctuation
                    for ch in ['.', ',', ';', '—', '--', ':']:
                        cut = rest.find(ch)
                        if 0 < cut < 60:
                            rest = rest[:cut]
                    rest = rest.strip().rstrip('.,;:!?')
                    if rest and 3 < len(rest) < 60:
                        return rest

        return ''
