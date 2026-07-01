"""
Structured Retriever
Query SVO triples, entity graph, and coreference chains at query time.
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
}


class StructuredRetriever:
    """Query structured knowledge tables at query time."""

    def __init__(self, db):
        self.db = db

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
        names_a = {r['related'] for r in related_a}
        names_b = {r['related'] for r in related_b}
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
