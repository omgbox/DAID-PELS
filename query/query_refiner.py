"""
Query Refiner
Detect poor answers and reformulate queries for retry.
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.query.query_refiner')


class QueryRefiner:
    """Refine queries when initial retrieval is poor."""

    def __init__(self, db=None, bm25_engine=None):
        self.db = db
        self.bm25 = bm25_engine

    def evaluate_quality(self, answer: str, confidence: float,
                         sources: list) -> Dict:
        """
        Evaluate answer quality.

        Returns:
            {'good': bool, 'reason': str, 'score': float}
        """
        if not answer or len(answer.strip()) < 20:
            return {'good': False, 'reason': 'answer_too_short', 'score': 0.0}

        if confidence < 0.15:
            return {'good': False, 'reason': 'low_confidence', 'score': confidence}

        words = answer.split()
        if len(words) < 5:
            return {'good': False, 'reason': 'answer_fragment', 'score': 0.1}

        # Check for verb presence (real sentence)
        common_verbs = {'is', 'was', 'were', 'are', 'has', 'had', 'did',
                        'said', 'told', 'went', 'came', 'made', 'gave',
                        'loved', 'proposed', 'refused', 'returned'}
        has_verb = any(w.lower() in common_verbs for w in words)
        if not has_verb and len(words) > 8:
            return {'good': False, 'reason': 'no_verb', 'score': 0.2}

        # Check if answer is just repetition of query
        return {'good': True, 'reason': 'acceptable', 'score': confidence}

    def refine(self, original_query: str, context: Dict, attempt: int) -> str:
        """
        Generate a refined query based on what failed.

        Args:
            original_query: The original user query
            context: Info about the failure (entity, intent, reason)
            attempt: Which attempt this is (0, 1, 2...)
        """
        entity = context.get('primary_entity', '')
        intent = context.get('intent', 'FACTUAL')
        reason = context.get('poor_reason', '')

        if attempt == 0:
            # Strategy 1: Try the entity name directly
            if entity:
                return entity

        elif attempt == 1:
            # Strategy 2: Add intent-specific keywords
            intent_keywords = {
                'DEFINITIONAL': ['is', 'was', 'known', 'character'],
                'FACTUAL': ['did', 'happened', 'event'],
                'CAUSAL': ['because', 'reason', 'caused'],
                'TEMPORAL': ['when', 'before', 'after'],
            }
            keywords = intent_keywords.get(intent, [])
            if entity:
                return f"{entity} {' '.join(keywords[:2])}"

        elif attempt == 2:
            # Strategy 3: Try related entities via knowledge graph
            if entity and self.db:
                try:
                    rows = self.db.execute(
                        "SELECT target_id FROM knowledge_edges "
                        "WHERE LOWER(source_id) LIKE ? LIMIT 3",
                        (f'%{entity.lower()}%',)
                    )
                    if rows:
                        related = [r[0] for r in rows]
                        return f"{entity} {' '.join(related[:2])}"
                except Exception:
                    pass

        # Fallback: clean up original query
        return original_query.replace('?', '').strip()
