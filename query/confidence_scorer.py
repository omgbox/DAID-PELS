"""
BookBot Confidence Scorer
Calibrated confidence estimation.
"""

import logging
from typing import Dict, List

from ..core.base_module import BaseModule

logger = logging.getLogger('bookbot.query.confidence_scorer')


class ConfidenceScorer(BaseModule):
    """Confidence scoring module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and compute confidence score.

        Args:
            input_data: Dict with 'query', 'answer', 'retrieval_results' keys

        Returns:
            Dict with 'confidence' key
        """
        query = input_data.get('query', '')
        answer = input_data.get('answer', {})
        retrieval_results = input_data.get('retrieval_results', [])

        # Compute confidence components
        retrieval_quality = self._compute_retrieval_quality(retrieval_results)
        answer_coverage = self._compute_answer_coverage(answer)
        consensus = self._compute_consensus(answer)
        entity_familiarity = self._compute_entity_familiarity(answer)
        intent_clarity = 1.0  # Default

        # Compute weighted confidence
        weights = self.config
        confidence = (
            weights.get('retrieval_weight', 0.30) * retrieval_quality +
            weights.get('coverage_weight', 0.25) * answer_coverage +
            weights.get('consensus_weight', 0.20) * consensus +
            weights.get('entity_weight', 0.15) * entity_familiarity +
            weights.get('intent_weight', 0.10) * intent_clarity
        )

        # Apply modifiers
        if answer.get('text'):
            # Boost if multiple sources
            if len(answer.get('sources', [])) > 1:
                confidence += 0.1

            # Penalty for long answers
            if len(answer['text'].split()) > 50:
                confidence -= 0.1

        # Clip to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        self._initialized = True

        return {'confidence': confidence}

    def _compute_retrieval_quality(self, retrieval_results: List[Dict]) -> float:
        """
        Compute retrieval quality score.

        Args:
            retrieval_results: List of retrieval result dicts

        Returns:
            Quality score (0.0 to 1.0)
        """
        if not retrieval_results:
            return 0.0

        # Use top 3 scores
        top_scores = sorted([r.get('score', 0) for r in retrieval_results[:3]], reverse=True)
        if not top_scores:
            return 0.0

        # Normalize (assuming max score around 10)
        avg_score = sum(top_scores) / len(top_scores)
        return min(1.0, avg_score / 10.0)

    def _compute_answer_coverage(self, answer: Dict) -> float:
        """
        Compute answer coverage score.

        Args:
            answer: Answer dict

        Returns:
            Coverage score (0.0 to 1.0)
        """
        sentences = answer.get('sentences', [])
        if not sentences:
            return 0.0

        # Coverage based on number of supporting sentences
        return min(1.0, len(sentences) / 5.0)

    def _compute_consensus(self, answer: Dict) -> float:
        """
        Compute consensus score.

        Args:
            answer: Answer dict

        Returns:
            Consensus score (0.0 to 1.0)
        """
        # Simplified: check if sources agree
        sources = answer.get('sources', [])
        if len(sources) <= 1:
            return 0.5

        # Simple text similarity check
        texts = [s.get('text', '') for s in sources]
        if len(texts) < 2:
            return 0.5

        # Count common words
        words_sets = [set(t.lower().split()) for t in texts]
        if not words_sets[0]:
            return 0.5

        common = words_sets[0]
        for ws in words_sets[1:]:
            common = common.intersection(ws)

        all_words = set()
        for ws in words_sets:
            all_words.update(ws)

        if not all_words:
            return 0.5

        return len(common) / len(all_words)

    def _compute_entity_familiarity(self, answer: Dict) -> float:
        """
        Compute entity familiarity score.

        Args:
            answer: Answer dict

        Returns:
            Familiarity score (0.0 to 1.0)
        """
        # Simplified: check if answer contains known entities
        text = answer.get('text', '')
        if not text:
            return 0.0

        # Default to moderate familiarity
        return 0.5
