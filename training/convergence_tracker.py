"""
BookBot Convergence Tracker
KL-divergence convergence criteria.
"""

import math
import logging
from typing import Dict, List, Optional
from collections import defaultdict

from ..core.base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.training.convergence_tracker')


class ConvergenceTracker(BaseModule):
    """Convergence tracking module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.previous_state = None
        self.pass_count = 0
        self.converged_count = 0

    def process(self, input_data) -> dict:
        """
        Process input data and check convergence.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'converged' key
        """
        if isinstance(input_data, PipelineContext):
            context = input_data
        else:
            context = None

        self.pass_count += 1

        # Get current state
        current_state = self._get_state(context)

        # Check convergence
        converged = False
        reason = None

        if self.previous_state:
            method = self.get_config('method', 'kl_divergence')

            if method == 'kl_divergence':
                kl_div = self._compute_kl_divergence(current_state)
                threshold = self.get_config('kl_threshold', 0.01)

                if kl_div < threshold:
                    self.converged_count += 1
                    if self.converged_count >= 2:
                        converged = True
                        reason = f"KL-divergence < {threshold} for 2 consecutive passes"
                else:
                    self.converged_count = 0

                logger.info(f"KL-divergence: {kl_div:.4f}")

            elif method == 'simple':
                new_rels = self._count_new_relationships(current_state)
                threshold = self.get_config('min_new_relationships', 5)

                if new_rels < threshold:
                    converged = True
                    reason = f"New relationships < {threshold}"

        # Check stability
        if not converged:
            stability = self._compute_stability(current_state)
            threshold = self.get_config('min_stability', 0.98)

            if stability > threshold:
                converged = True
                reason = f"Stability > {threshold}"

        # Check max passes
        max_passes = self.get_config('max_passes', 10)
        if self.pass_count >= max_passes:
            converged = True
            reason = f"Max passes ({max_passes}) reached"

        # Update state
        self.previous_state = current_state

        # Log convergence
        if converged:
            logger.info(f"CONVERGED: {reason}")

        return {
            'converged': converged,
            'reason': reason,
            'pass_count': self.pass_count,
        }

    def _get_state(self, context: PipelineContext = None) -> Dict:
        """
        Get current knowledge state.

        Args:
            context: Pipeline context

        Returns:
            State dict
        """
        state = {
            'entities': [],
            'svo_triples': [],
            'knowledge_edges': [],
            'topics': [],
        }

        if context:
            state['entities'] = context.entities or []
            state['svo_triples'] = context.svo_triples or []
            state['knowledge_edges'] = context.knowledge_edges or []
            state['topics'] = context.topics or []

        return state

    def _compute_kl_divergence(self, current_state: Dict) -> float:
        """
        Compute KL-divergence between current and previous state.

        Args:
            current_state: Current state dict

        Returns:
            KL-divergence value
        """
        if not self.previous_state:
            return float('inf')

        alpha = self.get_config('laplace_alpha', 0.01)

        # Compute for each knowledge type
        kl_total = 0.0
        n_types = 0

        for key in ['entities', 'svo_triples', 'knowledge_edges', 'topics']:
            current = self._to_distribution(current_state.get(key, []), alpha)
            previous = self._to_distribution(self.previous_state.get(key, []), alpha)

            kl = self._kl_divergence(current, previous)
            kl_total += kl
            n_types += 1

        return kl_total / n_types if n_types > 0 else 0.0

    def _to_distribution(self, items: List, alpha: float) -> Dict[str, float]:
        """
        Convert items to probability distribution.

        Args:
            items: List of items
            alpha: Laplace smoothing parameter

        Returns:
            Probability distribution
        """
        counts = defaultdict(int)
        for item in items:
            if isinstance(item, dict):
                key = str(item.get('edge_type', item.get('type', 'unknown')))
            else:
                key = str(item)
            counts[key] += 1

        total = sum(counts.values())
        V = len(counts)

        if total == 0:
            return {}

        return {k: (v + alpha) / (total + alpha * V) for k, v in counts.items()}

    def _kl_divergence(self, p: Dict[str, float], q: Dict[str, float]) -> float:
        """
        Compute KL-divergence between two distributions.

        Args:
            p: First distribution
            q: Second distribution

        Returns:
            KL-divergence value
        """
        if not p or not q:
            return 0.0

        kl = 0.0
        for key, p_val in p.items():
            q_val = q.get(key, 1e-10)
            if p_val > 0 and q_val > 0:
                kl += p_val * math.log(p_val / q_val)

        return kl

    def _count_new_relationships(self, current_state: Dict) -> int:
        """
        Count new relationships since last pass.

        Args:
            current_state: Current state dict

        Returns:
            Number of new relationships
        """
        if not self.previous_state:
            return len(current_state.get('knowledge_edges', []))

        current_edges = set()
        for edge in current_state.get('knowledge_edges', []):
            if isinstance(edge, dict):
                key = (edge.get('source_id'), edge.get('target_id'), edge.get('edge_type'))
                current_edges.add(key)

        previous_edges = set()
        for edge in self.previous_state.get('knowledge_edges', []):
            if isinstance(edge, dict):
                key = (edge.get('source_id'), edge.get('target_id'), edge.get('edge_type'))
                previous_edges.add(key)

        return len(current_edges - previous_edges)

    def _compute_stability(self, current_state: Dict) -> float:
        """
        Compute stability score.

        Args:
            current_state: Current state dict

        Returns:
            Stability score (0.0 to 1.0)
        """
        if not self.previous_state:
            return 0.0

        # Compare entity counts
        current_entities = len(current_state.get('entities', []))
        previous_entities = len(self.previous_state.get('entities', []))

        if current_entities == 0 and previous_entities == 0:
            return 1.0

        if current_entities == 0 or previous_entities == 0:
            return 0.0

        # Simple stability metric
        ratio = min(current_entities, previous_entities) / max(current_entities, previous_entities)
        return ratio
