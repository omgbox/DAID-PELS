"""
BookBot Entity Graph
Co-occurrence graph construction and centrality computation.
"""

import logging
from typing import Dict, List, Optional
from collections import defaultdict

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.entity_graph')


class EntityGraph(BaseModule):
    """Entity graph module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.edges = defaultdict(float)
        self.centrality = {}

    def process(self, input_data) -> dict:
        """
        Process input data and build entity graph.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'knowledge_edges' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
            entities = input_data.entities
            svo_triples = input_data.svo_triples
        else:
            sentences = input_data.get('sentences', [])
            entities = input_data.get('entities', [])
            svo_triples = input_data.get('svo_triples', [])

        # Build edges
        edges = []
        entity_names = {e.get('canonical_name', '') for e in entities}
        edges.extend(self._build_cooccurrence_edges(sentences, entities))
        edges.extend(self._build_svo_edges(svo_triples, entity_names))

        # Compute centrality
        centrality = self._compute_centrality(edges, entities)

        # Update entities with centrality
        for entity in entities:
            name = entity.get('canonical_name', '')
            entity['centrality'] = centrality.get(name, 0.0)

        self._initialized = True
        self.logger.info(f"Built graph with {len(edges)} edges")

        return {
            'knowledge_edges': edges,
            'entities': entities,
        }

    def _build_cooccurrence_edges(self, sentences: List[Dict],
                                   entities: List[Dict]) -> List[Dict]:
        """
        Build co-occurrence edges between entities with frequency weights.
        """
        entity_names = {e.get('canonical_name', '') for e in entities}
        cooccur_counts = defaultdict(int)

        for sent in sentences:
            sent_entities = []
            for token in sent.get('tokens', []):
                word = token.get('token', '')
                if word in entity_names:
                    sent_entities.append(word)

            # Count co-occurrences between entities in same sentence
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:]:
                    if e1 != e2:
                        # Normalize ordering for counting
                        pair = tuple(sorted([e1, e2]))
                        cooccur_counts[pair] += 1

        # Build edges with frequency weight
        edges = []
        for (e1, e2), count in cooccur_counts.items():
            edges.append({
                'source_type': 'entity',
                'source_id': e1,
                'target_type': 'entity',
                'target_id': e2,
                'edge_type': 'co_occurrence',
                'weight': float(count),
            })

        return edges

    def _build_svo_edges(self, svo_triples: List[Dict],
                         entity_names: set = None) -> List[Dict]:
        """
        Build edges from SVO triples, only for recognized entities.
        """
        edges = []
        entity_names = entity_names or set()

        for triple in svo_triples:
            subject = triple.get('subject', '')
            verb = triple.get('verb', '')
            object_ = triple.get('object', '')

            # Only create edges if both endpoints are actual entities
            if subject and object_ and subject in entity_names and object_ in entity_names:
                edges.append({
                    'source_type': 'entity',
                    'source_id': subject,
                    'target_type': 'entity',
                    'target_id': object_,
                    'edge_type': verb,
                    'weight': triple.get('confidence', 0.5),
                })

        return edges

    def _compute_centrality(self, edges: List[Dict],
                            entities: List[Dict]) -> Dict[str, float]:
        """
        Compute entity centrality using degree centrality.

        Args:
            edges: List of edge dicts
            entities: List of entity dicts

        Returns:
            Dict mapping entity names to centrality scores
        """
        # Count degree for each entity
        degree = defaultdict(int)
        for edge in edges:
            source = edge.get('source_id', '')
            target = edge.get('target_id', '')
            degree[source] += 1
            degree[target] += 1

        # Normalize by max degree
        max_degree = max(degree.values()) if degree else 1
        centrality = {name: count / max_degree for name, count in degree.items()}

        return centrality
