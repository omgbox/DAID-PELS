"""
BookBot Topic Modeler
Sentence-level TF-IDF clustering.
"""

import logging
import math
from typing import Dict, List, Optional
from collections import defaultdict

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.topic_modeler')


class TopicModeler(BaseModule):
    """Topic modeling module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        """
        Process input data and build topic clusters.

        Args:
            input_data: PipelineContext or dict

        Returns:
            Dict with 'topics' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Build TF-IDF vectors
        tfidf_vectors = self._build_tfidf_vectors(sentences)

        # Cluster sentences
        topics = self._cluster_sentences(tfidf_vectors, sentences)

        self._initialized = True
        self.logger.info(f"Found {len(topics)} topics")

        return {'topics': topics}

    def _build_tfidf_vectors(self, sentences: List[Dict]) -> List[Dict[str, float]]:
        """
        Build TF-IDF vectors for sentences.

        Args:
            sentences: List of sentence dicts

        Returns:
            List of TF-IDF vectors
        """
        # Compute document frequency
        doc_freq = defaultdict(int)
        for sent in sentences:
            words = set(t.get('token_lower', '') for t in sent.get('tokens', [])
                       if not t.get('is_punctuation'))
            for word in words:
                doc_freq[word] += 1

        # Compute TF-IDF for each sentence
        num_sentences = len(sentences)
        vectors = []

        for sent in sentences:
            # Compute term frequency
            tf = defaultdict(int)
            for token in sent.get('tokens', []):
                if not token.get('is_punctuation'):
                    word = token.get('token_lower', '')
                    tf[word] += 1

            # Compute TF-IDF
            vector = {}
            for word, count in tf.items():
                tf_score = count / len(sent.get('tokens', []))
                idf_score = math.log((num_sentences + 1) / (doc_freq.get(word, 0) + 1)) + 1
                vector[word] = tf_score * idf_score

            vectors.append(vector)

        return vectors

    def _cluster_sentences(self, tfidf_vectors: List[Dict[str, float]],
                           sentences: List[Dict]) -> List[Dict]:
        """
        Cluster sentences into topics using connected-component clustering.
        Auto-samples if sentence count exceeds threshold for speed.

        Args:
            tfidf_vectors: List of TF-IDF vectors
            sentences: List of sentence dicts

        Returns:
            List of topic dicts
        """
        threshold = self.get_config('topic_similarity_threshold', 0.3)
        max_clusters = self.get_config('topic_max_clusters', 50)

        # Auto-sample if too many sentences (O(n^2) scaling)
        max_for_clustering = self.get_config('topic_max_sentences', 500)
        total = len(tfidf_vectors)
        if total > max_for_clustering:
            step = total // max_for_clustering
            sampled_indices = list(range(0, total, step))[:max_for_clustering]
            tfidf_vectors = [tfidf_vectors[i] for i in sampled_indices]
            sentences = [sentences[i] for i in sampled_indices]
            self.logger.info(
                f"Clustering sampled from {total} to {len(tfidf_vectors)} sentences "
                f"(O(n²) would be {total:,}² = {total*total:,} pairs)"
            )

        n = len(tfidf_vectors)
        self.logger.info(f"Building topic clusters: {n} sentences, {n*n:,} pair comparisons...")

        # Simple connected-component clustering
        clusters = []
        assigned = set()
        last_log = 0

        for i, vec in enumerate(tfidf_vectors):
            if i in assigned:
                continue

            # Start new cluster
            cluster = [i]
            assigned.add(i)

            # Find similar sentences
            for j, other_vec in enumerate(tfidf_vectors):
                if j in assigned:
                    continue

                similarity = self._cosine_similarity(vec, other_vec)
                if similarity > threshold:
                    cluster.append(j)
                    assigned.add(j)

            clusters.append(cluster)

            # Progress every 20 clusters
            if len(clusters) - last_log >= 20:
                self.logger.info(
                    f"  Topic clusters: {len(clusters)} found, "
                    f"{len(assigned)}/{n} sentences assigned"
                )
                last_log = len(clusters)

            if len(clusters) >= max_clusters:
                self.logger.info(f"Reached max clusters ({max_clusters}), stopping")
                break

        # Build topic objects
        topics = []
        for cluster_id, cluster_indices in enumerate(clusters[:max_clusters]):
            # Get top terms
            term_scores = defaultdict(float)
            for idx in cluster_indices:
                for word, score in tfidf_vectors[idx].items():
                    term_scores[word] += score

            top_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            top_terms_str = ', '.join(word for word, _ in top_terms)

            topics.append({
                'topic_id': cluster_id,
                'label': top_terms_str,
                'top_terms': top_terms_str,
                'sentence_count': len(cluster_indices),
                'sentence_ids': [sentences[idx].get('sentence_id') for idx in cluster_indices],
            })

        return topics

    def _cosine_similarity(self, vec1: Dict[str, float],
                           vec2: Dict[str, float]) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity (0.0 to 1.0)
        """
        # Find common keys
        common_keys = set(vec1.keys()) & set(vec2.keys())
        if not common_keys:
            return 0.0

        # Compute dot product
        dot_product = sum(vec1[k] * vec2[k] for k in common_keys)

        # Compute magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)
