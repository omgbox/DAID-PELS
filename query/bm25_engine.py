"""
BookBot BM25 Engine
BM25 + FTS5 dual-index retrieval.
"""

import logging
from typing import Dict, List, Optional

from ..core.base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.query.bm25_engine')

# Try to import rank_bm25
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class BM25Engine(BaseModule):
    """BM25 + FTS5 retrieval module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.bm25 = None
        self.corpus = []

    def process(self, input_data) -> dict:
        """
        Process input data and retrieve relevant sentences.

        Args:
            input_data: Dict with 'query' and 'context' keys

        Returns:
            Dict with 'results' key
        """
        query = input_data.get('query', '')
        context = input_data.get('context')
        intent = input_data.get('intent', 'EXPLANATORY')

        # Build index if needed
        if context and not self.bm25:
            self._build_index(context)

        # Retrieve results
        results = []
        if self.bm25:
            bm25_results = self._retrieve_bm25(query)
            results.extend(bm25_results)

        if self.db:
            fts5_results = self._retrieve_fts5(query)
            results.extend(fts5_results)

        # Merge and rank
        merged = self._merge_results(results)

        self._initialized = True

        return {'results': merged}

    def _build_index(self, context: PipelineContext):
        """
        Build BM25 index from sentences.

        Args:
            context: Pipeline context
        """
        if not BM25_AVAILABLE:
            logger.warning("rank_bm25 not available")
            return

        sentences = context.sentences or []
        self.corpus = []
        self._sentence_ids = []
        self._sentence_texts = []

        for sent in sentences:
            # Try to get tokens from sentence
            tokens = sent.get('tokens', [])
            
            # If tokens are empty, tokenize the sentence text
            if not tokens:
                text = sent.get('text', '')
                if text:
                    # Simple tokenization
                    import re
                    words = re.findall(r'\b\w+\b', text.lower())
                    tokens = [{'token_lower': w} for w in words]
                else:
                    tokens = []

            # Extract token strings
            token_strings = [t.get('token_lower', '') for t in tokens
                           if not t.get('is_punctuation') and t.get('token_lower')]
            
            if token_strings:
                self.corpus.append(token_strings)
                self._sentence_ids.append(sent.get('sentence_id'))
                self._sentence_texts.append(sent.get('text', ''))

        if self.corpus and len(self.corpus) > 0:
            try:
                self.bm25 = BM25Okapi(self.corpus)
                logger.info(f"Built BM25 index with {len(self.corpus)} documents")
            except Exception as e:
                logger.warning(f"Failed to build BM25 index: {e}")
                self.bm25 = None

    def _retrieve_bm25(self, query: str) -> List[Dict]:
        """
        Retrieve results using BM25.

        Args:
            query: Query text

        Returns:
            List of result dicts
        """
        if not self.bm25:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        top_k = self.get_config('bm25_top_k', 50)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                # Map index to sentence_id
                sentence_id = idx  # Default to index
                if hasattr(self, '_sentence_ids') and idx < len(self._sentence_ids):
                    sentence_id = self._sentence_ids[idx]
                
                # Get sentence text for debugging
                sentence_text = ""
                if hasattr(self, '_sentence_texts') and idx < len(self._sentence_texts):
                    sentence_text = self._sentence_texts[idx]
                
                results.append({
                    'sentence_id': sentence_id,
                    'score': float(scores[idx]),
                    'source': 'bm25',
                    'text': sentence_text,
                })

        return results

    def _retrieve_fts5(self, query: str) -> List[Dict]:
        """
        Retrieve results using FTS5.

        Args:
            query: Query text

        Returns:
            List of result dicts
        """
        if not self.db:
            return []

        try:
            # Check if FTS5 table exists
            tables = self.db.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sentences_fts'")
            if not tables:
                return []

            # Build FTS5 query
            query_terms = query.split()
            fts5_query = ' OR '.join(query_terms)

            results = self.db.execute(
                "SELECT sentence_id, rank FROM sentences_fts WHERE sentences_fts MATCH ? ORDER BY rank LIMIT ?",
                (fts5_query, self.get_config('fts5_top_k', 50))
            )

            return [{'sentence_id': row[0], 'score': abs(row[1]), 'source': 'fts5'}
                    for row in results]
        except Exception as e:
            logger.debug(f"FTS5 query failed: {e}")
            return []

    def _merge_results(self, results: List[Dict]) -> List[Dict]:
        """
        Merge and rank results from multiple sources.

        Args:
            results: List of result dicts

        Returns:
            Merged and ranked results
        """
        bm25_weight = self.get_config('bm25_weight', 0.4)
        fts5_weight = self.get_config('fts5_weight', 0.3)

        # Group by sentence_id
        sentence_scores = {}
        for result in results:
            sid = result['sentence_id']
            if sid not in sentence_scores:
                sentence_scores[sid] = {'bm25': 0, 'fts5': 0}

            if result['source'] == 'bm25':
                sentence_scores[sid]['bm25'] = result['score']
            elif result['source'] == 'fts5':
                sentence_scores[sid]['fts5'] = result['score']

        # Compute merged scores
        merged = []
        for sid, scores in sentence_scores.items():
            merged_score = (scores['bm25'] * bm25_weight +
                          scores['fts5'] * fts5_weight)
            merged.append({
                'sentence_id': sid,
                'score': merged_score,
                'bm25_score': scores['bm25'],
                'fts5_score': scores['fts5'],
            })

        # Sort by merged score
        merged.sort(key=lambda x: x['score'], reverse=True)

        top_k = self.get_config('merged_top_k', 10)
        return merged[:top_k]
