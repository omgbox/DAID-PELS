"""
Lightweight Attention-Based Sentence Planner

Novel approach: instead of fixed templates, uses self-attention over SVO triples
to learn which facts are relevant and how to order them.

Architecture:
  1. SVO Embedding: average word2vec vectors for subject/verb/object
  2. Entity-Aware Attention: query-weighted attention over SVO triples
  3. Selection Network: scores each triple for inclusion
  4. Orderer: learned ordering based on semantic flow
  5. Realizer: converts selected triples to natural sentences

This is NOT a full transformer — it's a lightweight attention module
(~200 lines, CPU-friendly, no pretrained weights needed).
"""

import math
import random
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger('bookbot.query.attention_planner')


class AttentionPlanner:
    """
    Attention-based sentence planner that selects and orders SVO triples.

    Input: ranked SVO triples + entity + intent
    Output: ordered list of (subject, verb, object) with discourse roles

    Novel: uses entity-aware attention to weight triples by relevance,
    then a learned ordering network to sequence them naturally.
    """

    def __init__(self, embeddings, dim: int = 50):
        self.embeddings = embeddings
        self.dim = dim

        # Project word embeddings to planner dim
        self.proj_scale = math.sqrt(2.0 / (dim + dim))
        self.W_proj = np.random.randn(dim, dim).astype(np.float32) * self.proj_scale
        self.b_proj = np.zeros(dim, dtype=np.float32)

        # Attention: entity query -> SVO keys
        self.W_query = np.random.randn(dim, dim).astype(np.float32) * self.proj_scale
        self.W_key = np.random.randn(dim, dim).astype(np.float32) * self.proj_scale
        self.W_val = np.random.randn(dim, dim).astype(np.float32) * self.proj_scale

        # Selection scoring
        self.W_score = np.random.randn(dim, 1).astype(np.float32) * math.sqrt(2.0 / dim)
        self.b_score = np.zeros(1, dtype=np.float32)

        # Ordering: pairwise comparison network
        self.W_order = np.random.randn(dim * 2, 1).astype(np.float32) * math.sqrt(2.0 / (dim * 2))
        self.b_order = np.zeros(1, dtype=np.float32)

        # Discourse role classification (nucleus/satellite)
        self.W_disc = np.random.randn(dim, 3).astype(np.float32) * math.sqrt(2.0 / dim)
        self.b_disc = np.zeros(3, dtype=np.float32)

    def _embed_triple(self, subject: str, verb: str, obj: str) -> np.ndarray:
        """Embed an SVO triple as the average of its word vectors."""
        words = subject.lower().split() + verb.lower().split() + obj.lower().split()
        vecs = []
        for w in words:
            v = self.embeddings.get_vector(w)
            if v is not None:
                # Project to planner dim
                vecs.append(v @ self.W_proj + self.b_proj)
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def _embed_entity(self, entity: str) -> np.ndarray:
        """Embed entity name."""
        words = entity.lower().split()
        vecs = []
        for w in words:
            v = self.embeddings.get_vector(w)
            if v is not None:
                vecs.append(v @ self.W_proj + self.b_proj)
        if not vecs:
            return np.zeros(self.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max())
        return e / (e.sum() + 1e-10)

    def _attention(self, query: np.ndarray, keys: np.ndarray,
                   values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-head attention.
        query: (D,)
        keys: (N, D)
        values: (N, D)
        Returns: attended (D,), attention_weights (N,)
        """
        Q = query @ self.W_query    # (D,)
        K = keys @ self.W_key       # (N, D)
        V = values @ self.W_val     # (N, D)

        # Scaled dot-product attention
        scores = (K @ Q) / math.sqrt(self.dim)  # (N,)
        weights = self._softmax(scores)          # (N,)

        attended = weights @ V                   # (D,)
        return attended, weights

    def _select_triples(self, entity_vec: np.ndarray,
                        triple_vecs: np.ndarray) -> np.ndarray:
        """Score each triple for inclusion. Returns scores (N,)."""
        N = triple_vecs.shape[0]
        scores = np.zeros(N, dtype=np.float32)
        for i in range(N):
            # Concatenate entity+triple for scoring
            combined = np.concatenate([entity_vec, triple_vecs[i]])
            h = self._relu(combined @ np.concatenate([
                np.random.randn(self.dim, self.dim) * self.proj_scale,
                np.random.randn(self.dim, self.dim) * self.proj_scale
            ], axis=0))
            scores[i] = float((h @ self.W_score).item() + self.b_score.item())
        return self._softmax(scores)

    def _order_triples(self, triple_vecs: np.ndarray,
                       scores: np.ndarray) -> List[int]:
        """
        Order triples by learned pairwise comparison.
        Returns list of indices in display order.
        """
        n = len(triple_vecs)
        if n <= 1:
            return list(range(n))

        # Pairwise preference: for each pair (i,j), who should come first?
        # Higher-scored triples tend to come first, but also consider semantic flow
        order_scores = np.zeros(n, dtype=np.float32)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pair = np.concatenate([triple_vecs[i], triple_vecs[j]])
                pref = float((pair @ self.W_order).item() + self.b_order.item())
                # If pref > 0, i should come before j
                order_scores[i] += pref

        # Sort by order score (descending)
        return list(np.argsort(-order_scores))

    def _assign_discourse_roles(self, triple_vecs: np.ndarray,
                                order: List[int]) -> List[str]:
        """Assign discourse roles: nucleus (most important), satellite (supporting)."""
        roles = []
        for rank, idx in enumerate(order):
            h = self._relu(triple_vecs[idx])
            logits = h @ self.W_disc + self.b_disc
            probs = self._softmax(logits)
            role_idx = int(np.argmax(probs))
            role_map = {0: 'nucleus', 1: 'elaboration', 2: 'satellite'}
            roles.append(role_map.get(role_idx, 'elaboration'))
        return roles

    def plan(self, entity: str, triples: List,
             intent: str = 'DEFINITIONAL',
             max_triples: int = 4) -> List[Dict]:
        """
        Plan which triples to include and in what order.
        """
        if not triples:
            return []

        # Pre-filter: skip triples with bad verbs/objects
        SKIP_VERBS = {'is', 'are', 'was', 'were', 'has', 'have', 'had',
                      's', 'es', 'ses', 'been', 'being'}
        BAD_OBJ = {'i', 'me', 'she', 'he', 'it', 'we', 'they', 'you',
                    'him', 'her', 'us', 'them', 'his', 'its', 'my',
                    'your', 'their', 'our', 'a', 'an', 'the', 'mr', 'mrs'}

        clean_triples = []
        for t in triples:
            if isinstance(t, dict):
                s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                s, v, o = str(t[0]), str(t[1]), str(t[2])
            else:
                continue

            v_clean = v.strip().lower()
            o_clean = o.strip().lower().rstrip('.,;:!?')

            # Skip bad verbs
            if v_clean in SKIP_VERBS or len(v_clean) < 3:
                continue
            # Skip artifacts with underscores
            if '_' in v or '_' in o:
                continue
            # Skip bad objects
            if o_clean in BAD_OBJ or len(o_clean) < 2:
                continue
            # Skip very long objects (raw sentences)
            if len(o) > 40:
                continue
            # Skip objects that are just articles
            if o_clean in ('a', 'an', 'the', 'to', 'of', 'in', 'for'):
                continue

            clean_triples.append({'subject': s, 'verb': v.strip(), 'object': o.strip()})

        if not clean_triples:
            return []

        # Embed all triples
        triple_vecs = []
        for t in clean_triples:
            vec = self._embed_triple(t['subject'], t['verb'], t['object'])
            triple_vecs.append(vec)

        triple_vecs = np.array(triple_vecs)
        entity_vec = self._embed_entity(entity)

        # Attend over triples using entity as query
        attended, attn_weights = self._attention(entity_vec, triple_vecs, triple_vecs)

        # Score each triple for inclusion
        inclusion_scores = self._select_triples(entity_vec, triple_vecs)

        # Combine attention and inclusion scores
        combined_scores = 0.5 * attn_weights + 0.5 * inclusion_scores

        # Select top-K
        top_k = min(max_triples, len(clean_triples))
        top_indices = np.argsort(-combined_scores)[:top_k]

        # Order selected triples
        selected_vecs = triple_vecs[top_indices]
        order = self._order_triples(selected_vecs, combined_scores[top_indices])

        # Assign discourse roles
        roles = self._assign_discourse_roles(selected_vecs, order)

        # Build output
        result = []
        for rank, local_idx in enumerate(order):
            real_idx = top_indices[local_idx]
            result.append({
                'subject': clean_triples[real_idx]['subject'],
                'verb': clean_triples[real_idx]['verb'],
                'object': clean_triples[real_idx]['object'],
                'role': roles[rank],
                'score': float(combined_scores[real_idx]),
            })

        return result

    def realize(self, entity: str, plan: List[Dict],
                related: List[str] = None) -> str:
        """
        Convert a plan into a natural language answer.

        Uses templates but with neural-selected content and ordering.
        """
        if not plan:
            return f"{entity} is a character in the story."

        parts = []

        for i, item in enumerate(plan):
            s = item['subject']
            v = item['verb']
            o = item['object']
            role = item['role']

            # Skip noise
            if not v or not o or len(o) < 2:
                continue

            # Conjugate verb for third person
            v_conj = self._conjugate_verb(v)

            # Build sentence
            if o and len(o) > 1:
                sent = f"{s} {v_conj} {o}"
            else:
                sent = f"{s} {v_conj}"

            # Clean up
            sent = sent.strip()
            if sent and sent[-1] not in '.!?':
                sent += '.'

            # Capitalize
            if sent:
                sent = sent[0].upper() + sent[1:]

            parts.append(sent)

        # Add relationships
        if related:
            rel_names = [r for r in related[:4] if r.lower() != entity.lower()]
            if rel_names:
                if len(rel_names) == 1:
                    rel_text = rel_names[0]
                elif len(rel_names) == 2:
                    rel_text = f"{rel_names[0]} and {rel_names[1]}"
                else:
                    rel_text = f"{', '.join(rel_names[:3])} and others"
                parts.append(f"In the story, {entity} interacts with {rel_text}.")

        return ' '.join(parts) if parts else f"{entity} is a character in the story."

    def _conjugate_verb(self, verb: str) -> str:
        """Third-person singular conjugation."""
        v = verb.lower().strip()
        if not v:
            return v

        # Already conjugated or artifacts
        if v in ('is', 'are', 'was', 'were', 'has', 'have', 'had'):
            return v
        if '_' in v or len(v) < 3:
            return v

        # Don't conjugate already-past-tense verbs
        if v.endswith('ed') or v.endswith('ght'):
            return v
        # Don't conjugate if already ends in 's' (except 'ss')
        if v.endswith('s') and not v.endswith('ss'):
            return v

        # Irregular past tense (return past form for 3rd person narrative)
        IRREGULAR_PAST = {
            'know': 'knew', 'feel': 'felt', 'find': 'found',
            'say': 'said', 'tell': 'told', 'give': 'gave',
            'take': 'took', 'make': 'made', 'come': 'came',
            'go': 'went', 'see': 'saw', 'do': 'did',
            'get': 'got', 'think': 'thought', 'leave': 'left',
            'bring': 'brought', 'write': 'wrote', 'pay': 'paid',
            'read': 'read', 'begin': 'began', 'keep': 'kept',
            'hold': 'held', 'stand': 'stood', 'sit': 'sat',
            'run': 'ran', 'grow': 'grew', 'draw': 'drew',
            'break': 'broke', 'spend': 'spent', 'cut': 'cut',
            'put': 'put', 'let': 'let', 'set': 'set',
            'show': 'showed', 'lead': 'led', 'fall': 'fell',
            'speak': 'spoke', 'eat': 'ate', 'drive': 'drove',
            'wear': 'wore', 'win': 'won', 'lose': 'lost',
        }
        if v in IRREGULAR_PAST:
            return IRREGULAR_PAST[v]

        # Regular: just return base form (narrative uses past tense)
        return v

    def save(self, path: str):
        """Save model weights."""
        data = {
            'dim': self.dim,
            'W_proj': self.W_proj.tolist(),
            'b_proj': self.b_proj.tolist(),
            'W_query': self.W_query.tolist(),
            'W_key': self.W_key.tolist(),
            'W_val': self.W_val.tolist(),
            'W_score': self.W_score.tolist(),
            'b_score': self.b_score.tolist(),
            'W_order': self.W_order.tolist(),
            'b_order': self.b_order.tolist(),
            'W_disc': self.W_disc.tolist(),
            'b_disc': self.b_disc.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, embeddings) -> 'AttentionPlanner':
        """Load model weights."""
        with open(path, 'r') as f:
            data = json.load(f)
        planner = cls(embeddings, data['dim'])
        planner.W_proj = np.array(data['W_proj'], dtype=np.float32)
        planner.b_proj = np.array(data['b_proj'], dtype=np.float32)
        planner.W_query = np.array(data['W_query'], dtype=np.float32)
        planner.W_key = np.array(data['W_key'], dtype=np.float32)
        planner.W_val = np.array(data['W_val'], dtype=np.float32)
        planner.W_score = np.array(data['W_score'], dtype=np.float32)
        planner.b_score = np.array(data['b_score'], dtype=np.float32)
        planner.W_order = np.array(data['W_order'], dtype=np.float32)
        planner.b_order = np.array(data['b_order'], dtype=np.float32)
        planner.W_disc = np.array(data['W_disc'], dtype=np.float32)
        planner.b_disc = np.array(data['b_disc'], dtype=np.float32)
        return planner
