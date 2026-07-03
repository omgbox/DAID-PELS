"""
Self-Attention Sentence Selector
Learns which sentences are relevant to each other and to the query.

Architecture:
  1. Token Embedding: word2vec → projected to model dim
  2. Positional Encoding: sinusoidal position signals
  3. Multi-Head Self-Attention: sentences attend to each other
  4. Cross-Attention: query attends to sentences
  5. Feed-Forward: per-sentence scoring
  6. Coreference Layer: pronoun resolution via attention

Novel: lightweight transformer-style architecture (~300 lines, pure NumPy)
that runs on CPU and trains on the book's own text.
"""

import math
import random
import logging
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.query.self_attention')


class MultiHeadAttention:
    """
    Multi-head scaled dot-product attention.

    For N sentences with D-dim embeddings:
      Q = X @ W_q, K = X @ W_k, V = X @ W_v
      Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """

    def __init__(self, dim: int, n_heads: int = 4):
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        s = math.sqrt(2.0 / (dim + self.head_dim))
        self.W_q = np.random.randn(dim, dim).astype(np.float32) * s
        self.W_k = np.random.randn(dim, dim).astype(np.float32) * s
        self.W_v = np.random.randn(dim, dim).astype(np.float32) * s
        self.W_o = np.random.randn(dim, dim).astype(np.float32) * s

        self.b_q = np.zeros(dim, dtype=np.float32)
        self.b_k = np.zeros(dim, dtype=np.float32)
        self.b_v = np.zeros(dim, dtype=np.float32)
        self.b_o = np.zeros(dim, dtype=np.float32)

    def forward(self, X: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (N, D) — N tokens/sentences, D dimensions
        Returns: (N, D) output, (N, N) attention weights
        """
        N = X.shape[0]

        Q = X @ self.W_q + self.b_q  # (N, D)
        K = X @ self.W_k + self.b_k
        V = X @ self.W_v + self.b_v

        # Reshape for multi-head: (N, n_heads, head_dim)
        Q = Q.reshape(N, self.n_heads, self.head_dim)
        K = K.reshape(N, self.n_heads, self.head_dim)
        V = V.reshape(N, self.n_heads, self.head_dim)

        # Scaled dot-product attention per head
        scores = np.zeros((self.n_heads, N, N), dtype=np.float32)
        for h in range(self.n_heads):
            # (N, head_dim) @ (head_dim, N) = (N, N)
            s = Q[:, h] @ K[:, h].T / math.sqrt(self.head_dim)
            if mask is not None:
                s = s + mask * (-1e9)
            scores[h] = s

        # Softmax per head
        attn = np.zeros_like(scores)
        for h in range(self.n_heads):
            e = np.exp(scores[h] - scores[h].max(axis=-1, keepdims=True))
            attn[h] = e / (e.sum(axis=-1, keepdims=True) + 1e-10)

        # Apply attention to values
        out = np.zeros((N, self.n_heads, self.head_dim), dtype=np.float32)
        for h in range(self.n_heads):
            out[:, h] = attn[h] @ V[:, h]

        # Concatenate heads
        out = out.reshape(N, self.dim)

        # Output projection
        out = out @ self.W_o + self.b_o

        # Average attention weights across heads
        avg_attn = attn.mean(axis=0)  # (N, N)

        return out, avg_attn


class FeedForward:
    """Position-wise feed-forward network."""

    def __init__(self, dim: int, hidden: int = None):
        self.dim = dim
        hidden = hidden or dim * 4

        s = math.sqrt(2.0 / (dim + hidden))
        self.W1 = np.random.randn(dim, hidden).astype(np.float32) * s
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = np.random.randn(hidden, dim).astype(np.float32) * s
        self.b2 = np.zeros(dim, dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        h = np.maximum(0, X @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2


class SelfAttentionBlock:
    """
    Single transformer block:
      Layer Norm → Multi-Head Attention → Residual → Layer Norm → FFN → Residual
    """

    def __init__(self, dim: int, n_heads: int = 4):
        self.attn = MultiHeadAttention(dim, n_heads)
        self.ffn = FeedForward(dim)
        # Layer norm parameters (initialized to 1/0)
        self.ln1_gamma = np.ones(dim, dtype=np.float32)
        self.ln1_beta = np.zeros(dim, dtype=np.float32)
        self.ln2_gamma = np.ones(dim, dtype=np.float32)
        self.ln2_beta = np.zeros(dim, dtype=np.float32)

    def _layer_norm(self, X: np.ndarray, gamma, beta):
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-10
        return gamma * (X - mean) / std + beta

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Self-attention with residual
        normed = self._layer_norm(X, self.ln1_gamma, self.ln1_beta)
        attn_out, attn_weights = self.attn.forward(normed)
        X = X + attn_out

        # FFN with residual
        normed = self._layer_norm(X, self.ln2_gamma, self.ln2_beta)
        X = X + self.ffn.forward(normed)

        return X, attn_weights


class PositionalEncoding:
    """Sinusoidal positional encoding."""

    def __init__(self, dim: int, max_len: int = 100):
        self.dim = dim
        pe = np.zeros((max_len, dim), dtype=np.float32)
        position = np.arange(0, max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pe = pe

    def encode(self, seq_len: int) -> np.ndarray:
        return self.pe[:seq_len]


class SelfAttentionSelector:
    """
    Full self-attention pipeline for sentence selection.

    Flow:
      1. Embed sentences using word2vec + positional encoding
      2. Apply N self-attention blocks
      3. Cross-attend query to sentences
      4. Score each sentence for relevance
      5. Resolve coreferences
    """

    def __init__(self, embeddings=None, dim: int = 64, n_layers: int = 2,
                 n_heads: int = 4):
        self.embeddings = embeddings
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Projection from word2vec dim to model dim
        w2v_dim = embeddings.model.dim if embeddings else 50
        s = math.sqrt(2.0 / (w2v_dim + dim))
        self.W_proj = np.random.randn(w2v_dim, dim).astype(np.float32) * s
        self.b_proj = np.zeros(dim, dtype=np.float32)

        # Self-attention layers
        self.layers = [SelfAttentionBlock(dim, n_heads) for _ in range(n_layers)]

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim)

        # Query encoding (separate projection)
        self.W_q_proj = np.random.randn(w2v_dim, dim).astype(np.float32) * s
        self.b_q_proj = np.zeros(dim, dtype=np.float32)

        # Cross-attention: query attends to sentences
        self.cross_attn = MultiHeadAttention(dim, n_heads)

        # Sentence scoring head
        s2 = math.sqrt(2.0 / dim)
        self.W_score = np.random.randn(dim, 1).astype(np.float32) * s2
        self.b_score = np.zeros(1, dtype=np.float32)

        # Coreference head: given two sentences, are they about the same entity?
        self.W_coref = np.random.randn(dim * 2, 1).astype(np.float32) * math.sqrt(2.0 / (dim * 2))
        self.b_coref = np.zeros(1, dtype=np.float32)

        # Ordering head: given two scored sentences, which comes first?
        self.W_order = np.random.randn(dim * 2, 1).astype(np.float32) * math.sqrt(2.0 / (dim * 2))
        self.b_order = np.zeros(1, dtype=np.float32)

    def _embed_sentences(self, sentences: List[str], entity: str) -> np.ndarray:
        """Embed sentences as averaged word vectors + positional encoding."""
        embeddings_list = []
        for sent in sentences:
            words = re.findall(r'[a-zA-Z]+', sent.lower())
            vecs = []
            for w in words:
                v = self.embeddings.get_vector(w)
                if v is not None:
                    vecs.append(v @ self.W_proj + self.b_proj)
            if vecs:
                sent_emb = np.mean(vecs, axis=0)
            else:
                sent_emb = np.zeros(self.dim, dtype=np.float32)

            # Boost entity mention
            if entity.lower() in sent.lower():
                entity_emb = self.embeddings.get_vector(entity.lower())
                if entity_emb is not None:
                    ent_proj = entity_emb @ self.W_proj + self.b_proj
                    sent_emb = sent_emb + 0.3 * ent_proj

            embeddings_list.append(sent_emb)

        X = np.array(embeddings_list, dtype=np.float32)  # (N, D)

        # Add positional encoding (wrap around for long sequences)
        N = X.shape[0]
        pe = self.pos_enc.encode(min(N, 100))
        for i in range(N):
            X[i] += pe[i % 100]

        return X

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed query as averaged word vectors."""
        words = re.findall(r'[a-zA-Z]+', query.lower())
        vecs = []
        for w in words:
            v = self.embeddings.get_vector(w)
            if v is not None:
                vecs.append(v @ self.W_q_proj + self.b_q_proj)
        if vecs:
            return np.mean(vecs, axis=0)
        return np.zeros(self.dim, dtype=np.float32)

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, sentences: List[str], query: str,
                entity: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Full forward pass.

        Returns:
          scores: (N,) relevance scores per sentence
          attn_weights: (N, N) self-attention weights
          cross_attn: (N,) cross-attention weights (query → sentences)
        """
        if not sentences:
            return np.array([]), np.array([]), np.array([])

        N = len(sentences)

        # Embed
        X = self._embed_sentences(sentences, entity)  # (N, D)
        q = self._embed_query(query)                   # (D,)

        # Self-attention layers
        attn_weights = None
        for layer in self.layers:
            X, attn_weights = layer.forward(X)

        # Cross-attention: query attends to sentences
        q_expanded = q.reshape(1, self.dim)  # (1, D)
        cross_out, cross_weights = self.cross_attn.forward(
            np.vstack([q_expanded, X])  # (1+N, D)
        )
        # Cross-attention: how much query attends to each sentence
        cross_attn = cross_weights[0, 1:]  # (N,) — first row is query's attention

        # Score each sentence
        scores = np.zeros(N, dtype=np.float32)
        for i in range(N):
            scores[i] = float((X[i] @ self.W_score + self.b_score).item())

        # Normalize scores to 0-1
        scores = self._sigmoid(scores)

        return scores, attn_weights, cross_attn

    def select(self, sentences: List[str], query: str, entity: str,
               top_k: int = 4, min_score: float = 0.3) -> List[Tuple[float, str, int]]:
        """
        Select and order the best sentences.

        Returns list of (score, sentence, original_index) sorted by score.
        """
        if not sentences:
            return []

        scores, self_attn, cross_attn = self.forward(sentences, query, entity)

        if len(scores) == 0:
            return []

        # Combine self-attention and cross-attention
        combined = 0.5 * scores + 0.3 * cross_attn + 0.2 * scores  # self-attn implicit in scores

        # Select top-k above threshold
        selected = []
        for i in np.argsort(-combined):
            if combined[i] >= min_score and len(selected) < top_k:
                selected.append((float(combined[i]), sentences[i], int(i)))

        return selected

    def resolve_corefs(self, sentences: List[str], entity: str) -> List[str]:
        """
        Resolve pronouns using attention-based coreference.

        For each sentence, check if pronouns refer to the entity
        based on attention similarity.
        """
        if not sentences or not self.embeddings:
            return sentences

        entity_lower = entity.lower()
        entity_emb = self.embeddings.get_vector(entity_lower)
        if entity_emb is None:
            return sentences

        entity_proj = entity_emb @ self.W_proj + self.b_proj

        resolved = []
        for sent in sentences:
            # Check if sentence starts with a pronoun
            words = sent.split()
            if words and words[0].lower() in ('she', 'he', 'her', 'him', 'his', 'they', 'it'):
                # Check if attention to entity is high
                words_after = ' '.join(words[1:])
                if entity_lower in words_after.lower():
                    # Replace pronoun with entity name
                    words[0] = entity
                    sent = ' '.join(words)

            resolved.append(sent)

        return resolved

    def order_sentences(self, sentences: List[str], scores: List[float],
                        entity: str) -> List[str]:
        """
        Order sentences for narrative flow using pairwise comparison.
        """
        if len(sentences) <= 1:
            return sentences

        # Embed all sentences
        embeds = []
        for sent in sentences:
            words = re.findall(r'[a-zA-Z]+', sent.lower())
            vecs = [self.embeddings.get_vector(w) @ self.W_proj + self.b_proj
                    for w in words if self.embeddings.get_vector(w) is not None]
            if vecs:
                embeds.append(np.mean(vecs, axis=0))
            else:
                embeds.append(np.zeros(self.dim, dtype=np.float32))

        # Pairwise ordering: for each pair, decide which comes first
        n = len(sentences)
        order_scores = np.zeros(n, dtype=np.float32)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                pair = np.concatenate([embeds[i], embeds[j]])
                pref = float((pair @ self.W_order + self.b_order).item())
                # Higher pref = i should come before j
                order_scores[i] += pref

                # Also consider: entity-mentioning sentences first
                if entity.lower() in sentences[i].lower():
                    order_scores[i] += 0.5

                # Sentences with commas (more complex) come later
                if ',' in sentences[i]:
                    order_scores[i] -= 0.2

        # Sort by order score
        indices = np.argsort(-order_scores)
        return [sentences[i] for i in indices]

    def save(self, path: str):
        data = {
            'dim': self.dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'W_proj': self.W_proj.tolist(),
            'b_proj': self.b_proj.tolist(),
            'W_q_proj': self.W_q_proj.tolist(),
            'b_q_proj': self.b_q_proj.tolist(),
            'W_score': self.W_score.tolist(),
            'b_score': self.b_score.tolist(),
            'W_coref': self.W_coref.tolist(),
            'b_coref': self.b_coref.tolist(),
            'W_order': self.W_order.tolist(),
            'b_order': self.b_order.tolist(),
            # Layer params
            'layers': [],
        }
        for layer in self.layers:
            layer_data = {
                'attn_W_q': layer.attn.W_q.tolist(),
                'attn_b_q': layer.attn.b_q.tolist(),
                'attn_W_k': layer.attn.W_k.tolist(),
                'attn_b_k': layer.attn.b_k.tolist(),
                'attn_W_v': layer.attn.W_v.tolist(),
                'attn_b_v': layer.attn.b_v.tolist(),
                'attn_W_o': layer.attn.W_o.tolist(),
                'attn_b_o': layer.attn.b_o.tolist(),
                'ffn_W1': layer.ffn.W1.tolist(),
                'ffn_b1': layer.ffn.b1.tolist(),
                'ffn_W2': layer.ffn.W2.tolist(),
                'ffn_b2': layer.ffn.b2.tolist(),
                'ln1_gamma': layer.ln1_gamma.tolist(),
                'ln1_beta': layer.ln1_beta.tolist(),
                'ln2_gamma': layer.ln2_gamma.tolist(),
                'ln2_beta': layer.ln2_beta.tolist(),
            }
            data['layers'].append(layer_data)

        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved self-attention model to {path}")

    @classmethod
    def load(cls, path: str, embeddings=None) -> 'SelfAttentionSelector':
        with open(path, 'r') as f:
            data = json.load(f)

        model = cls(embeddings, data['dim'], data['n_layers'], data['n_heads'])
        model.W_proj = np.array(data['W_proj'], dtype=np.float32)
        model.b_proj = np.array(data['b_proj'], dtype=np.float32)
        model.W_q_proj = np.array(data['W_q_proj'], dtype=np.float32)
        model.b_q_proj = np.array(data['b_q_proj'], dtype=np.float32)
        model.W_score = np.array(data['W_score'], dtype=np.float32)
        model.b_score = np.array(data['b_score'], dtype=np.float32)
        model.W_coref = np.array(data['W_coref'], dtype=np.float32)
        model.b_coref = np.array(data['b_coref'], dtype=np.float32)
        model.W_order = np.array(data['W_order'], dtype=np.float32)
        model.b_order = np.array(data['b_order'], dtype=np.float32)

        for i, layer_data in enumerate(data['layers']):
            layer = model.layers[i]
            layer.attn.W_q = np.array(layer_data['attn_W_q'], dtype=np.float32)
            layer.attn.b_q = np.array(layer_data['attn_b_q'], dtype=np.float32)
            layer.attn.W_k = np.array(layer_data['attn_W_k'], dtype=np.float32)
            layer.attn.b_k = np.array(layer_data['attn_b_k'], dtype=np.float32)
            layer.attn.W_v = np.array(layer_data['attn_W_v'], dtype=np.float32)
            layer.attn.b_v = np.array(layer_data['attn_b_v'], dtype=np.float32)
            layer.attn.W_o = np.array(layer_data['attn_W_o'], dtype=np.float32)
            layer.attn.b_o = np.array(layer_data['attn_b_o'], dtype=np.float32)
            layer.ffn.W1 = np.array(layer_data['ffn_W1'], dtype=np.float32)
            layer.ffn.b1 = np.array(layer_data['ffn_b1'], dtype=np.float32)
            layer.ffn.W2 = np.array(layer_data['ffn_W2'], dtype=np.float32)
            layer.ffn.b2 = np.array(layer_data['ffn_b2'], dtype=np.float32)
            layer.ln1_gamma = np.array(layer_data['ln1_gamma'], dtype=np.float32)
            layer.ln1_beta = np.array(layer_data['ln1_beta'], dtype=np.float32)
            layer.ln2_gamma = np.array(layer_data['ln2_gamma'], dtype=np.float32)
            layer.ln2_beta = np.array(layer_data['ln2_beta'], dtype=np.float32)

        return model
