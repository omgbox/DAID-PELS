"""
Token-Level Self-Attention
Each word gets its own embedding. Attention operates at word level
across ALL sentences, enabling cross-sentence word relationships.

This is how LLMs work — not averaging words into sentence blobs,
but keeping every word as its own vector and letting attention
learn which words relate to which.

Architecture:
  1. Tokenize: each word → index
  2. Embed: word2vec → projected to model dim
  3. Add positional encoding (within-sentence position)
  4. Multi-head self-attention over ALL tokens
  5. Sentence pooling: aggregate token vectors back to sentence
  6. Score sentences based on pooled representations

Key insight: "She" in sentence 2 can attend to "Elizabeth" in sentence 1
because they're all in the same attention matrix.
"""

import math
import random
import logging
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.query.token_attention')


class TokenAttention:
    """
    Token-level multi-head self-attention.

    Input: (total_tokens, dim)
    Each token attends to every other token across all sentences.
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

    def forward(self, X: np.ndarray,
                sentence_mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        X: (T, D) — T tokens, D dimensions
        sentence_mask: (T, T) — -inf where tokens are from different sentences
                         (optional: if None, all tokens attend to all)

        Returns: (T, D) output, (T, T) attention weights
        """
        T = X.shape[0]

        Q = X @ self.W_q  # (T, D)
        K = X @ self.W_k
        V = X @ self.W_v

        # Reshape for multi-head: (T, n_heads, head_dim)
        Q = Q.reshape(T, self.n_heads, self.head_dim)
        K = K.reshape(T, self.n_heads, self.head_dim)
        V = V.reshape(T, self.n_heads, self.head_dim)

        # Compute attention scores
        scores = np.zeros((self.n_heads, T, T), dtype=np.float32)
        for h in range(self.n_heads):
            s = Q[:, h] @ K[:, h].T / math.sqrt(self.head_dim)
            if sentence_mask is not None:
                s = s + sentence_mask
            scores[h] = s

        # Softmax
        attn = np.zeros_like(scores)
        for h in range(self.n_heads):
            e = np.exp(scores[h] - scores[h].max(axis=-1, keepdims=True))
            attn[h] = e / (e.sum(axis=-1, keepdims=True) + 1e-10)

        # Apply attention to values
        out = np.zeros((T, self.n_heads, self.head_dim), dtype=np.float32)
        for h in range(self.n_heads):
            out[:, h] = attn[h] @ V[:, h]

        out = out.reshape(T, self.dim)
        out = out @ self.W_o

        avg_attn = attn.mean(axis=0)
        return out, avg_attn


class TokenLevelAttention:
    """
    Full token-level attention system.

    Flow:
      1. Tokenize all sentences
      2. Embed each token with word2vec
      3. Add position encoding
      4. Apply N layers of token-level self-attention
      5. Pool token vectors back to sentence vectors
      6. Score and select sentences
    """

    def __init__(self, embeddings=None, dim: int = 64,
                 n_layers: int = 2, n_heads: int = 4,
                 max_tokens: int = 2048):
        self.embeddings = embeddings
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.max_tokens = max_tokens

        w2v_dim = embeddings.model.dim if embeddings else 50
        s = math.sqrt(2.0 / (w2v_dim + dim))

        # Token embedding projection
        self.W_tok = np.random.randn(w2v_dim, dim).astype(np.float32) * s
        self.b_tok = np.zeros(dim, dtype=np.float32)

        # Sentence boundary embedding (marks where sentences start/end)
        self.W_sent = np.random.randn(1, dim).astype(np.float32) * s

        # Position encoding (per-token, within-sentence)
        pe = np.zeros((256, dim), dtype=np.float32)
        position = np.arange(0, 256)[:, np.newaxis]
        div_term = np.exp(np.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        self.pos_enc = pe

        # Attention layers
        self.layers = [TokenAttention(dim, n_heads) for _ in range(n_layers)]

        # Layer norms
        self.ln_gammas = [np.ones(dim, dtype=np.float32) for _ in range(n_layers)]
        self.ln_betas = [np.zeros(dim, dtype=np.float32) for _ in range(n_layers)]

        # Sentence pooling: attention-weighted pooling
        self.W_pool = np.random.randn(dim, 1).astype(np.float32) * math.sqrt(2.0 / dim)

        # Query encoding
        self.W_query = np.random.randn(w2v_dim, dim).astype(np.float32) * s
        self.b_query = np.zeros(dim, dtype=np.float32)

        # Cross-attention: query tokens attend to sentence tokens
        self.W_q_cross = np.random.randn(dim, dim).astype(np.float32) * s
        self.W_k_cross = np.random.randn(dim, dim).astype(np.float32) * s
        self.W_v_cross = np.random.randn(dim, dim).astype(np.float32) * s

        # Scoring
        self.W_score = np.random.randn(dim, 1).astype(np.float32) * math.sqrt(2.0 / dim)
        self.b_score = np.zeros(1, dtype=np.float32)

    def _layer_norm(self, X, gamma, beta):
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True) + 1e-10
        return gamma * (X - mean) / std + beta

    def tokenize(self, sentences: List[str]) -> Tuple[List[List[str]], List[int]]:
        """Tokenize sentences, return word lists and sentence boundaries."""
        all_tokens = []
        boundaries = []  # index where each sentence starts

        for sent in sentences:
            boundaries.append(len(all_tokens))
            words = re.findall(r'[a-zA-Z]+', sent.lower())
            all_tokens.append(words)

        return all_tokens, boundaries

    def embed_tokens(self, all_tokens: List[List[str]],
                     boundaries: List[int]) -> np.ndarray:
        """Embed all tokens with word2vec + position encoding."""
        # Flatten tokens
        flat_tokens = [w for sent_tokens in all_tokens for w in sent_tokens]
        T = len(flat_tokens)

        if T == 0:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Truncate if too long
        if T > self.max_tokens:
            flat_tokens = flat_tokens[:self.max_tokens]
            T = self.max_tokens

        # Embed each token
        X = np.zeros((T, self.dim), dtype=np.float32)
        for i, word in enumerate(flat_tokens):
            v = self.embeddings.get_vector(word)
            if v is not None:
                X[i] = v @ self.W_tok + self.b_tok

        # Add position encoding (within-sentence position)
        sent_idx = 0
        pos_in_sent = 0
        for i in range(T):
            if sent_idx < len(boundaries) - 1 and i == boundaries[sent_idx + 1]:
                sent_idx += 1
                pos_in_sent = 0
            X[i] += self.pos_enc[pos_in_sent % 256]
            pos_in_sent += 1

        # Add sentence boundary markers
        for b in boundaries:
            if b < T:
                X[b] += self.W_sent.flatten()

        return X

    def build_sentence_mask(self, boundaries: List[int], T: int) -> np.ndarray:
        """
        Build mask that allows cross-sentence attention.
        Returns (T, T) mask: 0 = can attend, -inf = cannot attend.

        For token-level attention, we allow ALL tokens to attend to all others
        (including cross-sentence). This is the key difference from sentence-level.
        """
        # Allow all tokens to attend to all others (full attention)
        # No masking — this enables cross-sentence relationships
        return np.zeros((T, T), dtype=np.float32)

    def pool_to_sentences(self, X: np.ndarray,
                          all_tokens: List[List[str]],
                          boundaries: List[int]) -> np.ndarray:
        """
        Pool token vectors back to sentence vectors.
        Uses attention-weighted pooling: each sentence's vector is a
        weighted average of its token vectors.
        """
        n_sents = len(boundaries)
        S = np.zeros((n_sents, self.dim), dtype=np.float32)

        for i in range(n_sents):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else X.shape[0]

            if start >= X.shape[0]:
                continue

            sent_tokens = X[start:end]
            if len(sent_tokens) == 0:
                continue

            # Attention-weighted pooling
            weights = sent_tokens @ self.W_pool  # (n_tokens, 1)
            weights = np.exp(weights - weights.max())
            weights = weights / (weights.sum() + 1e-10)

            S[i] = (weights * sent_tokens).sum(axis=0)

        return S

    def cross_attend_query(self, query_vec: np.ndarray,
                           sent_vecs: np.ndarray) -> np.ndarray:
        """
        Query tokens attend to sentence tokens.
        Returns per-sentence attention weights.
        """
        T = sent_vecs.shape[0]
        if T == 0:
            return np.array([])

        # Query as single token
        q = query_vec.reshape(1, self.dim)

        # Compute attention
        Q = q @ self.W_q_cross   # (1, D)
        K = sent_vecs @ self.W_k_cross  # (T, D)
        V = sent_vecs @ self.W_v_cross  # (T, D)

        scores = (Q @ K.T).flatten() / math.sqrt(self.dim)  # (T,)
        weights = np.exp(scores - scores.max())
        weights = weights / (weights.sum() + 1e-10)

        return weights

    def forward(self, sentences: List[str], query: str,
                entity: str) -> Tuple[np.ndarray, np.ndarray, List[List[str]], List[int]]:
        """
        Full forward pass at token level.

        Returns:
          sent_scores: (N,) per-sentence scores
          token_attn: (T, T) token-level attention weights
          all_tokens: tokenized sentences
          boundaries: sentence boundaries
        """
        if not sentences:
            return np.array([]), np.array([]), [], []

        # Tokenize
        all_tokens, boundaries = self.tokenize(sentences)

        # Embed tokens
        X = self.embed_tokens(all_tokens, boundaries)
        T = X.shape[0]

        if T == 0:
            return np.zeros(len(sentences)), np.zeros((0, 0)), all_tokens, boundaries

        # Build mask (full attention — no masking)
        mask = self.build_sentence_mask(boundaries, T)

        # Apply attention layers
        token_attn = None
        for i, layer in enumerate(self.layers):
            normed = self._layer_norm(X, self.ln_gammas[i], self.ln_betas[i])
            attn_out, token_attn = layer.forward(normed, mask)
            X = X + attn_out  # residual

        # Pool back to sentence vectors
        sent_vecs = self.pool_to_sentences(X, all_tokens, boundaries)

        # Embed query
        q_words = re.findall(r'[a-zA-Z]+', query.lower())
        q_vecs = []
        for w in q_words:
            v = self.embeddings.get_vector(w)
            if v is not None:
                q_vecs.append(v @ self.W_query + self.b_query)
        if q_vecs:
            q_vec = np.mean(q_vecs, axis=0)
        else:
            q_vec = np.zeros(self.dim, dtype=np.float32)

        # Cross-attend query to sentences
        cross_weights = self.cross_attend_query(q_vec, sent_vecs)

        # Score each sentence
        scores = np.zeros(len(sentences), dtype=np.float32)
        for i in range(len(sentences)):
            # Combine: self-attention pooled + cross-attention
            self_score = float((sent_vecs[i] @ self.W_score + self.b_score).item())
            cross_score = cross_weights[i] if i < len(cross_weights) else 0
            scores[i] = 0.6 * self_score + 0.4 * cross_score

        # Normalize to 0-1
        scores = 1.0 / (1.0 + np.exp(-scores))

        return scores, token_attn, all_tokens, boundaries

    def select(self, sentences: List[str], query: str, entity: str,
               top_k: int = 4, min_score: float = 0.3) -> List[Tuple[float, str, int]]:
        """Select best sentences using token-level attention."""
        if not sentences:
            return []

        scores, token_attn, all_tokens, boundaries = self.forward(
            sentences, query, entity
        )

        if len(scores) == 0:
            return []

        selected = []
        for i in np.argsort(-scores):
            if scores[i] >= min_score and len(selected) < top_k:
                selected.append((float(scores[i]), sentences[i], int(i)))

        return selected

    def resolve_corefs(self, sentences: List[str],
                       token_attn: np.ndarray,
                       all_tokens: List[List[str]],
                       boundaries: List[int],
                       entity: str) -> List[str]:
        """
        Resolve coreferences using token-level attention.

        If "she"/"he"/"they" attends strongly to the entity name
        in another sentence, replace the pronoun.
        """
        if token_attn is None or len(token_attn) == 0:
            return sentences

        entity_lower = entity.lower()
        PRONOUNS = {'she': 'f', 'he': 'm', 'her': 'f', 'him': 'm',
                     'his': 'm', 'hers': 'f', 'they': 'x', 'them': 'x'}

        resolved = []
        for sent_idx, sent_tokens in enumerate(all_tokens):
            start = boundaries[sent_idx]
            end = boundaries[sent_idx + 1] if sent_idx + 1 < len(boundaries) else len(all_tokens)

            new_tokens = list(sent_tokens)
            for local_idx, word in enumerate(sent_tokens):
                if word.lower() in PRONOUNS:
                    global_idx = start + local_idx
                    if global_idx >= token_attn.shape[0]:
                        continue

                    # Find which entity token this pronoun attends to most
                    attn_row = token_attn[global_idx]
                    best_target = -1
                    best_attn = 0.0

                    # Check attention to entity name tokens in other sentences
                    for other_idx in range(len(all_tokens)):
                        if other_idx == sent_idx:
                            continue
                        other_start = boundaries[other_idx]
                        other_end = boundaries[other_idx + 1] if other_idx + 1 < len(boundaries) else len(all_tokens)

                        for t_idx in range(other_start, min(other_end, len(attn_row))):
                            if t_idx < len(all_tokens_flat(all_tokens)):
                                target_word = _get_word(all_tokens, t_idx)
                                if target_word and entity_lower in target_word.lower():
                                    if attn_row[t_idx] > best_attn:
                                        best_attn = attn_row[t_idx]
                                        best_target = t_idx

                    # If strong attention to entity, replace pronoun
                    if best_attn > 0.1 and best_target >= 0:
                        new_tokens[local_idx] = entity

            resolved.append(' '.join(new_tokens))

        return resolved

    def save(self, path: str):
        data = {
            'dim': self.dim,
            'n_layers': self.n_layers,
            'n_heads': self.n_heads,
            'max_tokens': self.max_tokens,
            'W_tok': self.W_tok.tolist(),
            'b_tok': self.b_tok.tolist(),
            'W_sent': self.W_sent.tolist(),
            'W_query': self.W_query.tolist(),
            'b_query': self.b_query.tolist(),
            'W_q_cross': self.W_q_cross.tolist(),
            'W_k_cross': self.W_k_cross.tolist(),
            'W_v_cross': self.W_v_cross.tolist(),
            'W_pool': self.W_pool.tolist(),
            'W_score': self.W_score.tolist(),
            'b_score': self.b_score.tolist(),
            'ln_gammas': [g.tolist() for g in self.ln_gammas],
            'ln_betas': [b.tolist() for b in self.ln_betas],
            'layers': [],
        }
        for layer in self.layers:
            data['layers'].append({
                'W_q': layer.W_q.tolist(),
                'W_k': layer.W_k.tolist(),
                'W_v': layer.W_v.tolist(),
                'W_o': layer.W_o.tolist(),
            })
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved token-level attention model to {path}")

    @classmethod
    def load(cls, path: str, embeddings=None) -> 'TokenLevelAttention':
        with open(path, 'r') as f:
            data = json.load(f)
        model = cls(embeddings, data['dim'], data['n_layers'],
                    data['n_heads'], data['max_tokens'])
        model.W_tok = np.array(data['W_tok'], dtype=np.float32)
        model.b_tok = np.array(data['b_tok'], dtype=np.float32)
        model.W_sent = np.array(data['W_sent'], dtype=np.float32)
        model.W_query = np.array(data['W_query'], dtype=np.float32)
        model.b_query = np.array(data['b_query'], dtype=np.float32)
        model.W_q_cross = np.array(data['W_q_cross'], dtype=np.float32)
        model.W_k_cross = np.array(data['W_k_cross'], dtype=np.float32)
        model.W_v_cross = np.array(data['W_v_cross'], dtype=np.float32)
        model.W_pool = np.array(data['W_pool'], dtype=np.float32)
        model.W_score = np.array(data['W_score'], dtype=np.float32)
        model.b_score = np.array(data['b_score'], dtype=np.float32)
        model.ln_gammas = [np.array(g, dtype=np.float32) for g in data['ln_gammas']]
        model.ln_betas = [np.array(b, dtype=np.float32) for b in data['ln_betas']]
        for i, layer_data in enumerate(data['layers']):
            model.layers[i].W_q = np.array(layer_data['W_q'], dtype=np.float32)
            model.layers[i].W_k = np.array(layer_data['W_k'], dtype=np.float32)
            model.layers[i].W_v = np.array(layer_data['W_v'], dtype=np.float32)
            model.layers[i].W_o = np.array(layer_data['W_o'], dtype=np.float32)
        return model


def _get_word(all_tokens, global_idx):
    """Get word at global index."""
    count = 0
    for sent_tokens in all_tokens:
        for w in sent_tokens:
            if count == global_idx:
                return w
            count += 1
    return None
