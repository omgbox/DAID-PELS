"""
Token-Level Self-Attention (PyTorch)
Real gradients, real training, real learning.

Architecture:
  Tokenize → Embed → Position Encoding → N Attention Layers → Pool → Score

Training:
  Contrastive: positive pairs score higher than negatives
  Coreference: pronoun-entity pairs score higher
  Ordering: adjacent sentences score higher than random
"""

import math
import random
import logging
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.query.torch_attention')


class MultiHeadAttention(nn.Module):
    """Multi-head scaled dot-product attention with residual + layer norm."""

    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, X, mask=None):
        """
        X: (T, D)
        Returns: (T, D) output, (T, T) attention weights
        """
        T = X.shape[0]

        Q = self.W_q(X)  # (T, D)
        K = self.W_k(X)
        V = self.W_v(X)

        # Reshape: (T, n_heads, head_dim)
        Q = Q.reshape(T, self.n_heads, self.head_dim)
        K = K.reshape(T, self.n_heads, self.head_dim)
        V = V.reshape(T, self.n_heads, self.head_dim)

        # Attention scores: (n_heads, T, T)
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale

        if mask is not None:
            scores = scores + mask

        attn = F.softmax(scores, dim=-1)

        # Apply attention to values
        out = torch.bmm(attn, V)  # (n_heads, T, head_dim)
        out = out.reshape(T, self.dim)
        out = self.W_o(out)

        return out, attn.mean(dim=0)  # average across heads


class TransformerBlock(nn.Module):
    """Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, dim: int, n_heads: int = 4, ffn_dim: int = None):
        super().__init__()
        ffn_dim = ffn_dim or dim * 4

        self.attn = MultiHeadAttention(dim, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, X, mask=None):
        # Self-attention with residual
        normed = self.ln1(X)
        attn_out, attn_weights = self.attn(normed, mask)
        X = X + attn_out

        # FFN with residual
        X = X + self.ffn(self.ln2(X))

        return X, attn_weights


class TorchTokenAttention(nn.Module):
    """
    Token-level self-attention for sentence selection.

    Each word gets its own embedding. Attention operates across all
    words from all sentences, enabling cross-sentence relationships.
    """

    def __init__(self, w2v_dim: int = 50, dim: int = 64,
                 n_layers: int = 2, n_heads: int = 4,
                 max_tokens: int = 2048):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.max_tokens = max_tokens

        # Token embedding projection
        self.W_tok = nn.Linear(w2v_dim, dim, bias=True)

        # Sentence boundary embedding
        self.sent_embed = nn.Parameter(torch.randn(1, dim) * 0.02)

        # Position encoding (fixed, not learned)
        pe = torch.zeros(256, dim)
        position = torch.arange(0, 256).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads) for _ in range(n_layers)
        ])

        # Sentence pooling: attention-weighted
        self.W_pool = nn.Linear(dim, 1, bias=False)

        # Query encoding
        self.W_query = nn.Linear(w2v_dim, dim, bias=True)

        # Scoring
        self.W_score = nn.Linear(dim, 1, bias=True)

        # Cross-attention for query→sentence
        self.W_q_cross = nn.Linear(dim, dim, bias=False)
        self.W_k_cross = nn.Linear(dim, dim, bias=False)
        self.W_v_cross = nn.Linear(dim, dim, bias=False)

    def embed_tokens(self, all_tokens, boundaries, w2v_model):
        """Embed all tokens using word2vec."""
        flat_tokens = [w for sent in all_tokens for w in sent]
        T = len(flat_tokens)
        if T == 0:
            return torch.zeros(0, self.dim)

        # Truncate
        if T > self.max_tokens:
            flat_tokens = flat_tokens[:self.max_tokens]
            T = self.max_tokens

        # Get word2vec vectors
        vecs = []
        for word in flat_tokens:
            v = w2v_model.get_vector(word)
            if v is not None:
                vecs.append(v)
            else:
                vecs.append(np.zeros(w2v_model.model.dim, dtype=np.float32))

        X = torch.tensor(np.array(vecs), dtype=torch.float32)

        # Project to model dim
        X = self.W_tok(X)

        # Add position encoding
        pos = 0
        sent_idx = 0
        for i in range(T):
            if sent_idx < len(boundaries) - 1 and i == boundaries[sent_idx + 1]:
                sent_idx += 1
                pos = 0
            X[i] += self.pe[pos % 256]
            pos += 1

        # Add sentence boundary markers
        for b in boundaries:
            if b < T:
                X[b] = X[b] + self.sent_embed.squeeze()

        return X

    def embed_query(self, query, w2v_model):
        """Embed query tokens."""
        words = re.findall(r'[a-zA-Z]+', query.lower())
        vecs = []
        for w in words:
            v = w2v_model.get_vector(w)
            if v is not None:
                vecs.append(v)
        if vecs:
            q = torch.tensor(np.mean(vecs, axis=0), dtype=torch.float32)
            return self.W_query(q)
        return torch.zeros(self.dim)

    def forward(self, X, boundaries, query_vec=None):
        """
        X: (T, D) token embeddings
        boundaries: sentence boundary indices
        query_vec: (D,) query embedding

        Returns: per-sentence scores, token attention weights
        """
        if X.shape[0] == 0:
            return torch.tensor([]), torch.zeros(0, 0)

        # Apply transformer layers
        token_attn = None
        for layer in self.layers:
            X, token_attn = layer(X)

        # Pool to sentences
        n_sents = len(boundaries)
        sent_vecs = []
        for i in range(n_sents):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else X.shape[0]
            if start >= X.shape[0]:
                sent_vecs.append(torch.zeros(self.dim))
                continue
            sent_tokens = X[start:end]
            if len(sent_tokens) == 0:
                sent_vecs.append(torch.zeros(self.dim))
                continue

            # Attention-weighted pooling
            weights = self.W_pool(sent_tokens).squeeze(-1)  # (n_tokens,)
            weights = F.softmax(weights, dim=0)
            pooled = (weights.unsqueeze(-1) * sent_tokens).sum(dim=0)
            sent_vecs.append(pooled)

        sent_vecs = torch.stack(sent_vecs)  # (N, D)

        # Cross-attention: query → sentences
        if query_vec is not None:
            Q = self.W_q_cross(query_vec.unsqueeze(0))  # (1, D)
            K = self.W_k_cross(sent_vecs)  # (N, D)
            V = self.W_v_cross(sent_vecs)

            scores = (Q @ K.T).squeeze(0) / math.sqrt(self.dim)  # (N,)
            cross_weights = F.softmax(scores, dim=0)
        else:
            cross_weights = torch.ones(n_sents) / n_sents

        # Score each sentence
        self_scores = self.W_score(sent_vecs).squeeze(-1)  # (N,)

        # Combine self-attention and cross-attention
        combined = 0.6 * self_scores + 0.4 * cross_weights

        return combined, token_attn

    def select(self, sentences, query, w2v_model, entity='',
               top_k=4, min_score=0.3):
        """Select best sentences."""
        # Tokenize
        all_tokens = []
        boundaries = []
        for sent in sentences:
            boundaries.append(len(all_tokens))
            words = re.findall(r'[a-zA-Z]+', sent.lower())
            all_tokens.append(words)

        # Embed
        X = self.embed_tokens(all_tokens, boundaries, w2v_model)
        q_vec = self.embed_query(query, w2v_model)

        # Forward
        scores, _ = self.forward(X, boundaries, q_vec)

        # Select
        selected = []
        probs = torch.sigmoid(scores).detach().numpy()
        for i in np.argsort(-probs):
            if probs[i] >= min_score and len(selected) < top_k:
                selected.append((float(probs[i]), sentences[i], int(i)))

        return selected

    def save(self, path):
        torch.save(self.state_dict(), path)
        logger.info(f"Saved PyTorch model to {path}")

    @classmethod
    def load(cls, path, w2v_dim=50, dim=64, n_layers=2, n_heads=4):
        model = cls(w2v_dim, dim, n_layers, n_heads)
        model.load_state_dict(torch.load(path, weights_only=False))
        model.eval()
        return model


class AttentionTrainer:
    """
    Trains token-level attention with PyTorch autograd.
    """

    def __init__(self, model, lr: float = 0.001):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.losses = []

    def train(self, training_data: Dict, w2v_model,
              epochs: int = 10, batch_size: int = 16):
        """Train on all tasks."""
        logger.info("Starting PyTorch training...")

        query_pairs = training_data.get('query_pairs', [])
        coref_pairs = training_data.get('coref_pairs', [])
        order_pairs = training_data.get('order_pairs', [])

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            # Train query selection
            if query_pairs:
                random.shuffle(query_pairs)
                for i in range(0, min(len(query_pairs), 300), batch_size):
                    batch = query_pairs[i:i + batch_size]
                    loss = self._train_query_batch(batch, w2v_model)
                    if loss is not None:
                        epoch_loss += loss
                        n_batches += 1

            # Train coreference
            if coref_pairs:
                random.shuffle(coref_pairs)
                for i in range(0, min(len(coref_pairs), 200), batch_size):
                    batch = coref_pairs[i:i + batch_size]
                    loss = self._train_coref_batch(batch, w2v_model)
                    if loss is not None:
                        epoch_loss += loss
                        n_batches += 1

            # Train ordering
            if order_pairs:
                random.shuffle(order_pairs)
                for i in range(0, min(len(order_pairs), 200), batch_size):
                    batch = order_pairs[i:i + batch_size]
                    loss = self._train_order_batch(batch, w2v_model)
                    if loss is not None:
                        epoch_loss += loss
                        n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.losses.append(avg_loss)
            logger.info(f"  Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}")

    def _train_query_batch(self, batch, w2v_model):
        """Train query selection on a batch."""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n = 0

        for pair in batch:
            pos_text = pair['positive']
            neg_text = pair['negative']
            query = pair['query']

            if not pos_text or not neg_text:
                continue

            # Score positive
            pos_score = self._score_sentence(pos_text, query, w2v_model)
            # Score negative
            neg_score = self._score_sentence(neg_text, query, w2v_model)

            # Hinge loss
            loss = F.relu(0.3 - pos_score + neg_score)
            total_loss = total_loss + loss
            n += 1

        if n == 0:
            return None

        loss = total_loss / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _train_coref_batch(self, batch, w2v_model):
        """Train coreference on a batch."""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n = 0

        for pair in batch:
            context = pair['context']
            entity = pair['entity']

            score = self._score_sentence(context, entity, w2v_model)
            loss = F.relu(0.3 - score + 0.2)  # want score > 0.2
            total_loss = total_loss + loss
            n += 1

        if n == 0:
            return None

        loss = total_loss / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _train_order_batch(self, batch, w2v_model):
        """Train ordering on a batch."""
        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n = 0

        for pair in batch:
            s1, s2 = pair['sentence1'], pair['sentence2']
            label = pair['label']

            score1 = self._score_sentence(s1, '', w2v_model)
            score2 = self._score_sentence(s2, '', w2v_model)

            # Want score1 > score2 if label=1
            target = 2.0 * label - 1.0  # {0,1} → {-1,1}
            pred = score1 - score2
            loss = F.relu(0.3 - target * pred)
            total_loss = total_loss + loss
            n += 1

        if n == 0:
            return None

        loss = total_loss / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def _score_sentence(self, text, query, w2v_model):
        """Score a single sentence using full attention. Returns scalar tensor."""
        self.model.eval()

        # Tokenize
        words = re.findall(r'[a-zA-Z]+', text.lower())
        if not words:
            return torch.tensor(0.5, requires_grad=True)

        # Get word vectors
        vecs = []
        for w in words:
            v = w2v_model.get_vector(w)
            if v is not None:
                vecs.append(v)
            else:
                vecs.append(np.zeros(w2v_model.model.dim, dtype=np.float32))

        X = torch.tensor(np.array(vecs), dtype=torch.float32)
        X = self.model.W_tok(X)

        # Add position encoding
        for i in range(min(len(X), 256)):
            X[i] = X[i] + self.model.pe[i]

        # Apply transformer layers (with gradients)
        self.model.train()
        for layer in self.model.layers:
            X, _ = layer(X)

        # Pool
        pooled = X.mean(dim=0)  # (D,)

        # Score
        score = self.model.W_score(pooled).squeeze()
        return score

    def save(self, model_path, head_path=None):
        self.model.save(model_path)
        # Save optimizer state
        if head_path:
            torch.save({
                'optimizer': self.optimizer.state_dict(),
                'losses': self.losses,
            }, head_path)

    def load(self, model_path, head_path=None):
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        if head_path:
            try:
                data = torch.load(head_path, weights_only=False)
                self.optimizer.load_state_dict(data['optimizer'])
                self.losses = data.get('losses', [])
            except Exception:
                pass
