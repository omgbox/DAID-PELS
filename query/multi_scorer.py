"""
Multi-Dimensional Sentence Scorer
Scores sentences on 5 axes using separate neural heads.

Architecture:
  Input: sentence features (20-dim) + entity features (10-dim) + query features (8-dim)
  Shared encoder: 2-layer MLP → 64-dim latent
  5 scoring heads (each 1-layer):
    1. Relevance:     does this sentence answer the query?
    2. Fluency:       is this grammatical and well-formed?
    3. Informativeness: does this convey useful facts?
    4. Style:         is this well-written and readable?
    5. Coherence:     does this fit with other selected sentences?

  Final score: weighted combination (learnable weights)

Calibration: Platt scaling per head for well-calibrated 0-1 outputs.
"""

import math
import random
import logging
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.query.multi_scorer')


class MultiDimScorer:
    """
    Multi-dimensional sentence scorer with 5 neural heads.

    Each head learns a different quality dimension.
    Scores are calibrated via Platt scaling (sigmoid on logit).
    """

    def __init__(self, feature_dim: int = 38, latent_dim: int = 64):
        self.feature_dim = feature_dim    # 20 sentence + 10 entity + 8 query
        self.latent_dim = latent_dim
        self.n_heads = 5
        self.head_names = ['relevance', 'fluency', 'informativeness', 'style', 'coherence']

        # Shared encoder
        s = math.sqrt(2.0 / (feature_dim + latent_dim))
        self.W_enc = np.random.randn(feature_dim, latent_dim).astype(np.float32) * s
        self.b_enc = np.zeros(latent_dim, dtype=np.float32)

        # 5 scoring heads
        self.W_heads = []
        self.b_heads = []
        for _ in range(self.n_heads):
            s2 = math.sqrt(2.0 / latent_dim)
            self.W_heads.append(np.random.randn(latent_dim, 1).astype(np.float32) * s2)
            self.b_heads.append(np.zeros(1, dtype=np.float32))

        # Platt scaling per head (a, b params for sigmoid(a*x + b))
        self.platt_a = np.ones(self.n_heads, dtype=np.float32)
        self.platt_b = np.zeros(self.n_heads, dtype=np.float32)

        # Learnable combination weights
        self.combo_weights = np.ones(self.n_heads, dtype=np.float32) / self.n_heads

        # Cache for sentence features
        self._cache = {}

    def _relu(self, x):
        return np.maximum(0, x)

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def _extract_sentence_features(self, sentence: str, entity: str = '') -> np.ndarray:
        """Extract 20-dimensional feature vector from a sentence."""
        f = np.zeros(20, dtype=np.float32)
        words = sentence.split()
        wc = len(words)
        lower = sentence.lower()

        # Length features
        f[0] = min(1.0, wc / 30.0)                    # word count (normalized)
        f[1] = min(1.0, len(sentence) / 200.0)         # char count
        f[2] = 1.0 if 8 <= wc <= 25 else 0.0           # medium length
        f[3] = 1.0 if wc < 5 else 0.0                  # too short
        f[4] = 1.0 if wc > 30 else 0.0                 # too long

        # Entity features
        entity_lower = entity.lower()
        f[5] = 1.0 if lower.startswith(entity_lower) else 0.0  # entity as subject
        f[6] = lower.count(entity_lower) / max(wc, 1)          # entity density

        # Structural features
        f[7] = sentence.count(',') / max(wc, 1)        # comma density (complexity)
        f[8] = sentence.count(';') / max(wc, 1)        # semicolons
        f[9] = sentence.count(':')                      # colons (dialogue)
        f[10] = sentence.count('"') / 2.0              # dialogue presence

        # POS-like features (simple heuristics)
        f[11] = sum(1 for w in words if w[0].isupper()) / max(wc, 1)  # capital words
        f[12] = sum(1 for w in words if w.endswith(('ed', 'ing', 'ly'))) / max(wc, 1)  # verb/adv forms
        f[13] = sum(1 for w in words if w.endswith(('tion', 'ment', 'ness', 'ity'))) / max(wc, 1)  # noun forms

        # Descriptive features
        DESC_WORDS = {'very', 'quite', 'rather', 'extremely', 'absolutely',
                      'sensible', 'intelligent', 'beautiful', 'pleasing',
                      'delighted', 'pleased', 'anxious', 'gentle', 'firm',
                      'quietly', 'softly', 'calmly', 'eagerly', 'warmly'}
        f[14] = min(1.0, sum(1 for w in words if w.lower() in DESC_WORDS) / 3.0)

        # Action features
        ACTION_WORDS = {'felt', 'looked', 'turned', 'smiled', 'spoke',
                        'replied', 'said', 'told', 'gave', 'took', 'found',
                        'knew', 'thought', 'saw', 'walked', 'stood', 'sat',
                        'began', 'continued', 'seemed', 'noticed', 'reached'}
        f[15] = min(1.0, sum(1 for w in words if w.lower() in ACTION_WORDS) / 3.0)

        # Punctuation quality
        f[16] = 1.0 if sentence.rstrip().endswith(('.', '!', '?')) else 0.0
        f[17] = 1.0 if (sentence and sentence[0].isupper()) else 0.0

        # Readability (simple proxy)
        avg_word_len = np.mean([len(w) for w in words]) if words else 0
        f[18] = min(1.0, avg_word_len / 8.0)
        f[19] = 1.0 if re.search(r'[.!?]$', sentence.rstrip()) else 0.0

        return f

    def _extract_entity_features(self, entity: str, entity_info: Dict = None) -> np.ndarray:
        """Extract 10-dimensional entity context features."""
        f = np.zeros(10, dtype=np.float32)
        info = entity_info or {}

        f[0] = min(1.0, len(entity) / 15.0)                    # name length
        f[1] = 1.0 if entity[0].isupper() else 0.0             # proper noun
        f[2] = min(1.0, info.get('frequency', 0) / 100.0)      # frequency
        f[3] = min(1.0, info.get('n_svo', 0) / 50.0)           # SVO count
        f[4] = min(1.0, info.get('n_related', 0) / 10.0)       # related entities
        f[5] = info.get('has_definition', 0)                    # has definition
        f[6] = info.get('has_role', 0)                          # has role
        f[7] = min(1.0, info.get('avg_svo_len', 0) / 10.0)     # avg SVO length
        f[8] = 1.0 if entity.endswith(('a', 'e', 'i', 'o', 'u')) else 0.0  # ends vowel
        f[9] = min(1.0, info.get('centrality', 0))              # graph centrality

        return f

    def _extract_query_features(self, query: str, intent: str = '') -> np.ndarray:
        """Extract 8-dimensional query features."""
        f = np.zeros(8, dtype=np.float32)
        words = query.split()
        lower = query.lower()

        # Query type
        f[0] = 1.0 if lower.startswith(('who', 'whom')) else 0.0   # person query
        f[1] = 1.0 if lower.startswith('what') else 0.0            # thing query
        f[2] = 1.0 if lower.startswith(('why', 'how')) else 0.0    # causal query
        f[3] = 1.0 if lower.startswith('when') else 0.0            # temporal query

        # Intent encoding
        INTENTS = {'DEFINITIONAL': 0, 'FACTUAL': 1, 'CAUSAL': 2,
                   'TEMPORAL': 3, 'COMPARATIVE': 4, 'SUMMARIZATION': 5}
        intent_idx = INTENTS.get(intent, 6)
        f[4] = intent_idx / 6.0

        # Query complexity
        f[5] = min(1.0, len(words) / 10.0)
        f[6] = 1.0 if '?' in query else 0.0
        f[7] = min(1.0, len(query) / 50.0)

        return f

    def extract_features(self, sentence: str, entity: str,
                         query: str, intent: str,
                         entity_info: Dict = None) -> np.ndarray:
        """Extract full feature vector (38-dim)."""
        sent_f = self._extract_sentence_features(sentence, entity)
        ent_f = self._extract_entity_features(entity, entity_info)
        query_f = self._extract_query_features(query, intent)
        return np.concatenate([sent_f, ent_f, query_f])

    def score_heads(self, features: np.ndarray) -> Dict[str, float]:
        """Score on each head separately. Returns dict of head_name → score."""
        # Encode
        h = self._relu(features @ self.W_enc + self.b_enc)

        scores = {}
        for i, name in enumerate(self.head_names):
            logit = float((h @ self.W_heads[i]).item() + self.b_heads[i].item())
            # Platt scaling
            calibrated = self._sigmoid(self.platt_a[i] * logit + self.platt_b[i])
            scores[name] = float(calibrated)

        return scores

    def score(self, sentence: str, entity: str, query: str,
              intent: str = 'DEFINITIONAL',
              entity_info: Dict = None,
              weights: Dict[str, float] = None) -> Tuple[float, Dict[str, float]]:
        """
        Score a sentence. Returns (final_score, head_scores).

        Final score is weighted combination of head scores.
        """
        features = self.extract_features(sentence, entity, query, intent, entity_info)
        head_scores = self.score_heads(features)

        # Weighted combination
        w = weights or {name: float(self.combo_weights[i])
                        for i, name in enumerate(self.head_names)}
        total_w = sum(w.values()) or 1.0
        final = sum(head_scores[name] * w.get(name, 0) for name in self.head_names) / total_w

        return final, head_scores

    def score_with_cache(self, sentence: str, entity: str, query: str,
                         intent: str = 'DEFINITIONAL',
                         entity_info: Dict = None) -> Tuple[float, Dict[str, float]]:
        """Score with caching for repeated sentences."""
        key = (sentence[:80], entity, query[:40], intent)
        if key in self._cache:
            return self._cache[key]
        result = self.score(sentence, entity, query, intent, entity_info)
        self._cache[key] = result
        return result

    def rank_sentences(self, sentences: List[str], entity: str,
                       query: str, intent: str = 'DEFINITIONAL',
                       entity_info: Dict = None,
                       top_k: int = 4,
                       min_score: float = 0.3) -> List[Tuple[float, str, Dict]]:
        """
        Rank sentences by score. Returns list of (score, sentence, head_scores).
        """
        scored = []
        for sent in sentences:
            sc, heads = self.score_with_cache(sent, entity, query, intent, entity_info)
            if sc >= min_score:
                scored.append((sc, sent, heads))
        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def train(self, sentences: List[str], labels: List[float],
              entity: str, query: str, intent: str,
              epochs: int = 15, lr: float = 0.01):
        """
        Train the full model (encoder + heads + combination weights).
        Uses MSE loss on heuristic labels.
        """
        # Pre-compute features and head outputs
        all_features = []
        for sent in sentences:
            f = self.extract_features(sent, entity, query, intent)
            all_features.append(f)

        features = np.array(all_features)
        labels_arr = np.array(labels, dtype=np.float32)

        for epoch in range(epochs):
            perm = np.random.permutation(len(features))
            total_loss = 0.0

            for idx in perm:
                feat = features[idx]
                label = labels_arr[idx]

                # Forward
                h = self._relu(feat @ self.W_enc + self.b_enc)

                # Each head
                head_logits = []
                head_preds = []
                for i in range(self.n_heads):
                    logit = float((h @ self.W_heads[i]).item() + self.b_heads[i].item())
                    pred = self._sigmoid(logit)
                    head_logits.append(logit)
                    head_preds.append(pred)

                # Weighted combination
                w = self._sigmoid(self.combo_weights)  # normalize weights
                w = w / (w.sum() + 1e-8)
                final_pred = sum(head_preds[i] * w[i] for i in range(self.n_heads))

                # Loss
                loss = (final_pred - label) ** 2
                total_loss += loss

                # Backward (simplified — update heads and encoder)
                d_final = 2.0 * (final_pred - label)

                for i in range(self.n_heads):
                    # Gradient for head i
                    d_pred = d_final * w[i]
                    d_logit = d_pred * head_preds[i] * (1 - head_preds[i])

                    # Gradient for W_heads[i], b_heads[i]
                    d_W_head = np.outer(h, np.array([d_logit]))
                    d_b_head = np.array([d_logit])

                    # Gradient for encoder through head
                    d_h = self.W_heads[i].flatten() * d_logit
                    d_h[h <= 0] = 0  # ReLU

                    # Update head
                    self.W_heads[i] -= lr * np.clip(d_W_head, -1, 1)
                    self.b_heads[i] -= lr * np.clip(d_b_head, -1, 1)

                    # Update encoder
                    d_W_enc = np.outer(feat, d_h)
                    self.W_enc -= lr * np.clip(d_W_enc, -1, 1)
                    self.b_enc -= lr * np.clip(d_h, -1, 1)

                # Update combination weights
                for i in range(self.n_heads):
                    d_w = d_final * head_preds[i]
                    self.combo_weights[i] -= lr * 0.1 * d_w

            avg_loss = total_loss / len(features)
            if (epoch + 1) % 5 == 0:
                logger.info(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}")

    def train_platt_scaling(self, sentences: List[str], labels: List[float],
                            entity: str, query: str, intent: str):
        """
        Train Platt scaling for each head using labeled data.
        labels: list of 0-1 quality ratings per sentence.
        """
        all_features = []
        all_head_scores = []
        for sent in sentences:
            f = self.extract_features(sent, entity, query, intent)
            h = self.score_heads(f)
            all_features.append(f)
            all_head_scores.append([h[name] for name in self.head_names])

        all_features = np.array(all_features)
        all_head_scores = np.array(all_head_scores)
        labels = np.array(labels)

        # Simple Platt scaling per head
        for i in range(self.n_heads):
            logits = all_head_scores[:, i]
            # Fit logistic regression: platt_a * sigmoid(logit) + platt_b ≈ label
            best_a, best_b = 1.0, 0.0
            best_loss = float('inf')
            for a in np.arange(0.5, 3.0, 0.5):
                for b in np.arange(-2.0, 2.0, 0.5):
                    preds = self._sigmoid(a * logits + b)
                    loss = np.mean((preds - labels) ** 2)
                    if loss < best_loss:
                        best_loss = loss
                        best_a, best_b = a, b
            self.platt_a[i] = best_a
            self.platt_b[i] = best_b
            logger.info(f"  Head {self.head_names[i]}: platt_a={best_a:.2f}, "
                         f"platt_b={best_b:.2f}, loss={best_loss:.4f}")

    def save(self, path: str):
        data = {
            'feature_dim': self.feature_dim,
            'latent_dim': self.latent_dim,
            'n_heads': self.n_heads,
            'head_names': self.head_names,
            'W_enc': self.W_enc.tolist(),
            'b_enc': self.b_enc.tolist(),
            'W_heads': [w.tolist() for w in self.W_heads],
            'b_heads': [b.tolist() for b in self.b_heads],
            'platt_a': self.platt_a.tolist(),
            'platt_b': self.platt_b.tolist(),
            'combo_weights': self.combo_weights.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'MultiDimScorer':
        with open(path, 'r') as f:
            data = json.load(f)
        scorer = cls(data['feature_dim'], data['latent_dim'])
        scorer.head_names = data['head_names']
        scorer.W_enc = np.array(data['W_enc'], dtype=np.float32)
        scorer.b_enc = np.array(data['b_enc'], dtype=np.float32)
        scorer.W_heads = [np.array(w, dtype=np.float32) for w in data['W_heads']]
        scorer.b_heads = [np.array(b, dtype=np.float32) for b in data['b_heads']]
        scorer.platt_a = np.array(data['platt_a'], dtype=np.float32)
        scorer.platt_b = np.array(data['platt_b'], dtype=np.float32)
        scorer.combo_weights = np.array(data['combo_weights'], dtype=np.float32)
        return scorer
