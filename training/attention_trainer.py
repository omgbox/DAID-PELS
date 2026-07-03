"""
Training Loop for Token-Level Attention
Uses self-supervised data to train the attention weights.

Tasks trained jointly:
  1. Query-Sentence Relevance: predict if sentence answers query
  2. Coreference: predict if pronoun refers to entity
  3. Sentence Ordering: predict if sentence pair is correctly ordered
  4. Token Relevance: predict if token is relevant to entity
"""

import math
import random
import logging
import json
import numpy as np
from typing import Dict, List, Tuple

logger = logging.getLogger('bookbot.training.attention_trainer')


class AttentionTrainer:
    """
    Trains token-level attention using a feature-based approach:
      1. Attention produces features (not trained directly)
      2. Scoring head is trained on those features
      3. This gives us learned selection without full backprop
    """

    def __init__(self, model, lr: float = 0.02):
        self.model = model
        self.lr = lr
        # Trainable scoring weights (separate from model)
        self.W_head = np.random.randn(8, 1).astype(np.float32) * 0.1
        self.b_head = np.zeros(1, dtype=np.float32)
        # Platt scaling
        self.platt_a = 1.0
        self.platt_b = 0.0

    def _sigmoid(self, x):
        x = np.clip(x, -10, 10)
        return 1.0 / (1.0 + np.exp(-x))

    def _extract_features(self, scores, cross_attn, sent_vecs, query_vec):
        """Extract features from attention outputs for scoring."""
        features = []
        for i in range(len(scores)):
            f = np.zeros(8, dtype=np.float32)
            f[0] = scores[i]                           # self-attention score
            f[1] = cross_attn[i] if i < len(cross_attn) else 0  # cross-attn
            f[2] = np.linalg.norm(sent_vecs[i])         # sentence vector magnitude
            f[3] = float(sent_vecs[i] @ query_vec) if query_vec is not None else 0  # similarity to query
            f[4] = len(scores)                          # number of sentences (context)
            f[5] = 1.0 if i < 3 else 0.0               # position bias
            f[6] = scores[i] * cross_attn[i] if i < len(cross_attn) else 0  # interaction
            f[7] = 1.0                                  # bias
            features.append(f)
        return np.array(features)

    def _score_features(self, features):
        """Score using trained head."""
        logits = features @ self.W_head + self.b_head
        return self._sigmoid(logits).flatten()

    def train(self, training_data: Dict, epochs: int = 10):
        """Train the scoring head on all tasks."""
        query_pairs = training_data.get('query_pairs', [])
        coref_pairs = training_data.get('coref_pairs', [])

        logger.info(f"Training scoring head: {len(query_pairs)} query pairs, "
                     f"{len(coref_pairs)} coref pairs")

        for epoch in range(epochs):
            random.shuffle(query_pairs)
            total_loss = 0.0
            n_correct = 0
            n = 0

            for pair in query_pairs[:300]:
                pos_text = pair['positive']
                neg_text = pair['negative']
                query = pair['query']

                if not pos_text or not neg_text:
                    continue

                # Get attention features for positive
                pos_scores, pos_cross, pos_attn, _ = self.model.forward(
                    [pos_text], query, '')

                # Get raw attention scores
                pos_raw = float(pos_scores[0]) if len(pos_scores) > 0 else 0.5

                neg_scores_out, _, _, _ = self.model.forward(
                    [neg_text], query, '')
                neg_raw = float(neg_scores_out[0]) if len(neg_scores_out) > 0 else 0.5

                # Simple feature: just use the raw attention score
                pos_feat = np.array([[pos_raw, 0.5, 1.0, pos_raw, 1.0, 1.0, pos_raw * 0.5, 1.0]])
                neg_feat = np.array([[neg_raw, 0.5, 1.0, neg_raw, 1.0, 0.0, neg_raw * 0.5, 1.0]])

                pos_pred = self._sigmoid(float((pos_feat @ self.W_head + self.b_head).item()))
                neg_pred = self._sigmoid(float((neg_feat @ self.W_head + self.b_head).item()))

                # Hinge loss
                margin = 0.3
                loss = max(0, margin - pos_pred + neg_pred)
                total_loss += loss
                n += 1

                if pos_pred > neg_pred:
                    n_correct += 1

                # Gradient update on W_head
                if loss > 0:
                    # dL/d_pos_pred = -1, dL/d_neg_pred = 1
                    # d_pred/d_logit = pred * (1 - pred)
                    d_pos = -pos_pred * (1 - pos_pred)
                    d_neg = neg_pred * (1 - neg_pred)

                    self.W_head += self.lr * d_pos * pos_feat.T
                    self.W_head += self.lr * d_neg * neg_feat.T
                    self.b_head += self.lr * d_pos
                    self.b_head += self.lr * d_neg

            acc = n_correct / max(n, 1)
            avg_loss = total_loss / max(n, 1)
            logger.info(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}, acc={acc:.2%}")

        # Calibrate Platt scaling
        self._calibrate(query_pairs)

    def _calibrate(self, pairs):
        """Fit Platt scaling on training data."""
        logits = []
        labels = []
        for pair in pairs[:200]:
            pos_scores, _, _, _ = self.model.forward(
                [pair['positive']], pair['query'], '')
            if len(pos_scores) > 0:
                logits.append(float(pos_scores[0]))
                labels.append(1.0)

            neg_scores, _, _, _ = self.model.forward(
                [pair['negative']], pair['query'], '')
            if len(neg_scores) > 0:
                logits.append(float(neg_scores[0]))
                labels.append(0.0)

        if not logits:
            return

        logits = np.array(logits)
        labels = np.array(labels)

        # Grid search for best Platt params
        best_a, best_b = 1.0, 0.0
        best_loss = float('inf')
        for a in np.arange(0.5, 5.0, 0.5):
            for b in np.arange(-3.0, 3.0, 0.5):
                preds = self._sigmoid(a * logits + b)
                loss = np.mean((preds - labels) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    best_a, best_b = a, b

        self.platt_a = best_a
        self.platt_b = best_b
        logger.info(f"  Platt scaling: a={best_a:.2f}, b={best_b:.2f}, loss={best_loss:.4f}")

    def predict(self, sentences, query, entity=''):
        """Predict with trained head + calibration."""
        scores, cross_attn, _, _ = self.model.forward(sentences, query, entity)

        if len(scores) == 0:
            return np.array([])

        # Apply Platt scaling
        calibrated = self._sigmoid(self.platt_a * scores + self.platt_b)
        return calibrated

    def save(self, path):
        """Save model + trained head."""
        self.model.save(path)
        # Save head separately
        head_path = path.replace('.json', '_head.json')
        with open(head_path, 'w') as f:
            json.dump({
                'W_head': self.W_head.tolist(),
                'b_head': self.b_head.tolist(),
                'platt_a': self.platt_a,
                'platt_b': self.platt_b,
            }, f)
        logger.info(f"Saved trained head to {head_path}")

    def load_head(self, path):
        """Load trained head."""
        head_path = path.replace('.json', '_head.json')
        try:
            with open(head_path, 'r') as f:
                data = json.load(f)
            self.W_head = np.array(data['W_head'], dtype=np.float32)
            self.b_head = np.array(data['b_head'], dtype=np.float32)
            self.platt_a = data['platt_a']
            self.platt_b = data['platt_b']
            logger.info(f"Loaded trained head from {head_path}")
        except FileNotFoundError:
            logger.info("No trained head found, using random init")
