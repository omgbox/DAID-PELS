"""
SVO Quality Scorer
Scores each SVO triple by informativeness using word embeddings + features.

Uses a lightweight MLP (2 layers) trained on heuristics:
  - Verb informativeness (is "walks" useful? is "was" noise?)
  - Object completeness (empty = bad, pronoun = bad, long = maybe noise)
  - Semantic coherence (does verb-object pair make sense via embeddings?)
  - Entity relevance (does this tell us about the character?)

No labeled data needed — training uses heuristic labels.
"""

import math
import random
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.query.svo_scorer')

# Verbs that carry almost no information about an entity
UNINFORMATIVE_VERBS = {
    'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did',
    'says', 'said', 'tells', 'told', 'asks', 'asked',
    'seems', 'appears', 'looks', 'looks',
    'goes', 'went', 'comes', 'came',
    'makes', 'made', 'gets', 'got',
    'would', 'could', 'should', 'might', 'may', 'shall', 'can',
    'will', 'must',
}

# Pronouns/short words that make bad objects
BAD_OBJECTS = {
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    'what', 'which', 'who', 'whom',
    'something', 'anything', 'nothing', 'everything',
}

# Pronouns that make bad subjects for definitional answers
PRONOUNS = {
    'i', 'me', 'you', 'he', 'she', 'it', 'we', 'they',
    'him', 'her', 'us', 'them',
}


class SVOQualityScorer:
    """
    Scores SVO triples by informativeness.

    Features (8-dimensional):
      1. Verb informativeness (1 - freq in common verbs)
      2. Object completeness (length, not pronoun)
      3. Subject is entity name (not pronoun)
      4. Verb-object semantic similarity (from embeddings)
      5. Object is proper noun (capitalized)
      6. Verb length (longer = more specific)
      7. Object length (medium = good, too long = noise)
      8. Subject length (short = clean)

    Uses a small MLP with sigmoid output.
    """

    def __init__(self, embeddings=None):
        self.embeddings = embeddings
        self.feature_dim = 8
        self.hidden_dim = 16

        # Initialize weights (Xavier)
        scale1 = math.sqrt(2.0 / (self.feature_dim + self.hidden_dim))
        self.W1 = np.random.randn(self.feature_dim, self.hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        scale2 = math.sqrt(2.0 / (self.hidden_dim + 1))
        self.W2 = np.random.randn(self.hidden_dim, 1).astype(np.float32) * scale2
        self.b2 = np.zeros(1, dtype=np.float32)

    def _extract_features(self, subject: str, verb: str, obj: str,
                          entity: str = '') -> np.ndarray:
        """Extract feature vector for an SVO triple."""
        s_lower = subject.lower().strip()
        v_lower = verb.lower().strip()
        o_lower = obj.lower().strip()

        features = np.zeros(self.feature_dim, dtype=np.float32)

        # 1. Verb informativeness: 0 = common/useless, 1 = specific/informative
        features[0] = 0.0 if v_lower in UNINFORMATIVE_VERBS else 1.0
        # Bonus for more specific verbs
        if v_lower not in UNINFORMATIVE_VERBS and len(v_lower) > 4:
            features[0] = min(1.0, features[0] + 0.3)

        # 2. Object completeness
        if not o_lower or len(o_lower) < 2:
            features[1] = 0.0
        elif o_lower in BAD_OBJECTS:
            features[1] = 0.1
        elif len(o_lower) > 40:
            features[1] = 0.3  # too long = probably noise
        else:
            features[1] = min(1.0, len(o_lower) / 20.0)

        # 3. Subject is entity name
        if entity and entity.lower() in s_lower:
            features[2] = 1.0
        elif s_lower in PRONOUNS:
            features[2] = 0.0
        elif s_lower[0:1].isupper() if s_lower else False:
            features[2] = 0.8
        else:
            features[2] = 0.3

        # 4. Verb-object semantic similarity
        if self.embeddings and v_lower and o_lower:
            v_vec = self.embeddings.get_vector(v_lower)
            o_vec = self.embeddings.get_vector(o_lower)
            if v_vec is not None and o_vec is not None:
                norm_v = np.linalg.norm(v_vec)
                norm_o = np.linalg.norm(o_vec)
                if norm_v > 1e-8 and norm_o > 1e-8:
                    sim = float(np.dot(v_vec, o_vec) / (norm_v * norm_o))
                    features[3] = (sim + 1) / 2  # normalize to [0, 1]
                else:
                    features[3] = 0.5
            else:
                features[3] = 0.5
        else:
            features[3] = 0.5

        # 5. Object is proper noun
        features[4] = 1.0 if (o_lower and o_lower[0].isupper()) else 0.0

        # 6. Verb length (longer = more specific)
        features[5] = min(1.0, len(v_lower) / 10.0)

        # 7. Object length (medium is best)
        o_len = len(o_lower)
        if 3 <= o_len <= 20:
            features[6] = 1.0
        elif 2 <= o_len <= 30:
            features[6] = 0.7
        elif o_len > 30:
            features[6] = 0.3
        else:
            features[6] = 0.0

        # 8. Subject length (short is cleaner)
        s_len = len(s_lower)
        if s_len <= 10:
            features[7] = 1.0
        elif s_len <= 20:
            features[7] = 0.7
        else:
            features[7] = 0.3

        return features

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _sigmoid(self, x: np.ndarray) -> float:
        x = float(x)
        x = max(-10.0, min(10.0, x))
        return 1.0 / (1.0 + math.exp(-x))

    def score(self, subject: str, verb: str, obj: str,
              entity: str = '') -> float:
        """Score a single SVO triple. Returns 0-1 (higher = better)."""
        features = self._extract_features(subject, verb, obj, entity)

        # Forward pass
        hidden = self._relu(features @ self.W1 + self.b1)
        logit = float((hidden @ self.W2).item() + self.b2.item())
        return self._sigmoid(logit)

    def score_triple(self, triple: dict, entity: str = '') -> float:
        """Score a dict-format SVO triple."""
        if isinstance(triple, dict):
            s = triple.get('subject', '')
            v = triple.get('verb', '')
            o = triple.get('object', '')
        elif isinstance(triple, (list, tuple)) and len(triple) >= 3:
            s, v, o = str(triple[0]), str(triple[1]), str(triple[2])
        else:
            return 0.0
        return self.score(s, v, o, entity)

    def rank_triples(self, triples: List, entity: str = '',
                     threshold: float = 0.3) -> List[Tuple[float, any]]:
        """
        Score and rank triples. Returns list of (score, triple) sorted by score.
        Only returns triples above threshold.
        """
        scored = []
        for t in triples:
            sc = self.score_triple(t, entity)
            if sc >= threshold:
                scored.append((sc, t))
        scored.sort(key=lambda x: -x[0])
        return scored

    def train_heuristic(self, triples: List, entity: str = '',
                        epochs: int = 20, lr: float = 0.01):
        """
        Train the scorer using heuristic labels.

        Labels are generated by rules:
          - Good (1.0): specific verb + proper noun object + entity as subject
          - Medium (0.5): specific verb + decent object
          - Bad (0.0): common verb OR empty object OR pronoun object
        """
        training_data = []
        for t in triples:
            if isinstance(t, dict):
                s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                s, v, o = str(t[0]), str(t[1]), str(t[2])
            else:
                continue

            features = self._extract_features(s, v, o, entity)

            # Heuristic label
            v_lower = v.lower().strip()
            o_lower = o.lower().strip()

            if (v_lower not in UNINFORMATIVE_VERBS
                and o_lower and len(o_lower) >= 3
                and o_lower not in BAD_OBJECTS
                and entity.lower() in s.lower()):
                label = 0.9  # Good triple
            elif (v_lower not in UNINFORMATIVE_VERBS
                  and o_lower and len(o_lower) >= 2):
                label = 0.5  # Medium
            else:
                label = 0.1  # Bad triple

            training_data.append((features, label))

        if not training_data:
            return

        # Simple gradient descent
        for epoch in range(epochs):
            random.shuffle(training_data)
            total_loss = 0.0

            for features, label in training_data:
                # Forward
                hidden = self._relu(features @ self.W1 + self.b1)
                logit = float((hidden @ self.W2).item() + self.b2.item())
                pred = self._sigmoid(logit)

                # Loss: MSE
                loss = (pred - label) ** 2
                total_loss += loss

                # Backward
                d_logit = 2.0 * (pred - label) * pred * (1 - pred)
                d_W2 = hidden.reshape(-1, 1) * d_logit
                d_b2 = np.array([d_logit])
                d_hidden = self.W2.flatten() * d_logit
                d_hidden[hidden <= 0] = 0  # ReLU grad
                d_W1 = features.reshape(-1, 1) * d_hidden.reshape(1, -1)
                d_b1 = d_hidden

                # Clip gradients
                for g in [d_W1, d_b1, d_W2, d_b2]:
                    np.clip(g, -1.0, 1.0, out=g)

                # Update
                self.W1 -= lr * d_W1
                self.b1 -= lr * d_b1
                self.W2 -= lr * d_W2
                self.b2 -= lr * d_b2

            avg_loss = total_loss / len(training_data)
            if (epoch + 1) % 5 == 0:
                logger.info(f"  SVO Scorer epoch {epoch + 1}: loss={avg_loss:.4f}")

    def save(self, path: str):
        """Save model weights."""
        data = {
            'feature_dim': self.feature_dim,
            'hidden_dim': self.hidden_dim,
            'W1': self.W1.tolist(),
            'b1': self.b1.tolist(),
            'W2': self.W2.tolist(),
            'b2': self.b2.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str, embeddings=None) -> 'SVOQualityScorer':
        """Load model weights."""
        with open(path, 'r') as f:
            data = json.load(f)
        scorer = cls(embeddings)
        scorer.feature_dim = data['feature_dim']
        scorer.hidden_dim = data['hidden_dim']
        scorer.W1 = np.array(data['W1'], dtype=np.float32)
        scorer.b1 = np.array(data['b1'], dtype=np.float32)
        scorer.W2 = np.array(data['W2'], dtype=np.float32)
        scorer.b2 = np.array(data['b2'], dtype=np.float32)
        return scorer
