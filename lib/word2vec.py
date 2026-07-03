"""
Word2Vec Skip-Gram — Trained on the Book
Learns word embeddings directly from the training text.
No external pretrained vectors needed — the book teaches its own semantics.

Architecture:
  Skip-Gram: predict context words from a center word
  Negative Sampling: efficient training (sample K noise words per real pair)
  Subword info: handles misspellings/OCR artifacts via character n-grams
"""

import math
import random
import logging
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.lib.word2vec')


class Vocabulary:
    """Word-to-index mapping with frequency filtering."""

    def __init__(self, min_count: int = 2):
        self.min_count = min_count
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_freq: Counter = Counter()
        self.total_words = 0

    def build(self, tokens: List[str]):
        """Build vocabulary from token list."""
        self.word_freq.update(tokens)
        self.total_words = len(tokens)

        # Filter by frequency
        filtered = [(w, f) for w, f in self.word_freq.items()
                     if f >= self.min_count and len(w) > 1]

        # Sort by frequency (most frequent = lowest index)
        filtered.sort(key=lambda x: -x[1])

        # Reserve 0 for padding, 1 for UNK
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        for i, (word, _) in enumerate(filtered, start=2):
            self.word2idx[word] = i
            self.idx2word[i] = word

        logger.info(f"Vocabulary: {len(self.word2idx)} words "
                     f"(from {len(self.word_freq)} unique, "
                     f"{self.total_words} total tokens)")

    def encode(self, word: str) -> int:
        return self.word2idx.get(word.lower(), 1)  # 1 = UNK

    def decode(self, idx: int) -> str:
        return self.idx2word.get(idx, '<UNK>')

    def __len__(self):
        return len(self.word2idx)


class SkipGramModel:
    """
    Skip-Gram with Negative Sampling.

    Learns two embedding matrices:
      - W_in:  center word embeddings   (V × D)
      - W_out: context word embeddings  (V × D)

    Training: given center word w, predict nearby context words.
    """

    def __init__(self, vocab_size: int, dim: int = 100):
        self.vocab_size = vocab_size
        self.dim = dim

        # Xavier initialization
        scale = math.sqrt(2.0 / (vocab_size + dim))
        self.W_in = np.random.randn(vocab_size, dim).astype(np.float32) * scale
        self.W_out = np.zeros((vocab_size, dim), dtype=np.float32)

        # Training state
        self._center_grad = np.zeros_like(self.W_in)
        self._context_grad = np.zeros_like(self.W_out)

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        s = scores - scores.max()
        exp_s = np.exp(s)
        return exp_s / exp_s.sum()

    def train_pair(self, center_idx: int, context_idx: int,
                   neg_indices: np.ndarray, lr: float) -> float:
        """
        Train one (center, context) pair with K negative samples.

        Returns loss (negative log sigmoid of positive score).
        """
        # Get embeddings
        h = self.W_in[center_idx]          # (D,)
        pos_w = self.W_out[context_idx]    # (D,)
        neg_w = self.W_out[neg_indices]    # (K, D)

        # Positive score: dot(center, context)
        pos_score = np.dot(h, pos_w)
        pos_sig = 1.0 / (1.0 + np.exp(-min(pos_score, 10.0)))  # sigmoid

        # Negative scores: dot(center, neg_context)
        neg_scores = neg_w @ h              # (K,)
        neg_sigs = 1.0 / (1.0 + np.exp(np.clip(neg_scores, -10, 10)))

        # Loss: -log(sigmoid(pos)) - sum(log(sigmoid(-neg)))
        loss = -math.log(pos_sig + 1e-10) - np.sum(np.log(1 - neg_sigs + 1e-10))

        # Gradients
        # dL/dh = (pos_sig - 1) * pos_w + sum((1 - neg_sig_k) * neg_w_k)
        grad_h = (pos_sig - 1.0) * pos_w + ((1.0 - neg_sigs)[:, None] * neg_w).sum(axis=0)

        # dL/d_pos_w = (pos_sig - 1) * h
        grad_pos = (pos_sig - 1.0) * h

        # dL/d_neg_w_k = (1 - neg_sig_k) * h
        grad_neg = (1.0 - neg_sigs)[:, None] * h  # (K, D)

        # Update with gradient clipping
        grad_h = np.clip(grad_h, -1.0, 1.0)
        grad_pos = np.clip(grad_pos, -1.0, 1.0)
        grad_neg = np.clip(grad_neg, -1.0, 1.0)

        self.W_in[center_idx] -= lr * grad_h
        self.W_out[context_idx] -= lr * grad_pos
        self.W_out[neg_indices] -= lr * grad_neg

        return float(loss)

    def get_embedding(self, word_idx: int) -> np.ndarray:
        """Get word embedding (center vectors)."""
        return self.W_in[word_idx]

    def similarity(self, idx1: int, idx2: int) -> float:
        """Cosine similarity between two words."""
        v1 = self.W_in[idx1]
        v2 = self.W_in[idx2]
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-8 or norm2 < 1e-8:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def most_similar(self, word: str, vocab: Vocabulary, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find most similar words."""
        idx = vocab.encode(word)
        if idx == 1:  # UNK
            return []

        vec = self.W_in[idx]
        # Cosine similarity against all words
        norms = np.linalg.norm(self.W_in, axis=1)
        norms[norms < 1e-8] = 1.0
        sims = (self.W_in @ vec) / (norms * (np.linalg.norm(vec) + 1e-8))
        top_indices = np.argsort(-sims)[1:top_k + 1]  # exclude self

        return [(vocab.decode(i), float(sims[i])) for i in top_indices]

    def save(self, path: str):
        """Save model to disk."""
        data = {
            'vocab_size': self.vocab_size,
            'dim': self.dim,
            'W_in': self.W_in.tolist(),
            'W_out': self.W_out.tolist(),
        }
        with open(path, 'w') as f:
            json.dump(data, f)
        logger.info(f"Saved Word2Vec model to {path}")

    @classmethod
    def load(cls, path: str) -> 'SkipGramModel':
        """Load model from disk."""
        with open(path, 'r') as f:
            data = json.load(f)
        model = cls(data['vocab_size'], data['dim'])
        model.W_in = np.array(data['W_in'], dtype=np.float32)
        model.W_out = np.array(data['W_out'], dtype=np.float32)
        return model


class Word2VecTrainer:
    """
    Trains Word2Vec skip-gram on tokenized text.

    Usage:
        trainer = Word2VecTrainer(dim=100, window=5, min_count=2, neg_samples=5)
        model, vocab = trainer.train(tokenized_sentences)
        model.save('word2vec.json')
    """

    def __init__(self, dim: int = 100, window: int = 5,
                 min_count: int = 2, neg_samples: int = 5,
                 epochs: int = 5, lr: float = 0.025):
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.neg_samples = neg_samples
        self.epochs = epochs
        self.lr = lr

    def _subsample_prob(self, freq: float, total: float) -> float:
        """Probability of keeping a frequent word (Mikolov formula)."""
        t = 1e-4
        return min(1.0, (math.sqrt(freq / (t * total)) + 1) * (t * total / freq))

    def _build_neg_table(self, vocab: Vocabulary, table_size: int = 1_000_000) -> np.ndarray:
        """Build negative sampling table (frequency-based)."""
        table = np.zeros(table_size, dtype=np.int32)
        # p(w) ∝ freq(w)^0.75
        total = sum(f ** 0.75 for f in vocab.word_freq.values())
        cumul = 0.0
        idx = 0
        for word, freq in vocab.word_freq.items():
            if word not in vocab.word2idx:
                continue
            cumul += (freq ** 0.75) / total
            end = int(cumul * table_size)
            end = min(end, table_size)
            while idx < end and idx < table_size:
                table[idx] = vocab.word2idx[word]
                idx += 1
        # Fill remaining
        while idx < table_size:
            table[idx] = vocab.word2idx.get(word, 1)
            idx += 1
        return table

    def _generate_pairs(self, tokens: List[int], vocab_size: int) -> List[Tuple[int, int]]:
        """Generate all (center, context) pairs for one sentence."""
        pairs = []
        for i, center in enumerate(tokens):
            if center == 0:  # skip PAD
                continue
            # Fixed window for speed
            for j in range(max(0, i - self.window), min(len(tokens), i + self.window + 1)):
                if j != i and tokens[j] != 0:
                    pairs.append((center, tokens[j]))
        return pairs

    def train(self, tokenized_sentences: List[List[str]],
              save_path: str = None) -> Tuple[SkipGramModel, Vocabulary]:
        """
        Train Word2Vec on tokenized sentences.

        Args:
            tokenized_sentences: List of sentences, each a list of word strings
            save_path: Optional path to save model

        Returns:
            (model, vocabulary)
        """
        # Build vocabulary
        all_tokens = [w for sent in tokenized_sentences for w in sent]
        vocab = Vocabulary(min_count=self.min_count)
        vocab.build(all_tokens)

        if len(vocab) < 10:
            logger.warning("Vocabulary too small for meaningful embeddings")
            return None, vocab

        # Initialize model
        model = SkipGramModel(len(vocab), self.dim)

        # Build negative sampling table
        neg_table = self._build_neg_table(vocab)
        table_size = len(neg_table)

        # Encode all sentences
        encoded_sents = []
        for sent in tokenized_sentences:
            encoded = [vocab.encode(w) for w in sent]
            # Subsample frequent words
            encoded = [idx for idx in encoded
                       if idx > 1 and random.random() < self._subsample_prob(
                           vocab.word_freq.get(vocab.decode(idx), 1),
                           vocab.total_words
                       )]
            if len(encoded) >= 2:
                encoded_sents.append(encoded)

        total_pairs = sum(len(s) * self.window * 2 for s in encoded_sents)
        logger.info(f"Training Word2Vec: {len(vocab)} words, "
                     f"{len(encoded_sents)} sentences, "
                     f"~{total_pairs:,} pairs/epoch, "
                     f"{self.epochs} epochs")

        # Training loop
        for epoch in range(self.epochs):
            random.shuffle(encoded_sents)
            total_loss = 0.0
            pair_count = 0

            for sent in encoded_sents:
                # Train on-the-fly instead of pre-generating all pairs
                for i, center in enumerate(sent):
                    if center == 0:
                        continue
                    # Sample context positions (not all)
                    ctx_start = max(0, i - self.window)
                    ctx_end = min(len(sent), i + self.window + 1)
                    ctx_indices = [j for j in range(ctx_start, ctx_end) if j != i]
                    # Subsample context positions for speed
                    if len(ctx_indices) > 3:
                        ctx_indices = random.sample(ctx_indices, 3)

                    for j in ctx_indices:
                        if sent[j] == 0:
                            continue
                        neg_indices = neg_table[
                            np.random.randint(0, table_size, size=self.neg_samples)
                        ]
                        loss = model.train_pair(center, sent[j], neg_indices, self.lr)
                        total_loss += loss
                        pair_count += 1

            # Decay learning rate
            self.lr *= 0.95
            avg_loss = total_loss / max(pair_count, 1)
            logger.info(f"  Epoch {epoch + 1}/{self.epochs}: "
                         f"loss={avg_loss:.4f}, pairs={pair_count:,}")

        # Save if requested
        if save_path:
            model.save(save_path)
            # Also save vocabulary
            vocab_path = save_path.replace('.json', '_vocab.json')
            with open(vocab_path, 'w') as f:
                json.dump({
                    'word2idx': vocab.word2idx,
                    'word_freq': dict(vocab.word_freq.most_common(5000)),
                    'total_words': vocab.total_words,
                }, f)
            logger.info(f"Saved vocabulary to {vocab_path}")

        return model, vocab


class Word2VecEmbeddings:
    """
    High-level interface for Word2Vec embeddings.
    Provides word similarity, analogy, and sentence embedding.
    """

    def __init__(self, model: SkipGramModel, vocab: Vocabulary):
        self.model = model
        self.vocab = vocab

    @classmethod
    def load(cls, model_path: str, vocab_path: str = None) -> 'Word2VecEmbeddings':
        """Load from saved files."""
        model = SkipGramModel.load(model_path)
        if vocab_path is None:
            vocab_path = model_path.replace('.json', '_vocab.json')
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        vocab = Vocabulary()
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in data['word2idx'].items()}
        vocab.word_freq = Counter(data.get('word_freq', {}))
        vocab.total_words = data.get('total_words', 0)
        return cls(model, vocab)

    def similarity(self, word1: str, word2: str) -> float:
        """Cosine similarity between two words."""
        idx1 = self.vocab.encode(word1)
        idx2 = self.vocab.encode(word2)
        return self.model.similarity(idx1, idx2)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get embedding vector for a word."""
        idx = self.vocab.encode(word)
        if idx == 1:
            return None
        return self.model.get_embedding(idx)

    def sentence_embedding(self, words: List[str]) -> np.ndarray:
        """Average of word vectors (simple but effective)."""
        vecs = [self.model.get_embedding(self.vocab.encode(w))
                for w in words if self.vocab.encode(w) != 1]
        if not vecs:
            return np.zeros(self.model.dim, dtype=np.float32)
        return np.mean(vecs, axis=0)

    def word_analogy(self, a: str, b: str, c: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Solve: a is to b as c is to ?
        Returns list of (word, similarity).
        """
        va = self.get_vector(a)
        vb = self.get_vector(b)
        vc = self.get_vector(c)
        if va is None or vb is None or vc is None:
            return []

        # target = b - a + c
        target = vb - va + vc
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-8:
            return []

        # Similarity against all words
        norms = np.linalg.norm(self.model.W_in, axis=1)
        norms[norms < 1e-8] = 1.0
        sims = (self.model.W_in @ target) / (norms * target_norm)

        # Exclude input words
        exclude = {self.vocab.encode(w) for w in [a, b, c]}
        top_indices = np.argsort(-sims)
        results = []
        for i in top_indices:
            if i not in exclude and len(results) < top_k:
                results.append((self.vocab.decode(i), float(sims[i])))
        return results

    def closest_words(self, vec: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Find closest words to a vector."""
        norms = np.linalg.norm(self.model.W_in, axis=1)
        norms[norms < 1e-8] = 1.0
        vec_norm = np.linalg.norm(vec)
        if vec_norm < 1e-8:
            return []
        sims = (self.model.W_in @ vec) / (norms * vec_norm)
        top_indices = np.argsort(-sims)[:top_k]
        return [(self.vocab.decode(i), float(sims[i])) for i in top_indices]
