"""
MiniGPT — Next-Token Predictor Trained on the Book

A small GPT-style transformer (~2M params) trained on Pride and Prejudice.
Learns Jane Austen's sentence patterns, vocabulary, and style.

Architecture:
  Token embedding + positional encoding
  N transformer decoder blocks (causal attention)
  Linear head → vocabulary logits

Usage:
  Train: model.train_on_book(book_text)
  Generate: model.generate("Elizabeth felt", max_tokens=30)
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
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('bookbot.query.minigpt')


class CausalSelfAttention(nn.Module):
    """Causal (masked) multi-head self-attention — tokens only attend to previous tokens."""

    def __init__(self, dim: int, n_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.W_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.W_o = nn.Linear(dim, dim, bias=False)

        # Causal mask: upper triangular = -inf
        mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
        self.register_buffer('mask', mask)

        self.scale = math.sqrt(self.head_dim)

    def forward(self, X):
        B, T, _ = X.shape

        # Combined QKV
        qkv = self.W_qkv(X)  # (B, T, 3D)
        Q, K, V = qkv.chunk(3, dim=-1)

        # Reshape: (B, n_heads, T, head_dim)
        Q = Q.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.reshape(B, T, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, n_heads, T, T)
        scores = scores.masked_fill(self.mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)

        # Apply attention
        out = torch.matmul(attn, V)  # (B, n_heads, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.dim)
        out = self.W_o(out)

        return out


class GPTBlock(nn.Module):
    """Transformer decoder block: LayerNorm → Causal Attention → Residual → LayerNorm → FFN → Residual."""

    def __init__(self, dim: int, n_heads: int = 4, ffn_dim: int = None):
        super().__init__()
        ffn_dim = ffn_dim or dim * 4

        self.attn = CausalSelfAttention(dim, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, X):
        X = X + self.attn(self.ln1(X))
        X = X + self.ffn(self.ln2(X))
        return X


class MiniGPT(nn.Module):
    """
    Small GPT-style model trained on a single book.

    ~2M parameters: enough to learn sentence patterns from ~126K tokens.
    """

    def __init__(self, vocab_size: int = 8000, dim: int = 128,
                 n_layers: int = 6, n_heads: int = 4, max_len: int = 512):
        super().__init__()
        self.dim = dim
        self.max_len = max_len

        # Token embedding
        self.tok_emb = nn.Embedding(vocab_size, dim)

        # Positional encoding (learned)
        self.pos_emb = nn.Embedding(max_len, dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPTBlock(dim, n_heads) for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(dim)

        # Language model head
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying (embedding = output projection)
        self.head.weight = self.tok_emb.weight

        # Init weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        """
        idx: (B, T) token indices
        Returns: (B, T, vocab_size) logits
        """
        B, T = idx.shape

        tok_emb = self.tok_emb(idx)  # (B, T, D)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))  # (T, D)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

    @torch.no_grad()
    def generate(self, idx, max_tokens: int = 50, temperature: float = 0.8,
                 top_k: int = 40) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        for _ in range(max_tokens):
            # Crop to max_len
            idx_cond = idx[:, -self.max_len:]

            # Forward
            logits = self.forward(idx_cond)
            logits = logits[:, -1, :]  # last token logits

            # Temperature
            logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, idx_next], dim=1)

        return idx

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def load(cls, model_path, vocab_path):
        """Load a trained MiniGPT model."""
        import torch
        model = cls(vocab_size=4000, dim=64, n_layers=4, n_heads=4, max_len=128)
        model.load_state_dict(torch.load(model_path, weights_only=False))
        model.eval()
        model._vocab = Vocabulary.load(vocab_path)
        return model

    @torch.no_grad()
    def generate_from_text(self, prompt: str, max_tokens: int = 50,
                           temperature: float = 0.8) -> str:
        """Generate text from a prompt string."""
        self.eval()
        vocab = self._vocab
        tokens = vocab.encode(prompt)
        if not tokens:
            return ''
        idx = torch.tensor([tokens], dtype=torch.long)
        output = self.generate(idx, max_tokens=max_tokens, temperature=temperature)
        return vocab.decode(output[0].tolist())


class Vocabulary:
    """Simple token-to-index mapping with BPE-like tokenization."""

    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<BOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.idx2word = {0: '<PAD>', 1: '<BOS>', 2: '<EOS>', 3: '<UNK>'}
        self.word_freq = {}
        self._built = False

    def build(self, text: str):
        """Build vocabulary from text."""
        if self._built:
            return

        # Simple word-level tokenization
        words = text.lower().split()
        self.word_freq = {}
        for w in words:
            w_clean = w.strip('.,;:!?""\n\t')
            if w_clean:
                self.word_freq[w_clean] = self.word_freq.get(w_clean, 0) + 1

        # Sort by frequency, take top vocab_size
        sorted_words = sorted(self.word_freq.items(), key=lambda x: -x[1])
        for word, _ in sorted_words[:self.vocab_size - 4]:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        self._built = True
        logger.info(f"Vocabulary: {len(self.word2idx)} words "
                     f"(from {len(self.word_freq)} unique)")

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices."""
        words = text.lower().split()
        indices = [1]  # BOS
        for w in words:
            w_clean = w.strip('.,;:!?""\n\t')
            idx = self.word2idx.get(w_clean, 3)  # UNK
            indices.append(idx)
        indices.append(2)  # EOS
        return indices

    def decode(self, indices: List[int]) -> str:
        """Decode token indices to text."""
        words = []
        for idx in indices:
            if idx in (0, 1, 2):  # PAD, BOS, EOS
                continue
            word = self.idx2word.get(idx, '<UNK>')
            words.append(word)
        return ' '.join(words)

    def save(self, path):
        data = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            data = json.load(f)
        vocab = cls(data['vocab_size'])
        vocab.word2idx = data['word2idx']
        vocab.idx2word = {int(v): k for k, v in data['word2idx'].items()}
        return vocab


class MiniGPTTrainer:
    """Training loop for MiniGPT."""

    def __init__(self, model, vocab, lr: float = 3e-4):
        self.model = model
        self.vocab = vocab
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
        self.losses = []

    def create_training_data(self, text: str, seq_len: int = 128,
                              stride: int = 64) -> List[torch.Tensor]:
        """Create sliding window training sequences."""
        indices = self.vocab.encode(text)

        sequences = []
        for i in range(0, len(indices) - seq_len, stride):
            seq = indices[i:i + seq_len + 1]
            if len(seq) == seq_len + 1:
                sequences.append(torch.tensor(seq, dtype=torch.long))

        logger.info(f"Created {len(sequences)} training sequences "
                     f"(seq_len={seq_len}, stride={stride})")
        return sequences

    def train(self, text: str, epochs: int = 10, seq_len: int = 128,
              batch_size: int = 16):
        """Train on book text with visual progress."""
        import time
        import sys

        # Build vocab
        print("\n  Building vocabulary...")
        self.vocab.build(text)

        # Create training data
        sequences = self.create_training_data(text, seq_len)

        print(f"\n  Model: {self.model.count_parameters():,} parameters")
        print(f"  Data: {len(sequences)} sequences, seq_len={seq_len}")
        print(f"  Batch size: {batch_size}, Batches/epoch: {(len(sequences) + batch_size - 1) // batch_size}")
        print()

        for epoch in range(epochs):
            self.model.train()
            random.shuffle(sequences)

            total_loss = 0.0
            n_batches = 0
            epoch_start = time.time()

            batches = (len(sequences) + batch_size - 1) // batch_size

            for i in range(0, len(sequences), batch_size):
                batch = sequences[i:i + batch_size]
                if len(batch) < 2:
                    continue

                batch = torch.stack(batch)  # (B, T+1)

                x = batch[:, :-1]
                y = batch[:, 1:]

                logits = self.model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=0
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                # Progress bar
                pct = n_batches / batches * 100
                bar_len = 30
                filled = int(bar_len * n_batches / batches)
                bar = '#' * filled + '-' * (bar_len - filled)
                avg_loss = total_loss / n_batches
                elapsed = time.time() - epoch_start
                eta = elapsed / n_batches * (batches - n_batches) if n_batches > 0 else 0

                sys.stdout.write(
                    f"\r  Epoch {epoch+1:>2}/{epochs} |{bar}| "
                    f"{pct:>5.1f}% | loss: {avg_loss:.4f} | "
                    f"{elapsed:.0f}s elapsed | ETA: {eta:.0f}s"
                )
                sys.stdout.flush()

            avg_loss = total_loss / max(n_batches, 1)
            self.losses.append(avg_loss)
            perplexity = math.exp(avg_loss) if avg_loss < 10 else float('inf')
            elapsed = time.time() - epoch_start

            sys.stdout.write(
                f"\r  Epoch {epoch+1:>2}/{epochs} |{'#' * bar_len}| "
                f"100.0% | loss: {avg_loss:.4f} | "
                f"ppl: {perplexity:.1f} | {elapsed:.0f}s           \n"
            )
            sys.stdout.flush()

    def generate_from_prompt(self, prompt: str, max_tokens: int = 50,
                              temperature: float = 0.8) -> str:
        """Generate text from a prompt."""
        self.model.eval()
        indices = self.vocab.encode(prompt)
        x = torch.tensor([indices], dtype=torch.long)

        with torch.no_grad():
            output = self.model.generate(x, max_tokens=max_tokens,
                                          temperature=temperature)

        return self.vocab.decode(output[0].tolist())

    def save(self, model_path, vocab_path):
        torch.save(self.model.state_dict(), model_path)
        self.vocab.save(vocab_path)
        logger.info(f"Saved model to {model_path}")

    def load(self, model_path, vocab_path):
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.vocab = Vocabulary.load(vocab_path)
        logger.info(f"Loaded model from {model_path}")


def load_minigpt(model_path: str = 'C:/projects/bookbot/minigpt.pt',
                 vocab_path: str = 'C:/projects/bookbot/minigpt_vocab.json') -> Optional[MiniGPTTrainer]:
    """Load a trained MiniGPT model and return a trainer for generation."""
    try:
        vocab = Vocabulary.load(vocab_path)
        model = MiniGPT(
            vocab_size=vocab.vocab_size,
            dim=64,
            n_layers=4,
            n_heads=4,
            max_len=128
        )
        trainer = MiniGPTTrainer(model, vocab)
        trainer.load(model_path, vocab_path)
        return trainer
    except Exception as e:
        logger.warning(f"Failed to load MiniGPT: {e}")
        return None


class DistilGPT2Generator:
    """
    DistilGPT2-based text generator — drop-in replacement for MiniGPT.
    82M params vs 463K, dramatically better prose quality.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load DistilGPT2 model."""
        if self._loaded:
            return True

        try:
            import torch
            from transformers import GPT2LMHeadModel, GPT2Tokenizer

            logger.info("Loading DistilGPT2...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
            self.model = GPT2LMHeadModel.from_pretrained('distilgpt2')
            self.model.eval()

            # Move to CPU explicitly
            self.model = self.model.to('cpu')

            self._loaded = True
            logger.info("DistilGPT2 loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load DistilGPT2: {e}")
            return False

    def generate_from_prompt(self, prompt: str, max_tokens: int = 80,
                             temperature: float = 0.7) -> str:
        """Generate text from a prompt with repetition penalty."""
        if not self._loaded:
            if not self.load():
                return ''

        try:
            import torch

            # Encode prompt
            inputs = self.tokenizer(prompt, return_tensors='pt')
            input_ids = inputs['input_ids']

            # Generate with repetition penalty
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.3,  # Penalize repetition
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode
            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

            # Clean up: remove the prompt from the output if it's repeated
            if generated.startswith(prompt):
                generated = generated[len(prompt):].strip()

            return generated if len(generated) > 10 else ''

        except Exception as e:
            logger.debug(f"DistilGPT2 generation failed: {e}")
            return ''


def load_distilgpt2() -> Optional[DistilGPT2Generator]:
    """Load DistilGPT2 generator (preferred over MiniGPT)."""
    try:
        generator = DistilGPT2Generator()
        if generator.load():
            return generator
        return None
    except Exception as e:
        logger.warning(f"Failed to load DistilGPT2: {e}")
        return None
