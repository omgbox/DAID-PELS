"""
Initialize and save Token-Level Attention model.
Usage: python train_token_attention.py
"""
import sys
import json
import numpy as np
sys.path.insert(0, r'C:\projects')

from bookbot.lib.word2vec import Word2VecEmbeddings
from bookbot.query.token_attention import TokenLevelAttention


def train_token_attention():
    print("=" * 60)
    print("  INITIALIZING TOKEN-LEVEL ATTENTION")
    print("=" * 60)

    # Load Word2Vec embeddings
    print("\n  Loading Word2Vec embeddings...")
    try:
        embeddings = Word2VecEmbeddings.load(
            'C:/projects/bookbot/word2vec.json',
            'C:/projects/bookbot/word2vec_vocab.json'
        )
        print(f"  Loaded: {len(embeddings.vocab.word2idx):,} words")
    except Exception as e:
        print(f"  Error loading Word2Vec: {e}")
        print("  Train Word2Vec first: python train_word2vec.py")
        return

    # Create model
    print("\n  Creating TokenLevelAttention model...")
    model = TokenLevelAttention(
        embeddings=embeddings,
        dim=100,  # Match Word2Vec dimensions
        n_layers=2,
        n_heads=4,
        max_tokens=2048
    )

    # Save
    print("  Saving...")
    model.save('C:/projects/bookbot/token_attention.json')
    print(f"  Saved: token_attention.json")

    # Verify
    print("\n  Verifying...")
    loaded = TokenLevelAttention.load('C:/projects/bookbot/token_attention.json', embeddings)
    print(f"  Loaded successfully: dim={loaded.dim}, layers={loaded.n_layers}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == '__main__':
    train_token_attention()
