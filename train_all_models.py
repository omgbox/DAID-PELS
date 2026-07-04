"""
Train All Neural Models - MiniGPT, Word2Vec, MultiDimScorer
Usage: python train_all_models.py
"""
import sys
import time
import json
import logging
sys.path.insert(0, r'C:\projects')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('bookbot')

from bookbot.config import BOOK_PATH, DATABASE_PATH
from bookbot.database.db_manager import DBManager


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def load_book_text():
    """Load the book text with progress."""
    print("\n  Loading book...")
    start = time.time()
    with open(BOOK_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    elapsed = time.time() - start
    words = len(text.split())
    print(f"  Loaded {len(text):,} chars, {words:,} words in {elapsed:.1f}s")
    return text


def load_sentences_from_db():
    """Load tokenized sentences from database with progress."""
    print("\n  Loading sentences from database...")
    start = time.time()
    db = DBManager(DATABASE_PATH)
    db.connect()

    rows = db.execute(
        "SELECT sentence_id, raw_text FROM sentences "
        "WHERE LENGTH(raw_text) > 20 ORDER BY sentence_id"
    )
    sentences = [r[1] for r in rows]

    db.disconnect()
    elapsed = time.time() - start
    print(f"  Loaded {len(sentences):,} sentences in {elapsed:.1f}s")
    return sentences


def train_word2vec(book_text):
    """Train Word2Vec embeddings with visual progress."""
    print_header("TRAINING WORD2VEC")

    from bookbot.lib.word2vec import Word2VecTrainer, Vocabulary

    # Tokenize
    print("\n  Tokenizing...")
    start = time.time()
    words = book_text.lower().split()
    words = [w.strip('.,;:!?""\n\t') for w in words if w.strip('.,;:!?""\n\t')]
    print(f"  Tokenized {len(words):,} words in {time.time()-start:.1f}s")

    # Build vocab
    print("  Building vocabulary...")
    start = time.time()
    vocab = Vocabulary(min_count=2)
    vocab.build(words)
    print(f"  Vocabulary: {len(vocab.word2idx):,} words in {time.time()-start:.1f}s")

    # Create sentence-like chunks for training
    chunk_size = 50
    sentences = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    print(f"  Training chunks: {len(sentences):,}")

    # Train
    trainer = Word2VecTrainer(dim=100, window=5, min_count=2, epochs=5)
    model, vocab = trainer.train(sentences, save_path='C:/projects/bookbot/word2vec.json')
    return model, vocab


def train_minigpt(book_text):
    """Train MiniGPT model with visual progress."""
    print_header("TRAINING MINIGPT")

    import torch
    from bookbot.query.minigpt import MiniGPT, MiniGPTTrainer, Vocabulary

    # Create model
    print("\n  Creating model...")
    vocab = Vocabulary(vocab_size=4000)
    model = MiniGPT(vocab_size=4000, dim=64, n_layers=4, n_heads=4, max_len=128)

    params = model.count_parameters()
    print(f"  Parameters: {params:,}")
    print(f"  Size: ~{params * 4 / 1024 / 1024:.1f} MB (float32)")

    # Train
    trainer = MiniGPTTrainer(model, vocab, lr=3e-4)
    trainer.train(book_text, epochs=25, seq_len=64, batch_size=32)

    # Save
    print("\n  Saving...")
    trainer.save('C:/projects/bookbot/minigpt.pt', 'C:/projects/bookbot/minigpt_vocab.json')
    print(f"  Saved: minigpt.pt, minigpt_vocab.json")

    # Test generation
    print("\n  " + "-" * 50)
    print("  GENERATION TESTS")
    print("  " + "-" * 50)
    prompts = ["Elizabeth felt", "Darcy was", "She said"]
    for prompt in prompts:
        output = trainer.generate_from_prompt(prompt, max_tokens=20, temperature=0.7)
        print(f"    '{prompt}' -> '{output}'")

    return trainer


def train_multi_scorer(sentences):
    """Train MultiDimScorer with visual progress."""
    print_header("TRAINING MULTI-DIM SCORER")

    from bookbot.query.multi_scorer import MultiDimScorer

    scorer = MultiDimScorer()

    # Create training data
    print("\n  Preparing training data...")
    start = time.time()
    training_sentences = []
    training_labels = []

    for sent in sentences[:3000]:
        words = sent.split()
        lower = sent.lower()

        # Find potential entities (capitalized words)
        entity = None
        for word in words:
            if word[0:1].isupper() and len(word) > 2 and word.lower() not in {
                'the', 'and', 'but', 'for', 'not', 'you', 'that', 'this',
                'with', 'from', 'they', 'have', 'been', 'were', 'said',
                'mr', 'mrs', 'miss', 'she', 'her', 'his', 'him',
            }:
                entity = word
                break

        if not entity:
            continue

        # Heuristic label based on sentence quality
        score = 0.5  # base

        # Boost for good sentences
        if lower.startswith(entity.lower()):
            score += 0.2  # entity as subject
        if 8 <= len(words) <= 25:
            score += 0.15  # good length
        if ',' in sent:
            score += 0.05  # complex sentence
        if any(w in lower for w in ['felt', 'looked', 'turned', 'smiled', 'said', 'replied']):
            score += 0.1  # action verbs

        # Penalize bad sentences
        if len(words) < 5:
            score -= 0.3  # too short
        if len(words) > 40:
            score -= 0.1  # too long

        score = max(0.0, min(1.0, score))
        training_sentences.append(sent)
        training_labels.append(score)

    print(f"  Prepared {len(training_sentences)} samples in {time.time()-start:.1f}s")

    # Train
    scorer.train(training_sentences, training_labels, entity="Elizabeth",
                 query="Who is Elizabeth?", intent="DEFINITIONAL", epochs=20, lr=0.01)

    # Save
    print("\n  Saving...")
    scorer.save('C:/projects/bookbot/multi_scorer.json')
    print(f"  Saved: multi_scorer.json")
    return scorer


def main():
    print_header("BOOKBOT - TRAIN ALL MODELS")
    print("\n  This will train all neural components:")
    print("    1. Word2Vec (word embeddings)")
    print("    2. MiniGPT (prose generation)")
    print("    3. MultiDimScorer (sentence scoring)")

    total_start = time.time()

    # Load data
    book_text = load_book_text()
    sentences = load_sentences_from_db()

    # Train each model
    train_word2vec(book_text)
    train_minigpt(book_text)
    train_multi_scorer(sentences)

    # Summary
    total = time.time() - total_start
    print_header("TRAINING COMPLETE")
    print(f"\n  Total time: {total:.0f}s ({total/60:.1f} minutes)")
    print("\n  Model files saved to C:/projects/bookbot/:")
    print("    - word2vec.json")
    print("    - minigpt.pt + minigpt_vocab.json")
    print("    - multi_scorer.json")
    print("\n  Restart your query to use the new models!")
    print()


if __name__ == '__main__':
    main()
