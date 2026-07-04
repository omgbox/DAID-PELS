"""Train Word2Vec on the book's tokenized sentences."""
import sys
sys.path.insert(0, r'C:\projects')

from bookbot.config import DATABASE_PATH
from bookbot.database.db_manager import DBManager
from bookbot.lib.word2vec import Word2VecTrainer


def train_word2vec():
    print("Loading tokens from database...")
    db = DBManager(DATABASE_PATH)
    db.connect()

    # Get all tokens grouped by sentence
    rows = db.execute(
        "SELECT sentence_id, token FROM sentence_tokens "
        "WHERE is_punctuation = 0 ORDER BY sentence_id, position"
    )

    # Group tokens into sentences
    sentences = []
    current_sid = None
    current_tokens = []
    for sid, token in rows:
        if sid != current_sid:
            if current_tokens:
                sentences.append(current_tokens)
            current_sid = sid
            current_tokens = [token.lower()]
        else:
            current_tokens.append(token.lower())
    if current_tokens:
        sentences.append(current_tokens)

    db.disconnect()

    print(f"Loaded {len(sentences)} sentences")
    print("Training Word2Vec (this may take a minute)...")

    trainer = Word2VecTrainer(dim=100, window=5, min_count=2, epochs=5)
    model, vocab = trainer.train(sentences, save_path='C:/projects/bookbot/word2vec.json')
    print(f"Saved word2vec.json ({vocab.total_words} words, {len(vocab.word2idx)} vocab)")


if __name__ == '__main__':
    train_word2vec()
