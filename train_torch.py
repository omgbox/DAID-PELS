"""
Train Token-Level Attention (PyTorch)
Self-supervised training on Pride and Prejudice.
"""
import sys
sys.path.insert(0, 'C:/projects')

from bookbot.database.db_manager import DBManager
from bookbot.lib.word2vec import Word2VecEmbeddings
from bookbot.query.torch_attention import TorchTokenAttention, AttentionTrainer
from bookbot.training.self_supervised_data import SelfSupervisedDataGenerator

print("=== PyTorch Token-Level Attention Training ===\n")

# 1. Load data
print("1. Loading book data...")
db = DBManager('C:/projects/bookbot/bookbot.db')
db.connect()

rows = db.execute("SELECT sentence_id, raw_text FROM sentences WHERE LENGTH(raw_text) > 20")
sentences = [{'sentence_id': r[0], 'text': r[1]} for r in rows]
print("   Sentences:", len(sentences))

rows = db.execute("SELECT entity_id, canonical_name FROM entities")
entities = [{'entity_id': r[0], 'canonical_name': r[1]} for r in rows]
print("   Entities:", len(entities))

rows = db.execute("SELECT subject, verb, object, sentence_id FROM svo_triples LIMIT 5000")
svo_triples = [{'subject': r[0], 'verb': r[1], 'object': r[2], 'sentence_id': r[3]} for r in rows]
print("   SVO triples:", len(svo_triples))

db.disconnect()

# 2. Generate training data
print("\n2. Generating self-supervised training data...")
generator = SelfSupervisedDataGenerator()
training_data = generator.generate_all(sentences, entities, svo_triples)

# 3. Load embeddings
print("\n3. Loading Word2Vec embeddings...")
emb = Word2VecEmbeddings.load('C:/projects/bookbot/word2vec.json')

# 4. Create model
print("4. Creating PyTorch model...")
model = TorchTokenAttention(
    w2v_dim=emb.model.dim,
    dim=64,
    n_layers=2,
    n_heads=4
)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {total_params:,}")

# 5. Train
print("\n5. Training with PyTorch autograd...")
trainer = AttentionTrainer(model, lr=0.001)
trainer.train(training_data, emb, epochs=15, batch_size=8)

# 6. Save
print("\n6. Saving...")
trainer.save(
    'C:/projects/bookbot/torch_attention.pt',
    'C:/projects/bookbot/torch_attention_state.pt'
)

# 7. Test
print("\n7. Testing...")
model.eval()
test_sentences = [
    "Elizabeth felt all the impertinence of her questions, but answered them very composedly.",
    "Darcy was delighted with their engagement.",
    "Lydia was exceedingly fond of him.",
    "Jane looked a little paler than usual.",
    "Mr. Collins was not a sensible man.",
    "Wickham was a charming young man.",
]

for query in ["Who is Elizabeth?", "Who is Darcy?", "Tell me about Lydia"]:
    selected = model.select(test_sentences, query, emb, top_k=3)
    print("\n  Query: %s" % query)
    for score, sent, idx in selected:
        print("    [%.3f] %s" % (score, sent[:60]))

print("\nDone!")
