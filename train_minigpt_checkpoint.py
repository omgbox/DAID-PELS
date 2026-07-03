"""
Train MiniGPT with checkpointing - saves after each epoch.
"""
import sys
import torch
import time
sys.path.insert(0, 'C:/projects')

from bookbot.query.minigpt import MiniGPT, Vocabulary, MiniGPTTrainer

print("=== MiniGPT Training (with checkpoints) ===\n")

MODEL_PATH = 'C:/projects/bookbot/minigpt.pt'
VOCAB_PATH = 'C:/projects/bookbot/minigpt_vocab.json'

# 1. Load book
print("1. Loading book...")
with open('C:/projects/books/pride_and_prejudice_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(f"   Characters: {len(text):,}")
print(f"   Words: {len(text.split()):,}")

# 2. Create model
print("\n2. Creating model...")
vocab = Vocabulary(vocab_size=4000)
model = MiniGPT(
    vocab_size=4000,
    dim=64,
    n_layers=4,
    n_heads=4,
    max_len=128
)
params = model.count_parameters()
print(f"   Parameters: {params:,}")

# 3. Train with checkpointing
print("\n3. Training (25 epochs, saving after each)...")
trainer = MiniGPTTrainer(model, vocab, lr=3e-4)

start_time = time.time()
for epoch in range(25):
    # Train one epoch at a time
    trainer.train(text, epochs=1, seq_len=64, batch_size=32)
    
    # Save checkpoint after each epoch
    torch.save(model.state_dict(), MODEL_PATH)
    vocab.save(VOCAB_PATH)
    
    elapsed = time.time() - start_time
    print(f"   [Checkpoint saved after epoch {epoch+1}] Total time: {elapsed:.0f}s")

print(f"\n4. Training complete! Total time: {time.time() - start_time:.0f}s")
print(f"   Model saved to: {MODEL_PATH}")
print(f"   Vocab saved to: {VOCAB_PATH}")

# 5. Test generation
print("\n" + "=" * 60)
print("  GENERATION TESTS")
print("=" * 60)
prompts = [
    "Elizabeth felt",
    "Darcy was",
    "She said",
    "He looked",
]

for prompt in prompts:
    output = trainer.generate_from_prompt(prompt, max_tokens=30, temperature=0.8)
    print(f"\n  Prompt:  '{prompt}'")
    print(f"  Output:  '{output}'")

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
