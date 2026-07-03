"""
Train MiniGPT on Pride and Prejudice
~2M param model that learns to write like Jane Austen.
"""
import sys
import torch
sys.path.insert(0, 'C:/projects')

from bookbot.query.minigpt import MiniGPT, Vocabulary, MiniGPTTrainer

print("=== MiniGPT Training ===\n")

# 1. Load book
print("1. Loading book...")
with open('C:/projects/books/pride_and_prejudice_clean.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print("   Characters:", len(text))
print("   Words:", len(text.split()))

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
print(f"   ~{params * 4 / 1024 / 1024:.1f} MB in float32")

# 3. Train
print("\n3. Training...")
trainer = MiniGPTTrainer(model, vocab, lr=3e-4)
trainer.train(text, epochs=25, seq_len=64, batch_size=32)

# 4. Save
print("\n4. Saving...")
torch.save(model.state_dict(), 'C:/projects/bookbot/minigpt.pt')
vocab.save('C:/projects/bookbot/minigpt_vocab.json')
print("   Saved model + vocab")

# 5. Test generation
print("\n" + "=" * 60)
print("  GENERATION TESTS")
print("=" * 60)
prompts = [
    "Elizabeth felt",
    "Darcy was",
    "Lydia is a",
    "Jane looked",
    "He said to her",
    "She had never",
    "The young woman",
    "In the morning",
]

for prompt in prompts:
    output = trainer.generate_from_prompt(prompt, max_tokens=30, temperature=0.8)
    print("\n  Prompt:  '%s'" % prompt)
    print("  Output:  '%s'" % output)

# 6. Test with lower temperature
print("\n" + "=" * 60)
print("  LOWER TEMPERATURE (more coherent)")
print("=" * 60)
for prompt in ["Elizabeth felt", "Darcy said", "She was"]:
    output = trainer.generate_from_prompt(prompt, max_tokens=40, temperature=0.5)
    print("\n  Prompt:  '%s'" % prompt)
    print("  Output:  '%s'" % output)

print("\n" + "=" * 60)
print("  DONE")
print("=" * 60)
