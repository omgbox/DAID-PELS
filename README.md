# DAID-PELS

### Dictionary-Anchored Iterative Deepening with Phonetic-Enhanced Lexical Search

> **⚠️ EXPERIMENTAL — RESEARCH PROJECT ONLY**
> This is a research prototype, not production software.
> Built for learning and experimentation, not for real-world use.

A **conversational AI chatbot** that combines rule-based NLP with neural components to answer questions about books and the world. Runs entirely on CPU with no API calls.

---

## What Is This?

DAID-PELS is a chatbot that:
1. **Reads books** and builds a knowledge base through iterative analysis
2. **Answers questions** about the book content using retrieved passages
3. **Searches Wikipedia** for general knowledge questions
4. **Has natural conversations** — greetings, personal statements, emotional expressions
5. **Maintains context** across conversation turns
6. **Rewrites responses** naturally using T5 Paraphrase model
7. **Shows source attribution** — tells you where information comes from
8. **Resolves pronouns** — understands "it", "he", "she" from context
9. **Scores response quality** — picks the best response from multiple candidates
10. **Learns on the fly** — 4 neural networks that improve with every query
11. **Web interface** — chat via browser with stats panel
12. **Trained on Wikipedia** — 2,279 learned query→page mappings

```
> hi
Hey! What's on your mind?

> what is rust
Rust is a multi-paradigm, general-purpose programming language emphasizing
performance, type safety, and memory safety.

— Source: Wikipedia

> who created it?
Graydon Hoare created Rust in 2006 while working at Mozilla Research.

— Source: Wikipedia

> tell me about alice
Alice's Adventures in Wonderland is an English children's novel by Lewis Carroll
from 1865 and tells the story of a little girl named Alice who falls into a
fantasy world of anthropomorphic creatures through a rabbit hole.

— Source: Book database

> what are ufos?
An unidentified flying object (UFO) is an object or phenomenon seen in the sky
but not yet identified or explained.

— Source: Wikipedia
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/omgbox/DAID-PELS.git
cd DAID-PELS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download assets
python download_assets.py

# 4. Train on books
python train_pride.py

# 5. Run web interface
cd C:\projects
python run_server.py
# Open http://localhost:5000
```

---

## Neural Networks (4 trained on Wikipedia)

| Component | Architecture | Weights | Training Data |
|-----------|--------------|---------|---------------|
| Topic Extractor | 3-layer (24→256→128→1) | 49,152 | 494 words |
| Wikipedia Mapper | 3-layer (20→512→256→1) | 196,608 | 2,279 mappings |
| Intent Classifier | 3-layer (20→256→128→1) | 49,152 | 2,272 patterns |
| Response Selector | 3-layer (16→128→64→1) | 12,288 | Learned preferences |

---

## Full LLM Pipeline

```
User Message
    ↓
Pronoun Resolution (context from history)
    ↓
Entity Extraction (track entities)
    ↓
Intent Classification (neural network)
    ↓
Topic Extraction (neural network)
    ↓
Wikipedia Search (neural mapper predicts page)
    ↓
Book Search (BM25 retrieval)
    ↓
Combine Sources (Wikipedia + Books)
    ↓
Generate Response:
  ├── T5 Paraphrase (candidate 1)
  ├── T5 Simplify (candidate 2)
  ├── DistilGPT2 generation (candidate 3)
  └── Original text (candidate 4)
    ↓
Neural Response Scoring (picks best)
    ↓
Source Attribution
    ↓
Response
```

---

## Training on Wikipedia

```bash
# Crawl Wikipedia and train neural networks
python wiki_crawler.py

# Each run:
# - Crawls ~250 pages
# - Trains ~2,000 queries
# - Saves to JSON files
```

---

## Web Interface

```bash
cd C:\projects
python run_server.py
# Open http://localhost:5000
```

**Features:**
- Sidebar with session history
- Processing status indicator
- Snackbar notifications (Wikipedia/Books/Local)
- Stats panel (neural networks, memory, queries)
- Mobile responsive design

---

## What's Persistent

| Component | Storage | Survives Restart |
|-----------|---------|------------------|
| Wiki mappings | `wiki_mappings.json` | ✅ Yes |
| Topic scores | `topic_scores.json` | ✅ Yes |
| Intent patterns | `intent_scores.json` | ✅ Yes |
| Response preferences | `response_scores.json` | ✅ Yes |
| Wikipedia facts | `learned_knowledge` table | ✅ Yes |
| Conversation history | `learned_knowledge` table | ✅ Yes |

---

## Disclaimer

**This project is experimental and for research purposes only.**

- Not intended for production use
- Not tested for security or reliability
- May contain bugs or unexpected behavior
- Use at your own risk
- For learning and experimentation only

---

## License

MIT License
