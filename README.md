# DAID-PELS

### Dictionary-Anchored Iterative Deepening with Phonetic-Enhanced Lexical Search

A **conversational AI chatbot** that combines rule-based NLP with neural components to answer questions about books and the world. Runs entirely on CPU with no API calls.

---

## What Is This?

DAID-PELS is a chatbot that:
1. **Reads books** and builds a knowledge base through iterative analysis
2. **Answers questions** about the book content using retrieved passages
3. **Searches Wikipedia** for general knowledge questions
4. **Has natural conversations** — greetings, personal statements, emotional expressions
5. **Maintains context** across conversation turns

```
> hi
Hey! What's on your mind?

> what is rust
Rust is a general-purpose programming language which emphasizes performance,
type safety, concurrency, and memory safety.

> who is elizabeth
[Book passages about Elizabeth Bennet from Pride and Prejudice]

> how many islands does japan have
Japan... consists of four major islands alongside 14,121 smaller islands.

> I love cooking
That's great! Cooking is interesting. Tell me more about it!

> when was the first ufo reported
An unidentified flying object (UFO) is an object or phenomenon seen in the sky...
```

---

## What's New (v2.0)

### Conversational AI
- **Natural conversations** — not just book Q&A
- **Wikipedia integration** — answers any general knowledge question
- **Context tracking** — remembers what was discussed
- **Personal statements** — "I like cooking" → remembers your preferences
- **Emotional expressions** — "I'm feeling happy" → responds appropriately

### Visual Progress
- Loading progress bars for AI model and Wikipedia
- Training progress with time estimates
- Clean, informative output

### Multi-Book Training
- Train on multiple books simultaneously
- 20+ classic books included
- Comprehensive knowledge base

---

## Quick Start (5 minutes)

```bash
# 1. Clone
git clone https://github.com/omgbox/DAID-PELS.git
cd DAID-PELS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install VC++ Redistributable (REQUIRED for PyTorch)
# Download: https://aka.ms/vs/17/release/vc_redist.x64.exe

# 4. Download NLTK data
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('words'); nltk.download('averaged_perceptron_tagger_eng')"

# 5. Download assets (dictionary + sample book)
python download_assets.py

# 6. Train book knowledge (~40 seconds)
python train_pride.py

# 7. Train neural models (~30-40 minutes, optional)
python train_all_models.py

# 8. Chat!
python -m bookbot.main query
```

---

## What's Been Tested

| Feature | Status | Notes |
|---------|--------|-------|
| Book Q&A | ✅ Tested | Works with Pride and Prejudice |
| Wikipedia Q&A | ✅ Tested | General knowledge questions work |
| Conversations | ✅ Tested | Greetings, personal statements, emotions |
| Context tracking | ✅ Tested | Follow-up questions work |
| Multi-book training | ✅ Tested | 20+ books can be trained |
| DistilGPT2 | ✅ Tested | 82M params, good for general queries |
| DistilGPT2 fallback | ✅ Tested | Falls back to Wikipedia when GPT2 fails |
| Progress bars | ✅ Tested | Clean loading visualization |

---

## Use Cases

| Use Case | Example | How It Works |
|----------|---------|--------------|
| **Book comprehension** | "Who is Elizabeth?" | Retrieves passages from the book |
| **General knowledge** | "What is Python?" | Searches Wikipedia |
| **Conversational AI** | "Hi, how are you?" | Natural language responses |
| **Personal assistant** | "I like cooking" | Remembers preferences |
| **Educational tool** | "Explain quantum computing" | Wikipedia-based answers |
| **Historical research** | "Who invented the telephone?" | Wikipedia + book context |
| **Character analysis** | "Tell me about Darcy" | Book-based character info |
| **World knowledge** | "What is the capital of France?" | Wikipedia lookup |

---

## Architecture

### Training Pipeline
```
Raw Text → OCR Normalize → Tokenize → POS Tag → NER → SVO Extract
         → Entity Graph → Coreference → Topic Model → Temporal Reason
         → Convergence Check (KL-divergence)
```

### Query Pipeline
```
User Message → Intent Classification → Route Decision
    ↓
    ├── Book Query → Book DB Search → Return Passages
    ├── General Knowledge → Wikipedia Search → Return Facts
    ├── Personal Statement → Store Preference → Acknowledge
    ├── Emotional → Respond Empathetically
    └── Greeting/Farewell → Social Response
```

---

## Module Map

| Layer | Directory | Modules |
|-------|-----------|---------|
| **Libraries** | `lib/` | `phonetic_matcher`, `word2vec`, `edit_distance`, `fst_engine`, `phonetics`, `gematria`, `anagram`, `text_utils` |
| **Core NLP** | `core/` | `tokenizer`, `pos_tagger`, `pos_guesser`, `ner_extractor`, `svo_extractor`, `coreference`, `entity_graph`, `topic_modeler`, `temporal_reasoner` |
| **Query** | `query/` | `trilateral_bm25`, `query_classifier`, `answer_engine`, `conversation_memory`, `contextual_rewriter`, `response_formatter`, `advanced_answer`, `style_realizer`, `prose_realizer` |
| **Conversational** | `query/` | `conversational_ai`, `conversational_responder`, `personal_statement_handler`, `general_knowledge_retriever`, `user_profile`, `conversation_router`, `response_personalizer` |
| **Neural** | `query/` | `minigpt` (DistilGPT2 82M params), `multi_scorer` (5-head), `svo_scorer`, `self_attention`, `token_attention`, `torch_attention` |
| **Training** | `training/` | `convergence_tracker`, `pass_manager`, `visualizer`, `attention_trainer`, `self_supervised_data` |
| **Database** | `database/` | `db_manager`, `schema.sql` (25 tables) |

---

## What's Novel?

### 1. Dictionary-Anchored Learning
Uses a 175K-entry dictionary as a **semantic scaffold** — definitions provide grounding for the words encountered in the book.

### 2. Phonetic-Enhanced Lexical Search (Trilateral BM25)
Combines BM25 + Soundex + Double Metaphone for robust matching:
- "Darcy" matches "Darsay"
- "Bennet" matches "Bennett"
- OCR errors like "titanic" match "titamc"

### 3. Hybrid Rule-Based + Neural Architecture
- **Rule-based NLP**: SVO extraction, coreference, entity graphs, phonetic matching
- **Neural components**: DistilGPT2 for generation, Word2Vec for similarity
- All running on CPU with no API calls — fully offline

### 4. Wikipedia Integration
Searches Wikipedia for general knowledge questions — no fine-tuning needed.

### 5. Self-Supervised Training
MiniGPT and Word2Vec are trained directly on the book — no external training data needed.

---

## Configuration

All settings in `config.py`. Environment variables override paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKBOT_BOOK_PATH` | `books/pride_and_prejudice_clean.txt` | Training book path |
| `BOOKBOT_DICT_PATH` | `English_dictionary.csv` | Dictionary CSV path |
| `BOOKBOT_DB_PATH` | `bookbot.db` | SQLite database path |
| `BOOKBOT_LOG_LEVEL` | `INFO` | Logging level |

---

## Performance

| Metric | Value |
|--------|-------|
| Book training time (724K chars) | ~40 seconds |
| Neural models training (all) | ~30-40 minutes (CPU) |
| Word2Vec training (5 epochs) | ~2 minutes |
| DistilGPT2 loading | ~2 seconds |
| Wikipedia search | ~1 second |
| Query response time | ~1-2 seconds |
| Database size (20 books) | ~170MB |
| Dictionary entries | 206,956 |
| CMU pronunciations | 134,000+ words |
| DistilGPT2 parameters | 82M |

---

## Database Schema

The system uses SQLite with 25+ tables:

| Table | Purpose |
|-------|---------|
| `definitions` | 206K dictionary entries |
| `sentences` | Tokenized sentences from books |
| `sentence_tokens` | Individual tokens with POS tags |
| `entities` | Named entities |
| `svo_triples` | Subject-Verb-Object relationship triples |
| `knowledge_edges` | Entity relationship graph |
| `coreference_chains` | Pronoun resolution chains |
| `topics` | Topic clusters |
| `user_preferences` | User preferences and facts |
| `learned_knowledge` | Facts learned from conversations |

---

## Project Structure

```
DAID-PELS/
├── main.py                 # Entry point (train/query/stats)
├── pipeline.py             # Training + query orchestration
├── config.py               # All settings (portable paths)
├── train_pride.py          # Fast training script
├── train_all_books.py      # Train on all books
├── train_all_models.py     # Train neural models
├── download_assets.py      # Download dictionary + sample book
├── download_book.py        # Download books from Project Gutenberg
├── requirements.txt        # Python dependencies
├── core/                   # NLP modules
│   ├── tokenizer.py        # pysbd sentence splitting
│   ├── pos_tagger.py       # Batch-optimized NLTK POS
│   ├── ner_extractor.py    # Gazetteer + regex NER
│   ├── svo_extractor.py    # Subject-Verb-Object triples
│   ├── coreference.py      # Pronoun resolution
│   └── entity_graph.py     # Entity relationship graph
├── query/                  # Query pipeline
│   ├── conversational_ai.py    # Main conversational pipeline
│   ├── conversational_responder.py  # Chat responses
│   ├── personal_statement_handler.py  # Handle personal statements
│   ├── general_knowledge_retriever.py  # Wikipedia search
│   ├── user_profile.py     # User preferences
│   ├── trilateral_bm25.py  # Phonetic-enhanced BM25
│   ├── minigpt.py          # DistilGPT2 integration
│   └── ...
├── lib/                    # Standalone utilities
│   ├── phonetic_matcher.py # CMU + Double Metaphone + Jaro-Winkler
│   ├── word2vec.py         # Skip-gram trained on book
│   └── ...
├── training/               # Training management
├── database/               # SQLite schema + manager
├── books/                  # Downloaded books (20+ classics)
└── data/                   # Data files
```

---

## License

MIT License

## Citation

If you use this in research, please cite:

```
DAID-PELS: Dictionary-Anchored Iterative Deepening with
Phonetic-Enhanced Lexical Search. omgbox, 2026.
```
