# DAID-PELS

### Dictionary-Anchored Iterative Deepening with Phonetic-Enhanced Lexical Search

A rule-based NLP system that trains on a book and answers questions about it — no LLM, no GPU, no neural networks. All intelligence comes from statistical methods, phonetic matching, and graph analysis.

---

## What Is This?

DAID-PELS is a chatbot that reads a book, builds a knowledge base through iterative analysis, and answers conversational questions about the content. It uses a **dictionary as a semantic scaffold** — the system "understands" words through their definitions, not through neural embeddings.

```
$ python -m bookbot.main query
============================================================
  BOOKBOT - Pride and Prejudice
============================================================
  Loaded 15,743 sentences
  Type your questions, or 'quit' to exit.
  Try: 'Who is Elizabeth?', 'Tell me more about Darcy'
============================================================

> Who is Elizabeth?
Based on the text: Elizabeth were herself, has an exposition, catching her eye.
Connected to: Bennet, Bingley, Catherine, Charlotte.

> What is Pemberley?
Based on the text: Pemberley is a powerful motive, admit his society.
Connected to: Darcy, Elizabeth, Gardiner.

> Tell me about Darcy
Do not you, Darcy?.

> quit
Goodbye!
```

## Quick Start (2 minutes)

```bash
# 1. Clone
git clone https://github.com/omgbox/DAID-PELS.git
cd DAID-PELS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download NLTK data
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('words'); nltk.download('averaged_perceptron_tagger_eng')"

# 4. Download assets (dictionary + sample book)
python download_assets.py

# 5. Train (~60 seconds)
python train_pride.py

# 6. Chat!
python -m bookbot.main query
```

## How to Chat

### Interactive mode (multi-turn conversation)
```bash
python -m bookbot.main query
```
Then type questions at the `>` prompt. The system remembers context across turns.

### Single question mode
```bash
python -m bookbot.main query --single "Who is Elizabeth?"
python -m bookbot.main query --single "What is Pemberley?"
python -m bookbot.main query --single "Tell me about Darcy"
```

### Example conversation
```
> Who is Elizabeth?
Based on the text: Elizabeth were herself, has an exposition.
Connected to: Bennet, Bingley, Catherine, Charlotte.

> What about her sister?
Here's more about Jane: Connected to: Bennet, Bingley, Elizabeth.

> Tell me more about Darcy
Do not you, Darcy?.

> What did he do?
Darcy proposed, Darcy helped, Darcy returned.
```

## Installation

### Prerequisites
- Python 3.10+ (3.11 or 3.12 recommended)
- ~200MB disk space for dictionary + database
- Optional: conda (for pynini FST support)

### Step-by-Step

#### 1. Clone the repository
```bash
git clone https://github.com/omgbox/DAID-PELS.git
cd DAID-PELS
```

#### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

This installs:
- `nltk` — POS tagging, NER, WordNet
- `pysbd` — sentence boundary detection
- `rank-bm25` — BM25Okapi ranking
- `inflect` — singularization/pluralization
- `regex` — enhanced regex
- `pronouncing` — CMU Pronouncing Dictionary (134K words)
- `jellyfish` — Jaro-Winkler, Soundex, Metaphone
- `abydos` — Double Metaphone, phonetic algorithms

#### 3. Download NLTK data
```bash
python -c "
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
"
```

#### 4. Download dictionary and sample book
```bash
python download_assets.py
```

This downloads:
- `English_dictionary.csv` — 175K-entry English dictionary (Word, POS, Definition)
- `books/pride_and_prejudice_clean.txt` — Clean Gutenberg text of Pride and Prejudice

**Manual download** if the script fails:
- Dictionary: https://raw.githubusercontent.com/vijayvamsi28/English-Dictionary/refs/heads/main/English_dictionary.csv
- Book: Download from Project Gutenberg (https://www.gutenberg.org/ebooks/1342) and clean with `python preprocess_gutenberg.py`

#### 5. (Optional) Install pynini for FST support
```bash
conda install -c conda-forge pynini
```

## Usage

### Train on a Book
```bash
# Fast training (Pride and Prejudice, ~60 seconds)
python train_pride.py

# Or use the main entry point
python -m bookbot.main train --passes 1

# Train on your own book
python -m bookbot.main train --book books/my_book.txt

# Train with more passes
python -m bookbot.main train --passes 3
```

### Query
```bash
# Interactive mode (multi-turn with conversation memory)
python -m bookbot.main query

# Single question
python -m bookbot.main query --single "Who is Elizabeth?"
python -m bookbot.main query --single "What is Pemberley?"
python -m bookbot.main query --single "Tell me about Darcy"
```

### Show Database Stats
```bash
python -m bookbot.main stats
```

### Preprocess Gutenberg Books
```bash
python preprocess_gutenberg.py books/raw_book.txt -o books/clean_book.txt
```

## Architecture

### Training Pipeline
```
Raw Text → OCR Normalize → Tokenize → POS Tag → NER → SVO Extract
         → Entity Graph → Coreference → Topic Model → Temporal Reason
         → Convergence Check (KL-divergence)
```

### Query Pipeline
```
User Query → Contextual Rewrite → Classify Intent → Trilateral BM25
           → Answer Assembly (DB fallback) → Confidence Score → Format Response
```

### Module Map

| Layer | Directory | Modules |
|-------|-----------|---------|
| **Libraries** | `lib/` | `phonetic_matcher`, `edit_distance`, `fst_engine`, `phonetics`, `gematria`, `anagram`, `text_utils` |
| **Core NLP** | `core/` | `tokenizer`, `pos_tagger`, `pos_guesser`, `ner_extractor`, `svo_extractor`, `coreference`, `entity_graph`, `topic_modeler`, `temporal_reasoner` |
| **Query** | `query/` | `trilateral_bm25`, `query_classifier`, `answer_engine`, `conversation_memory`, `contextual_rewriter`, `response_formatter` |
| **Training** | `training/` | `convergence_tracker`, `pass_manager`, `visualizer` |
| **Database** | `database/` | `db_manager`, `schema.sql` (25 tables) |

## What's Novel?

### 1. Dictionary-Anchored Learning
Most text comprehension systems learn from the text alone. DAID-PELS uses a 175K-entry dictionary as a **semantic scaffold** — definitions provide grounding for the words encountered in the book. The system links words in the text to their dictionary definitions, creating a semantic layer on top of raw statistics.

### 2. Phonetic-Enhanced Lexical Search (Trilateral BM25)
Standard BM25 matches exact words. Trilateral BM25 adds two phonetic dimensions:

| Signal | Source | Purpose |
|--------|--------|---------|
| **BM25** | TF-IDF ranking | Exact keyword matching |
| **Soundex bitmask** | Consonant skeleton encoding | Fast phonetic similarity |
| **Double Metaphone** | CMU Pronouncing Dictionary (134K words) | Precise sound-alike detection |

This means "Darcy" can match "Darsay", "Bennet" can match "Bennett", and OCR errors like "titanic" can match "titamc".

### 3. POS Guesser from Definition Patterns
Words with empty POS fields in the dictionary get their part-of-speech inferred by analyzing the grammatical structure of their definitions. For example, if a definition follows the pattern "a [adjective] [noun] that...", the word is likely a noun.

### 4. Multi-Sentence Answer Assembly
For "Who is X?" questions, the system queries the database for longer sentences mentioning the entity and combines them into a richer answer, rather than returning a single short fragment.

## Use Cases

| Use Case | Description |
|----------|-------------|
| **Book comprehension chatbot** | Ask questions about Pride and Prejudice, Sherlock Holmes, or any plain text book |
| **OCR-robust search** | Find passages even when words are misspelled or OCR-corrupted |
| **Educational tool** | Students can query classic literature conversationally |
| **Research baseline** | No-LLM baseline for comparing against neural QA systems |
| **Offline NLP** | Runs entirely locally — no API calls, no internet required after training |
| **Low-resource deployment** | No GPU needed, runs on any machine with Python 3 |

## Configuration

All settings in `config.py`. Environment variables override paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKBOT_BOOK_PATH` | `books/pride_and_prejudice_clean.txt` | Training book path |
| `BOOKBOT_DICT_PATH` | `English_dictionary.csv` | Dictionary CSV path |
| `BOOKBOT_DB_PATH` | `bookbot.db` | SQLite database path |
| `BOOKBOT_LOG_LEVEL` | `INFO` | Logging level |

Or use command-line flags:
```bash
python -m bookbot.main train --book books/my_book.txt --dict my_dict.csv --db my.db
```

## Performance

| Metric | Value |
|--------|-------|
| Training time (724K chars) | ~62 seconds |
| POS tagging (15K sentences) | ~15 seconds |
| NER (15K sentences) | ~0.5 seconds |
| Query response time | ~1-2 seconds |
| Database size | ~50MB |
| Dictionary entries | 175,721 |
| CMU pronunciations | 134,000+ words |

## Database Schema

The system uses SQLite with 25 tables:

| Table | Purpose |
|-------|---------|
| `definitions` | 175K dictionary entries |
| `sentences` | Tokenized sentences from the book |
| `sentence_tokens` | Individual tokens with POS tags |
| `entities` | Named entities |
| `svo_triples` | Subject-Verb-Object relationship triples |
| `entity_mentions` | Entity occurrences in sentences |
| `knowledge_edges` | Entity relationship graph |
| `coreference_chains` | Pronoun resolution chains |
| `topics` | Topic clusters |
| `temporal_events` | Time-related events |
| `convergence_log` | Training pass statistics |

Full schema: `database/schema.sql`

## Research

The system implements ideas from:

- **BM25** (Robertson et al., 1994) — probabilistic ranking function
- **Soundex** (American Soundex, 1918) — phonetic encoding
- **Double Metaphone** (Philips, 2000) — ambiguous pronunciation handling
- **KL-divergence** — training convergence measurement
- **Trilateral BM25** — novel combination of BM25 + phonetic signals

Research notes: `research/` directory

## Project Structure

```
DAID-PELS/
├── main.py                 # Entry point (train/query/stats)
├── pipeline.py             # Training + query orchestration
├── pipeline_context.py     # Shared context between modules
├── config.py               # All settings (portable paths)
├── train_pride.py          # Fast training script
├── download_assets.py      # Download dictionary + sample book
├── preprocess_gutenberg.py # Clean Gutenberg texts
├── requirements.txt        # Python dependencies
├── core/                   # NLP modules
│   ├── tokenizer.py        # pysbd sentence splitting
│   ├── pos_tagger.py       # Batch-optimized NLTK POS
│   ├── pos_guesser.py      # POS from definition patterns
│   ├── ner_extractor.py    # Gazetteer + regex NER
│   ├── svo_extractor.py    # Subject-Verb-Object triples
│   ├── coreference.py      # Pronoun resolution
│   ├── entity_graph.py     # Entity relationship graph
│   └── ...
├── query/                  # Query pipeline
│   ├── trilateral_bm25.py  # Phonetic-enhanced BM25
│   ├── answer_engine.py    # Multi-sentence assembly
│   ├── conversation_memory.py  # Multi-turn context
│   └── ...
├── lib/                    # Standalone utilities
│   ├── phonetic_matcher.py # CMU + Double Metaphone + Jaro-Winkler
│   ├── fst_engine.py       # Pynini FST support
│   └── ...
├── training/               # Training management
├── database/               # SQLite schema + manager
├── books/                  # Place your books here
├── research/               # Research notes
└── docs/                   # Full specification
```

## License

MIT License

## Citation

If you use this in research, please cite:

```
DAID-PELS: Dictionary-Anchored Iterative Deepening with
Phonetic-Enhanced Lexical Search. omgbox, 2026.
```
