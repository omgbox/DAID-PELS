# DAID-PELS

### Dictionary-Anchored Iterative Deepening with Phonetic-Enhanced Lexical Search

A rule-based NLP system that trains on a book and answers questions about it — no LLM, no GPU, no neural networks. All intelligence comes from statistical methods, phonetic matching, and graph analysis.

---

## What Is This?

DAID-PELS is a chatbot that reads a book, builds a knowledge base through iterative analysis, and answers conversational questions about the content. It uses a **dictionary as a semantic scaffold** — the system "understands" words through their definitions, not through neural embeddings.

```
$ python -m bookbot.main query
> Who is Elizabeth?
According to the book, Miss Elizabeth herself, were, I am inclined to think,
rather hard on him... When Jane and Elizabeth were alone, the former, who
had been cautious in.

> What is Pemberley?
With all my heart: I will buy Pemberley itself, if Darcy will sell it...
impossible for her to see the word without thinking of Pemberley and its
Pemberley House, situated on the opposite side of the valley.
```

## Architecture

The system has two main pipelines:

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

## Quick Start

### Prerequisites
- Python 3.10+
- conda (for pynini support, optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/omgbox/DAID-PELS.git
cd DAID-PELS

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords'); nltk.download('words')"

# Optional: install pynini for FST support
conda install -c conda-forge pynini
```

### Train on a Book

```bash
# Train on Pride and Prejudice (default)
python train_pride.py

# Or use the main entry point
python -m bookbot.main train --passes 1
```

Training takes ~60 seconds for a 724K-character book (15,743 sentences, 276 entities).

### Query

```bash
# Interactive mode
python -m bookbot.main query

# Single question
python -m bookbot.main query --single "Who is Elizabeth?"
```

### Use Your Own Book

```bash
# Place a plain text file at books/my_book.txt, then:
python -m bookbot.main train --book books/my_book.txt
```

For Gutenberg books, use the preprocessor:
```bash
python preprocess_gutenberg.py books/pride_and_prejudice.txt books/pride_and_prejudice_clean.txt
```

## Database Schema

The system uses SQLite with 25 tables:

| Table | Purpose |
|-------|---------|
| `definitions` | 175K dictionary entries |
| `sentences` | Tokenized sentences from the book |
| `sentence_tokens` | Individual tokens with POS tags |
| `entities` | Named entities (276 for Pride and Prejudice) |
| `svo_triples` | Subject-Verb-Object relationship triples |
| `entity_mentions` | Entity occurrences in sentences |
| `knowledge_edges` | Entity relationship graph |
| `coreference_chains` | Pronoun resolution chains |
| `topics` | Topic clusters |
| `temporal_events` | Time-related events |
| `convergence_log` | Training pass statistics |

Full schema: `database/schema.sql`

## Configuration

All settings in `config.py`. Environment variables override paths:

| Variable | Default | Description |
|----------|---------|-------------|
| `BOOKBOT_BOOK_PATH` | `books/pride_and_prejudice_clean.txt` | Training book path |
| `BOOKBOT_DICT_PATH` | `English_dictionary.csv` | Dictionary CSV path |
| `BOOKBOT_DB_PATH` | `bookbot.db` | SQLite database path |
| `BOOKBOT_LOG_LEVEL` | `INFO` | Logging level |

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
bookbot/
├── main.py                 # Entry point (train/query/stats)
├── pipeline.py             # Training + query orchestration
├── pipeline_context.py     # Shared context between modules
├── config.py               # All settings
├── train_pride.py          # Fast training script
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
