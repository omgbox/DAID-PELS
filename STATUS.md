# BookBot Status — July 2026

## Project Overview
Conversational chatbot with book QA capabilities. Can handle greetings, personal statements, emotional expressions, general knowledge questions (via Wikipedia), and book-specific queries. Hybrid pipeline: rule-based NLP + neural components (Word2Vec, self-attention, multi-dim scorer, style realizer, DistilGPT2).

## Environment
- **Python**: 3.14.6 (system, `C:\Users\voo\AppData\Local\Python\bin\python.exe`)
- **OS**: Windows, PowerShell 5.1
- **DB**: `C:\projects\bookbot\bookbot.db` (SQLite, WAL mode)
- **Key deps installed**: `torch` 2.12.1+cpu, `numpy` 2.5.0, `nltk`, `pysbd`, `rank-bm25`, `inflect`, `pronouncing`, `jellyfish`, `abydos`, `wikipedia-api` 0.15.0
- **VC++ Redistributable**: **REQUIRED** — install from https://aka.ms/vs/17/release/vc_redist.x64.exe
  - Without this, PyTorch will fail to load DLLs

## Python 3.14 Compatibility

### Removed Dependencies
- `regex` package removed — not needed, Python's built-in `re` module is used everywhere
- All code uses `import re` (standard library)

### Why?
- `regex` has no pre-built wheels for Python 3.14
- Building from source requires Visual C++ Build Tools
- The codebase already uses Python's built-in `re` module exclusively

## Installation (Quick Start)

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

# 7. Train neural models (~30-40 minutes)
python train_all_models.py

# 8. Chat!
python -m bookbot.main query
```

## What's Done ✅

### Training Pipeline
- Full pipeline: OCR → tokenize → POS → NER → SVO → entity graph → coreference → topics → temporal → convergence
- DB: 3958 sentences, 29180 SVO triples, 143 entities, 1020 knowledge edges
- Training takes ~40s

### Neural Components
| Component | File | Weights | Training |
|-----------|------|---------|----------|
| Word2Vec skip-gram (100-dim) | `lib/word2vec.py` | `word2vec.json`, `word2vec_vocab.json` | `train_all_models.py` |
| Multi-Dim Scorer (5-head, 38-dim) | `query/multi_scorer.py` | `multi_scorer.json` | `train_all_models.py` |
| DistilGPT2 (82M params) | `query/minigpt.py` | Pre-trained (auto-download) | Fine-tune optional |
| SVO Quality Scorer (2-layer MLP) | `query/svo_scorer.py` | `svo_scorer.json` | — |
| Self-Attention Selector (2 layers, 4 heads) | `query/self_attention.py` | `self_attention.json` | — |
| Token-Level Attention | `query/token_attention.py` | `token_attention.json` | — |
| PyTorch Token Attention | `query/torch_attention.py` | `torch_attention.pt`, `torch_attention_state.pt` | — |
| T5-small (paraphraser, abandoned — too small) | `t5_small/` | tokenizer + model files | — |

### Query Pipeline (wired into `pipeline.py`)
- Step 1: Contextual query rewriting (follow-up detection)
- Step 2: Query classification (intent detection)
- Step 3: Entity extraction (with DB case-insensitive fallback)
- Step 4: Trilateral BM25 retrieval
- Step 5: Structured knowledge retrieval (SVO, relationships, descriptions)
- Step 6: Answer engine
- Step 6b: Advanced prose generation (4-option output: Professional/Formal/Neutral/MiniGPT)
- Step 7: Confidence scoring
- Step 8: Iterative refinement (skipped if good answer already synthesized)
- Step 9: Response formatting (expanded `already_good` keywords to preserve prose)
- Step 10: Conversation memory update

### Answer Generation
- `query/advanced_answer.py`: orchestrates scoring + styling + description handler + MiniGPT
- `query/prose_realizer.py`: selects/formats original book sentences (scores by entity-as-subject, action verbs)
- `query/style_realizer.py`: 3 styles — Professional (concise), Formal (literary), Neutral (reference)
- `query/minigpt.py`: GPT-style transformer generating prose from prompts (Option 4)
- `query/response_formatter.py`: expanded `already_good` keyword list so prose answers aren't overwritten
- `query/answer_engine.py`: for "Who is X?" queries, entity-name boosting

### Bugs Fixed (this session)
- Removed duplicate `save` method in `training/attention_trainer.py` (was overriding head save)
- Fixed MiniGPT loading in `advanced_answer.py` — used classmethod `MiniGPT.load()` instead of broken instance method
- Added `load` and `generate_from_text` methods to `MiniGPT` class for pipeline integration
- Removed unused `regex` dependency (Python 3.14 compatibility)
- Fixed `_simple_rank` function stuck inside `_extract_traits` in `advanced_answer.py`
- Added visual progress bars to Word2Vec and MultiDimScorer training
- Replaced MiniGPT (463K params) with DistilGPT2 (82M params) for dramatically better prose
- Improved sentence selection: filter fragments, boost complete thoughts, remove dialogue
- Improved action extraction: filter fragments with pronouns/determiners, stop at conjunctions
- Improved DistilGPT2 prompting: use source sentences as grounding, add repetition penalty
- Improved trait extraction: only meaningful character traits, require context patterns
- Fixed emotional expression regex patterns (needed `\s*` between groups)
- Added conversational chatbot capabilities (Phases 1-3 complete)

## What's Broken / Incomplete ❌

### 1. Token attention training produces uniform scores
- Full backprop through attention layers needs larger training data
- `torch_attention.py` has PyTorch autograd version but scores are uniform
- Self-supervised training from book alone insufficient

### 2. DistilGPT2 can hallucinate
- May generate content not from the book (Wikipedia-style text)
- Mitigated by grounding prompts with source sentences and filtering hallucinated content
- Could be improved with fine-tuning on the book

## Conversational Chatbot Features (COMPLETE)

### What Works Now
- **Greetings**: "Hello!" → "Hi there! What would you like to talk about?"
- **Personal statements**: "I like cooking" → "Noted — you're into cooking. How did you get started?"
- **Emotional expressions**: "I'm feeling happy" → "That's great! Happy is a beautiful feeling."
- **General knowledge**: "What is the capital of France?" → Wikipedia answer
- **Book QA**: "Who is Elizabeth?" → Book-based answer (existing functionality)
- **User profile**: Stores name, preferences, facts about the user
- **Learning**: Learns from conversations, stores knowledge for future reference
- **Personalization**: References user preferences and conversation history

### New Files Created
| File | Purpose |
|------|---------|
| `query/conversation_router.py` | Routes intents to appropriate handlers |
| `query/conversational_responder.py` | Handles greetings, farewells, emotional expressions |
| `query/personal_statement_handler.py` | Detects and stores user preferences |
| `query/user_profile.py` | Stores user preferences and facts |
| `query/general_knowledge_retriever.py` | Wikipedia + local knowledge base retrieval |
| `query/response_personalizer.py` | Personalizes responses based on user context |

### New Dependencies
- `wikipedia-api>=0.6.0` — Wikipedia access for open-domain questions

### Intent Types (expanded from 9 to 17+)
- GREETING, FAREWELL, HELP
- PERSONAL_STATEMENT, EMOTIONAL, OPINION
- GENERAL_KNOWN (Wikipedia)
- Book intents: DEFINITIONAL, FACTUAL, CAUSAL, TEMPORAL, COMPARATIVE, SUMMARIZATION, EXPLANATORY, LISTING

## Next Steps (in order)

1. ✅ All phases complete - conversational chatbot is fully functional
2. (Optional) Port token-level attention training to proper PyTorch with larger dataset
3. (Optional) Cross-book training for better generalization
4. (Optional) Add more knowledge sources (DuckDuckGo, local KB expansion)

## Key Gotchas
- `find_entity_actions()` returns tuples `(subject, verb, object, confidence, sentence_id, raw_text)`, not dicts — pipeline code handles both
- SVO triples have `sentence_id` linking to `sentences` table — key for prose retrieval
- MiniGPT vocab size is 4000 (BPE-like), trained on book text
- `response_formatter.py` has `already_good` keyword list — don't let refiner overwrite good prose
- `pipeline.py` line 398-399: `has_synthesized` check skips refinement for good answers
- MiniGPT uses `generate_from_text()` method (not raw `generate()` which takes token indices)
- Junction at `C:\projects\bookbot` points to `C:\projects\DAID-PELS` — run queries from `C:\projects`
- Visual progress bars added to Word2Vec and MultiDimScorer training

## File Map

```
bookbot/
├── pipeline.py                  # Main orchestration (512 lines)
├── train_pride.py               # Book training script (~40s)
├── train_all_models.py          # Train all neural models (~30-40 min)
├── train_minigpt.py             # MiniGPT training script
├── query/
│   ├── minigpt.py               # GPT-style transformer (463K params) ✅ TRAINED + INTEGRATED
│   ├── advanced_answer.py       # Orchestrates scoring + styling + MiniGPT
│   ├── multi_scorer.py          # 5-head neural scorer
│   ├── svo_scorer.py            # SVO quality scorer
│   ├── prose_realizer.py        # Book sentence selection
│   ├── style_realizer.py        # 3-style output
│   ├── self_attention.py        # Sentence-level attention
│   ├── token_attention.py       # Token-level attention
│   ├── torch_attention.py       # PyTorch port with autograd
│   ├── response_formatter.py    # Final formatting
│   ├── answer_engine.py         # "Who is X?" handler
│   ├── trilateral_bm25.py       # Phonetic-enhanced BM25
│   └── ...
├── lib/
│   └── word2vec.py              # Skip-gram trained on book
├── training/
│   ├── self_supervised_data.py  # Auto-generate training pairs
│   └── attention_trainer.py     # Token attention trainer (duplicate save fixed)
├── minigpt.pt                   # Trained MiniGPT weights (2.78MB)
├── minigpt_vocab.json           # MiniGPT vocabulary
├── word2vec.json                # Trained embeddings
├── word2vec_vocab.json          # Word2Vec vocabulary
├── multi_scorer.json            # Trained multi-dim scorer
├── svo_scorer.json              # Trained SVO scorer
├── self_attention.json          # Trained self-attention
├── token_attention.json         # Trained token attention
├── torch_attention.pt           # PyTorch attention weights
└── bookbot.db                   # SQLite database
```
