# BookBot Architecture Research & TODO
# Transformation: Book QA System → General Conversational Chatbot

## Current Architecture Summary

The system has **decorative neural components** — Word2Vec, self-attention, and MiniGPT exist but aren't properly integrated:

| Component | Status | Issue |
|-----------|--------|-------|
| Word2Vec | Trained | Never used in retrieval (BM25 is purely lexical) |
| Self-attention | Random weights | Only the scoring head is trained, not the attention layers |
| DistilGPT2 | Pre-trained | Replaced MiniGPT (82M params vs 463K) |
| Style Realizer | Rule-based | Hardcoded word lists, no learning |
| Idiom Detector | Binary lookup | Exact hash-table match, no fuzzy matching |
| Sentiment Tracker | Missing | No per-entity emotional trajectory |

The system's actual intelligence comes from the **rule-based pipeline**: BM25 + SVO extraction + entity graph + templates. The neural components are loaded but don't contribute meaningfully to answer quality.

---

## CHATBOT TRANSFORMATION PHASES

### Phase 1: Expand Intent Classifier + Conversation Router (Foundation)
**Goal**: Make the system recognize that not every input is a book question.

**Files to modify:**
- `query/query_classifier.py` — Add 8+ new intent types
- `pipeline.py` — Add routing layer at top of `run_query()`
- `config.py` — Add conversational config section

**New files to create:**
- `query/conversation_router.py` — Routes intents to handlers

**New intents:**
- GREETING, FAREWELL
- PERSONAL_STATEMENT ("I like gardening")
- EMOTIONAL ("I'm feeling sad")
- OPINION ("What do you think about X?")
- GENERAL_KNOWN ("What is the capital of France?")
- HELP

**Status**: IN PROGRESS

---

### Phase 2: Personal Statement Handler + User Profile
**Goal**: Handle "I like gardening" and remember it.

**Files to modify:**
- `database/schema.sql` — Add new tables
- `database/db_manager.py` — Add user preference methods

**New files to create:**
- `query/personal_statement_handler.py` — Detect and respond to personal statements
- `query/user_profile.py` — Store/retrieve user preferences

**New DB tables:**
- `user_preferences` — Store user likes/dislikes
- `user_facts` — Store personal facts
- `learned_knowledge` — Store facts from conversations

**Status**: PENDING

---

### Phase 3: General Knowledge Integration (Wikipedia)
**Goal**: Answer questions not covered by the book.

**Files to modify:**
- `pipeline.py` — Add knowledge retrieval path
- `requirements.txt` — Add wikipedia-api

**New files to create:**
- `query/general_knowledge_retriever.py` — Wikipedia + local KB retrieval
- `query/conversational_responder.py` — General conversation responses
- `query/knowledge_base.py` — General fact store

**Knowledge sources:**
1. Wikipedia API (highest quality)
2. Local knowledge base (learned from conversations)
3. DistilGPT2 generation (fallback)

**Status**: PENDING

---

### Phase 4: Learning from Conversations
**Goal**: The chatbot improves over time.

**Files to modify:**
- `query/conversation_memory.py` — Add learning capability
- `query/general_knowledge_retriever.py` — Search learned facts

**New capabilities:**
- Extract and store personal facts from user statements
- Store factual Q&A pairs for future reference
- Confidence scoring for learned facts

**Status**: PENDING

---

### Phase 5: Polish and Integration
**Goal**: Seamless experience across all conversation types.

**Files to modify:**
- `main.py` — Remove book-specific branding
- `query/response_formatter.py` — Generalize source formatting
- `query/response_personalizer.py` — Personalize with user preferences

**New files to create:**
- `query/response_personalizer.py` — Inject user preferences into responses

**Features:**
- Graceful degradation (book → knowledge → templates → "I don't know")
- Response personalization using user profile
- Update UI to be generic (not Pride and Prejudice specific)

**Status**: PENDING

---

## ARCHITECTURE DECISIONS

### Decision 1: Routing vs. Blending
Route early based on intent. Don't run full book pipeline for "I like gardening."

### Decision 2: Keep Book Pipeline Intact
Don't modify existing book-QA code. Create new modules alongside and add routing.

### Decision 3: Database Extension
Add new tables to existing SQLite. Keep everything in one place.

### Decision 4: Model Upgrade Path
DistilGPT2 for now. Consider `microsoft/DialoGPT-medium` or `facebook/blenderbot-400M-distill` later.

### Decision 5: Graceful Degregadation
Book → Knowledge → Templates → "I don't know" with follow-up suggestion.

---

## RESEARCH QUESTIONS (Original)

### 1. Why don't the attention layers learn?
- `self_attention.py` and `token_attention.py` have Q/K/V projections that are NEVER updated
- Only the scoring head (8 weights) is trained via hinge loss
- **Question:** Is this by design (frozen feature extractor) or an oversight?

### 2. Why are Word2Vec embeddings unused in retrieval?
- `word2vec.json` is trained and loaded
- But `trilateral_bm25.py` is purely lexical — no embedding similarity
- **Question:** Was semantic retrieval attempted and removed?

### 3. Why is MiniGPT so small?
- 463K params, dim=64, 4 layers, vocab=4000
- **Replaced with DistilGPT2 (82M params)**

### 4. Why no subword embeddings?
- Docstring claims subword support but code has none
- **Question:** Was this planned but never implemented?

### 5. Why is the style realizer rule-based?
- Hardcoded word lists, no learning from book
- **Question:** Was a learned style transfer model considered?

### 6. Why no discourse coherence model?
- Ordering head is never trained
- **Question:** How to learn discourse patterns from a single book?

### 7. Why is idiom detection binary?
- Exact hash-table lookup only
- **Question:** Was fuzzy matching attempted?

### 8. Why no sentiment/tonal tracking?
- No per-entity emotional trajectory
- **Question:** Was this considered in the original design?

---

## TOP 5 IMPROVEMENTS (Impact Order)

### 1. Semantic Retrieval — Biggest Impact
BM25 is purely lexical. Word2Vec embeddings exist but are never used in retrieval.

### 2. Train Attention End-to-End
Self-attention weights are randomly initialized and never updated.

### 3. Replace MiniGPT with DistilGPT2
✅ DONE — DistilGPT2 (82M params) now used instead of MiniGPT (463K params).

### 4. Fix Double-Counted Score Bug
`self_attention.py:327` — `0.5 * scores + 0.3 * cross_attn + 0.2 * scores` = `0.7 * scores + 0.3 * cross_attn`.

### 5. Add Subword Embeddings
Word2Vec claims subword support but has none. OOV words map to UNK.

---

## METRICS TO TRACK

| Metric | Current | Target |
|--------|---------|--------|
| Intent types | 9 (book only) | 17+ (conversational) |
| Personal statements handled | 0% | 100% |
| General knowledge sources | 0 | 3+ (Wikipedia, local KB, generated) |
| User preferences stored | 0 | Unlimited |
| Conversation quality | Book QA only | Natural conversation |
| Training required | 40s per book | 0s (optional) |

---

## FILES TO INVESTIGATE

| File | Why |
|------|-----|
| `training/attention_trainer.py` | Why only head is trained, not attention |
| `training/self_supervised_data.py` | Quality of auto-generated training pairs |
| `query/torch_attention.py` | Why scores are uniform |
| `pipeline.py:290-380` | Why advanced answer falls back to raw sentences |
| `query/prose_realizer.py` | Why sentence scoring is so basic |
| `query/response_formatter.py` | How `already_good` keyword list works |
