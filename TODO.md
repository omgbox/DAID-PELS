# BookBot Architecture Research & TODO
# Transformation: Book QA System → True LLM-Style Chatbot

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

## TRUE LLM CHATBOT IMPROVEMENTS

### 1. Better Response Generation (T5 Paraphrase)
**Goal**: Generate natural, fluent prose instead of copying Wikipedia text

**Current**: "Rust is a programming language..."
**Target**: "Rust is actually a really cool language! It's known for being fast and safe."

**How to implement:**
- Use `Vamsi/T5_Paraphrase_Paws` for rewriting
- T5 is fine-tuned specifically for paraphrasing on PAWS dataset
- Encoder-decoder architecture prevents hallucination
- ~60M params, fast inference

**Why T5 Paraphrase over DistilGPT2/Pegasus:**
- DistilGPT2 hallucinates — ignores input, generates unrelated text
- Pegasus has position embedding issues — hangs on loading
- T5 Paraphrase is specifically fine-tuned for paraphrasing
- Works out of the box, no additional training needed

**Priority**: HIGH

---

### 2. Knowledge Synthesis
**Goal**: Combine information from multiple sources into coherent answers

**Current**: Returns single Wikipedia sentence
**Target**: "Python is great for beginners, Rust is for systems programming, JavaScript runs in browsers..."

**How to implement:**
- Query multiple Wikipedia pages
- Synthesize facts into unified response
- Use DistilGPT2 to generate natural transitions

**Priority**: HIGH

---

### 3. Conversational Personality
**Goal**: Track user interests and adapt responses

**Current**: Generic responses for everyone
**Target**: Personalized based on conversation history

**How to implement:**
- Extend user_profile with conversation patterns
- Track topics discussed
- Adapt tone based on user's style

**Priority**: MEDIUM

---

### 4. Source Attribution (Current Priority)
**Goal**: Tell users where information comes from

**Current**: No source attribution
**Target**: "According to Pride and Prejudice..." or "Wikipedia states that..."

**How to implement:**
- Track source in response generation (book, Wikipedia, or general knowledge)
- Add attribution templates at the end of responses
- Differentiate book vs Wikipedia vs general knowledge

**Priority**: HIGH (Current)

---

### 5. Better Context Handling
**Goal**: Understand pronouns and references across turns

**Current**: "Who created it?" → "I don't know" (doesn't remember "it" = Rust)
**Target**: "Graydon Hoare created it in 2006"

**How to implement:**
- Expand pronoun resolution
- Track entities mentioned in conversation
- Build entity cache for quick lookup

**Priority**: HIGH

---

### 6. Multi-Source Knowledge Base
**Goal**: Combine book knowledge with Wikipedia

**Current**: Book OR Wikipedia
**Target**: Both sources combined

**How to implement:**
- Search book database first
- Supplement with Wikipedia if needed
- Merge information coherently

**Priority**: MEDIUM

---

### 7. Response Quality Scoring
**Goal**: Rank responses by quality and pick the best

**Current**: Returns first result
**Target**: Returns best result based on quality

**How to implement:**
- Score responses on fluency, relevance, completeness
- Use DistilGPT2 to generate multiple options
- Pick highest-scoring response

**Priority**: LOW

---

### 8. Fact Verification
**Goal**: Cross-reference information for accuracy

**Current**: Single source only
**Target**: Multiple sources for verification

**How to implement:**
- Query multiple sources
- Compare facts
- Flag inconsistencies

**Priority**: LOW

---

## IMPLEMENTATION PLAN

### Phase 6: Response Quality (Current Priority)
1. Rewrite Wikipedia text with DistilGPT2 for natural style
2. Add multi-sentence synthesis
3. Add source attribution

### Phase 7: Context Enhancement
1. Expand pronoun resolution
2. Track conversation entities
3. Build entity cache

### Phase 8: Personality & Personalization
1. Extend user profile with patterns
2. Track topics discussed
3. Adapt response style

### Phase 9: Advanced Features
1. Multi-source knowledge synthesis
2. Fact verification
3. Response quality scoring

---

## WHAT'S ALREADY WORKING

| Feature | Status |
|---------|--------|
| Book Q&A | ✅ Working |
| Wikipedia search | ✅ Working |
| Conversations | ✅ Working |
| Context tracking | ✅ Working |
| Personal statements | ✅ Working |
| Emotional responses | ✅ Working |
| Multi-book training | ✅ Working |
| Progress bars | ✅ Working |

---

## NEXT STEPS (Immediate)

1. **Improve response generation** — Use Pegasus for paraphrasing
2. **Add source attribution** — Tell users where info comes from
3. **Fix context handling** — Better pronoun resolution

---

## ALTERNATIVE PARAPHRASING ENGINES (To Try Later)

| Model | Size | Quality | Speed | Notes |
|-------|------|---------|-------|-------|
| `tuner007/pegasus_paraphrase` | ~500MB | Good | Medium | **Current choice** |
| `Vamsi/T5_Paraphrase_Paws` | ~60M | Good | Fast | Lightweight option |
| `eugenesiow/bart-paraphrase` | ~1.6GB | Best | Slow | Highest quality |
| `Ateeqq/Text-Rewriter-Paraphraser` | ~223M | Best | Medium | 430k training examples |
| `parrot` | ~500MB | Good | Medium | Augmentation framework |

### When to try alternatives:
- **T5 Paraphrase**: If Pegasus is too slow, try this lightweight option
- **BART Paraphrase**: If quality is priority over speed
- **Text Rewriter**: If need best paraphrasing quality
- **Parrot**: If need data augmentation for training

---

## WHAT MAKES A TRUE LLM CHATBOT

| Feature | Current | LLM |
|---------|---------|-----|
| Knowledge | 20 books + Wikipedia | Trained on internet |
| Response quality | Wikipedia text + templates | Natural, fluent prose |
| Context length | 10 turns | 100+ turns |
| Reasoning | Simple lookup | Chain-of-thought |
| Personality | Generic | Adaptive tone |
| Memory | Last 10 turns | Long-term preferences |
| Source attribution | None | Always cited |
| Multi-source synthesis | No | Yes |
