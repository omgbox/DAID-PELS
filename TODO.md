# DAID-PELS Architecture & TODO
# Transformation: Book QA System → True LLM-Style Chatbot

## Current Architecture Summary

The system now has **fully integrated neural components** that learn on the fly:

| Component | Architecture | Weights | Purpose |
|-----------|--------------|---------|---------|
| Topic Extractor | 3-layer (24→256→128→1) | 49,152 | Extracts key topics from queries |
| Wikipedia Mapper | 3-layer (20→512→256→1) | 196,608 | Maps queries to Wikipedia pages |
| Intent Classifier | 3-layer (20→256→128→1) | 49,152 | Classifies user intent |
| Response Selector | 3-layer (16→128→64→1) | 12,288 | Picks best response |
| DistilGPT2 | 82M params | 82,000,000 | Text generation |
| T5 Paraphrase | 60M params | 60,000,000 | Response rewriting |

**Total neural parameters: ~142M** (all running on CPU)

---

## WEB INTERFACE

### Server Module (`server/`)
```
server/
  __init__.py      # Module exports
  __main__.py      # Entry point (python -m server)
  app.py           # Flask application factory
  templates/
    chat.html      # Chat interface
  static/
    style.css      # Dark theme styling
    script.js      # Chat functionality
```

### Run Web Interface
```bash
cd C:\projects
python run_server.py
# Open http://localhost:5000
```

### Features
- **Chat interface** — modern dark-themed UI
- **Stats panel** — click Stats button to view:
  - System stats (uptime, memory, queries, avg response time)
  - Neural network info (architecture, weights, training count)
- **Response time** — shown on each message
- **Mobile responsive** — works on phone/tablet

### API Endpoints
- `GET /` — Chat interface
- `POST /chat` — Send message, get response
- `GET /stats` — System statistics
- `GET /health` — Health check

---

## NEURAL NETWORKS INVENTORY

### 1. Neural Topic Extractor (`neural_topic_extractor.py`)
**Architecture**: 3-layer feedforward network
- Input: 24 features per word (position, length, case, stop word, verb detection, etc.)
- Hidden1: 256 neurons (ReLU)
- Hidden2: 128 neurons (ReLU)
- Output: 1 (sigmoid score)
- **Total weights: 49,152**

**Features extracted:**
1. Position normalized
2. Word length normalized
3. Is capitalized (proper noun)
4. Is all caps (abbreviation)
5. Is stop word
6. Contains vowels
7. Is short (≤2 chars)
8. Is long (≥5 chars)
9. Is first word
10. Is last word
11. Previous word is stop word
12. Next word is stop word
13. Learned word score
14. Title case in original
15. Is question word
16. Is verb-like
17. Is common verb
18. Is noun-like
19. Is entity (proper noun)
20. Context: previous is question word
21. Is subject candidate
22. Is object candidate
23. Starts with consonant cluster
24. Word frequency hint

**Online learning:**
- Trains on successful Wikipedia lookups
- Saves word scores to `topic_scores.json`
- Gets smarter with every query

---

### 2. Neural Wikipedia Mapper (`neural_wiki_mapper.py`)
**Architecture**: 3-layer feedforward network
- Input: 20 features (word overlap, n-grams, edit distance, Soundex, etc.)
- Hidden1: 256 neurons (ReLU)
- Hidden2: 128 neurons (ReLU)
- Output: 1 (sigmoid score)
- **Total weights: 38,016**

**Features extracted:**
1. Word overlap (Jaccard)
2. Query contains title
3. Title contains query
4. Character n-gram similarity (3-grams)
5. Edit distance (normalized)
6. Length ratio
7. First word match
8. Last word match
9. Word count difference
10. Contains parentheses (disambiguation)
11. Title starts with query
12. Query starts with title
13. Common words overlap
14. Character overlap ratio
15. Exact match after removing disambiguation
16. Query words in title (proportion)
17. Title words in query (proportion)
18. Longest common substring
19. Soundex similarity
20. Word similarity

**Online learning:**
- Trains on successful Wikipedia lookups
- Saves mappings to `wiki_mappings.json`
- Remembers query→page mappings

---

### 3. Neural Intent Classifier (`neural_intent_classifier.py`)
**Architecture**: 3-layer feedforward network
- Input: 20 features per message
- Hidden1: 128 neurons (ReLU)
- Hidden2: 64 neurons (ReLU)
- Output: 1 (sigmoid score)
- **Total weights: 28,736**

**Features extracted:**
1. Message length
2. Word count
3. Starts with question word
4. Ends with question mark
5. Starts with I/my (personal)
6. Contains greeting
7. Contains farewell
8. Contains emotional word
9. Contains command word
10. Contains book-related word
11. Average word length
12. Has exclamation
13. Is very short (< 5 words)
14. Is long (> 15 words)
15. Contains pronoun
16. Contains verb-like word
17. Contains negation
18. Starts with tell/show/explain
19. Learned intent score
20. Word overlap with common intents

**Intent types:**
- greeting, farewell, question, statement, personal, emotional, command, book_query

---

### 4. Neural Response Selector (`neural_response_selector.py`)
**Architecture**: 3-layer feedforward network
- Input: 16 features per response
- Hidden1: 64 neurons (ReLU)
- Hidden2: 32 neurons (ReLU)
- Output: 1 (sigmoid score)
- **Total weights: 9,281**

**Features extracted:**
1. Response length
2. Word count
3. Average word length
4. Has proper sentence structure
5. Ends with punctuation
6. Has multiple sentences
7. Query word overlap
8. Has numbers (factual)
9. Has attribution
10. Has emotional tone
11. Has technical terms
12. Has examples
13. Has comparison
14. Has causation
15. Has timeline
16. Learned response score

**Online learning:**
- Trains on user feedback (implicit or explicit)
- Saves preferences to `response_scores.json`
- Replaces random.choice with learned selection

---

## WHAT'S ALREADY WORKING

| Feature | Status | Neural Network |
|---------|--------|----------------|
| Book Q&A | ✅ Working | - |
| Wikipedia search | ✅ Working | Wikipedia Mapper |
| Conversations | ✅ Working | Intent Classifier |
| Context tracking | ✅ Working | - |
| Personal statements | ✅ Working | Intent Classifier |
| Emotional responses | ✅ Working | Intent Classifier |
| Multi-book training | ✅ Working | - |
| Progress bars | ✅ Working | - |
| T5 Paraphrase rewriter | ✅ Working | T5 Paraphrase |
| Source attribution | ✅ Working | - |
| Pronoun resolution | ✅ Working | - |
| Response quality scoring | ✅ Working | Response Selector |
| Multi-source synthesis | ✅ Working | - |
| 466K word list | ✅ Integrated | - |
| Old English dictionary | ✅ Integrated | - |
| Neural Wikipedia mapper | ✅ Working | Wikipedia Mapper |
| Neural topic extractor | ✅ Working | Topic Extractor |
| Neural intent classifier | ✅ Working | Intent Classifier |
| Neural response selector | ✅ Working | Response Selector |
| Follow-up context | ✅ Working | - |
| Dynamic Wikipedia search | ✅ Working | Wikipedia Mapper |
| Online learning | ✅ Working | All neural components |
| HF_TOKEN authenticated | ✅ Working | - |
| Compound topic expansion | ✅ Working | - |
| 3-layer deep networks | ✅ Working | All neural components |

---

## NEXT STEPS (Immediate)

1. **More training data** — Use the bot more to train neural networks
2. **Test Old English queries** — "What does wyrd mean?" → "fate, destiny"
3. **Add more follow-up patterns** — "and also", "what else", etc.
4. **Improve verb detection** — Better extraction of action words
5. **Add more neural components** — Sentiment analysis, coreference, etc.

---

## NEURAL NETWORK COMPARISON

| Component | Before | After | Improvement |
|-----------|--------|-------|-------------|
| Topic extraction | Regex patterns | 3-layer neural (128→64) | Learns from usage |
| Wikipedia mapping | Hard-coded rules | 3-layer neural (256→128) | No hard-coded mappings |
| Intent classification | Regex patterns | 3-layer neural (128→64) | Handles edge cases |
| Response selection | random.choice | 3-layer neural (64→32) | Learned preferences |

---

## WHAT MAKES A TRUE LLM CHATBOT

| Feature | Current | LLM |
|---------|---------|-----|
| Knowledge | 20 books + Wikipedia + 491K dictionary | Trained on internet |
| Response quality | T5 Paraphrase (natural, fluent) | Natural, fluent prose |
| Context length | 10 turns | 100+ turns |
| Reasoning | Neural networks + rules | Chain-of-thought |
| Personality | Generic | Adaptive tone |
| Memory | Last 10 turns + learned scores | Long-term preferences |
| Source attribution | ✅ Always cited | Always cited |
| Multi-source synthesis | ✅ Yes | Yes |
| Online learning | ✅ All neural components | N/A |
| Word validation | ✅ 466K words | Full dictionary |
| Old English | ✅ 42K OE words | N/A |
| Wikipedia mapping | ✅ Neural mapper (learns) | N/A |
| Intent classification | ✅ Neural classifier (learns) | N/A |
| Response selection | ✅ Neural selector (learns) | N/A |

---

## FILES CREATED/UPDATED

| File | Purpose |
|------|---------|
| `query/neural_topic_extractor.py` | Neural topic extraction |
| `query/neural_wiki_mapper.py` | Neural Wikipedia mapping |
| `query/neural_intent_classifier.py` | Neural intent classification |
| `query/neural_response_selector.py` | Neural response selection |
| `query/conversational_ai.py` | Main conversational pipeline |
| `config.py` | Updated dictionary paths |
| `TODO.md` | This file |
| `README.md` | Updated documentation |

---

## ONLINE LEARNING REFERENCES

Online learning is a well-established field in machine learning:
- **Stochastic Gradient Descent (SGD)** — Most common online learning algorithm
- **Spam filters** — Bayesian online learning from each email
- **Recommendation engines** — Learn from user clicks
- **Search engines** — Learn from query patterns
- **RLHF** — Reinforcement Learning from Human Feedback

**What's different about our approach:**
1. Pure online learning — no pre-training needed
2. Multiple neural networks working together
3. Runs on CPU — no GPU required
4. Learns from successful lookups
5. Persists learning to JSON files
