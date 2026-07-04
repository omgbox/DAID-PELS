# DAID-PELS Architecture & TODO

## Current Status: v2.2 — Trained on Wikipedia

### Neural Networks (all trained)

| Component | Architecture | Weights | Training |
|-----------|--------------|---------|----------|
| Topic Extractor | 3-layer (24→256→128→1) | 49,152 | 494 words |
| Wikipedia Mapper | 3-layer (20→512→256→1) | 196,608 | 2,279 mappings |
| Intent Classifier | 3-layer (20→256→128→1) | 49,152 | 2,272 patterns |
| Response Selector | 3-layer (16→128→64→1) | 12,288 | Learned prefs |
| DistilGPT2 | Transformer | 82M | Pre-trained |
| T5 Paraphrase | Encoder-Decoder | 60M | Pre-trained |

**Total: ~142M parameters** (all running on CPU)

---

## What's Working

| Feature | Status | How |
|---------|--------|-----|
| Wikipedia Q&A | ✅ | Neural mapper → Wikipedia API |
| Book Q&A | ✅ | BM25 + database |
| Conversations | ✅ | Intent classifier |
| Follow-ups | ✅ | Context tracking |
| Pronoun resolution | ✅ | Entity tracking |
| T5 Paraphrase | ✅ | Rewrites responses |
| DistilGPT2 | ✅ | Generates responses |
| Response scoring | ✅ | Neural selector |
| Source attribution | ✅ | Wikipedia/Books |
| Web interface | ✅ | Flask + HTML/CSS/JS |
| Session persistence | ✅ | JSON + SQLite |
| Wikipedia training | ✅ | 2,279 mappings |

---

## Training Commands

```bash
# Train on books
python train_pride.py          # One book (~40 sec)
python train_all_books.py      # All books (~10 min)

# Train neural networks on Wikipedia
python wiki_crawler.py         # Crawls ~250 pages per run
python run_crawler.py          # Runs crawler 5 times

# Run web interface
cd C:\projects
python run_server.py           # Open http://localhost:5000
```

---

## Web Interface

```bash
cd C:\projects
python run_server.py
# Open http://localhost:5000
```

**Features:**
- Sidebar with sessions
- Processing status indicator
- Snackbar notifications
- Stats panel
- Mobile responsive

---

## What's Persistent

| File | Contents |
|------|----------|
| `wiki_mappings.json` | 2,279 query→page mappings |
| `topic_scores.json` | 494 word importance scores |
| `intent_scores.json` | 2,272 intent patterns |
| `response_scores.json` | Response preferences |
| `learned_knowledge` | Wikipedia facts in SQLite |

---

## Next Steps

1. **Run wiki_crawler.py more** — Accumulate more training data
2. **Improve disambiguation** — Better handling of ambiguous queries
3. **Add more follow-up patterns** — "and also", "what else"
4. **Improve verb detection** — Better action word extraction
5. **Add sentiment analysis** — Emotional response detection

---

## Files

```
DAID-PELS/
├── query/
│   ├── conversational_ai.py    # Main chat pipeline
│   ├── neural_topic_extractor.py
│   ├── neural_wiki_mapper.py
│   ├── neural_intent_classifier.py
│   ├── neural_response_selector.py
│   ├── minigpt.py              # DistilGPT2 + T5
│   └── ...
├── server/
│   ├── app.py                  # Flask app
│   ├── templates/chat.html
│   └── static/
├── database/
│   ├── schema.sql
│   └── db_manager.py
├── wiki_crawler.py             # Wikipedia training
├── wiki_mappings.json          # Trained mappings
├── topic_scores.json           # Trained word scores
└── intent_scores.json          # Trained intent patterns
```
