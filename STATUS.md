# BookBot Status — July 2026

## Project Overview
Natural-language QA system over Pride and Prejudice. Answers like a narrator — synthesizes prose from book data, not quoting raw text. Hybrid pipeline: rule-based NLP + neural components (Word2Vec, self-attention, multi-dim scorer, style realizer, MiniGPT).

## Environment
- **Python**: 3.14.6 (system, `C:\Users\voo\AppData\Local\Python\bin\python.exe`)
- **OS**: Windows, PowerShell 5.1
- **DB**: `C:\projects\bookbot\bookbot.db` (SQLite, WAL mode)
- **Key deps installed**: `torch` (CPU-only), `transformers` 5.12.1, `sentencepiece`, `nltk`, `pysbd`, `rank-bm25`
- **VC++ Redistributable**: installed (required for torch DLL loading)

## What's Done ✅

### Training Pipeline
- Full pipeline: OCR → tokenize → POS → NER → SVO → entity graph → coreference → topics → temporal → convergence
- DB: 3958 sentences, 29180 SVO triples, 143 entities, 1020 knowledge edges
- Training takes ~40s

### Neural Components (all trained, saved)
| Component | File | Weights |
|-----------|------|---------|
| Word2Vec skip-gram (50-dim) | `lib/word2vec.py` | `word2vec.json`, `word2vec_vocab.json` |
| SVO Quality Scorer (2-layer MLP) | `query/svo_scorer.py` | `svo_scorer.json` |
| Multi-Dim Scorer (5-head, 38-dim) | `query/multi_scorer.py` | `multi_scorer.json` |
| Self-Attention Selector (2 layers, 4 heads) | `query/self_attention.py` | `self_attention.json` |
| Token-Level Attention | `query/token_attention.py` | `token_attention.json` |
| PyTorch Token Attention | `query/torch_attention.py` | `torch_attention.pt`, `torch_attention_state.pt` |
| MiniGPT (463K params, 4 layers) | `query/minigpt.py` | `minigpt.pt`, `minigpt_vocab.json` |
| T5-small (paraphraser, abandoned — too small) | `t5_small/` | tokenizer + model files |

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

## What's Broken / Incomplete ❌

### 1. Token attention training produces uniform scores
- Full backprop through attention layers needs larger training data
- `torch_attention.py` has PyTorch autograd version but scores are uniform
- Self-supervised training from book alone insufficient

### 2. MiniGPT quality
- Trained 25 epochs: loss 8.3→5.6, perplexity 2058→~300
- Generation is functional but not fully coherent (includes UNK tokens)
- Needs 50+ epochs or larger model for better prose

## Next Steps (in order)

1. (Optional) Port token-level attention training to proper PyTorch with larger dataset
2. (Optional) Cross-book training for better generalization
3. (Optional) Train MiniGPT longer (50+ epochs) for better coherence

## Key Gotchas
- `find_entity_actions()` returns tuples `(subject, verb, object, confidence, sentence_id, raw_text)`, not dicts — pipeline code handles both
- SVO triples have `sentence_id` linking to `sentences` table — key for prose retrieval
- MiniGPT vocab size is 4000 (BPE-like), trained on book text
- `response_formatter.py` has `already_good` keyword list — don't let refiner overwrite good prose
- `pipeline.py` line 398-399: `has_synthesized` check skips refinement for good answers
- MiniGPT uses `generate_from_text()` method (not raw `generate()` which takes token indices)

## File Map

```
bookbot/
├── pipeline.py                  # Main orchestration (512 lines)
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
├── svo_scorer.json              # Trained SVO scorer
├── multi_scorer.json            # Trained multi-dim scorer
├── self_attention.json          # Trained self-attention
├── token_attention.json         # Trained token attention
├── torch_attention.pt           # PyTorch attention weights
└── bookbot.db                   # SQLite database
```
