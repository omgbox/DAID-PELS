"""
T5 Answer Generator
Uses t5-small to paraphrase structured evidence into natural language answers.
Instead of quoting the book, it synthesizes evidence into fluent prose.
"""

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.query.t5_answer_generator')

# Lazy-loaded model to avoid startup delay
_model = None
_tokenizer = None


def _load_model(model_path: str = 'C:/projects/bookbot/t5_small'):
    global _model, _tokenizer
    if _model is not None:
        return

    try:
        import os
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        token = os.environ.get('HF_TOKEN')
        logger.info(f"Loading T5 model from {model_path}...")
        _tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, token=token)
        _model = AutoModelForSeq2SeqLM.from_pretrained(model_path, token=token)
        logger.info("T5 model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load T5 model: {e}")
        _model = None
        _tokenizer = None


def _build_evidence_context(entity: str, evidence: Dict, query: str = '') -> str:
    """Build a text summary of all available evidence for the entity."""
    parts = []

    # Role/definition
    for desc in evidence.get('descriptions', [])[:3]:
        if desc and len(desc) > 5:
            parts.append(desc.strip())

    # Attributes
    for attr in evidence.get('attributes', [])[:3]:
        if attr and len(attr) > 5:
            parts.append(f"{entity} is described as {attr.strip()}")

    # SVO actions - use clean triples from training if available
    svo = evidence.get('svo_triples', evidence.get('actions', []))
    actions = []
    for t in svo[:10]:
        if isinstance(t, dict):
            s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
        elif isinstance(t, (list, tuple)) and len(t) >= 3:
            s, v, o = str(t[0]), str(t[1]), str(t[2])
        else:
            continue
        s_lower = s.lower().strip()
        if entity.lower() not in s_lower:
            continue
        v = v.strip()
        o = o.strip()
        if len(v) < 3 or len(o) < 2:
            continue
        # Skip noise
        if v.lower() in ('is', 'are', 'was', 'were', 'has', 'have', 'had'):
            if not o or len(o) < 3:
                continue
        # Skip very long objects (raw sentences, not clean SVO)
        if len(o) > 50:
            continue
        actions.append(f"{entity} {v} {o}".rstrip('.'))
    if actions:
        parts.extend(actions[:6])

    # Relationships
    related = evidence.get('related_entities', [])
    names = []
    for r in related[:6]:
        if isinstance(r, dict):
            name = r.get('related', '') or r.get('source_id', '') or r.get('target_id', '')
        elif isinstance(r, (list, tuple)):
            name = r[0] if r else ''
        else:
            name = str(r)
        if name and name.lower() != entity.lower() and len(name) > 1:
            names.append(name)
    if names:
        parts.append(f"{entity} interacts with {', '.join(names[:5])}")

    # Definition from DB (if available in evidence)
    definition = evidence.get('definition', '')
    if definition and len(definition) > 10:
        parts.append(definition)

    return '. '.join(parts)


def generate_natural_answer(entity: str, evidence: Dict, query: str,
                            intent: str = 'DEFINITIONAL') -> Optional[str]:
    """
    Generate a natural language answer using t5-small.

    Takes structured evidence (SVO triples, descriptions, relationships)
    and produces a fluent paragraph that answers the query.
    """
    _load_model()
    if _model is None or _tokenizer is None:
        return None

    # Build context from evidence
    context = _build_evidence_context(entity, evidence, query)
    if not context or len(context) < 15:
        return None

    # Build the prompt - use summarize which T5 handles well
    # T5 was trained on specific task prefixes
    if intent == 'DEFINITIONAL':
        prompt = f"summarize in complete sentences: {context}"
    elif intent == 'FACTUAL':
        prompt = f"summarize in complete sentences: {context}"
    elif intent == 'CAUSAL':
        prompt = f"summarize cause and effect: {context}"
    elif intent == 'SUMMARIZATION':
        prompt = f"summarize: {context}"
    else:
        prompt = f"summarize in complete sentences: {context}"

    try:
        inputs = _tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = _model.generate(
            **inputs,
            max_new_tokens=120,
            min_length=15,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            length_penalty=1.2,
        )
        result = _tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up the result
        result = result.strip()
        if not result or len(result) < 15:
            return None

        # Ensure entity is mentioned
        if entity.lower() not in result.lower():
            result = f"{entity}. {result}"

        # Capitalize first letter
        if result and result[0].islower():
            result = result[0].upper() + result[1:]

        # Ensure period ending
        if result and result[-1] not in '.!?':
            result += '.'

        return result

    except Exception as e:
        logger.error(f"T5 generation failed: {e}")
        return None


class T5AnswerGenerator:
    """Wrapper class for compatibility with the pipeline module system."""

    def __init__(self, config=None, db_manager=None, logger_instance=None):
        self.config = config or {}
        self.db = db_manager

    def process(self, input_data: Dict) -> Dict:
        """Process answer generation request."""
        entity = input_data.get('entity', '')
        evidence = input_data.get('evidence', {})
        query = input_data.get('query', '')
        intent = input_data.get('intent', 'DEFINITIONAL')

        answer = generate_natural_answer(entity, evidence, query, intent)
        return {'answer': answer or ''}
