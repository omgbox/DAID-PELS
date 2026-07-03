"""
Advanced Answer Engine
Combines multi-dimensional scoring, style realization, and query-aware selection
to produce high-quality, multi-option answers.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger('bookbot.query.advanced_answer')

# Lazy-loaded components
_multi_scorer = None
_style_realizer = None
_token_attn = None
_minigpt = None


def _load_components():
    global _multi_scorer, _style_realizer, _token_attn, _minigpt
    if _multi_scorer is not None:
        return

    try:
        from .multi_scorer import MultiDimScorer
        _multi_scorer = MultiDimScorer.load('C:/projects/bookbot/multi_scorer.json')
        logger.info("Loaded multi-dim scorer")
    except Exception as e:
        logger.info(f"Multi-scorer not available: {e}")
        _multi_scorer = None

    try:
        from .style_realizer import StyleRealizer
        _style_realizer = StyleRealizer()
        logger.info("Loaded style realizer")
    except Exception as e:
        logger.info(f"Style realizer not available: {e}")
        _style_realizer = None

    try:
        from .token_attention import TokenLevelAttention
        from ..lib.word2vec import Word2VecEmbeddings
        emb = Word2VecEmbeddings.load('C:/projects/bookbot/word2vec.json')
        _token_attn = TokenLevelAttention.load('C:/projects/bookbot/token_attention.json', emb)
        logger.info("Loaded token-level attention")
    except Exception as e:
        logger.info(f"Token attention not available: {e}")
        _token_attn = None

    try:
        from .minigpt import load_minigpt
        _minigpt = load_minigpt()
        if _minigpt:
            logger.info("Loaded MiniGPT")
        else:
            logger.info("MiniGPT not available (load failed)")
    except Exception as e:
        logger.info(f"MiniGPT not available: {e}")
        _minigpt = None


def generate_answer(entity: str, original_sentences: List[str],
                    query: str, intent: str = 'DEFINITIONAL',
                    related: List[str] = None,
                    entity_info: Dict = None,
                    max_sentences: int = 4) -> Dict:
    """
    Generate a multi-option answer with advanced scoring.

    Returns dict with:
      - 'options': dict of style_name → answer text
      - 'best': the highest-scoring option
      - 'scores': per-option scoring breakdown
    """
    _load_components()

    # Handle character description queries (e.g., "is Lydia a girl or a boy?")
    desc_result = _handle_description_query(entity, query, original_sentences, related)
    if desc_result:
        return desc_result

    if not original_sentences:
        return {
            'options': {'neutral': f"{entity} is a character in the story."},
            'best': f"{entity} is a character in the story.",
            'scores': {},
        }

    related = related or []
    entity_info = entity_info or {}

    # Step 1: Score sentences — prefer token-level attention if available
    if _token_attn and len(original_sentences) > 2:
        # Use token-level attention for sentence selection
        selected_raw = _token_attn.select(
            original_sentences, query, entity,
            top_k=max_sentences + 2, min_score=0.2
        )
        selected = [sent for _, sent, _ in selected_raw]

        # Resolve coreferences using token attention
        # (skip for now — needs token_attn output from forward pass)
        pass

        # Order for narrative flow
        selected = selected[:max_sentences]

        # Also score with multi-dim for style selection
        if _multi_scorer:
            score_details = {}
            for sent in selected:
                _, heads = _multi_scorer.score_with_cache(sent, entity, query, intent, entity_info)
                score_details[sent[:50]] = heads
        else:
            score_details = {}
    elif _multi_scorer:
        ranked = _multi_scorer.rank_sentences(
            original_sentences, entity, query, intent,
            entity_info=entity_info, top_k=max_sentences + 2,
            min_score=0.25
        )
        selected = [sent for _, sent, _ in ranked]
        score_details = {sent[:50]: heads for _, sent, heads in ranked}
    else:
        selected = _simple_rank(original_sentences, entity)[:max_sentences]
        score_details = {}

    if not selected:
        selected = original_sentences[:max_sentences]

    # Step 2: Generate styled versions
    if _style_realizer:
        options = _style_realizer.realize(entity, selected, related)
    else:
        # Fallback: just join sentences
        text = ' '.join(selected)
        options = {'neutral': text}

    # Step 2b: MiniGPT prose generation
    if _minigpt and selected:
        try:
            # Build prompt from entity and first sentence context
            prompt = f"{entity} "
            if selected:
                first_words = ' '.join(selected[0].split()[:5])
                prompt = first_words
            generated = _minigpt.generate_from_prompt(prompt, max_tokens=60, temperature=0.7)
            if generated and len(generated) > 20:
                options['minigpt'] = generated
        except Exception as e:
            logger.debug(f"MiniGPT generation failed: {e}")

    # Step 3: Pick best option
    best_style = 'neutral'
    best_score = 0.0

    for style_name, text in options.items():
        if _multi_scorer:
            sc, heads = _multi_scorer.score(text, entity, query, intent, entity_info)
        else:
            sc = len(text) / 200.0  # crude fallback
            heads = {}

        score_details[style_name] = {'final': sc, **heads}

        if sc > best_score:
            best_score = sc
            best_style = style_name

    best = options.get(best_style, options.get('neutral', ''))

    return {
        'options': options,
        'best': best,
        'scores': score_details,
    }


def _handle_description_query(entity: str, query: str,
                               sentences: List[str],
                               related: List[str]) -> Optional[Dict]:
    """
    Handle queries asking about character attributes (gender, role, etc.)
    e.g., "is Lydia a girl or a boy?"
    """
    lower_query = query.lower()
    entity_lower = entity.lower()

    # Detect description/gender queries — only specific patterns
    DESC_PATTERNS = [
        r'is\s+\w+\s+(?:a|an)\s+(?:girl|boy|man|woman|male|female)',
        r'(?:girl|boy|man|woman|male|female)\s+or\s+(?:a\s+)?(?:girl|boy|man|woman|male|female)',
        r'describe\s+\w+',
        r'what\s+(?:is|are)\s+\w+\s+(?:like|about)',
    ]
    is_desc_query = any(re.search(p, lower_query) for p in DESC_PATTERNS)

    if not is_desc_query:
        return None

    if not sentences:
        return None

    # Build a natural description from the sentences
    # Find gender clues
    gender = _detect_gender(entity, sentences)

    # Find character traits
    traits = _extract_traits(entity, sentences)

    # Find role/relationships
    rel_text = ''
    if related:
        names = [r for r in related[:4] if r.lower() != entity_lower and len(r) > 1]
        if names:
            if len(names) == 1:
                rel_text = f"{entity} interacts with {names[0]}."
            elif len(names) == 2:
                rel_text = f"{entity} interacts with {names[0]} and {names[1]}."
            else:
                rel_text = f"{entity} interacts with {', '.join(names[:3])}."

    # Build options
    options = {}

    # Professional
    prof_parts = [f"{entity} is {gender}."]
    if traits:
        prof_parts.append(f"{entity} is {', '.join(traits[:3])}.")
    if rel_text:
        prof_parts.append(rel_text)
    options['professional'] = ' '.join(prof_parts)

    # Formal
    formal_parts = []
    if gender in ('a young woman', 'a woman', 'female'):
        formal_parts.append(
            f"Within the narrative, {entity} is portrayed as {gender} "
            f"of notable character."
        )
    elif gender in ('a young man', 'a man', 'male'):
        formal_parts.append(
            f"Within the narrative, {entity} is portrayed as {gender} "
            f"of notable standing."
        )
    else:
        formal_parts.append(f"Within the narrative, {entity} is a character of significance.")
    if traits:
        formal_parts.append(f"{entity} is characterized by {', '.join(traits[:3])}.")
    if rel_text:
        formal_parts.append(rel_text)
    options['formal'] = ' '.join(formal_parts)

    # Neutral
    neut_parts = [f"{entity} is {gender}."]
    if traits:
        neut_parts.append(f"Key traits include {', '.join(traits[:3])}.")
    if rel_text:
        neut_parts.append(rel_text)
    options['neutral'] = ' '.join(neut_parts)

    # Pick best (professional is most direct for this type)
    best = options['professional']

    return {
        'options': options,
        'best': best,
        'scores': {},
    }


def _detect_gender(entity: str, sentences: List[str]) -> str:
    """Detect gender from context clues in sentences."""
    entity_lower = entity.lower()

    MALE_PRONOUNS = {'he', 'him', 'his', 'himself', 'mr', 'sir'}
    FEMALE_PRONOUNS = {'she', 'her', 'hers', 'herself', 'mrs', 'miss', 'ms'}
    MALE_WORDS = {'man', 'boy', 'gentleman', 'father', 'son', 'brother', 'husband', 'uncle'}
    FEMALE_WORDS = {'woman', 'girl', 'lady', 'mother', 'daughter', 'sister', 'wife', 'aunt', 'miss'}

    male_score = 0
    female_score = 0

    for sent in sentences:
        lower = sent.lower()
        if entity_lower not in lower:
            continue

        # Check pronouns
        words = set(lower.split())
        male_score += len(words & MALE_PRONOUNS)
        female_score += len(words & FEMALE_PRONOUNS)

        # Check gendered words near entity
        for w in MALE_WORDS:
            if w in lower:
                male_score += 2
        for w in FEMALE_WORDS:
            if w in lower:
                female_score += 2

        # Check possessives: "her brother" near entity = entity is female
        if re.search(rf'{entity_lower}\s+(?:is|was)\s+(?:a\s+)?(?:young\s+)?(?:woman|girl|lady)', lower):
            female_score += 5
        if re.search(rf'{entity_lower}\s+(?:is|was)\s+(?:a\s+)?(?:young\s+)?(?:man|boy|gentleman)', lower):
            male_score += 5

    if female_score > male_score:
        return 'a young woman' if female_score > 3 else 'female'
    elif male_score > female_score:
        return 'a young man' if male_score > 3 else 'male'
    else:
        return 'a character'


def _extract_traits(entity: str, sentences: List[str]) -> List[str]:
    """Extract character traits from sentences."""
    entity_lower = entity.lower()
    traits = []
    TRAIT_WORDS = {
        'sensible', 'intelligent', 'beautiful', 'handsome', 'pleasing',
        'gentle', 'firm', 'quiet', 'warm', 'cold', 'kind', 'clever', 'witty',
        'spirited', 'lively', 'bright', 'proud', 'humble', 'modest',
        'elegant', 'graceful', 'charming', 'engaging', 'obstinate',
        'stubborn', 'willful', 'determined', 'resolute', 'amiable',
        'agreeable', 'reserved', 'haughty', 'conceited', 'vain', 'silly',
    }
    for sent in sentences:
        lower = sent.lower()
        if entity_lower in lower:
            for w in TRAIT_WORDS:
                if w in lower and w not in traits:
                    traits.append(w)
    return traits[:5]
    """Simple sentence ranking fallback."""
    scored = []
    entity_lower = entity.lower()
    for sent in sentences:
        score = 0.0
        lower = sent.lower()
        words = sent.split()

        if lower.startswith(entity_lower):
            score += 4.0
        if 8 <= len(words) <= 25:
            score += 2.0
        if ',' in sent:
            score += 1.0
        if any(w in lower for w in ['felt', 'looked', 'turned', 'smiled', 'said', 'spoke']):
            score += 2.0
        if len(words) < 5:
            score -= 3.0

        scored.append((score, sent))

    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]
