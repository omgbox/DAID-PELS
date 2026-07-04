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
        from .minigpt import load_distilgpt2
        _minigpt = load_distilgpt2()
        if _minigpt:
            logger.info("Loaded DistilGPT2")
        else:
            logger.info("DistilGPT2 not available (load failed)")
    except Exception as e:
        logger.info(f"DistilGPT2 not available: {e}")
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

    # Step 2b: DistilGPT2 prose generation
    if _minigpt and selected:
        try:
            # Build rich prompt from selected sentences
            prompt = _build_gpt2_prompt(entity, selected, related)
            generated = _minigpt.generate_from_prompt(prompt, max_tokens=80, temperature=0.7)
            if generated and len(generated) > 30:
                # Clean up repetitive text
                generated = _clean_gpt2_output(generated, entity)
                if generated:
                    options['gpt2'] = generated
        except Exception as e:
            logger.debug(f"DistilGPT2 generation failed: {e}")

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


def _build_gpt2_prompt(entity: str, sentences: List[str], related: List[str]) -> str:
    """Build a grounded prompt for DistilGPT2 generation."""
    parts = []
    
    # Use the best source sentences directly as context
    if sentences:
        # Take top 2-3 best sentences and clean them
        good_sentences = []
        for sent in sentences[:3]:
            # Clean the sentence
            sent = sent.strip()
            if sent and len(sent.split()) >= 8:
                good_sentences.append(sent)
        
        if good_sentences:
            # Join source sentences as context
            context = ' '.join(good_sentences)
            # Truncate if too long
            if len(context) > 200:
                context = context[:200].rsplit(' ', 1)[0] + '.'
            parts.append(context)
    
    # Add entity and relationships for grounding
    if related:
        rel_str = ', '.join(related[:2])
        parts.append(f"{entity} interacts with {rel_str}.")
    
    # Final prompt: use source text as grounding
    parts.append(f"In summary, {entity}")
    
    return ' '.join(parts)


def _clean_gpt2_output(text: str, entity: str) -> str:
    """Clean DistilGPT2 output to remove hallucination and repetition."""
    import re
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter out hallucinated content
    cleaned = []
    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 15:
            continue
            
        # Skip if it looks like Wikipedia/hallucinated content
        hallucination_markers = [
            'episode', 'season', 'aired', 'interview', 'published',
            'released', 'directed', 'produced', 'written by',
            'according to', 'cited', 'reference', 'source',
            'www.', 'http', 'ISBN', 'doi:',
        ]
        if any(marker.lower() in sent.lower() for marker in hallucination_markers):
            continue
            
        # Skip if entity name appears too often (repetitive)
        entity_count = sent.lower().count(entity.lower())
        if entity_count > 2 and len(sent.split()) < 15:
            continue
            
        # Skip if it's just the entity name repeated
        words = sent.split()
        if len(words) < 5:
            continue
            
        cleaned.append(sent)
    
    # Take only first 2-3 sentences to keep it concise
    result = ' '.join(cleaned[:3])
    
    # Ensure it ends with punctuation
    if result and result[-1] not in '.!?':
        result += '.'
    
    # Verify it mentions the entity
    if entity.lower() not in result.lower():
        return ''
    
    return result if len(result) > 30 else ''


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


def _simple_rank(sentences: List[str], entity: str) -> List[str]:
    """Simple sentence ranking with quality filtering."""
    scored = []
    entity_lower = entity.lower()
    
    for sent in sentences:
        score = 0.0
        lower = sent.lower()
        words = sent.split()
        wc = len(words)
        
        # FILTER: Skip fragments and low-quality sentences
        if wc < 6:
            continue  # Too short
        if wc > 40:
            continue  # Too long (run-on)
        if lower.startswith(('and ', 'but ', 'or ', 'so ', 'then ')):
            continue  # Fragment starting with conjunction
        if not any(c in sent for c in '.!?'):
            if wc < 10:
                continue  # No punctuation and short = fragment
                
        # BOOST: Entity as subject (natural descriptions)
        if lower.startswith(entity_lower):
            score += 5.0
            
        # BOOST: Complete thoughts with verbs
        has_verb = any(w in lower for w in [
            'was', 'were', 'is', 'are', 'had', 'has', 'felt', 'looked',
            'turned', 'smiled', 'said', 'replied', 'thought', 'knew',
            'saw', 'heard', 'found', 'gave', 'took', 'made', 'came',
            'went', 'told', 'asked', 'answered', 'declared', 'seemed',
        ])
        if has_verb:
            score += 3.0
            
        # BOOST: Descriptive sentences (adjectives + entity)
        DESCRIPTIVE_WORDS = [
            'beautiful', 'handsome', 'kind', 'gentle', 'clever', 'witty',
            'lively', 'amiable', 'agreeable', 'sensible', 'intelligent',
            'charming', 'elegant', 'graceful', 'proud', 'humble', 'modest',
        ]
        if any(w in lower for w in DESCRIPTIVE_WORDS):
            score += 2.0
            
        # BOOST: Medium length (sweet spot for answers)
        if 10 <= wc <= 20:
            score += 2.0
            
        # BOOST: Complex sentences (clauses, commas)
        if ',' in sent:
            score += 1.0
        if ';' in sent or ':' in sent:
            score += 0.5
            
        # PENALIZE: Dialogue (less useful for answers)
        if '"' in sent or '"' in sent or '"' in sent:
            score -= 1.0
            
        # PENALIZE: Starts with lowercase (likely continuation)
        if sent[0:1].islower():
            score -= 2.0
            
        # PENALIZE: Repetitive structure
        if lower.count(entity_lower) > 2:
            score -= 1.0  # Too many entity mentions
            
        scored.append((score, sent))
    
    # Sort by score, filter out negatives
    scored.sort(key=lambda x: -x[0])
    return [s for score, s in scored if score > 0]
