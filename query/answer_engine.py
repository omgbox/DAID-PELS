"""
BookBot Answer Engine
Combines retrieval results with knowledge graph lookups.
"""

import re
import logging
from typing import Dict, List, Optional

from ..core.base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.query.answer_engine')

# Stopwords to filter from query scoring
STOPWORDS = {
    'who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was', 'were',
    'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'into', 'about', 'between', 'through', 'during', 'before',
    'after', 'above', 'below', 'under', 'over', 'do', 'does', 'did', 'has',
    'have', 'had', 'can', 'could', 'will', 'would', 'should', 'may', 'might',
    'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
    'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
    'its', 'our', 'their', 'and', 'or', 'but', 'not', 'so', 'if', 'then',
    'than', 'too', 'very', 'just', 'tell', 'me', 'more', 'about', 'give',
    'know', 'think', 'say', 'said', 'like', 'make', 'made',
}


def clean_answer_text(text: str) -> str:
    """
    Clean OCR artifacts from answer text.
    """
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' ;', ';')
    text = text.replace(' :', ':')
    text = text.replace(' !', '!')
    text = text.replace(' ?', '?')
    return text.strip()


def extract_question_target(query: str) -> Optional[str]:
    """
    Extract the entity target from a question like 'Who is Elizabeth?'.
    
    Returns the proper noun or key noun phrase being asked about.
    """
    query = query.strip().rstrip('?')
    words = query.split()
    
    # Who/What is X? / Who was X?
    question_words = {'who', 'what', 'when', 'where', 'why', 'how'}
    be_verbs = {'is', 'are', 'was', 'were'}
    
    for i, word in enumerate(words):
        if word.lower() in question_words and i + 1 < len(words):
            # Check next words for be-verb + entity
            if words[i + 1].lower() in be_verbs and i + 2 < len(words):
                # "Who is Elizabeth?" -> "Elizabeth"
                target_words = []
                for j in range(i + 2, len(words)):
                    w = words[j]
                    if w[0:1].isupper() or w.lower() not in STOPWORDS:
                        target_words.append(w)
                    else:
                        break
                if target_words:
                    return ' '.join(target_words)
    
    # Fallback: find capitalized words not in question position
    for word in words:
        if word[0:1].isupper() and word.lower() not in STOPWORDS:
            return word
    
    return None


def score_sentence_relevance(query: str, sentence_text: str, 
                              retrieval_score: float,
                              question_target: Optional[str] = None) -> float:
    """
    Score how relevant a sentence is to the query.
    
    Combines BM25 score with entity-name matching.
    """
    text_lower = sentence_text.lower()
    query_lower = query.lower()
    
    # Filter stopwords from query for scoring
    query_terms = [w for w in query_lower.split() if w.lower() not in STOPWORDS]
    
    # Base term overlap score
    term_score = 0.0
    for term in query_terms:
        if term in text_lower:
            term_score += 1.0
    
    # Normalize by number of query terms
    if query_terms:
        term_score /= len(query_terms)
    
    # Entity name boost: if the question asks about "Elizabeth",
    # sentences containing "Elizabeth" get a big boost
    entity_boost = 0.0
    if question_target:
        target_lower = question_target.lower()
        if target_lower in text_lower:
            entity_boost = 3.0
        # Also check for partial match (surname)
        target_parts = question_target.split()
        for part in target_parts:
            if len(part) > 2 and part.lower() in text_lower:
                entity_boost = max(entity_boost, 2.0)
    
    # Combined score
    score = retrieval_score * 0.3 + term_score * 1.0 + entity_boost
    
    return score


class AnswerEngine(BaseModule):
    """Answer assembly module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)

    def process(self, input_data) -> dict:
        query = input_data.get('query', '')
        intent = input_data.get('intent', 'EXPLANATORY')
        retrieval_results = input_data.get('retrieval_results', [])
        context = input_data.get('context')
        structured_evidence = input_data.get('structured_evidence', {})
        query_entities = input_data.get('query_entities', [])

        sentences = context.sentences if context else []
        top_sentences = self._get_top_sentences(retrieval_results, sentences, query)

        # Collect all structured evidence across entities
        all_svo = []
        all_related = []
        for entity in query_entities:
            ev = structured_evidence.get(entity, {})
            all_svo.extend(ev.get('actions', []))
            all_related.extend(ev.get('related_entities', []))

        answer = self._build_answer(query, intent, top_sentences, context,
                                    all_svo, all_related, query_entities)

        self._initialized = True
        return {'answer': answer}

    def _get_top_sentences(self, retrieval_results: List[Dict],
                           sentences: List[Dict], query: str) -> List[Dict]:
        """Get top sentences, boosted by entity-name matching."""
        max_sentences = self.get_config('max_answer_sentences', 5)
        question_target = extract_question_target(query)
        
        scored = []

        for result in retrieval_results[:max_sentences * 3]:  # Over-fetch for re-ranking
            sid = result.get('sentence_id')
            text = result.get('text', '')
            retrieval_score = result.get('score', 0)
            
            # Try to find full sentence in context
            found_text = text
            found_chapter = None
            for sent in sentences:
                if sent.get('sentence_id') == sid:
                    found_text = sent.get('text', text)
                    found_chapter = sent.get('chapter_id')
                    break
            
            found_text = clean_answer_text(found_text)
            
            # Score with entity-name boost
            combined_score = score_sentence_relevance(
                query, found_text, retrieval_score, question_target
            )
            
            scored.append({
                'sentence_id': sid,
                'text': found_text,
                'retrieval_score': retrieval_score,
                'combined_score': combined_score,
                'chapter_id': found_chapter,
            })
        
        # Sort by combined score, take top N
        scored.sort(key=lambda x: x['combined_score'], reverse=True)
        return scored[:max_sentences]

    def _build_answer(self, query: str, intent: str,
                      sentences: List[Dict], context: PipelineContext = None,
                      svo_triples: List = None, related_entities: List = None,
                      query_entities: List = None) -> Dict:
        if not sentences:
            return {
                'text': "I couldn't find relevant information in the book.",
                'sources': [],
                'confidence': 0.0,
            }

        if intent == 'DEFINITIONAL':
            answer_text = self._build_definitional_answer(query, sentences,
                                                          svo_triples, related_entities)
        elif intent == 'FACTUAL':
            answer_text = self._build_factual_answer(query, sentences)
        elif intent == 'CAUSAL':
            answer_text = self._build_causal_answer(query, sentences)
        elif intent == 'TEMPORAL':
            answer_text = self._build_temporal_answer(query, sentences)
        elif intent == 'COMPARATIVE':
            answer_text = self._build_comparative_answer(query, sentences)
        elif intent == 'SUMMARIZATION':
            answer_text = self._build_summarization_answer(query, sentences)
        else:
            answer_text = self._build_explanatory_answer(query, sentences)

        sources = []
        for sent in sentences[:3]:
            sources.append({
                'text': sent.get('text', ''),
                'chapter': sent.get('chapter_id'),
            })

        return {
            'text': answer_text,
            'sources': sources,
            'sentences': sentences,
        }

    def _build_definitional_answer(self, query: str, sentences: List[Dict],
                                    svo_triples: List = None,
                                    related_entities: List = None) -> str:
        """Build definitional answer using text + structured evidence."""
        question_target = extract_question_target(query)
        target_lower = question_target.lower() if question_target else None
        svo = svo_triples or []
        related = related_entities or []

        parts = []

        # 1. Best descriptive sentence from BM25 results
        if target_lower and sentences:
            for sent in sentences:
                text = sent.get('text', '')
                text_lower = text.lower()
                word_count = len(text.split())
                if (target_lower in text_lower and
                    6 <= word_count <= 40 and
                    any(w in text_lower for w in ['is', 'was', 'were', 'are'])):
                    cleaned = clean_answer_text(text)
                    if cleaned and len(cleaned) > 20:
                        parts.append(cleaned)
                        break

        # 2. Related entities
        if related:
            names = set()
            for r in related[:6]:
                if isinstance(r, dict):
                    name = r.get('related', '') or r.get('source_id', '') or r.get('target_id', '')
                elif isinstance(r, (list, tuple)):
                    name = r[0] if r else ''
                else:
                    name = str(r)
                if name and name.lower() != target_lower and len(name) > 1:
                    names.add(name)
            if names:
                parts.append(f"In the story, {question_target or 'the character'} interacts with {', '.join(sorted(names))}.")

        # 3. Fallback: DB lookup for descriptive sentences
        if not parts and target_lower:
            db = getattr(self, 'db', None)
            if db is not None:
                try:
                    rows = db.execute(
                        "SELECT raw_text FROM sentences "
                        "WHERE raw_text LIKE ? AND LENGTH(raw_text) > 20 "
                        "AND LENGTH(raw_text) < 200 "
                        "AND (raw_text LIKE ? OR raw_text LIKE ? OR raw_text LIKE ?) "
                        "ORDER BY LENGTH(raw_text) DESC LIMIT 3",
                        (f'%{question_target}%',
                         f'%{question_target} is %',
                         f'%{question_target} was %',
                         f'%{question_target} has %')
                    )
                    for row in (rows or []):
                        text = clean_answer_text(row[0])
                        if text and len(text.split()) >= 4:
                            parts.append(text)
                            break
                except Exception:
                    pass

        # 4. Final fallback: any BM25 result mentioning the entity
        if not parts and target_lower:
            for sent in sentences:
                text = sent.get('text', '')
                if target_lower in text.lower():
                    cleaned = clean_answer_text(text)
                    if cleaned:
                        parts.append(cleaned)
                        break

        return ' '.join(parts) if parts else f"I couldn't find enough information about {question_target or 'that topic'}."

    def _build_factual_answer(self, query: str, sentences: List[Dict]) -> str:
        """Build factual answer - best term overlap."""
        # Already sorted by combined_score, use the best
        if sentences:
            return clean_answer_text(sentences[0].get('text', ''))
        return "I couldn't find the answer."

    def _build_causal_answer(self, query: str, sentences: List[Dict]) -> str:
        causal_words = ['because', 'therefore', 'consequently', 'as a result',
                       'thus', 'hence', 'due to', 'caused by', 'led to']
        for sent in sentences:
            text = sent.get('text', '').lower()
            if any(word in text for word in causal_words):
                return clean_answer_text(sent.get('text', ''))
        return clean_answer_text(sentences[0].get('text', '')) if sentences else "I couldn't find the cause."

    def _build_temporal_answer(self, query: str, sentences: List[Dict]) -> str:
        temporal_words = ['when', 'before', 'after', 'during', 'while',
                         'then', 'next', 'first', 'finally', 'eventually']
        for sent in sentences:
            text = sent.get('text', '').lower()
            if any(word in text for word in temporal_words):
                return clean_answer_text(sent.get('text', ''))
        return clean_answer_text(sentences[0].get('text', '')) if sentences else "I couldn't find temporal information."

    def _build_comparative_answer(self, query: str, sentences: List[Dict]) -> str:
        if sentences:
            texts = [clean_answer_text(s.get('text', '')) for s in sentences[:3]]
            return ' '.join(texts)
        return "I couldn't find comparison information."

    def _build_summarization_answer(self, query: str, sentences: List[Dict]) -> str:
        if sentences:
            texts = [clean_answer_text(s.get('text', '')) for s in sentences[:5]]
            return ' '.join(texts)
        return "I couldn't find a summary."

    def _build_explanatory_answer(self, query: str, sentences: List[Dict]) -> str:
        """Build explanatory answer - best combined score."""
        if sentences:
            return clean_answer_text(sentences[0].get('text', ''))
        return "I couldn't find an explanation."
