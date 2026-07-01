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

        sentences = context.sentences if context else []
        top_sentences = self._get_top_sentences(retrieval_results, sentences, query)
        answer = self._build_answer(query, intent, top_sentences, context)

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
                      sentences: List[Dict], context: PipelineContext = None) -> Dict:
        if not sentences:
            return {
                'text': "I couldn't find relevant information in the book.",
                'sources': [],
                'confidence': 0.0,
            }

        if intent == 'DEFINITIONAL':
            answer_text = self._build_definitional_answer(query, sentences)
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

    def _build_definitional_answer(self, query: str, sentences: List[Dict]) -> str:
        """Build definitional answer - combine multiple sentences for character questions."""
        question_target = extract_question_target(query)
        target_lower = question_target.lower() if question_target else None
        
        # First: try to find a sentence where the entity is the subject + be-verb
        # Only use if sentence is reasonably long (not a fragment)
        if target_lower:
            for sent in sentences:
                text = sent.get('text', '')
                text_lower = text.lower()
                if (text_lower.startswith(target_lower) and 
                    len(text.split()) >= 6 and
                    any(w in text_lower for w in ['is', 'was', 'were', 'are'])):
                    return clean_answer_text(text)
        
        # Second: for character questions, use DB to find longer, more informative sentences
        if target_lower:
            db = getattr(self, 'db', None)
            if db is not None:
                try:
                    rows = db.execute(
                        "SELECT sentence_id, raw_text FROM sentences "
                        "WHERE raw_text LIKE ? AND LENGTH(raw_text) > 30 "
                        "ORDER BY LENGTH(raw_text) DESC LIMIT 5",
                        (f'%{question_target}%',)
                    )
                    combined = []
                    for row in (rows or []):
                        text = clean_answer_text(row[1])
                        words = text.split()
                        if len(words) >= 4:
                            combined.append(text)
                    if combined:
                        return ' '.join(combined[:3])
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.logger.debug(f"DB lookup failed: {e}")
        
        # Third: combine from BM25 results
        if target_lower:
            entity_sentences = []
            for sent in sentences:
                text = sent.get('text', '')
                if target_lower in text.lower():
                    entity_sentences.append(clean_answer_text(text))
            if entity_sentences:
                combined = []
                for s in entity_sentences[:5]:
                    if len(s.split()) >= 3:
                        combined.append(s)
                if combined:
                    return ' '.join(combined)
        
        # Fourth: look for definitional patterns in any sentence
        for sent in sentences:
            text = sent.get('text', '').lower()
            if any(word in text for word in ['is', 'was', 'means', 'refers']):
                return clean_answer_text(sent.get('text', ''))
        
        return clean_answer_text(sentences[0].get('text', '')) if sentences else ""

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
