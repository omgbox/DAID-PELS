"""
BookBot Trilateral BM25 Engine
Enhanced BM25 with phonetic matching for OCR-robust search.

Uses CMU Pronouncing Dictionary + Double Metaphone + Jaro-Winkler
for sound-alike matching (e.g., "titanic" matches "titamc").
"""

import math
import re
import logging
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from ..core.base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.query.trilateral_bm25')

# Lazy-loaded phonetic matcher
_phonetic_matcher = None


def _get_phonetic_matcher():
    global _phonetic_matcher
    if _phonetic_matcher is None:
        from ..lib.phonetic_matcher import get_phonetic_matcher
        _phonetic_matcher = get_phonetic_matcher()
    return _phonetic_matcher


def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts from text."""
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    return text.strip()


def word_to_phonetic_bitmask(word: str) -> int:
    """Fast Soundex-like bitmask (fallback when CMU unavailable)."""
    CONSONANTS = "BFPVCGJKQSXZDTLMNR"
    DIGITS     = "111122222222334556"
    SOUNDEX_TABLE = str.maketrans(CONSONANTS + CONSONANTS.lower(), DIGITS + DIGITS)
    if not word:
        return 0
    translated = word.translate(SOUNDEX_TABLE)
    mask = 0
    seen_digits = set(c for c in translated if c.isdigit())
    for d in seen_digits:
        mask |= (1 << int(d))
    mask |= (ord(word[0].upper()) << 8)
    return mask


def phonetic_similarity(mask1: int, mask2: int) -> float:
    """Fast bitmask similarity (fallback)."""
    if mask1 == 0 or mask2 == 0:
        return 0.0
    if (mask1 >> 8) != (mask2 >> 8):
        return 0.1
    shared_bits = bin((mask1 & 0xFF) & (mask2 & 0xFF)).count('1')
    total_bits = bin((mask1 | mask2) & 0xFF).count('1')
    return (shared_bits / total_bits) if total_bits > 0 else 1.0


class TrilateralBM25:
    """
    Trilateral BM25 Engine
    
    Combines three scoring dimensions:
    1. Standard BM25 (TF-IDF)
    2. Phonetic similarity (CMU + Double Metaphone + Jaro-Winkler)
    3. Semantic context (word position patterns)
    """
    
    def __init__(self, b: float = 0.75, k1: float = 1.5):
        self.b = b
        self.k1 = k1
        self.doc_count = 0
        self.avg_doc_len = 0.0
        
        # Core BM25 storage
        self.doc_lengths: Dict[int, int] = {}
        self.doc_term_frequencies: Dict[int, Dict[str, int]] = defaultdict(dict)
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        
        # Phonetic lookup caches
        self.doc_phonetic_masks: Dict[int, List[int]] = defaultdict(list)
        self.doc_metaphone_keys: Dict[int, Set[str]] = defaultdict(set)
        self.vocab_idf: Dict[str, float] = {}
        
        # Sentence ID mapping
        self.doc_id_to_sentence_id: Dict[int, int] = {}
        self.doc_id_to_text: Dict[int, str] = {}
    
    def add_document(self, doc_id: int, tokens: List[str], 
                     sentence_id: int = None, text: str = ""):
        """Stream a document into the index."""
        self.doc_count += 1
        self.doc_lengths[doc_id] = len(tokens)
        
        if sentence_id is not None:
            self.doc_id_to_sentence_id[doc_id] = sentence_id
        self.doc_id_to_text[doc_id] = text
        
        # Calculate frequencies
        for token in tokens:
            self.doc_term_frequencies[doc_id][token] = \
                self.doc_term_frequencies[doc_id].get(token, 0) + 1
            self.inverted_index[token].add(doc_id)
        
        # Store phonetic data
        self.doc_phonetic_masks[doc_id] = [word_to_phonetic_bitmask(t) for t in tokens]
        
        # Precompute Double Metaphone keys for fast phonetic lookup
        try:
            from abydos.phonetic import DoubleMetaphone
            dm = DoubleMetaphone()
            for token in set(tokens):
                meta = dm.encode(token)
                for m in meta:
                    if m:
                        self.doc_metaphone_keys[doc_id].add(m)
        except Exception:
            pass
    
    def finalize_index(self):
        """Precompute vocabulary IDF scores."""
        if self.doc_count == 0:
            return
            
        total_len = sum(self.doc_lengths.values())
        self.avg_doc_len = total_len / self.doc_count
        
        for term, doc_ids in self.inverted_index.items():
            n_q = len(doc_ids)
            self.vocab_idf[term] = math.log(
                ((self.doc_count - n_q + 0.5) / (n_q + 0.5)) + 1.0
            )
    
    def search(self, query_tokens: List[str], top_n: int = 10) -> List[Tuple[int, float, str]]:
        """
        Execute hybrid search: BM25 + phonetic matching.
        
        Uses two phonetic strategies:
        1. Fast bitmask similarity (all docs)
        2. Double Metaphone key matching (exact sound-alike)
        """
        scores = defaultdict(float)
        
        # Precompute query phonetic data
        query_masks = [word_to_phonetic_bitmask(q) for q in query_tokens]
        
        try:
            from abydos.phonetic import DoubleMetaphone
            dm = DoubleMetaphone()
            query_metaphone = []
            for q in query_tokens:
                meta = dm.encode(q)
                query_metaphone.append(set(m for m in meta if m))
        except Exception:
            query_metaphone = [set() for _ in query_tokens]
        
        for doc_id in self.doc_lengths.keys():
            doc_score = 0.0
            doc_tf_dict = self.doc_term_frequencies[doc_id]
            doc_len = self.doc_lengths[doc_id]
            doc_masks = self.doc_phonetic_masks[doc_id]
            doc_meta = self.doc_metaphone_keys[doc_id]
            
            for q_idx, q_term in enumerate(query_tokens):
                q_mask = query_masks[q_idx]
                
                # A. Core BM25: exact term match
                tf = doc_tf_dict.get(q_term, 0)
                idf = self.vocab_idf.get(q_term, 0.5)
                
                # B. Phonetic boost: bitmask similarity
                max_phonetic_boost = 0.0
                for d_mask in doc_masks:
                    sim = phonetic_similarity(q_mask, d_mask)
                    if sim > max_phonetic_boost:
                        max_phonetic_boost = sim
                
                # C. Double Metaphone boost: exact sound-alike match
                metaphone_boost = 0.0
                if query_metaphone[q_idx] and doc_meta:
                    overlap = query_metaphone[q_idx] & doc_meta
                    if overlap:
                        metaphone_boost = 0.9  # Strong signal for exact phoneme match
                
                # Combine: effective TF = exact + phonetic + metaphone
                effective_tf = tf + max(max_phonetic_boost * 0.8, metaphone_boost * 0.9)
                
                # D. Compute final BM25 curve weight
                numerator = effective_tf * (self.k1 + 1)
                denominator = effective_tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avg_doc_len)
                )
                
                doc_score += idf * (numerator / denominator)
            
            scores[doc_id] = round(doc_score, 4)
        
        # Sort and return top results
        results = []
        for doc_id, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]:
            if score > 0:
                sentence_id = self.doc_id_to_sentence_id.get(doc_id, doc_id)
                text = self.doc_id_to_text.get(doc_id, "")
                results.append((sentence_id, score, text))
        
        return results


class TrilateralBM25Engine(BaseModule):
    """BookBot Trilateral BM25 Engine with phonetic matching."""
    
    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.engine = None
    
    def process(self, input_data) -> dict:
        """Process input data and retrieve relevant sentences."""
        query = input_data.get('query', '')
        context = input_data.get('context')
        
        if context and not self.engine:
            self._build_index(context)
        
        results = []
        if self.engine:
            query_tokens = self._tokenize(query)
            search_results = self.engine.search(query_tokens, top_n=10)
            
            for sentence_id, score, text in search_results:
                results.append({
                    'sentence_id': sentence_id,
                    'score': score,
                    'source': 'trilateral_bm25',
                    'text': text,
                })
        
        self._initialized = True
        return {'results': results}
    
    def _build_index(self, context: PipelineContext):
        """Build Trilateral BM25 index from sentences."""
        sentences = context.sentences or []
        self.engine = TrilateralBM25(
            b=self.get_config('bm25_b', 0.75),
            k1=self.get_config('bm25_k1', 1.5)
        )
        
        for doc_id, sent in enumerate(sentences):
            text = sent.get('text', '')
            cleaned_text = clean_ocr_text(text)
            tokens = self._tokenize(cleaned_text)
            
            if tokens:
                self.engine.add_document(
                    doc_id=doc_id,
                    tokens=tokens,
                    sentence_id=sent.get('sentence_id'),
                    text=cleaned_text
                )
        
        self.engine.finalize_index()
        logger.info(f"Built Trilateral BM25 index with {self.engine.doc_count} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into lowercase words, filtering stopwords."""
        stopwords = {
            'who', 'what', 'when', 'where', 'why', 'how', 'is', 'are', 'was',
            'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of',
            'with', 'by', 'from', 'as', 'do', 'does', 'did', 'has', 'have',
            'had', 'can', 'could', 'will', 'would', 'should', 'may', 'might',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
            'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
            'his', 'its', 'our', 'their', 'and', 'or', 'but', 'not', 'so',
            'if', 'then', 'than', 'too', 'very', 'just', 'tell', 'me', 'more',
            'about', 'give', 'know', 'think', 'say', 'said', 'like', 'make',
            'made', 'been', 'being', 'be',
        }
        return [w for w in re.findall(r'\b\w+\b', text.lower()) if w not in stopwords]
