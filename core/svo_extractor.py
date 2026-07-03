"""
BookBot SVO Extractor
Subject-Verb-Object triple extraction using multiple methods.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

from .base_module import BaseModule
from ..pipeline_context import PipelineContext
from ..config import CLAUSE

logger = logging.getLogger('bookbot.core.svo_extractor')

# Try to import NLTK
try:
    import nltk
    from nltk import RegexpParser
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# Clause boundary markers (from config)
COORD_CONJUNCTIONS = CLAUSE['coord_conjunctions']
SUBORD_CONJUNCTIONS = CLAUSE['subord_conjunctions']
CLAUSE_BOUNDARY_POS = CLAUSE['clause_boundary_pos']
CLAUSE_PUNCT = CLAUSE['clause_punctuation']


def _is_clause_boundary(tokens: List[Dict], idx: int, clause_config: dict = None) -> bool:
    """Check if token at idx is a clause boundary."""
    if idx < 0 or idx >= len(tokens):
        return False
    token = tokens[idx]
    word = token.get('token', '').lower()
    pos = token.get('pos_tag', '')
    # Use provided config or defaults
    config = clause_config or {}
    
    # Subordinating conjunctions (configurable)
    subj_conj = config.get('subord_conjunctions', SUBORD_CONJUNCTIONS)
    if word in subj_conj:
        return True
    
    # Coordinating conjunctions preceded by comma/semicolon (configurable)
    coord_conj = config.get('coord_conjunctions', COORD_CONJUNCTIONS)
    clause_punct = config.get('clause_punctuation', CLAUSE_PUNCT)
    if word in coord_conj:
        if idx > 0 and tokens[idx - 1].get('token', '') in (',', ';', '--', '—'):
            return True
    
    # Clause punctuation (configurable)
    if word in clause_punct:
        return True
    
    # Relative pronouns (configurable POS)
    rel_pros = config.get('clause_boundary_pos', CLAUSE_BOUNDARY_POS)
    if pos in rel_pros and word not in ('what', 'whatever'):
        return True
    
    return False


class SVOExtractor(BaseModule):
    """Subject-Verb-Object extraction module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.chunk_parser = None

    def process(self, input_data) -> dict:
        """
        Process input data and extract SVO triples.

        Args:
            input_data: PipelineContext or dict with 'sentences' key

        Returns:
            Dict with 'svo_triples' key
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Initialize chunk parser
        if NLTK_AVAILABLE:
            grammar = r"""
                NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}   
                    {<NNP>+}                        
                    {<PRP>}                         
                VP: {<MD>?<VB.*><RP>?}              
                    {<VB.*><PRT>?<NP|PP>*}          
                PP: {<IN><NP>}                      
            """
            self.chunk_parser = RegexpParser(grammar)

        # Extract SVO triples
        svo_triples = []
        max_length = self.get_config('svo_max_sentence_length', 300)

        for sent in sentences:
            tokens = sent.get('tokens', [])
            if len(tokens) > max_length:
                continue

            triples = self._extract_triples(tokens, sent)
            svo_triples.extend(triples)

        self._initialized = True
        self.logger.info(f"Extracted {len(svo_triples)} SVO triples")

        return {'svo_triples': svo_triples}

    def _extract_triples(self, tokens: List[Dict], sent: Dict) -> List[Dict]:
        """
        Extract SVO triples from tokens.

        Args:
            tokens: List of token dicts
            sent: Sentence dict

        Returns:
            List of SVO triple dicts
        """
        triples = []

        # Method 1: Pattern-based extraction (most reliable)
        pattern_triples = self._extract_with_patterns(tokens, sent)
        triples.extend(pattern_triples)

        # Method 2: Chunk-based extraction
        if self.chunk_parser:
            chunk_triples = self._extract_with_chunks(tokens, sent)
            triples.extend(chunk_triples)

        # Method 3: Simple SVO pattern (fallback)
        simple_triples = self._extract_simple_svo(tokens, sent)
        triples.extend(simple_triples)

        # Remove duplicates
        triples = self._deduplicate_triples(triples)

        return triples

    def _extract_with_patterns(self, tokens: List[Dict], sent: Dict) -> List[Dict]:
        """
        Extract SVO using POS tag patterns.

        Args:
            tokens: List of token dicts
            sent: Sentence dict

        Returns:
            List of SVO triple dicts
        """
        triples = []
        sentence_id = sent.get('sentence_id')

        # Find all verbs
        verb_indices = []
        for i, token in enumerate(tokens):
            pos = token.get('pos_tag', '')
            if pos.startswith('VB') or pos == 'MD':
                verb_indices.append(i)

        # For each verb, try to find subject and object
        for verb_idx in verb_indices:
            verb_token = tokens[verb_idx]
            verb = verb_token.get('token', '')
            verb_pos = verb_token.get('pos_tag', '')

            # Skip auxiliary verbs (is, was, has, etc.) unless they're main verbs
            if verb.lower() in ('is', 'are', 'was', 'were', 'has', 'have', 'had', 'do', 'does', 'did'):
                # Check if there's a main verb after
                if verb_idx + 1 < len(tokens):
                    next_pos = tokens[verb_idx + 1].get('pos_tag', '')
                    if next_pos.startswith('VB') and next_pos != 'VBZ':
                        continue

            # Find subject (before verb)
            subject = self._find_subject(tokens, verb_idx)

            # Find object (after verb)
            object_ = self._find_object(tokens, verb_idx)

            if subject and verb:
                triple = {
                    'subject': subject,
                    'verb': verb,
                    'object': object_ or '',
                    'sentence_id': sentence_id,
                    'confidence': 0.6,
                    'passive': False,
                }
                triples.append(triple)

        # Check for passive voice
        passive_triples = self._extract_passive(tokens, sent)
        triples.extend(passive_triples)

        return triples

    # Auxiliary verbs to skip when searching for subject
    AUX_VERBS = {
        'is', 'am', 'are', 'was', 'were', 'been', 'being', 'be',
        'has', 'have', 'had', 'having',
        'do', 'does', 'did', 'doing',
        'will', 'would', 'can', 'could', 'shall', 'should',
        'may', 'might', 'must', 'need', 'dare', 'used',
    }

    def _find_subject(self, tokens: List[Dict], verb_idx: int) -> Optional[str]:
        """
        Find subject of a verb using dependency-inspired search.
        Skips auxiliary verbs (was, had, would) and adverbs to reach the subject,
        mimicking how dependency trees connect nsubj directly past auxiliaries.

        Args:
            tokens: List of token dicts
            verb_idx: Index of the verb

        Returns:
            Subject text or None
        """
        subject_tokens = []
        i = verb_idx - 1
        clause_config = self.get_config('clause', {})

        # Skip adverbs and auxiliary verbs before the main verb
        # (dependency parsing: nsubj connects past aux nodes)
        while i >= 0:
            if _is_clause_boundary(tokens, i, clause_config):
                break
            pos = tokens[i].get('pos_tag', '')
            word = tokens[i].get('token', '').lower()
            if pos in ('RB', 'RBR', 'RBS'):  # Adverbs
                i -= 1
                continue
            if word in self.AUX_VERBS and pos.startswith('VB'):
                i -= 1
                continue
            if pos == 'TO':  # "to" before infinitive
                i -= 1
                continue
            break

        # Collect noun phrase, stopping at clause boundaries
        while i >= 0:
            if _is_clause_boundary(tokens, i, clause_config):
                break
            pos = tokens[i].get('pos_tag', '')
            token = tokens[i].get('token', '')

            if pos.startswith('NN') or pos == 'PRP' or pos in ('DT', 'JJ', 'JJR', 'JJS', 'PRP$', 'POS', 'CD'):
                subject_tokens.insert(0, token)
                i -= 1
            else:
                break

        return ' '.join(subject_tokens) if subject_tokens else None

    def _find_object(self, tokens: List[Dict], verb_idx: int) -> Optional[str]:
        """
        Find object of a verb, respecting clause boundaries.

        Args:
            tokens: List of token dicts
            verb_idx: Index of the verb

        Returns:
            Object text or None
        """
        object_tokens = []
        i = verb_idx + 1
        clause_config = self.get_config('clause', {})

        # Skip particles, adverbs, and auxiliary verbs after the main verb
        while i < len(tokens):
            if _is_clause_boundary(tokens, i, clause_config):
                break
            pos = tokens[i].get('pos_tag', '')
            word = tokens[i].get('token', '').lower()
            if pos == 'RP' or pos in ('RB', 'RBR', 'RBS'):  # Particles/adverbs
                i += 1
                continue
            if word in self.AUX_VERBS and pos.startswith('VB'):
                i += 1
                continue
            if pos == 'TO':  # "to" before infinitive
                i += 1
                continue
            break

        # Collect noun phrase, stopping at clause boundaries
        while i < len(tokens):
            if _is_clause_boundary(tokens, i, clause_config):
                break
            pos = tokens[i].get('pos_tag', '')
            token = tokens[i].get('token', '')

            if pos.startswith('NN') or pos == 'PRP' or pos in ('DT', 'JJ', 'JJR', 'JJS', 'PRP$', 'CD'):
                object_tokens.append(token)
                i += 1
            elif pos == 'IN':  # Preposition - might be part of phrasal verb
                break
            else:
                break

        return ' '.join(object_tokens) if object_tokens else None

    def _extract_passive(self, tokens: List[Dict], sent: Dict) -> List[Dict]:
        """
        Extract passive voice constructions.

        Pattern: NP + be-verb + past-participle + (by NP)

        Args:
            tokens: List of token dicts
            sent: Sentence dict

        Returns:
            List of SVO triple dicts
        """
        triples = []
        sentence_id = sent.get('sentence_id')
        clause_config = self.get_config('clause', {})

        # Look for passive patterns
        for i in range(len(tokens) - 2):
            # Skip if at a clause boundary
            if _is_clause_boundary(tokens, i, clause_config):
                continue
            # Check for be-verb + past participle
            if (tokens[i].get('pos_tag', '').startswith('VB') and
                tokens[i].get('token', '').lower() in ('is', 'are', 'was', 'were', 'been', 'being')):
                
                # Check if next token is past participle
                if i + 1 < len(tokens) and tokens[i + 1].get('pos_tag') == 'VBN':
                    verb = tokens[i + 1].get('token', '')
                    
                    # Find subject (before be-verb)
                    subject = self._find_subject(tokens, i)
                    
                    # Look for "by" phrase (agent)
                    agent = None
                    for j in range(i + 2, min(i + 5, len(tokens))):
                        if _is_clause_boundary(tokens, j, clause_config):
                            break
                        if tokens[j].get('token', '').lower() == 'by':
                            agent = self._find_object(tokens, j)
                            break

                    if subject and verb:
                        triples.append({
                            'subject': agent or 'someone',
                            'verb': verb,
                            'object': subject,
                            'sentence_id': sentence_id,
                            'confidence': 0.5,
                            'passive': True,
                        })

        return triples

    def _extract_with_chunks(self, tokens: List[Dict], sent: Dict) -> List[Dict]:
        """
        Extract SVO using chunk parsing with clause-level isolation.
        Splits chunk sequence at clause boundaries (constituency-inspired),
        then extracts NP-VP-NP patterns from each clause independently.

        Args:
            tokens: List of token dicts
            sent: Sentence dict

        Returns:
            List of SVO triple dicts
        """
        triples = []
        sentence_id = sent.get('sentence_id')

        try:
            # Prepare POS-tagged words with original indices
            indexed_words = [(i, t['token'], t.get('pos_tag', 'NN'))
                             for i, t in enumerate(tokens) if not t.get('is_punctuation')]

            if len(indexed_words) < 3:
                return triples

            # Split into clause groups at clause boundaries
            clause_groups = []
            current = []
            clause_config = self.get_config('clause', {})
            for idx, word, pos in indexed_words:
                if current and _is_clause_boundary(tokens, idx, clause_config):
                    if len(current) >= 3:
                        clause_groups.append(current)
                    current = []
                current.append((word, pos))
            if len(current) >= 3:
                clause_groups.append(current)

            # Parse each clause group independently
            for words in clause_groups:
                tree = self.chunk_parser.parse(words)

                # Extract chunks from this clause
                nps = []
                vps = []
                pps = []

                for subtree in tree:
                    if hasattr(subtree, 'label'):
                        chunk_type = subtree.label()
                        chunk_words = ' '.join(word for word, tag in subtree.leaves())
                        if chunk_type == 'NP':
                            nps.append(chunk_words)
                        elif chunk_type == 'VP':
                            vps.append(chunk_words)
                        elif chunk_type == 'PP':
                            pps.append(chunk_words)

                # NP + VP + NP pattern
                if len(nps) >= 2 and len(vps) >= 1:
                    triples.append({
                        'subject': nps[0],
                        'verb': vps[0].split()[0] if vps[0] else '',
                        'object': nps[1],
                        'sentence_id': sentence_id,
                        'confidence': 0.7,
                        'passive': False,
                    })

                # NP + VP + PP pattern
                if len(nps) >= 1 and len(vps) >= 1 and len(pps) >= 1:
                    pp_words = pps[0].split()
                    if len(pp_words) >= 2:
                        prep = pp_words[0]
                        obj = ' '.join(pp_words[1:])
                        triples.append({
                            'subject': nps[0],
                            'verb': f"{vps[0].split()[0]}_{prep}" if vps[0] else prep,
                            'object': obj,
                            'sentence_id': sentence_id,
                            'confidence': 0.6,
                            'passive': False,
                        })

        except Exception as e:
            self.logger.debug(f"Chunk extraction failed: {e}")

        return triples

    def _extract_simple_svo(self, tokens: List[Dict], sent: Dict) -> List[Dict]:
        """
        Extract simple SVO patterns: Noun Verb Noun, respecting clause boundaries.

        Args:
            tokens: List of token dicts
            sent: Sentence dict

        Returns:
            List of SVO triple dicts
        """
        triples = []
        sentence_id = sent.get('sentence_id')
        clause_config = self.get_config('clause', {})

        # Simple pattern: find verb and look for nouns around it
        for i in range(1, len(tokens) - 1):
            pos = tokens[i].get('pos_tag', '')
            if pos.startswith('VB'):
                verb = tokens[i].get('token', '')
                
                # Look for subject (noun before verb, same clause)
                subject = None
                for j in range(i - 1, max(0, i - 4), -1):
                    if _is_clause_boundary(tokens, j, clause_config):
                        break
                    if tokens[j].get('pos_tag', '').startswith('NN'):
                        subject = tokens[j].get('token', '')
                        break

                # Look for object (noun after verb, same clause)
                object_ = None
                for j in range(i + 1, min(len(tokens), i + 4)):
                    if _is_clause_boundary(tokens, j, clause_config):
                        break
                    if tokens[j].get('pos_tag', '').startswith('NN'):
                        object_ = tokens[j].get('token', '')
                        break

                if subject and verb:
                    triples.append({
                        'subject': subject,
                        'verb': verb,
                        'object': object_ or '',
                        'sentence_id': sentence_id,
                        'confidence': 0.5,
                        'passive': False,
                    })

        return triples

    def _deduplicate_triples(self, triples: List[Dict]) -> List[Dict]:
        """
        Remove duplicate triples.

        Args:
            triples: List of SVO triple dicts

        Returns:
            Deduplicated list
        """
        seen = set()
        unique = []

        for triple in triples:
            key = (triple['sentence_id'], triple['subject'], triple['verb'], triple['object'])
            if key not in seen:
                seen.add(key)
                unique.append(triple)

        return unique
