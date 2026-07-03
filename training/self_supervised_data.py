"""
Self-Supervised Training Data Generator
Creates training pairs from the book itself — no human labeling needed.

Tasks:
  1. Entity-Query Matching: "Who is X?" → sentences about X (positive)
                            random sentences (negative)
  2. Coreference Pairs: "she/he" → entity it refers to
  3. Sentence Ordering: sentence N → sentence N+1 (positive)
                         sentence N → random sentence (negative)
  4. Token Relevance: tokens near entity name (positive)
                      random tokens (negative)
"""

import re
import random
import logging
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger('bookbot.training.self_supervised')


class SelfSupervisedDataGenerator:
    """
    Generates training data from a tokenized book and entity annotations.
    """

    def __init__(self):
        pass

    def generate_all(self, sentences: List[Dict], entities: List[Dict],
                     svo_triples: List[Dict]) -> Dict:
        """
        Generate all training tasks from book data.

        Args:
            sentences: list of dicts with 'sentence_id', 'text', 'tokens'
            entities: list of dicts with 'entity_id', 'canonical_name'
            svo_triples: list of dicts with 'subject', 'verb', 'object', 'sentence_id'

        Returns:
            Dict with training data for each task
        """
        logger.info(f"Generating training data from {len(sentences)} sentences, "
                     f"{len(entities)} entities, {len(svo_triples)} SVO triples")

        # Build lookups
        entity_names = {e['canonical_name'].lower(): e['canonical_name']
                        for e in entities if e.get('canonical_name')}

        # Group sentences by entity
        entity_sentences = defaultdict(list)
        for sent in sentences:
            text = sent.get('text', '').lower()
            for name_lower, name in entity_names.items():
                if name_lower in text:
                    entity_sentences[name_lower].append(sent)

        # Task 1: Entity-Query Matching
        query_pairs = self._generate_query_pairs(
            sentences, entity_names, entity_sentences
        )

        # Task 2: Coreference Pairs
        coref_pairs = self._generate_coref_pairs(
            sentences, entity_names, entity_sentences
        )

        # Task 3: Sentence Ordering
        order_pairs = self._generate_order_pairs(sentences)

        # Task 4: Token Relevance
        token_pairs = self._generate_token_pairs(
            sentences, entity_names, entity_sentences
        )

        total = (len(query_pairs) + len(coref_pairs) +
                 len(order_pairs) + len(token_pairs))

        logger.info(f"Generated {total} training pairs:")
        logger.info(f"  Query pairs: {len(query_pairs)}")
        logger.info(f"  Coref pairs: {len(coref_pairs)}")
        logger.info(f"  Order pairs: {len(order_pairs)}")
        logger.info(f"  Token pairs: {len(token_pairs)}")

        return {
            'query_pairs': query_pairs,
            'coref_pairs': coref_pairs,
            'order_pairs': order_pairs,
            'token_pairs': token_pairs,
        }

    def _generate_query_pairs(self, sentences: List[Dict],
                               entity_names: Dict,
                               entity_sentences: Dict) -> List[Dict]:
        """
        Generate "Who is X?" → sentence relevance pairs.

        Positive: sentence contains entity name
        Negative: random sentence that doesn't contain entity name
        """
        pairs = []
        query_templates = [
            "Who is {entity}?",
            "Tell me about {entity}.",
            "What do we know about {entity}?",
            "Describe {entity}.",
            "Who was {entity}?",
        ]

        for name_lower, name in entity_names.items():
            pos_sents = entity_sentences.get(name_lower, [])
            if not pos_sents:
                continue

            neg_sents = [s for s in sentences
                         if name_lower not in s.get('text', '').lower()]

            for template in random.sample(query_templates, min(3, len(query_templates))):
                query = template.format(entity=name)

                for pos_sent in pos_sents[:5]:  # max 5 per entity
                    pairs.append({
                        'query': query,
                        'positive': pos_sent.get('text', ''),
                        'negative': random.choice(neg_sents).get('text', '') if neg_sents else '',
                        'label': 1.0,
                    })

                # Add a negative pair
                if neg_sents:
                    neg = random.choice(neg_sents)
                    pairs.append({
                        'query': query,
                        'positive': pos_sents[0].get('text', ''),
                        'negative': neg.get('text', ''),
                        'label': 0.0,
                    })

        random.shuffle(pairs)
        return pairs[:500]  # Cap at 500

    def _generate_coref_pairs(self, sentences: List[Dict],
                               entity_names: Dict,
                               entity_sentences: Dict) -> List[Dict]:
        """
        Generate coreference pairs: pronoun → entity.

        Heuristic: if a sentence has "entity. She/he ..." → "She" refers to entity
        """
        pairs = []
        PRONOUNS = {'she': 'f', 'he': 'm', 'him': 'm', 'her': 'f',
                     'his': 'm', 'hers': 'f'}

        # Sort sentences by ID
        sorted_sents = sorted(sentences, key=lambda s: s.get('sentence_id', 0))

        for i, sent in enumerate(sorted_sents[:-1]):
            text = sent.get('text', '')
            next_text = sorted_sents[i + 1].get('text', '')

            # Find entity in current sentence
            for name_lower, name in entity_names.items():
                if name_lower in text.lower():
                    # Check if next sentence starts with a pronoun
                    next_words = next_text.split()
                    if next_words and next_words[0].lower() in PRONOUNS:
                        pronoun = next_words[0].lower()
                        gender = PRONOUNS[pronoun]

                        # Check gender match
                        if (gender == 'f' and name_lower in {'elizabeth', 'jane', 'lydia', 'kitty', 'mary', 'caroline', 'charlotte', 'catherine', 'anne'}):
                            pairs.append({
                                'context': text,
                                'pronoun': pronoun,
                                'entity': name,
                                'label': 1.0,
                            })
                        elif (gender == 'm' and name_lower in {'darcy', 'bingley', 'wickham', 'collins', 'bennet', 'hurst', 'gardiner'}):
                            pairs.append({
                                'context': text,
                                'pronoun': pronoun,
                                'entity': name,
                                'label': 1.0,
                            })

        random.shuffle(pairs)
        return pairs[:300]

    def _generate_order_pairs(self, sentences: List[Dict]) -> List[Dict]:
        """
        Generate sentence ordering pairs.

        Positive: sentence N, sentence N+1 (adjacent)
        Negative: sentence N, random sentence
        """
        pairs = []
        sorted_sents = sorted(sentences, key=lambda s: s.get('sentence_id', 0))

        for i in range(len(sorted_sents) - 1):
            s1 = sorted_sents[i].get('text', '')
            s2 = sorted_sents[i + 1].get('text', '')

            if s1 and s2 and len(s1) > 20 and len(s2) > 20:
                # Positive: adjacent
                pairs.append({
                    'sentence1': s1,
                    'sentence2': s2,
                    'label': 1.0,  # correct order
                })

                # Negative: random
                j = random.randint(0, len(sorted_sents) - 1)
                if j != i and j != i + 1:
                    s_rand = sorted_sents[j].get('text', '')
                    if s_rand and len(s_rand) > 20:
                        pairs.append({
                            'sentence1': s1,
                            'sentence2': s_rand,
                            'label': 0.0,  # wrong order
                        })

        random.shuffle(pairs)
        return pairs[:500]

    def _generate_token_pairs(self, sentences: List[Dict],
                               entity_names: Dict,
                               entity_sentences: Dict) -> List[Dict]:
        """
        Generate token relevance pairs.

        Positive: tokens near entity mention (within 5 words)
        Negative: random tokens in the sentence
        """
        pairs = []

        for name_lower, name in entity_names.items():
            for sent in entity_sentences.get(name_lower, [])[:10]:
                text = sent.get('text', '')
                words = text.split()
                lower_words = [w.lower().strip('.,;:!?') for w in words]

                # Find entity position
                for idx, w in enumerate(lower_words):
                    if name_lower in w:
                        # Positive: tokens within 5 words
                        for j in range(max(0, idx - 5), min(len(words), idx + 6)):
                            if j != idx:
                                pairs.append({
                                    'context': text,
                                    'target_token': words[j],
                                    'entity_token': words[idx],
                                    'position_diff': abs(j - idx),
                                    'label': 1.0,
                                })

                        # Negative: random token far from entity
                        far_indices = [k for k in range(len(words))
                                       if abs(k - idx) > 8]
                        if far_indices:
                            neg_idx = random.choice(far_indices)
                            pairs.append({
                                'context': text,
                                'target_token': words[neg_idx],
                                'entity_token': words[idx],
                                'position_diff': abs(neg_idx - idx),
                                'label': 0.0,
                            })
                        break

        random.shuffle(pairs)
        return pairs[:500]


class TrainingDataLoader:
    """Load generated training data from disk."""

    @staticmethod
    def save(data: Dict, path: str):
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved training data to {path}")

    @staticmethod
    def load(path: str) -> Dict:
        with open(path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded training data from {path}")
        return data
