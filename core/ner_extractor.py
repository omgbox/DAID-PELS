"""
BookBot NER Extractor (Optimized)
Named Entity Recognition using gazetteer + regex (no slow NLTK ne_chunk).
"""

import re
import logging
from typing import Dict, List, Optional

from .base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.core.ner_extractor')


class NERExtractor(BaseModule):
    """Named Entity Recognition module (optimized for speed)."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.gazetteer = set()

    def process(self, input_data) -> dict:
        """
        Process input data and extract named entities.

        Args:
            input_data: PipelineContext or dict with 'sentences' key

        Returns:
            Dict with 'entities' and 'entity_mentions' keys
        """
        if isinstance(input_data, PipelineContext):
            sentences = input_data.sentences
        else:
            sentences = input_data.get('sentences', [])

        # Build gazetteer from proper nouns
        self._build_gazetteer(sentences)

        # Extract entities
        entities = []
        entity_mentions = []

        for sent in sentences:
            sent_entities, sent_mentions = self._extract_entities(sent)
            entities.extend(sent_entities)
            entity_mentions.extend(sent_mentions)

        # Merge duplicate entities
        entities = self._merge_entities(entities)

        # Update mentions with entity IDs
        entity_map = {e['canonical_name']: e['entity_id'] for e in entities}
        for mention in entity_mentions:
            name = mention.get('mention_text', '')
            if name in entity_map:
                mention['entity_id'] = entity_map[name]

        self._initialized = True
        self.logger.info(f"Found {len(entities)} entities, {len(entity_mentions)} mentions")

        return {
            'entities': entities,
            'entity_mentions': entity_mentions,
        }

    def _build_gazetteer(self, sentences: List[Dict]):
        """
        Build gazetteer from proper nouns in text.
        Excludes common English words that happen to start sentences.
        """
        COMMON_WORDS = {
            'ah', 'oh', 'alas', 'well', 'why', 'indeed', 'aye', 'nay',
            'dear', 'look', 'yes', 'no', 'so', 'now', 'then', 'here', 'there',
            'chapter', 'volume', 'book', 'part', 'section', 'page',
            'preface', 'introduction', 'appendix', 'index', 'contents',
            'illustrations', 'contents', 'list', 'note', 'notes',
            'mr', 'mrs', 'ms', 'miss', 'dr', 'prof', 'st', 'rev',
            'vol', 'ch', 'p', 'pp', 'no',
        }

        proper_nouns = {}
        lowercase_count = {}
        for sent in sentences:
            for token in sent.get('tokens', []):
                pos = token.get('pos_tag', '')
                word = token.get('token', '')
                if pos in ('NNP', 'NNPS'):
                    proper_nouns[word] = proper_nouns.get(word, 0) + 1
                elif word.isalpha():
                    lower = word.lower()
                    lowercase_count[lower] = lowercase_count.get(lower, 0) + 1

        min_freq = self.get_config('min_entity_frequency', 2)
        self.gazetteer = set()
        for word, freq in proper_nouns.items():
            lower = word.lower()
            if freq < min_freq:
                continue
            if lower in COMMON_WORDS:
                continue
            # Exclude if the word appears as a common noun more often than as proper noun
            lower_freq = lowercase_count.get(lower, 0)
            if lower_freq > freq:
                continue
            self.gazetteer.add(word)

        self.logger.info(f"Built gazetteer with {len(self.gazetteer)} entries")

    def _extract_entities(self, sent: Dict) -> tuple:
        """
        Extract entities from a sentence (fast method).

        Args:
            sent: Sentence dict

        Returns:
            Tuple of (entities, mentions)
        """
        entities = []
        mentions = []

        tokens = sent.get('tokens', [])
        if not tokens:
            return entities, mentions

        # Method 1: Gazetteer-based (fast)
        gaz_entities = self._extract_with_gazetteer(tokens, sent)
        entities.extend(gaz_entities[0])
        mentions.extend(gaz_entities[1])

        # Method 2: Regex patterns (fast)
        regex_entities = self._extract_with_regex(sent)
        entities.extend(regex_entities[0])
        mentions.extend(regex_entities[1])

        return entities, mentions

    def _extract_with_gazetteer(self, tokens: List[Dict], sent: Dict) -> tuple:
        """
        Extract entities using gazetteer.

        Args:
            tokens: List of token dicts
            sent: Sentence dict

        Returns:
            Tuple of (entities, mentions)
        """
        entities = []
        mentions = []

        # Look for proper noun sequences
        current_entity = []
        for token in tokens:
            pos = token.get('pos_tag', '')
            if pos in ('NNP', 'NNPS'):
                current_entity.append(token['token'])
            else:
                if current_entity:
                    entity_text = ' '.join(current_entity)
                    if entity_text in self.gazetteer:
                        entities.append({
                            'canonical_name': entity_text,
                            'entity_type': 'PERSON',  # Default type
                            'confidence': 0.7,
                        })
                        mentions.append({
                            'mention_text': entity_text,
                            'sentence_id': sent.get('sentence_id'),
                            'is_pronoun': False,
                        })
                    current_entity = []

        # Handle last entity
        if current_entity:
            entity_text = ' '.join(current_entity)
            if entity_text in self.gazetteer:
                entities.append({
                    'canonical_name': entity_text,
                    'entity_type': 'PERSON',
                    'confidence': 0.7,
                })
                mentions.append({
                    'mention_text': entity_text,
                    'sentence_id': sent.get('sentence_id'),
                    'is_pronoun': False,
                })

        return entities, mentions

    def _extract_with_regex(self, sent: Dict) -> tuple:
        """
        Extract entities using regex patterns.

        Args:
            sent: Sentence dict

        Returns:
            Tuple of (entities, mentions)
        """
        entities = []
        mentions = []
        text = sent.get('text', '')

        # Date patterns
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        dates = re.findall(date_pattern, text)
        for date in dates:
            entities.append({
                'canonical_name': date,
                'entity_type': 'DATE',
                'confidence': 0.9,
            })

        # Title + Name patterns
        title_pattern = r'\b(Mr|Mrs|Dr|Captain|Sir|Lady)\.?\s+([A-Z][a-z]+)\b'
        titles = re.findall(title_pattern, text)
        for title, name in titles:
            full_name = f"{title} {name}"
            entities.append({
                'canonical_name': full_name,
                'entity_type': 'PERSON',
                'confidence': 0.85,
            })
            mentions.append({
                'mention_text': full_name,
                'sentence_id': sent.get('sentence_id'),
                'is_pronoun': False,
            })

        # Money patterns
        money_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        money = re.findall(money_pattern, text)
        for amount in money:
            entities.append({
                'canonical_name': amount,
                'entity_type': 'MONEY',
                'confidence': 0.9,
            })

        return entities, mentions

    def _merge_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge duplicate entities.

        Args:
            entities: List of entity dicts

        Returns:
            Merged list of entities
        """
        merged = {}
        mention_counts = {}
        for entity in entities:
            name = entity['canonical_name']
            mention_counts[name] = mention_counts.get(name, 0) + 1
            if name in merged:
                # Keep higher confidence
                if entity.get('confidence', 0) > merged[name].get('confidence', 0):
                    merged[name] = entity
            else:
                merged[name] = entity

        # Assign entity IDs, frequency, and gender
        for i, entity in enumerate(merged.values()):
            entity['entity_id'] = i + 1
            entity['frequency'] = mention_counts.get(entity['canonical_name'], 1)
            entity['gender'] = self._extract_gender(entity)

        return list(merged.values())

    def _extract_gender(self, entity: Dict) -> str:
        """
        Extract gender from entity name or definition.

        Args:
            entity: Entity dict with canonical_name

        Returns:
            Gender string: 'masculine', 'feminine', 'neuter', 'plural', or 'unknown'
        """
        name = entity.get('canonical_name', '').lower()

        # Common masculine names ending
        masculine_endings = {'mr.', 'mr', 'he', 'his'}
        # Common feminine names ending
        feminine_endings = {'mrs.', 'mrs', 'ms.', 'miss', 'she', 'her', 'hers'}
        # Common neutral/neuter
        neuter_words = {'it', 'its', 'the doctor', 'the gentleman', 'the lady'}

        # Check for obvious gender markers in name
        if name in masculine_endings:
            return 'masculine'
        elif name in feminine_endings:
            return 'feminine'
        elif name in neuter_words:
            return 'neuter'
        # Check for plural forms
        elif name.endswith('s') and name not in {'and', 'has', 'was', 'his', 'hers', 'its'}:
            return 'plural'

        # Try to determine from context
        if self.db:
            try:
                # Check if name appears with gender indicators in definitions
                definitions = self.db.select(
                    'definitions',
                    columns='definition',
                    where='word_lower = ?',
                    where_params=(name,),
                    limit=3
                )
                for definition, in definitions:
                    if 'man' in definition.lower():
                        return 'masculine'
                    elif 'woman' in definition.lower() or 'lady' in definition.lower():
                        return 'feminine'
                    elif 'person' in definition.lower() and 'gentleman' in definition.lower():
                        return 'masculine'
                    elif 'person' in definition.lower() and 'lady' in definition.lower():
                        return 'feminine'
            except:
                pass

        return 'unknown'
