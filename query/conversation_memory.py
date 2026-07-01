"""
BookBot Conversation Memory
Tracks entities, topics, and context across conversation turns.
"""

import re
import logging
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger('bookbot.query.conversation_memory')


@dataclass
class EntityMention:
    """A tracked entity with metadata."""
    name: str
    normalized: str
    mention_count: int = 1
    last_mentioned_turn: int = 0
    gender: str = 'unknown'  # 'masculine', 'feminine', 'neuter', 'unknown'
    source: str = 'user'     # 'user', 'answer', 'both'


class EntityStack:
    """Tracks entities across conversation turns with recency weighting."""
    
    def __init__(self, max_size: int = 20):
        self._entities: Dict[str, EntityMention] = {}
        self._recency_order: deque = deque(maxlen=max_size)
        self._max_size = max_size
    
    def push(self, name: str, turn_id: int, gender: str = 'unknown', source: str = 'user'):
        """Add or update an entity."""
        normalized = name.lower().strip()
        
        if normalized in self._entities:
            ent = self._entities[normalized]
            ent.mention_count += 1
            ent.last_mentioned_turn = turn_id
            ent.source = 'both' if ent.source != source else source
            # Move to front of recency
            if normalized in self._recency_order:
                self._recency_order.remove(normalized)
            self._recency_order.appendleft(normalized)
        else:
            self._entities[normalized] = EntityMention(
                name=name,
                normalized=normalized,
                mention_count=1,
                last_mentioned_turn=turn_id,
                gender=gender,
                source=source,
            )
            self._recency_order.appendleft(normalized)
        
        # Evict oldest if over capacity
        while len(self._recency_order) > self._max_size:
            evicted = self._recency_order.pop()
            self._entities.pop(evicted, None)
    
    def get_salient(self, n: int = 5) -> List[EntityMention]:
        """Get the N most salient entities."""
        if not self._entities:
            return []
        
        current_turn = max(e.last_mentioned_turn for e in self._entities.values())
        
        def salience(ent: EntityMention) -> float:
            recency = 1.0 / (1.0 + (current_turn - ent.last_mentioned_turn))
            frequency = min(ent.mention_count / 5.0, 1.0)
            return recency * 0.7 + frequency * 0.3
        
        ranked = sorted(self._entities.values(), key=salience, reverse=True)
        return ranked[:n]
    
    def find_by_gender(self, gender: str) -> Optional[EntityMention]:
        """Find the most recent entity matching a gender."""
        for normalized in self._recency_order:
            ent = self._entities.get(normalized)
            if ent and ent.gender == gender:
                return ent
        return None
    
    def find_by_name(self, name: str) -> Optional[EntityMention]:
        """Find entity by name."""
        return self._entities.get(name.lower().strip())


class TopicTracker:
    """Tracks conversation topic continuity."""
    
    def __init__(self):
        self.current_keywords: List[str] = []
        self.current_entity: Optional[str] = None
    
    def update(self, query_terms: List[str], entities: List[str]):
        """Update topic based on new query."""
        if entities:
            self.current_entity = entities[0]
        if query_terms:
            self.current_keywords = query_terms[:10]
    
    def get_context_terms(self) -> List[str]:
        """Get current topic terms for context expansion."""
        terms = list(self.current_keywords)
        if self.current_entity:
            terms.insert(0, self.current_entity)
        return terms[:5]


class ConversationMemory:
    """
    Manages conversation state including entity tracking,
    topic continuity, and pronoun resolution.
    """
    
    def __init__(self, max_history: int = 10):
        self.history: deque = deque(maxlen=max_history)
        self.entity_stack = EntityStack(max_size=20)
        self.topic_tracker = TopicTracker()
        self._turn_counter = 0
    
    def add_turn(self, user_query: str, answer: str, 
                 entities: List[str] = None, intent: str = 'UNKNOWN'):
        """Add a completed turn to history."""
        turn = {
            'turn_id': self._turn_counter,
            'timestamp': datetime.now(),
            'user_query': user_query,
            'answer': answer,
            'entities': entities or [],
            'intent': intent,
        }
        self.history.append(turn)
        self._turn_counter += 1
        
        # Update entity stack
        for ent in (entities or []):
            self.entity_stack.push(ent, self._turn_counter, 
                                   gender=self._infer_gender(ent),
                                   source='user')
        
        # Extract entities from answer
        answer_entities = self._extract_entities(answer)
        for ent in answer_entities:
            self.entity_stack.push(ent, self._turn_counter,
                                   gender=self._infer_gender(ent),
                                   source='answer')
        
        # Update topic tracker
        self.topic_tracker.update(
            query_terms=self._extract_terms(user_query),
            entities=entities or []
        )
    
    def get_last_turn(self) -> Optional[Dict]:
        """Get the most recent turn."""
        return self.history[-1] if self.history else None
    
    def get_prior_n(self, n: int = 3) -> List[Dict]:
        """Get the N most recent turns."""
        return list(self.history)[-n:]
    
    def resolve_pronoun(self, pronoun: str) -> Optional[str]:
        """Resolve a pronoun to an entity name."""
        pronoun_lower = pronoun.lower().strip()
        
        gender_map = {
            'he': 'masculine', 'him': 'masculine', 'his': 'masculine',
            'she': 'feminine', 'her': 'feminine', 'hers': 'feminine',
            'it': 'neuter', 'its': 'neuter',
            'they': 'plural', 'them': 'plural', 'their': 'plural',
        }
        
        gender = gender_map.get(pronoun_lower)
        if gender:
            entity = self.entity_stack.find_by_gender(gender)
            if entity:
                return entity.name
        
        return None
    
    def detect_followup(self, query: str) -> Optional[Dict]:
        """Detect if query is a follow-up to prior context."""
        query_lower = query.lower().strip()
        
        followup_patterns = [
            (r'^tell me more', 'tell_more'),
            (r'^what about (him|her|them|it)', 'what_about'),
            (r'^and (him|her|them|it)', 'and_reference'),
            (r'^how about (him|her|them|it)', 'how_about'),
            (r'^explain (that|this|it)', 'explain_prior'),
            (r'^(yes|yeah|ok|sure|go on|continue)', 'affirmation'),
            (r'^why\??$', 'causal_followup'),
            (r'^what else', 'elaboration'),
            (r'^did (he|she|they|it)', 'verification'),
        ]
        
        for pattern, followup_type in followup_patterns:
            match = re.match(pattern, query_lower)
            if match:
                referenced = None
                if match.lastindex:
                    pronoun = match.group(1)
                    referenced = self.resolve_pronoun(pronoun)
                
                return {
                    'type': followup_type,
                    'referenced_entity': referenced,
                }
        
        return None
    
    def expand_query(self, query: str) -> str:
        """Expand query with conversation context."""
        # Check for follow-up
        followup = self.detect_followup(query)
        
        if followup:
            if followup['type'] == 'tell_more':
                # Expand with prior context
                last = self.get_last_turn()
                if last:
                    entities = last.get('entities', [])
                    if entities:
                        return f"{query} about {entities[0]}"
            
            elif followup['type'] == 'what_about' and followup['referenced_entity']:
                # Replace pronoun with entity
                return re.sub(
                    r'\b(him|her|them|it)\b',
                    followup['referenced_entity'],
                    query, flags=re.IGNORECASE
                )
        
        # Check for pronouns in query
        pronouns = ['he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'its', 'their']
        words = query.split()
        expanded_words = []
        
        for word in words:
            clean = re.sub(r'[^\w]', '', word).lower()
            if clean in pronouns:
                resolved = self.resolve_pronoun(clean)
                if resolved:
                    expanded_words.append(resolved)
                else:
                    expanded_words.append(word)
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def get_context_entities(self) -> List[str]:
        """Get current context entities for query expansion."""
        return [e.name for e in self.entity_stack.get_salient(5)]
    
    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using simple heuristics."""
        entities = []
        words = text.split()
        for word in words:
            cleaned = word.strip('.,;:!?()[]"\'')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                if cleaned.lower() not in {
                    'the', 'what', 'when', 'where', 'who', 'why', 'how',
                    'this', 'that', 'these', 'those', 'tell', 'please',
                    'according', 'chapter', 'book',
                }:
                    entities.append(cleaned)
        return entities
    
    def _extract_terms(self, text: str) -> List[str]:
        """Extract content terms from text."""
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'of', 'in', 'to', 'for',
            'with', 'on', 'at', 'from', 'by', 'about', 'as', 'what',
            'which', 'who', 'whom', 'whose', 'when', 'where', 'why', 'how',
            'tell', 'me', 'more', 'and', 'or', 'but', 'not', 'it', 'its',
        }
        words = text.lower().split()
        return [w.strip('.,;:!?') for w in words 
                if w.strip('.,;:!?') not in stop_words and len(w) > 2]
    
    def _infer_gender(self, name: str) -> str:
        """Infer gender for an entity name."""
        # Pride and Prejudice characters
        masculine_names = {
            'darcy', 'bingley', 'wickham', 'collins', 'bennet',
            'mr.', 'mr', 'william', 'charles', 'george', 'fitzwilliam',
            'gardiner', 'phillips', 'lucas', 'denny', 'forster',
        }
        feminine_names = {
            'elizabeth', 'jane', 'lydia', 'kitty', 'mary', 'bennet',
            'mrs.', 'mrs', 'miss', 'charlotte', 'anne', 'catherine',
            'georgiana', 'caroline', 'louisa', 'hurley',
        }
        
        normalized = name.lower().strip()
        
        # Check direct matches
        if normalized in masculine_names:
            return 'masculine'
        elif normalized in feminine_names:
            return 'feminine'
        
        # Check prefixes
        if normalized.startswith('mr') or normalized.startswith('sir'):
            return 'masculine'
        elif normalized.startswith('mrs') or normalized.startswith('miss') or normalized.startswith('ms'):
            return 'feminine'
        
        return 'unknown'
    
    def reset(self):
        """Reset conversation state."""
        self.history.clear()
        self.entity_stack = EntityStack(max_size=20)
        self.topic_tracker = TopicTracker()
        self._turn_counter = 0
