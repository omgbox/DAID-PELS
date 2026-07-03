"""
BookBot Contextual Query Rewriter
Rewrites user queries using conversation context before retrieval.
"""

import re
import logging
from typing import Dict, List, Optional

from ..core.base_module import BaseModule
from .conversation_memory import ConversationMemory

logger = logging.getLogger('bookbot.query.contextual_rewriter')


class ContextualQueryRewriter(BaseModule):
    """
    Rewrites user queries using conversation context.
    
    This module:
    1. Detects follow-up queries
    2. Resolves pronouns
    3. Expands queries with context
    4. Carries over intent when appropriate
    """
    
    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.conversation_memory: Optional[ConversationMemory] = None
    
    def set_conversation_memory(self, memory: ConversationMemory):
        """Set the conversation memory instance."""
        self.conversation_memory = memory
    
    def process(self, input_data) -> dict:
        """
        Process a query with conversation context.
        
        Args:
            input_data: Dict with 'query', 'intent' keys
            
        Returns:
            Dict with 'rewritten_query', 'is_followup', 'resolutions' keys
        """
        query = input_data.get('query', '')
        intent = input_data.get('intent', 'EXPLANATORY')
        
        result = {
            'original_query': query,
            'rewritten_query': query,
            'is_followup': False,
            'followup_type': None,
            'intent_carryover': None,
            'resolutions': [],
            'expanded_terms': [],
        }
        
        if not self.conversation_memory:
            self._initialized = True
            return result
        
        # Step 1: Detect follow-up patterns
        followup = self.conversation_memory.detect_followup(query)
        
        if followup:
            result['is_followup'] = True
            result['followup_type'] = followup['type']
            
            # Step 2: Expand the follow-up query
            expanded = self.conversation_memory.expand_query(query)
            result['rewritten_query'] = expanded
            
            # Step 3: Intent carryover for certain follow-up types
            if followup['type'] in ('tell_more', 'elaboration', 'and_reference'):
                last_turn = self.conversation_memory.get_last_turn()
                if last_turn:
                    result['intent_carryover'] = last_turn.get('intent')
        
        # Step 4: Resolve pronouns even in non-follow-up queries
        if not result['is_followup']:
            expanded = self.conversation_memory.expand_query(query)
            if expanded != query:
                result['rewritten_query'] = expanded
                result['resolutions'].append({
                    'original': query,
                    'expanded': expanded,
                })
        
        # Step 5: Add contextual terms from entity stack and topic
        context_terms = self._gather_context_terms(query)
        if context_terms:
            result['expanded_terms'] = context_terms
        
        self._initialized = True
        return result
    
    def _gather_context_terms(self, query: str) -> List[str]:
        """Gather additional retrieval terms from conversation context."""
        terms = []
        query_lower = set(query.lower().split())
        
        # Add salient entities not already in query
        if self.conversation_memory:
            for ent_name in self.conversation_memory.get_context_entities():
                if ent_name.lower() not in query_lower:
                    terms.append(ent_name)
            
            # Add topic keywords
            topic_terms = self.conversation_memory.topic_tracker.get_context_terms()
            for term in topic_terms:
                if term.lower() not in query_lower:
                    terms.append(term)
        
        return terms[:5]  # Limit to 5 extra terms
