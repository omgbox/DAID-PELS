"""
BookBot Response Formatter
Generates natural-sounding responses with conversation awareness.
"""

import re
import random
import logging
from typing import Dict, List

from ..core.base_module import BaseModule

logger = logging.getLogger('bookbot.query.response_formatter')


def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts from text."""
    # Collapse multiple spaces
    text = re.sub(r'  +', ' ', text)
    # Remove spaces around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    # Remove spaces after opening quotes
    text = re.sub(r'"\s+', '"', text)
    # Remove spaces before closing quotes
    text = re.sub(r'\s+"', '"', text)
    return text.strip()


class ResponseFormatter(BaseModule):
    """Response formatting module with natural language generation."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        
        # Response templates for different intents
        self.templates = {
            'FACTUAL': [
                "{answer}",
                "According to the book, {answer}",
                "Based on the text, {answer}",
                "The book tells us that {answer}",
            ],
            'DEFINITIONAL': [
                "{answer}",
                "Based on the book, {answer}",
                "The text describes it as: {answer}",
            ],
            'CAUSAL': [
                "{answer}",
                "The reason is: {answer}",
                "According to the text, {answer}",
            ],
            'TEMPORAL': [
                "{answer}",
                "This happened: {answer}",
                "The timeline shows: {answer}",
            ],
            'EXPLANATORY': [
                "{answer}",
                "Here's what I found: {answer}",
                "The book explains: {answer}",
            ],
            'UNKNOWN': [
                "{answer}",
                "I found this: {answer}",
                "Based on my search: {answer}",
            ],
        }
        
        # Follow-up acknowledgments
        self.followup_acknowledgments = {
            'tell_more': [
                "Here's more about {entity}: ",
                "Let me tell you more about {entity}. ",
                "Regarding {entity}, ",
            ],
            'what_about': [
                "Regarding {entity}, ",
                "About {entity}: ",
                "As for {entity}, ",
            ],
            'causal_followup': [
                "The reason is: ",
                "This is because: ",
                "According to the text, ",
            ],
            'elaboration': [
                "Here's additional information: ",
                "Also, ",
                "Furthermore, ",
            ],
        }

    def process(self, input_data) -> dict:
        """
        Process input data and format response.

        Args:
            input_data: Dict with 'query', 'answer', 'confidence', 'intent' keys

        Returns:
            Dict with formatted response
        """
        query = input_data.get('query', '')
        answer = input_data.get('answer', {})
        confidence = input_data.get('confidence', 0.0)
        intent = input_data.get('intent', 'EXPLANATORY')
        conversation = input_data.get('conversation', {})

        # Handle nested answer structure
        if 'answer' in answer:
            answer = answer['answer']

        # Get answer text
        answer_text = answer.get('text', '')
        if not answer_text:
            answer_text = "I couldn't find relevant information in the book."

        # Clean OCR artifacts
        answer_text = clean_ocr_text(answer_text)

        # Format based on intent
        formatted_answer = self._format_answer(answer_text, intent)

        # Add follow-up acknowledgment if applicable
        if conversation.get('is_followup'):
            formatted_answer = self._add_followup_acknowledgment(
                formatted_answer, conversation
            )

        # Format sources
        sources = self._format_sources(answer.get('sources', []))

        # Generate follow-up suggestions
        followups = self._generate_followups(query, answer, intent)

        # Format final response
        response = {
            'answer': formatted_answer,
            'confidence': confidence,
            'sources': sources,
            'suggested_followups': followups,
        }

        self._initialized = True
        return response

    def _format_answer(self, answer_text: str, intent: str) -> str:
        """Format answer based on intent using templates."""
        templates = self.templates.get(intent, self.templates['EXPLANATORY'])
        template = random.choice(templates)
        
        # Clean up the answer text
        answer_text = answer_text.strip()
        
        # Ensure answer starts with capital letter
        if answer_text and answer_text[0].islower():
            answer_text = answer_text[0].upper() + answer_text[1:]
        
        # Ensure answer ends with punctuation
        if answer_text and answer_text[-1] not in '.!?':
            answer_text += '.'
        
        return template.format(answer=answer_text)

    def _add_followup_acknowledgment(self, answer: str, conversation: Dict) -> str:
        """Add acknowledgment for follow-up queries."""
        followup_type = conversation.get('followup_type')
        entities = conversation.get('context_entities', [])
        
        if followup_type and followup_type in self.followup_acknowledgments:
            acknowledgments = self.followup_acknowledgments[followup_type]
            acknowledgment = random.choice(acknowledgments)
            
            # Fill in entity if available
            if '{entity}' in acknowledgment and entities:
                acknowledgment = acknowledgment.format(entity=entities[0])
            elif '{entity}' in acknowledgment:
                acknowledgment = acknowledgment.replace('{entity}', 'this topic')
            
            return acknowledgment + answer
        
        return answer

    def _format_sources(self, sources: List[Dict]) -> List[str]:
        """Format source citations."""
        formatted = []
        for source in sources[:3]:
            text = source.get('text', '')
            chapter = source.get('chapter', '')

            # Truncate long texts
            if len(text) > 100:
                text = text[:97] + '...'

            citation = f'"{text}"'
            if chapter:
                citation += f' (Chapter {chapter})'

            formatted.append(citation)

        return formatted

    def _generate_followups(self, query: str, answer: Dict, intent: str) -> List[str]:
        """Generate follow-up question suggestions."""
        followups = []
        max_followups = self.get_config('max_followup_suggestions', 3)

        # Extract entities from answer
        text = answer.get('text', '')
        entities = self._extract_entities(text)

        # Generate follow-ups based on entities
        for entity in entities[:2]:
            followups.append(f"Tell me more about {entity}.")
            if len(followups) >= max_followups:
                break

        # Add intent-specific follow-ups
        if intent == 'FACTUAL' and len(followups) < max_followups:
            followups.append("What else happened?")
        elif intent == 'DEFINITIONAL' and len(followups) < max_followups:
            followups.append("Can you explain more?")
        elif intent == 'CAUSAL' and len(followups) < max_followups:
            followups.append("What were the consequences?")

        # Add generic follow-up if needed
        if len(followups) < max_followups:
            followups.append("What else can you tell me?")

        return followups[:max_followups]

    def _extract_entities(self, text: str) -> List[str]:
        """Extract entities from text."""
        entities = []
        words = text.split()
        for word in words:
            cleaned = word.strip('.,;:!?()[]"\'')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                if cleaned.lower() not in {
                    'the', 'what', 'when', 'where', 'who', 'why', 'how',
                    'this', 'that', 'these', 'those', 'tell', 'please',
                    'according', 'chapter', 'book', 'based', 'found',
                }:
                    entities.append(cleaned)
        return entities
