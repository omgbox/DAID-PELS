"""
BookBot Response Formatter
Generates natural-sounding responses using the Reasoning Engine.
"""

import re
import logging
from typing import Dict, List

from ..core.base_module import BaseModule
from .reasoning_engine import ReasoningEngine

logger = logging.getLogger('bookbot.query.response_formatter')


def clean_ocr_text(text: str) -> str:
    """Clean OCR artifacts from text."""
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r'\s+"', '"', text)
    return text.strip()


class ResponseFormatter(BaseModule):
    """Response formatting module using the Reasoning Engine."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.reasoning_engine = ReasoningEngine()

    def process(self, input_data) -> dict:
        query = input_data.get('query', '')
        answer = input_data.get('answer', {})
        confidence = input_data.get('confidence', 0.0)
        intent = input_data.get('intent', 'EXPLANATORY')
        conversation = input_data.get('conversation', {})
        structured_evidence = input_data.get('structured_evidence', {})
        query_entities = input_data.get('query_entities', [])

        # Handle nested answer structure
        if 'answer' in answer:
            answer = answer['answer']

        # Build evidence dict for ReasoningEngine
        entity = query_entities[0] if query_entities else ''
        all_svo = []
        all_related = []
        all_descriptions = []
        all_attributes = []
        all_definitions = []
        for ent in query_entities:
            ev = structured_evidence.get(ent, {})
            all_svo.extend(ev.get('actions', []))
            all_related.extend(ev.get('related_entities', []))
            all_descriptions.extend(ev.get('descriptions', []))
            all_attributes.extend(ev.get('attributes', []))
            if ev.get('definition'):
                all_definitions.append(ev['definition'])

        evidence = {
            'svo_triples': all_svo,
            'related_entities': all_related,
            'text_sentences': answer.get('sentences', []),
            'descriptions': all_descriptions,
            'attributes': all_attributes,
            'definition': all_definitions[0] if all_definitions else '',
        }

        # Use existing answer if it's already substantive (avoid re-synthesis)
        answer_text = answer.get('text', '') if isinstance(answer, dict) else str(answer)
        already_good = (answer_text and len(answer_text) > 30
                        and entity.lower() in answer_text.lower()
                        and any(w in answer_text.lower() for w in [
                            'interact', 'felt', 'feel', 'read', 'walk',
                            'is described', 'is known', 'was a', 'is a',
                            'took', 'gave', 'said', 'spoke', 'looked',
                            'turned', 'smiled', 'began', 'continued',
                            'seemed', 'appeared', 'noticed', 'reached',
                            'drew', 'lifted', 'held', 'kept', 'left',
                            'returned', 'joined', 'called', 'answered',
                        ]))

        if not already_good:
            # Re-synthesize using reasoning engine
            answer_text = self.reasoning_engine.generate(
                intent, entity, evidence, conversation
            )

        # Generate follow-ups
        followups = self.reasoning_engine.generate_followups(
            entity, evidence, conversation
        )

        # Format sources
        sources = self._format_sources(answer.get('sources', []))

        # Final cleanup
        answer_text = answer_text.strip()
        if answer_text and answer_text[0].islower():
            answer_text = answer_text[0].upper() + answer_text[1:]
        if answer_text and answer_text[-1] not in '.!?':
            answer_text += '.'

        response = {
            'answer': answer_text,
            'options': answer.get('options', {}),
            'confidence': confidence,
            'sources': sources,
            'suggested_followups': followups,
        }

        self._initialized = True
        return response

    def _format_sources(self, sources: List[Dict]) -> List[str]:
        """Format source citations."""
        formatted = []
        for source in sources[:3]:
            text = source.get('text', '')
            chapter = source.get('chapter', '')
            if len(text) > 120:
                text = text[:117] + '...'
            citation = f'"{text}"'
            if chapter:
                citation += f' (Chapter {chapter})'
            formatted.append(citation)
        return formatted

    def _generate_followups_fallback(self, query: str, answer: Dict,
                                      intent: str) -> List[str]:
        """Fallback follow-up generation."""
        followups = []
        text = answer.get('text', '') if isinstance(answer, dict) else str(answer)
        entities = self._extract_entities(text)

        for entity in entities[:2]:
            followups.append(f"Tell me more about {entity}.")
            if len(followups) >= 3:
                break

        if len(followups) < 3:
            if intent == 'FACTUAL':
                followups.append("What else happened?")
            elif intent == 'CAUSAL':
                followups.append("What were the consequences?")

        if len(followups) < 3:
            followups.append("What else can you tell me?")

        return followups[:3]

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
