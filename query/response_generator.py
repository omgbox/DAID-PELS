"""
Response Generator
Dynamic, natural responses instead of template fill-in.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger('bookbot.query.response_generator')


class ResponseGenerator:
    """Generate natural, dynamic responses based on intent and evidence."""

    def generate(self, intent: str, entity: str, evidence: Dict,
                 conversation_context: Dict = None) -> str:
        """
        Generate a complete response.

        Args:
            intent: Query intent
            entity: Primary entity
            evidence: Dict with 'text_sentences', 'svo_triples', 'related_entities'
            conversation_context: Prior conversation state
        """
        conv = conversation_context or {}

        if intent == 'DEFINITIONAL':
            return self._gen_definitional(entity, evidence, conv)
        elif intent == 'FACTUAL':
            return self._gen_factual(entity, evidence, conv)
        elif intent == 'CAUSAL':
            return self._gen_causal(entity, evidence, conv)
        elif intent == 'TEMPORAL':
            return self._gen_temporal(entity, evidence, conv)
        elif intent == 'COMPARATIVE':
            return self._gen_comparative(entity, evidence, conv)
        elif intent == 'SUMMARIZATION':
            return self._gen_summarization(entity, evidence, conv)
        else:
            return self._gen_general(entity, evidence, conv)

    def _gen_definitional(self, entity: str, evidence: Dict, conv: Dict) -> str:
        parts = []
        svo = evidence.get('svo_triples', [])
        sentences = evidence.get('text_sentences', [])
        related = evidence.get('related_entities', [])
        descriptions = evidence.get('descriptions', [])
        attributes = evidence.get('attributes', [])
        definition = evidence.get('definition', '')

        # Opening based on conversation state
        if conv.get('is_followup'):
            parts.append(f"Adding to what we discussed about {entity}:")
        else:
            parts.append(f"Based on the text:")

        # Dictionary definition if available
        if definition:
            parts.append(f"Dictionary: {definition}")

        # Entity attributes (e.g., "Alice is a young girl")
        if attributes:
            parts.append(f"{entity} is {attributes[0]}.")

        # Descriptive sentences from the text
        if descriptions:
            for desc in descriptions[:2]:
                desc = self._clean(desc)
                if desc and len(desc.split()) >= 4:
                    parts.append(desc)

        # Key actions from SVO
        if svo:
            actions = []
            for t in svo[:4]:
                if isinstance(t, dict):
                    s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
                elif isinstance(t, (list, tuple)) and len(t) >= 3:
                    s, v, o = t[0], t[1], t[2]
                else:
                    continue
                if entity.lower() in str(s).lower():
                    action = f"{v}"
                    if o:
                        action += f" {o}"
                    actions.append(action)
            if actions:
                parts.append(f"In the story, {entity} {', '.join(actions[:3])}.")

        # Relationships
        if related:
            names = []
            for r in related[:5]:
                if isinstance(r, dict):
                    name = r.get('related', '')
                elif isinstance(r, (list, tuple)):
                    name = r[0] if r else ''
                else:
                    name = str(r)
                if name and name.lower() != entity.lower():
                    names.append(name)
            if names:
                parts.append(f"Connected to: {', '.join(names)}.")

        # Text evidence (only if we don't have descriptions already)
        if not descriptions and sentences:
            for sent in sentences[:2]:
                text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
                text = self._clean(text)
                if text and len(text.split()) >= 4:
                    parts.append(text)

        return ' '.join(parts) if parts else f"I don't have enough information about {entity}."

    def _gen_factual(self, entity: str, evidence: Dict, conv: Dict) -> str:
        svo = evidence.get('svo_triples', [])
        sentences = evidence.get('text_sentences', [])

        # Direct answer from SVO
        if svo:
            t = svo[0]
            if isinstance(t, dict):
                s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                s, v, o = t[0], t[1], t[2]
            else:
                return self._best_sentence(sentences)
            answer = f"{s} {v}"
            if o:
                answer += f" {o}"
            return answer + '.'

        return self._best_sentence(sentences)

    def _gen_causal(self, entity: str, evidence: Dict, conv: Dict) -> str:
        sentences = evidence.get('text_sentences', [])
        svo = evidence.get('svo_triples', [])

        # Look for causal language in sentences
        causal_words = ['because', 'therefore', 'thus', 'consequently', 'due to']
        for sent in sentences:
            text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
            if any(w in text.lower() for w in causal_words):
                return self._clean(text)

        # Build causal chain from SVO
        if len(svo) >= 2:
            t1, t2 = svo[0], svo[1]
            s1 = t1.get('subject', '') if isinstance(t1, dict) else t1[0] if isinstance(t1, (list, tuple)) else ''
            v1 = t1.get('verb', '') if isinstance(t1, dict) else t1[1] if isinstance(t1, (list, tuple)) else ''
            s2 = t2.get('subject', '') if isinstance(t2, dict) else t2[0] if isinstance(t2, (list, tuple)) else ''
            v2 = t2.get('verb', '') if isinstance(t2, dict) else t2[1] if isinstance(t2, (list, tuple)) else ''
            if s1 and v1 and s2 and v2:
                return f"{s1} {v1}, leading to {s2} {v2}."

        return self._best_sentence(sentences)

    def _gen_temporal(self, entity: str, evidence: Dict, conv: Dict) -> str:
        sentences = evidence.get('text_sentences', [])
        temporal_words = ['when', 'before', 'after', 'during', 'then', 'first', 'finally']
        for sent in sentences:
            text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
            if any(w in text.lower() for w in temporal_words):
                return self._clean(text)
        return self._best_sentence(sentences)

    def _gen_comparative(self, entity: str, evidence: Dict, conv: Dict) -> str:
        sentences = evidence.get('text_sentences', [])
        parts = []
        for sent in sentences[:3]:
            text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
            text = self._clean(text)
            if text:
                parts.append(text)
        return ' '.join(parts) if parts else "I couldn't find comparison information."

    def _gen_summarization(self, entity: str, evidence: Dict, conv: Dict) -> str:
        sentences = evidence.get('text_sentences', [])
        parts = []
        for sent in sentences[:5]:
            text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
            text = self._clean(text)
            if text and len(text.split()) >= 4:
                parts.append(text)
        return ' '.join(parts) if parts else "I couldn't generate a summary."

    def _gen_general(self, entity: str, evidence: Dict, conv: Dict) -> str:
        svo = evidence.get('svo_triples', [])
        sentences = evidence.get('text_sentences', [])

        parts = []
        if svo:
            for t in svo[:3]:
                if isinstance(t, dict):
                    s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
                elif isinstance(t, (list, tuple)) and len(t) >= 3:
                    s, v, o = t[0], t[1], t[2]
                else:
                    continue
                line = f"{s} {v}"
                if o:
                    line += f" {o}"
                parts.append(line + '.')

        if not parts:
            return self._best_sentence(sentences)

        return ' '.join(parts)

    def _best_sentence(self, sentences: list) -> str:
        """Pick the best sentence from a list."""
        for sent in sentences:
            text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
            text = self._clean(text)
            if text and len(text.split()) >= 4:
                return text
        return "I don't have enough information to answer that."

    def _clean(self, text: str) -> str:
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        return text.strip()

    def generate_followups(self, entity: str, evidence: Dict,
                           conversation_context: Dict = None) -> List[str]:
        """Generate relevant follow-up suggestions."""
        suggestions = []
        conv = conversation_context or {}
        discussed = conv.get('entities_discussed', [])

        # From related entities
        related = evidence.get('related_entities', [])
        for r in related[:5]:
            if isinstance(r, dict):
                name = r.get('related', '')
            elif isinstance(r, (list, tuple)):
                name = r[0] if r else ''
            else:
                name = str(r)
            if name and name.lower() not in [d.lower() for d in discussed]:
                suggestions.append(f"Tell me more about {name}.")
                if len(suggestions) >= 2:
                    break

        # From SVO actions
        svo = evidence.get('svo_triples', [])
        for t in svo[:3]:
            if isinstance(t, dict):
                v = t.get('verb', '')
            elif isinstance(t, (list, tuple)) and len(t) >= 2:
                v = t[1]
            else:
                continue
            if entity and v:
                suggestions.append(f"What did {entity} {v}?")
                if len(suggestions) >= 3:
                    break

        return suggestions[:3]
