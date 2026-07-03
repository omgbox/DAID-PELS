"""
Answer Synthesizer
Combine multiple evidence sources into coherent paragraphs.
"""

import re
import logging
from typing import Dict, List

logger = logging.getLogger('bookbot.query.answer_synthesizer')


class AnswerSynthesizer:
    """Synthesize coherent answers from multiple evidence sources."""

    def synthesize(self, sentences: List[Dict], svo_triples: List[Dict],
                   related_entities: List[Dict], entity: str, intent: str) -> str:
        """
        Combine text evidence + structured knowledge into a coherent answer.

        Args:
            sentences: Retrieved sentences from BM25
            svo_triples: SVO triples related to the entity
            related_entities: Connected entities from knowledge graph
            entity: Main entity being asked about
            intent: Query intent (DEFINITIONAL, FACTUAL, etc.)
        """
        claims = []
        claims.extend(self._extract_svo_claims(svo_triples, entity))
        claims.extend(self._extract_text_claims(sentences, entity))
        claims = self._deduplicate(claims)
        claims = self._order(claims, intent)
        return self._generate(claims, entity, related_entities, intent)

    def _extract_svo_claims(self, triples: List[Dict], entity: str) -> List[Dict]:
        """Extract claims from SVO triples."""
        claims = []
        for t in triples[:10]:
            subject = t.get('subject', '') if isinstance(t, dict) else t[0] if isinstance(t, (list, tuple)) else ''
            verb = t.get('verb', '') if isinstance(t, dict) else t[1] if isinstance(t, (list, tuple)) else ''
            obj = t.get('object', '') if isinstance(t, dict) else t[2] if isinstance(t, (list, tuple)) else ''
            confidence = t.get('confidence', 0.5) if isinstance(t, dict) else t[3] if isinstance(t, (list, tuple)) and len(t) > 3 else 0.5

            if subject and verb:
                text = f"{subject} {verb}"
                if obj:
                    text += f" {obj}"
                claims.append({
                    'type': 'svo',
                    'text': text,
                    'subject': subject,
                    'verb': verb,
                    'object': obj,
                    'confidence': confidence,
                    'is_entity_action': entity.lower() in subject.lower() if entity else False,
                })
        return claims

    def _extract_text_claims(self, sentences: List[Dict], entity: str) -> List[Dict]:
        """Extract claims from retrieved sentences."""
        claims = []
        entity_lower = entity.lower() if entity else ''

        for sent in sentences[:8]:
            text = sent.get('text', '') if isinstance(sent, dict) else str(sent)
            text = self._clean(text)
            if not text or len(text.split()) < 3:
                continue

            # Boost sentences that mention the entity
            is_entity_match = entity_lower in text.lower() if entity_lower else False

            claims.append({
                'type': 'text',
                'text': text,
                'confidence': 0.6 if is_entity_match else 0.4,
                'is_entity_action': is_entity_match,
            })
        return claims

    def _deduplicate(self, claims: List[Dict]) -> List[Dict]:
        """Remove overlapping claims."""
        seen_texts = set()
        unique = []
        for claim in claims:
            # Normalize for dedup
            key = claim['text'].lower()[:80]
            if key not in seen_texts:
                seen_texts.add(key)
                unique.append(claim)
        return unique

    def _order(self, claims: List[Dict], intent: str) -> List[Dict]:
        """Order claims logically based on intent."""
        if intent == 'CAUSAL':
            # SVO actions first (causes), then text descriptions (effects)
            svo = [c for c in claims if c['type'] == 'svo']
            text = [c for c in claims if c['type'] == 'text']
            return svo + text
        elif intent == 'DEFINITIONAL':
            # Entity actions first, then general descriptions
            entity_actions = [c for c in claims if c.get('is_entity_action')]
            other = [c for c in claims if not c.get('is_entity_action')]
            return entity_actions + other
        else:
            # Sort by confidence
            return sorted(claims, key=lambda c: c.get('confidence', 0), reverse=True)

    def _generate(self, claims: List[Dict], entity: str,
                  related_entities: List[Dict], intent: str) -> str:
        """Generate a coherent paragraph from ordered claims."""
        if not claims:
            return ""

        parts = []

        # Extract SVO-based facts
        svo_claims = [c for c in claims if c['type'] == 'svo']
        text_claims = [c for c in claims if c['type'] == 'text']

        # Use SVO triples to build factual statements
        if svo_claims:
            for claim in svo_claims[:5]:
                parts.append(claim['text'] + '.')

        # Use text sentences for descriptive context
        if text_claims:
            for claim in text_claims[:3]:
                text = claim['text']
                if not text.endswith('.'):
                    text += '.'
                parts.append(text)

        # Add relationships if available
        if related_entities:
            rel_names = []
            for r in related_entities[:5]:
                name = r.get('related', '') if isinstance(r, dict) else str(r)
                if name:
                    rel_names.append(name)
            if rel_names:
                parts.append(f"Related to: {', '.join(rel_names)}.")

        answer = ' '.join(parts)
        return answer[:1500] if len(answer) > 1500 else answer

    def _clean(self, text: str) -> str:
        """Clean text artifacts."""
        text = re.sub(r'  +', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        return text.strip()
