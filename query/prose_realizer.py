"""
Prose Realizer
Transforms SVO triples into narrative prose by selecting and formatting
the best original sentences from the book.

Key insight: the original sentences already contain modifiers, emotion,
dialogue, and detail. We select the best ones and format them nicely.
"""

import re
import random
import logging
from typing import Dict, List, Optional

logger = logging.getLogger('bookbot.query.prose_realizer')


class ProseRealizer:
    """
    Realizes SVO triples into flowing narrative prose.

    Strategy: retrieve original sentences from the book, score them
    for quality, and present the best ones with proper formatting.
    """

    def realize(self, entity: str, svo_data: List[Dict],
                related: List[str] = None,
                original_sentences: Dict[int, str] = None) -> str:
        """
        Convert SVO triples with source sentences into narrative prose.
        """
        if not svo_data:
            return f"{entity} is a character in the story."

        original_sentences = original_sentences or {}

        # Collect candidate sentences
        candidates = []
        seen = set()

        for item in svo_data:
            sid = item.get('sentence_id')
            orig = original_sentences.get(sid, '') if sid else ''
            if not orig:
                continue

            clean = self._clean_sentence(orig)
            if not clean or len(clean) < 15:
                continue
            if entity.lower() not in clean.lower():
                continue

            key = clean.lower()[:50]
            if key in seen:
                continue
            seen.add(key)

            score = self._score_sentence(clean, entity)
            candidates.append((score, clean))

        # Sort by score, take top 4
        candidates.sort(key=lambda x: -x[0])
        selected = [s for _, s in candidates[:4]]

        if not selected:
            return f"{entity} is a character in the story."

        # Build prose with sentence variety
        parts = []
        for i, sent in enumerate(selected):
            sent = self._format_dialogue(sent)
            # Vary sentence openers
            if i > 0 and random.random() < 0.3:
                sent = self._vary_opener(sent, entity)
            parts.append(sent)

        # Add relationship closing
        if related:
            rel = self._format_relationships(entity, related)
            if rel:
                parts.append(rel)

        return ' '.join(parts)

    def _clean_sentence(self, sentence: str) -> str:
        """Clean a sentence for prose output."""
        # Remove chapter markers
        sentence = re.sub(r'CHAPTER\s+\w+\.?', '', sentence)
        sentence = re.sub(r'Chapter\s+\w+\.?', '', sentence)

        # Remove brackets and their content
        sentence = re.sub(r'\[.+?\]', '', sentence)

        # Clean whitespace
        sentence = re.sub(r'\s+', ' ', sentence).strip()

        # Remove unbalanced leading quotes
        while sentence and sentence[0] in '""' and sentence.count(sentence[0]) % 2 != 0:
            sentence = sentence[1:].strip()

        # Trim if too long — cut at natural boundary
        if len(sentence) > 150:
            for sep in ['; ', ', ', ' and ', ' but ', ' which ', ' who ']:
                idx = sentence.find(sep, 40)
                if 40 < idx < 150:
                    sentence = sentence[:idx].strip()
                    break
            else:
                sentence = sentence[:147].strip() + '...'

        # Ensure proper ending
        if sentence and sentence[-1] not in '.!?':
            sentence += '.'

        # Capitalize first letter
        if sentence and sentence[0].islower():
            sentence = sentence[0].upper() + sentence[1:]

        return sentence

    def _score_sentence(self, sentence: str, entity: str) -> float:
        """Score a sentence for inclusion in the answer."""
        score = 0.0
        lower = sentence.lower()
        words = sentence.split()
        wc = len(words)

        # Entity as subject (starts with entity)
        if lower.startswith(entity.lower()):
            score += 4.0

        # Descriptive patterns
        DESC = [' is a ', ' was a ', ' is an ', ' was an ',
                ' is the ', ' was the ', ' is described ', ' is known ']
        for d in DESC:
            if d in lower:
                score += 3.0
                break

        # Action verbs (more interesting than "is/was")
        ACTION = ['felt', 'looked', 'turned', 'smiled', 'laughed',
                  'spoke', 'replied', 'said', 'told', 'gave',
                  'took', 'found', 'knew', 'thought', 'saw',
                  'walked', 'stood', 'sat', 'ran', 'began',
                  'continued', 'seemed', 'appeared', 'observed',
                  'noticed', 'reached', 'drew', 'lifted', 'held',
                  'kept', 'left', 'returned', 'joined', 'called',
                  'answered', 'declared', 'exclaimed', 'whispered']
        for v in ACTION:
            if v in lower:
                score += 2.0
                break

        # Has descriptive words
        DESCRIBERS = ['sensible', 'intelligent', 'beautiful', 'pleasing',
                      'delighted', 'pleased', 'anxious', 'happy', 'sad',
                      'gentle', 'firm', 'quiet', 'warm', 'cold',
                      'very', 'quite', 'rather', 'extremely',
                      'gently', 'firmly', 'quietly', 'softly', 'calmly',
                      'eagerly', 'reluctantly', 'warmly']
        for d in DESCRIBERS:
            if d in lower:
                score += 1.0
                break

        # Commas = more complex = more descriptive
        if ',' in sentence:
            score += 1.0

        # Medium length best
        if 8 <= wc <= 25:
            score += 2.0
        elif 6 <= wc <= 35:
            score += 1.0
        elif wc < 5:
            score -= 3.0
        elif wc > 35:
            score -= 1.0

        # Penalize dialogue-heavy sentences
        if sentence.count('"') > 2:
            score -= 1.0

        return score

    def _format_dialogue(self, sentence: str) -> str:
        """Format dialogue with proper punctuation."""
        # Fix spacing around quotes
        sentence = re.sub(r'\s+"', ' "', sentence)
        sentence = re.sub(r'"\s+', '" ', sentence)
        sentence = re.sub(r'\s+,', ',', sentence)
        return sentence

    def _vary_opener(self, sentence: str, entity: str) -> str:
        """Vary sentence opener for readability."""
        # If sentence starts with entity name, sometimes use pronoun
        if sentence.startswith(entity):
            pronouns = {
                'Elizabeth': 'She', 'Darcy': 'He', 'Jane': 'She',
                'Bingley': 'He', 'Wickham': 'He', 'Collins': 'He',
                'Bennet': 'He', 'Lydia': 'She', 'Kitty': 'She',
                'Mary': 'She', 'Charlotte': 'She',
            }
            pronoun = pronouns.get(entity, 'They')
            rest = sentence[len(entity):].strip()
            if rest and rest[0] in ',.':
                rest = rest[1:].strip()
            if rest:
                return f"{pronoun} {rest[0].lower()}{rest[1:]}"
        return sentence

    def _format_relationships(self, entity: str, related: List[str]) -> str:
        """Format relationship sentence with variety."""
        names = [r for r in related[:4]
                 if r.lower() != entity.lower() and len(r) > 1]
        if not names:
            return ''

        if len(names) == 1:
            rel = names[0]
        elif len(names) == 2:
            rel = f"{names[0]} and {names[1]}"
        else:
            rel = f"{', '.join(names[:3])}, and others"

        patterns = [
            f"Throughout the story, {entity} moves among {rel}, "
            f"each interaction shaped by wit and feeling.",

            f"{entity}'s world revolves around {rel}, "
            f"each relationship carrying its own weight.",

            f"In the world of the story, {entity} interacts with {rel}, "
            f"weaving connections that drive the narrative forward.",

            f"The bonds between {entity} and {rel} "
            f"form the heart of the narrative.",
        ]

        return random.choice(patterns)
