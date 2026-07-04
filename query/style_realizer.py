"""
Style Realizer v2
Produces genuinely different styled outputs by:
  1. Extracting core facts from source sentences
  2. Rebuilding sentences in each style from those facts

Styles:
  - Professional: concise, direct, 1-2 short sentences
  - Formal: literary, complex, elevated vocabulary
  - Neutral: balanced, clear, like a reference entry
"""

import re
import random
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('bookbot.query.style_realizer')


class StyleRealizer:
    """
    Extracts core facts from source sentences and rebuilds them
    in different styles.
    """

    def __init__(self):
        pass

    def realize(self, entity: str, sentences: List[str],
                related: List[str] = None,
                styles: List[str] = None) -> Dict[str, str]:
        """
        Generate 3 genuinely different styled answers.
        """
        if not sentences:
            base = f"{entity} is a character in the story."
            return {s: base for s in (styles or ['professional', 'formal', 'neutral'])}

        styles = styles or ['professional', 'formal', 'neutral']
        related = related or []

        # Step 1: Extract core facts from source sentences
        facts = self._extract_facts(entity, sentences, related)

        # Step 2: Build each style from facts
        results = {}
        for style in styles:
            results[style] = self._build_style(entity, facts, style)

        return results

    def _extract_facts(self, entity: str, sentences: List[str],
                       related: List[str]) -> Dict:
        """
        Extract structured facts from source sentences.
        Returns dict with: traits, actions, moods, relationships, quotes
        """
        facts = {
            'traits': [],       # descriptive adjectives
            'actions': [],      # things the entity did
            'moods': [],        # emotional states
            'relationships': related[:4],
            'quotes': [],       # dialogue
            'descriptions': [], # "is a..." patterns
            'scene_details': [], # vivid scene fragments
        }

        entity_lower = entity.lower()

        for sent in sentences[:30]:
            lower = sent.lower()

            # Extract traits (adjectives describing entity)
            # Only include meaningful character traits
            TRAIT_WORDS = {
                # Positive traits
                'sensible', 'intelligent', 'beautiful', 'handsome', 'pleasing',
                'gentle', 'kind', 'clever', 'witty', 'spirited', 'lively',
                'bright', 'proud', 'humble', 'modest', 'elegant', 'graceful',
                'charming', 'engaging', 'amiable', 'agreeable', 'reserved',
                'determined', 'resolute', 'brave', 'courageous', 'faithful',
                'loyal', 'generous', 'compassionate', 'thoughtful', 'wise',
                # Negative traits (for contrast)
                'obstinate', 'stubborn', 'willful', 'haughty', 'conceited',
                'vain', 'silly', 'ignorant', 'foolish', 'proud', 'cold',
                'distant', 'aloof', 'arrogant', 'selfish', 'cruel',
                # Physical traits (only if meaningful)
                'young', 'beautiful', 'handsome', 'elegant', 'graceful',
            }
            # Filter out non-descriptive uses
            PHYSICAL_ONLY = {'tall', 'short', 'old', 'fat', 'thin', 'small'}
            for w in TRAIT_WORDS:
                if w in lower and w not in facts['traits']:
                    # Check context: "is [trait]" or "was [trait]" pattern
                    pattern = rf'{entity_lower}\s+(?:is|was|seemed|appeared)\s+(?:a\s+)?(?:very\s+)?{w}'
                    if re.search(pattern, lower):
                        facts['traits'].append(w)
                    elif w not in PHYSICAL_ONLY:
                        # For non-physical traits, be more lenient
                        if w in lower and entity_lower in lower:
                            facts['traits'].append(w)

            # Extract actions (verb phrases about entity)
            if entity_lower in lower:
                # Find the main verb phrase AFTER entity name
                ACTION_VERBS = (
                    'felt', 'looked', 'turned', 'smiled', 'spoke', 'replied',
                    'said', 'took', 'gave', 'found', 'knew', 'thought', 'saw',
                    'walked', 'stood', 'sat', 'began', 'continued', 'seemed',
                    'appeared', 'noticed', 'reached', 'drew', 'lifted', 'held',
                    'kept', 'left', 'returned', 'joined', 'called', 'answered',
                    'declared', 'exclaimed', 'whispered', 'observed', 'regarded',
                    'accepted', 'refused', 'received', 'engaged', 'married',
                    'loved', 'hated', 'desired', 'wished', 'hoped', 'feared',
                )
                
                # Words that indicate incomplete/fragment actions
                FRAGMENT_WORDS = {'anything', 'something', 'nothing', 'everything',
                                  'him', 'her', 'them', 'it', 'this', 'that'}
                
                # Find entity position
                ent_pos = lower.find(entity_lower)
                if ent_pos >= 0:
                    # Look for verb after entity
                    after_entity = lower[ent_pos + len(entity_lower):].strip()
                    for verb in ACTION_VERBS:
                        if after_entity.startswith(verb):
                            # Get the phrase after verb
                            rest = after_entity[len(verb):].strip()
                            # Trim to ~6 words
                            rest_words = rest.split()[:6]
                            action = f"{verb} {' '.join(rest_words)}".strip()
                            action = action.rstrip('.,;:!?')
                            
                            # Filter out fragments
                            action_words = set(action.lower().split())
                            if action_words & FRAGMENT_WORDS:
                                continue  # Skip fragments with pronouns/determiners
                            
                            # Require minimum quality
                            if len(action) > 12 and len(action.split()) >= 3:
                                if action not in facts['actions']:
                                    facts['actions'].append(action)
                            break

            # Also look for actions in other positions in the sentence
            if entity_lower in lower and len(facts['actions']) < 5:
                for verb in ACTION_VERBS:
                    verb_pos = lower.find(verb)
                    if verb_pos > 0:
                        # Check it's near entity
                        if abs(verb_pos - lower.find(entity_lower)) < 60:
                            rest = lower[verb_pos + len(verb):].strip()
                            rest_words = rest.split()[:5]
                            action = f"{verb} {' '.join(rest_words)}".strip()
                            action = action.rstrip('.,;:!?')
                            
                            # Filter out fragments
                            action_words_set = set(action.lower().split())
                            if action_words_set & FRAGMENT_WORDS:
                                continue  # Skip fragments
                            
                            if (len(action) > 12
                                and len(action.split()) >= 3
                                and action not in facts['actions']
                                and not any(a.startswith(verb) for a in facts['actions'])):
                                facts['actions'].append(action)
                            break

            # Extract moods (emotional context)
            MOOD_WORDS = {
                'ashamed': 'shame', 'vexed': 'vexation', 'pleased': 'pleasure',
                'delighted': 'delight', 'annoyed': 'annoyance', 'amused': 'amusement',
                'surprised': 'surprise', 'disappointed': 'disappointment',
                'embarrassed': 'embarrassment', 'concerned': 'concern',
                'interested': 'interest', 'bored': 'boredom',
                'angry': 'anger', 'afraid': 'fear', 'confused': 'confusion',
            }
            for word, mood in MOOD_WORDS.items():
                if word in lower and mood not in facts['moods']:
                    facts['moods'].append(mood)

            # Extract quotes
            quote_match = re.search(r'["\u201c](.+?)["\u201d]', sent)
            if quote_match and len(quote_match.group(1)) > 5:
                q = quote_match.group(1).strip()
                if q not in facts['quotes'] and len(facts['quotes']) < 3:
                    facts['quotes'].append(q)

            # Extract "is a" descriptions
            desc_pat = re.search(
                rf'{entity_lower}\s+(?:is|was)\s+(?:a|an)\s+(.+?)[.,]',
                lower
            )
            if desc_pat:
                d = desc_pat.group(1).strip()
                if d and d not in facts['descriptions']:
                    facts['descriptions'].append(d)

            # Extract vivid scene details (short, descriptive fragments)
            if entity_lower in lower:
                words = sent.split()
                wc = len(words)
                if 8 <= wc <= 25:
                    # Check for descriptive elements
                    has_desc = any(w.lower() in TRAIT_WORDS for w in words)
                    has_action = any(w.lower() in {
                        'felt', 'looked', 'turned', 'smiled', 'spoke',
                        'said', 'took', 'found', 'thought', 'saw',
                        'walked', 'stood', 'began', 'seemed', 'noticed'
                    } for w in words)
                    if (has_desc or has_action) and sent not in facts['scene_details']:
                        facts['scene_details'].append(sent.strip())

        return facts

    def _build_style(self, entity: str, facts: Dict, style: str) -> str:
        """Build a styled answer from extracted facts."""
        if style == 'professional':
            return self._build_professional(entity, facts)
        elif style == 'formal':
            return self._build_formal(entity, facts)
        else:
            return self._build_neutral(entity, facts)

    def _build_professional(self, entity: str, facts: Dict) -> str:
        """
        Professional: concise, direct, 1-2 short sentences.
        Like a character bio in a review.
        """
        parts = []

        # Core identity
        traits = facts.get('traits', [])
        if traits:
            trait_str = ', '.join(traits[:3])
            parts.append(f"{entity} is {trait_str}.")
        elif facts.get('descriptions'):
            parts.append(f"{entity} is {facts['descriptions'][0]}.")
        else:
            parts.append(f"{entity} is a key character in the story.")

        # Key action (one, max)
        actions = facts.get('actions', [])
        if actions:
            a = actions[0]
            # Make it concise
            a = self._shorten_action(a)
            # Don't duplicate entity name
            if not a.lower().startswith(entity.lower()) and len(a) > 10:
                parts.append(f"{entity} {a}.")
        elif facts.get('scene_details'):
            # Use a scene detail as fallback
            scene = facts['scene_details'][0]
            if entity.lower() in scene.lower():
                parts.append(scene.rstrip('.') + '.')
        
        # Add relationship context
        rel = facts.get('relationships', [])
        if rel:
            if len(rel) == 1:
                parts.append(f"Interacts with {rel[0]}.")
            elif len(rel) == 2:
                parts.append(f"Interacts with {rel[0]} and {rel[1]}.")
            else:
                parts.append(f"Interacts with {', '.join(rel[:3])}.")

        return ' '.join(parts)

    def _build_formal(self, entity: str, facts: Dict) -> str:
        """
        Formal/Literary: elevated, complex, literary sentences.
        Like a literary analysis.
        """
        parts = []

        # Opening: character introduction with traits
        traits = facts.get('traits', [])
        if traits:
            if len(traits) >= 2:
                trait_str = f"{traits[0]} and {traits[1]}"
            else:
                trait_str = traits[0]
            parts.append(
                f"Throughout the narrative, {entity} emerges as a figure "
                f"of {trait_str} character."
            )
        elif facts.get('descriptions'):
            parts.append(
                f"Within the story, {entity} is portrayed as "
                f"{facts['descriptions'][0]}."
            )
        else:
            parts.append(f"{entity} occupies a central position in the narrative.")

        # Emotional dimension
        moods = facts.get('moods', [])
        if moods:
            mood_str = moods[0] if len(moods) == 1 else f"{moods[0]}, among other states"
            parts.append(
                f"The narrative reveals {entity} experiencing {mood_str}, "
                f"adding depth to the character."
            )

        # Actions with elevated language
        actions = facts.get('actions', [])
        if actions:
            a = self._shorten_action(actions[0])
            if len(a) > 10:  # Only use if meaningful
                a = self._elevate_action(a)
                parts.append(f"{entity} {a}.")

        # Relationships
        rel = facts.get('relationships', [])
        if rel:
            if len(rel) <= 2:
                rel_str = ' and '.join(rel)
            else:
                rel_str = f"{', '.join(rel[:3])}, among others"
            parts.append(
                f"The bonds between {entity} and {rel_str} "
                f"form the emotional core of the story."
            )

        return ' '.join(parts)

    def _build_neutral(self, entity: str, facts: Dict) -> str:
        """
        Neutral: balanced, clear, informational.
        Like a reference entry or encyclopedia.
        """
        parts = []

        # Core description
        traits = facts.get('traits', [])
        if traits:
            trait_str = ', '.join(traits[:3])
            parts.append(f"{entity} is characterized as {trait_str}.")
        elif facts.get('descriptions'):
            parts.append(f"{entity} is described as {facts['descriptions'][0]}.")
        else:
            parts.append(f"{entity} is a central character in the narrative.")

        # Actions
        actions = facts.get('actions', [])
        if actions:
            a = self._shorten_action(actions[0])
            if len(a) > 10:  # Only use if meaningful
                parts.append(f"Throughout the story, {entity} {a}.")

        # Relationships
        rel = facts.get('relationships', [])
        if rel:
            if len(rel) == 1:
                parts.append(f"Key relationships include interactions with {rel[0]}.")
            elif len(rel) == 2:
                parts.append(f"Key relationships include interactions with {rel[0]} and {rel[1]}.")
            else:
                parts.append(
                    f"Key relationships include interactions with "
                    f"{', '.join(rel[:3])}."
                )

        return ' '.join(parts)

    def _shorten_action(self, action: str) -> str:
        """Shorten an action phrase to its core."""
        import re
        
        # Remove leading entity name if present
        words = action.split()
        if len(words) > 1 and words[0][0].isupper():
            words = words[1:]
        result = ' '.join(words).strip()
        
        # Stop at conjunctions or other entity names
        # This prevents "looked well, and elizabeth had little opportunity"
        stop_patterns = [
            r'\s+and\s+',  # "and"
            r'\s+but\s+',  # "but"
            r'\s+or\s+',   # "or"
            r'\s+so\s+',   # "so"
            r'\s+then\s+', # "then"
            r',\s+[A-Z]',  # comma followed by capital letter (new entity)
        ]
        for pattern in stop_patterns:
            match = re.search(pattern, result)
            if match:
                result = result[:match.start()]
                break
        
        # Trim to ~6 words max for conciseness
        words = result.split()
        if len(words) > 6:
            result = ' '.join(words[:6])
        
        # Remove trailing punctuation and clean up
        result = result.rstrip('.,;:!?')
        
        return result.strip()

    def _elevate_action(self, action: str) -> str:
        """Elevate action language for formal style."""
        ELEVATIONS = {
            'felt': 'experienced',
            'looked': 'regarded',
            'turned': 'oriented',
            'spoke': 'addressed',
            'said': 'remarked',
            'told': 'informed',
            'asked': 'inquired',
            'thought': 'contemplated',
            'saw': 'observed',
            'knew': 'understood',
            'began': 'commenced',
            'continued': 'persisted',
            'seemed': 'appeared',
            'noticed': 'perceived',
            'reached': 'extended toward',
            'took': 'received',
            'gave': 'bestowed',
            'found': 'discovered',
            'walked': 'proceeded',
            'stood': 'remained',
            'sat': 'settled',
            'smiled': 'offered a smile',
            'laughed': 'laughed gently',
        }
        words = action.split()
        if words:
            verb = words[0].lower()
            if verb in ELEVATIONS:
                words[0] = ELEVATIONS[verb]
        return ' '.join(words)

    def format_options(self, results: Dict[str, str]) -> str:
        """Format all options for display."""
        LABELS = {
            'professional': 'Professional, concise',
            'formal': 'Formal, literary',
            'neutral': 'Neutral summary',
            'gpt2': 'DistilGPT2 prose',
        }
        parts = []
        for i, (style, text) in enumerate(results.items(), 1):
            label = LABELS.get(style, style.title())
            parts.append(f"Option {i} -- {label}\n{text}")
        return '\n\n'.join(parts)
