"""
Template Engine
Parameterized templates per intent with grammatical features.
Provides coherent sentence construction from structured facts.
"""

import re
import logging
from typing import Dict, List, Optional
from .verbalizer import Verbalizer

logger = logging.getLogger('bookbot.query.template_engine')

verb = Verbalizer()


class TemplateSpec:
    """A single template with slots and grammatical constraints."""

    def __init__(self, intent: str, template_str: str, priority: int = 0,
                 min_facts: int = 1, conditions: Dict = None):
        self.intent = intent
        self.template_str = template_str
        self.priority = priority
        self.min_facts = min_facts
        self.conditions = conditions or {}

    def applicable(self, facts: Dict) -> bool:
        """Check if template can be filled with available facts."""
        for slot in re.findall(r'\{(\w+)\}', self.template_str):
            if slot not in facts or not facts[slot]:
                return False
        for cond, val in self.conditions.items():
            if val is True:
                if not facts.get(cond):
                    return False
            elif facts.get(cond) != val:
                return False
        return True

    def fill(self, facts: Dict, entity: str, conv: Dict) -> str:
        """Fill template slots and apply grammatical normalization."""
        filled = self.template_str

        # Fill all slots from facts
        for key, val in facts.items():
            placeholder = '{' + key + '}'
            if placeholder in filled:
                filled = filled.replace(placeholder, str(val or ''))

        # Apply verbalizer to specific slots
        filled = self._apply_grammar(filled, entity, conv)

        return filled.strip()

    def _apply_grammar(self, text: str, entity: str, conv: Dict) -> str:
        """Post-process for grammar: conjugation, articles, pronouns."""
        # Fix bare verb forms: "Elizabeth be clever" -> "Elizabeth is clever"
        gender = verb.infer_gender(entity)

        # Handle {entity} her/his -> resolve pronoun
        text = re.sub(
            r'\{entity:pronoun:(\w+)\}',
            lambda m: verb.pronoun(gender, m.group(1)),
            text
        )

        # Handle {entity:name} -> the entity name
        text = text.replace('{entity:name}', entity)

        # Fix subject-verb agreement: entity + verb
        # Pattern: "Entity verb" where verb is base form
        conjugations = {
            r'\b(am|are|is|was|were)\b': None,  # already correct
        }

        # Fix articles before nouns (a/an)
        text = re.sub(
            r'\ba ([aeiouAEIOU]\w+)',
            lambda m: 'an ' + m.group(1),
            text
        )

        return text


TEMPLATE_LIBRARY = {
    'DEFINITIONAL': [
        TemplateSpec('DEFINITIONAL',
            '{entity} is {def_np}.',
            priority=10, min_facts=1,
            conditions={'has_definition': True}),

        TemplateSpec('DEFINITIONAL',
            '{entity} is {role}. {entity:pronoun:subject} is described as {attr} in the story.',
            priority=9, min_facts=2,
            conditions={'has_role': True, 'has_attr': True}),

        TemplateSpec('DEFINITIONAL',
            '{entity} is {role}.',
            priority=8, min_facts=1,
            conditions={'has_role': True}),

        TemplateSpec('DEFINITIONAL',
            '{entity} {verb} {object}. {entity:pronoun:subject} is a {descriptive_noun} in the story.',
            priority=7, min_facts=2),

        TemplateSpec('DEFINITIONAL',
            '{entity} is described as {trait}. {entity:pronoun:subject} {verb} {object}.',
            priority=6, min_facts=2),

        TemplateSpec('DEFINITIONAL',
            '{entity} is a {descriptive_noun} in the story.',
            priority=6, min_facts=1),

        TemplateSpec('DEFINITIONAL',
            '{entity} is {role} who {verb} {object}. {entity:pronoun:subject} moves among {related} in the story.',
            priority=5, min_facts=3),

        TemplateSpec('DEFINITIONAL',
            '{text_context}',
            priority=4, min_facts=1,
            conditions={'text_context': True}),

        TemplateSpec('DEFINITIONAL',
            '{entity} {verb} {object}.',
            priority=3, min_facts=1),
    ],

    'FACTUAL': [
        TemplateSpec('FACTUAL',
            '{entity} {verb} {object}.',
            priority=10, min_facts=1),

        TemplateSpec('FACTUAL',
            '{entity} {verb} {object} in {context}.',
            priority=9, min_facts=2),

        TemplateSpec('FACTUAL',
            'According to the text, {entity} {verb} {object}.',
            priority=8, min_facts=1),
    ],

    'CAUSAL': [
        TemplateSpec('CAUSAL',
            '{entity} {cause_verb} {cause}. This happens because {reason}.',
            priority=10, min_facts=2),

        TemplateSpec('CAUSAL',
            'Because {reason}, {entity} {verb} {object}.',
            priority=9, min_facts=2),

        TemplateSpec('CAUSAL',
            '{result} leads to {consequence}.',
            priority=8, min_facts=2),
    ],

    'TEMPORAL': [
        TemplateSpec('TEMPORAL',
            'First, {entity} {verb_first} {obj_first}. Then, {entity:pronoun:subject} {verb_second} {obj_second}.',
            priority=10, min_facts=2),

        TemplateSpec('TEMPORAL',
            '{temporal_marker}, {entity} {verb} {object}.',
            priority=9, min_facts=1),
    ],

    'COMPARATIVE': [
        TemplateSpec('COMPARATIVE',
            'Both {entity_a} and {entity_b} are mentioned in the story. '
            '{entity_a} {verb_a} {obj_a}, while {entity_b} {verb_b} {obj_b}.',
            priority=10, min_facts=4),

        TemplateSpec('COMPARATIVE',
            '{entity_a} and {entity_b} {shared_relation}.',
            priority=9, min_facts=1),
    ],

    'SUMMARIZATION': [
        TemplateSpec('SUMMARIZATION',
            '{entity} is a {role} who {verb} {object}. {entity:pronoun:subject} interacts with {related} in the story.',
            priority=10, min_facts=3),

        TemplateSpec('SUMMARIZATION',
            'The story follows {entity}, a {role}. {entity:pronoun:subject} {verb} {object} throughout the narrative.',
            priority=9, min_facts=2),
    ],

    'GENERAL': [
        TemplateSpec('GENERAL',
            '{entity} {verb} {object}.',
            priority=5, min_facts=1),

        TemplateSpec('GENERAL',
            'In the story, {entity} is described as {trait}.',
            priority=4, min_facts=1),

        TemplateSpec('GENERAL',
            '{entity} {verb} {object} and {verb2} {obj2}.',
            priority=3, min_facts=2),
    ],
}


class TemplateEngine:
    """Select and fill templates based on intent and available facts."""

    def __init__(self):
        self.templates = TEMPLATE_LIBRARY

    def get_templates(self, intent: str) -> List[TemplateSpec]:
        """Get all templates for an intent, sorted by priority."""
        if intent in self.templates:
            return sorted(self.templates[intent], key=lambda t: -t.priority)
        return sorted(self.templates['GENERAL'], key=lambda t: -t.priority)

    def select_best(self, intent: str, facts: Dict) -> Optional[TemplateSpec]:
        """Select the best applicable template for given facts."""
        for tmpl in self.get_templates(intent):
            if tmpl.applicable(facts):
                return tmpl
        return None

    def render(self, intent: str, facts: Dict, entity: str = '',
               conv: Dict = None) -> str:
        """Select and fill the best template."""
        conv = conv or {}
        tmpl = self.select_best(intent, facts)
        if tmpl:
            return tmpl.fill(facts, entity, conv)
        return ''
