"""
Reasoning Engine
Transforms raw structured evidence into coherent natural language answers.
Pipeline: FactExtractor -> AnswerPlanner -> SentenceConstructor -> CoherenceAssembler -> QualityAssessor
"""

import re
import logging
from typing import Dict, List, Optional
from .verbalizer import Verbalizer
from .template_engine import TemplateEngine

logger = logging.getLogger('bookbot.query.reasoning_engine')
verb = Verbalizer()
template_engine = TemplateEngine()

DISCOURSE_CONNECTIVES = {
    'elaboration': ['also', 'in addition', 'furthermore', 'moreover'],
    'contrast': ['however', 'nevertheless', 'on the other hand', 'conversely'],
    'cause': ['therefore', 'as a result', 'consequently', 'thus'],
    'sequence': ['then', 'next', 'after that', 'subsequently'],
    'concession': ['although', 'despite this', 'even so'],
    'summary': ['in short', 'overall', 'in summary'],
    'example': ['for instance', 'for example', 'such as'],
}


REFLEXIVE_PRONOUNS = {'herself', 'himself', 'itself', 'themself',
                       'themselves', 'yourself', 'myself', 'ourselves'}
BAD_OBJECTS = {'', ' ', '-', '--', '...', 'the', 'a', 'an', 'it'}
BAD_VERB_OBJECT = {
    'have': {'exposition', 'exposition.', 'the causes', 'causes', '',
             'offensive viraginous', 'nothing offensive', 'nothing'},
    'be': {'', 'the', 'a', 'an'},
    'remain': {'cordial feelings', ''},
}
# Verbs too generic for definitional answers
GENERIC_VERBS = {'be', 'have', 'do', 'say', 'get', 'make', 'go', 'come', 'take'}
GENERIC_VERB_FORMS = {'is', 'are', 'was', 'were', 'has', 'have', 'had', 'does',
                      'do', 'did', 'says', 'said', 'gets', 'got', 'makes',
                      'made', 'goes', 'went', 'comes', 'came', 'takes', 'took'}
SKIP_VERBS = {'be', 'have', 'do', 'were', 'was', 's', 're', 've', 'll', 'd'}

STOPWORDS = {
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'above', 'below', 'between', 'under', 'over',
    'and', 'or', 'but', 'not', 'so', 'if', 'then', 'than', 'too',
    'very', 'just', 'about', 'more', 'some', 'any', 'all', 'each',
    'again', 'also', 'thus', 'hence', 'else', 'even', 'still', 'yet',
    'now', 'here', 'there', 'though', 'although', 'however',
    'indeed', 'perhaps', 'maybe', 'almost', 'quite', 'rather', 'well',
    'ah', 'oh', 'alas', 'why', 'aye', 'nay', 'dear', 'look',
    'yes', 'no', 'so', 'then', 'here', 'there',
}


class Fact:

    def __init__(self, fact_type: str, subject: str, predicate: str,
                 object_val: str = '', confidence: float = 0.5,
                 tense: str = 'pres', negated: bool = False,
                 source: str = ''):
        self.fact_type = fact_type
        self.subject = subject
        self.predicate = predicate
        self.object_val = object_val
        self.confidence = confidence
        self.tense = tense
        self.negated = negated
        self.source = source

    def __repr__(self):
        obj = f' {self.object_val}' if self.object_val else ''
        neg = ' not' if self.negated else ''
        return f'Fact({self.subject}{neg} {self.predicate}{obj})'


class FactExtractor:
    """Extract and normalize facts from raw evidence."""

    def extract(self, evidence: Dict, entity: str) -> Dict:
        """Convert raw evidence into structured facts dictionary."""
        facts = {
            'has_definition': False,
            'has_role': False,
            'has_attr': False,
            'entity': entity,
            'def_np': '',
            'role': '',
            'attr': '',
            'verb': '',
            'object': '',
            'verb2': '',
            'obj2': '',
            'descriptive_noun': '',
            'context': '',
            'related': '',
            'trait': '',
            'actions': [],
            'fact_list': [],
            'text_context': '',
        }

        self._extract_definition(facts, evidence, entity)
        self._extract_attributes(facts, evidence, entity)
        self._extract_actions(facts, evidence, entity)
        self._extract_relations(facts, evidence, entity)
        self._extract_descriptive_noun(facts, evidence, entity)
        self._extract_text_context(facts, evidence, entity)

        return facts

    def _extract_definition(self, facts: Dict, evidence: Dict, entity: str):
        definition = evidence.get('definition', '')
        if not definition:
            return

        # Skip dictionary definitions for named characters who are clearly
        # from the book text (they have descriptions, SVO, or sentences)
        if entity and entity[0].isupper():
            has_book_context = (
                bool(evidence.get('svo_triples')) or
                bool(evidence.get('descriptions')) or
                any(True for s in evidence.get('text_sentences', [])
                    if entity.lower() in (s.get('text', '') if isinstance(s, dict) else str(s)).lower())
            )
            if has_book_context:
                return

        facts['has_definition'] = True
        words = definition.split()[:10]
        np = ' '.join(words)
        if len(np) > 5:
            facts['def_np'] = np
            facts['role'] = words[0] if words else ''

    def _extract_attributes(self, facts: Dict, evidence: Dict, entity: str):
        attrs = evidence.get('attributes', [])
        if attrs:
            facts['has_attr'] = True
            # Strip leading adverbs from attributes (e.g., "again deep in thought" → "deep in thought")
            LEADING_ADVERBS = re.compile(
                r'^(again|still|now|then|also|never|always|even|quite|very|just|already|'
                r'once|yet|often|ever|hardly|scarcely|barely|merely|simply|nearly|almost|'
                r'really|truly|indeed|perhaps|maybe|probably|certainly|surely|not|no|'
                r'rather|somewhat|slightly|barely|scarcely|hardly)\s+', re.IGNORECASE
            )
            best = ''
            for a in attrs:
                a_clean = a.strip('.,;:!?')
                # Strip leading adverbs
                while LEADING_ADVERBS.match(a_clean):
                    a_clean = LEADING_ADVERBS.sub('', a_clean).strip()
                if entity.lower() not in a_clean.lower() and len(a_clean) > 5 and len(a_clean) < 60:
                    best = a_clean
                    break
            if not best and attrs:
                best = attrs[0].strip('.,;:!?')
            if best:
                facts['attr'] = best
                facts['trait'] = best[:40]

        descriptions = evidence.get('descriptions', [])
        if descriptions and not facts['role']:
            # Try to extract "is a / was a" pattern from descriptions
            pat = re.compile(rf'{re.escape(entity)}\s+(?:is|was)\s+(?:a|an)\s+(.+?)[,.]', re.IGNORECASE)
            for desc in descriptions:
                m = pat.search(desc)
                if m:
                    role = m.group(1).strip()
                    if role and len(role) > 2:
                        facts['role'] = role[:40]
                        facts['has_role'] = True
                        break

        if not facts['role']:
            # Try "is described as" or "is known as"
            for desc in descriptions:
                pat = re.compile(rf'{re.escape(entity)}\s+is\s+(?:described|known|regarded)\s+as\s+(.+?)[,.]', re.IGNORECASE)
                m = pat.search(desc)
                if m:
                    facts['role'] = m.group(1).strip()[:40]
                    facts['has_role'] = True
                    break

    def _extract_actions(self, facts: Dict, evidence: Dict, entity: str):
        svo = evidence.get('svo_triples', [])
        if not svo:
            return

        entity_lower = entity.lower()

        def _is_bad_svo(s, v, o):
            s_lower = s.lower().strip()
            v_lower = v.lower().strip()
            o_lower = o.lower().strip()
            # Skip pronouns as subjects (I, she, he, etc.)
            if s_lower in {'i', 'she', 'he', 'it', 'we', 'they', 'you', 'me', 'him', 'her', 'us', 'them'}:
                return True
            # Skip if subject doesn't mention entity
            if entity_lower not in s_lower:
                return True
            # Skip if subject is just "entity name + chapter garbage"
            if 'chapter' in s_lower:
                return True
            # Skip garbage objects
            if o_lower in REFLEXIVE_PRONOUNS | BAD_OBJECTS:
                return True
            # Skip objects that start with 's ' (OCR artifacts)
            if len(o) > 1 and o_lower.startswith('s '):
                return True
            # Skip very long objects (probably merged sentences)
            if len(o) > 60:
                return True
            # Skip very long subjects (merged sentences)
            if len(s) > 40:
                return True
            # Skip generic verbs with no useful object
            base_v = verb.normalize_verb_form(v_lower)
            if base_v in SKIP_VERBS and not o_lower:
                return True
            if base_v in ('s', 'ses', 's_'):
                return True
            if base_v in BAD_VERB_OBJECT:
                obj_no_article = o_lower.replace('a ', '').replace('an ', '').replace('the ', '')
                if o_lower in BAD_VERB_OBJECT[base_v] or obj_no_article in BAD_VERB_OBJECT[base_v]:
                    return True
            # Skip empty or whitespace-only
            if not v_lower.strip() or not s_lower.strip():
                return True
            return False

        # Get actions where entity is the subject
        entity_actions = []
        for t in svo[:20]:
            if isinstance(t, dict):
                s, v, o = t.get('subject', ''), t.get('verb', ''), t.get('object', '')
            elif isinstance(t, (list, tuple)) and len(t) >= 3:
                s, v, o = str(t[0]), str(t[1]), str(t[2])
            else:
                continue
            if _is_bad_svo(s, v, o):
                continue
            base_v = verb.normalize_verb_form(v.strip())
            obj_clean = o.strip('.,;:!?')
            entity_actions.append({
                'verb': base_v,
                'verb_original': v,
                'object': obj_clean,
                'raw': t,
            })

        if not entity_actions:
            return

        # Store first action
        first = entity_actions[0]
        facts['verb'] = verb.conjugate(first['verb'], 'pres', '3', 's')
        facts['object'] = first['object']

        # Store second action if available
        if len(entity_actions) > 1:
            second = entity_actions[1]
            facts['verb2'] = verb.conjugate(second['verb'], 'pres', '3', 's')
            facts['obj2'] = second['object']

        # Build verb list for "who verb+objects"
        action_phrases = []
        for a in entity_actions[:3]:
            conj_verb = verb.conjugate(a['verb'], 'pres', '3', 's')
            phrase = conj_verb
            if a['object']:
                phrase += f' {a["object"]}'
            action_phrases.append(phrase)

        facts['actions'] = action_phrases

    def _extract_relations(self, facts: Dict, evidence: Dict, entity: str):
        related = evidence.get('related_entities', [])
        if not related:
            return

        names = []
        for r in related[:8]:
            if isinstance(r, dict):
                name = r.get('related', '') or r.get('source_id', '') or r.get('target_id', '')
            elif isinstance(r, (list, tuple)):
                name = r[0] if r else ''
            else:
                name = str(r)
            if name and name.lower() != entity.lower() and len(name) > 1:
                # Skip stopwords
                if name.lower() not in STOPWORDS:
                    names.append(name)

        if names:
            if len(names) == 1:
                facts['related'] = names[0]
            elif len(names) == 2:
                facts['related'] = f'{names[0]} and {names[1]}'
            else:
                facts['related'] = f'{", ".join(names[:3])} and others'

    def _extract_descriptive_noun(self, facts: Dict, evidence: Dict, entity: str):
        """Find a descriptive noun like 'woman', 'girl', 'man' from dictionary or context."""
        descs = evidence.get('descriptions', [])
        for desc in descs[:5]:
            lower = desc.lower()
            for pattern in [f'{entity.lower()} is a ', f'{entity.lower()} was a ']:
                idx = lower.find(pattern)
                if idx >= 0:
                    rest = desc[idx + len(pattern):].strip()
                    noun = rest.split()[0] if rest else ''
                    noun = noun.strip('.,;:!?"\'')
                    if noun and noun[0].isalpha():
                        facts['descriptive_noun'] = noun
                        break
            if facts['descriptive_noun']:
                break

    def _extract_text_context(self, facts: Dict, evidence: Dict, entity: str):
        """Extract best descriptive sentence as text context fallback."""
        descriptions = evidence.get('descriptions', [])
        sentences = evidence.get('text_sentences', [])

        best = ''
        best_score = 0

        # Score patterns: prefer entity-starting, descriptive, medium-length sentences
        GOOD_STARTS = [f'{entity.lower()} is ', f'{entity.lower()} was ',
                       f'{entity.lower()} has ', f'{entity.lower()} had ']

        for source in descriptions + [s.get('text', '') if isinstance(s, dict) else str(s) for s in sentences]:
            if not source:
                continue
            lower = source.lower()
            wc = len(source.split())
            if wc < 5 or len(source) > 250:
                continue
            if entity.lower() not in lower:
                continue

            score = 0
            # Prefer sentences starting with entity name
            if lower.startswith(entity.lower()):
                score += 4
            # Prefer be-verb patterns (most descriptive)
            for pat in GOOD_STARTS:
                if pat in lower:
                    score += 3
                    break
            # Prefer medium-length sentences (more info)
            if 8 <= wc <= 30:
                score += 2
            elif 6 <= wc <= 40:
                score += 1
            # Prefer sentences with commas (more complex = more descriptive)
            if ',' in source:
                score += 1
            # Penalize very short or very long
            if wc < 6:
                score -= 2
            if wc > 40:
                score -= 1

            if score > best_score:
                best_score = score
                best = source

        if best:
            facts['text_context'] = best.strip()

    def extract_sentences_context(self, evidence: Dict) -> str:
        """Extract best descriptive sentences for context."""
        sentences = evidence.get('text_sentences', [])
        descriptions = evidence.get('descriptions', [])

        candidates = []
        for d in descriptions:
            wc = len(d.split())
            if 6 <= wc <= 40 and len(d) < 200:
                candidates.append(d)

        for s in sentences:
            text = s.get('text', '') if isinstance(s, dict) else str(s)
            wc = len(text.split())
            if 6 <= wc <= 40 and len(text) < 200:
                candidates.append(text)

        # Prefer shorter, cleaner sentences
        candidates.sort(key=lambda x: len(x))
        return ' '.join(candidates[:2])


class AnswerPlanner:
    """Plan answer structure based on intent and available facts."""

    def plan(self, intent: str, facts: Dict) -> List[Dict]:
        """Build a rhetorical plan as list of sentence specs."""
        if intent == 'DEFINITIONAL':
            return self._plan_definitional(facts)
        elif intent == 'FACTUAL':
            return self._plan_factual(facts)
        elif intent == 'CAUSAL':
            return self._plan_causal(facts)
        elif intent == 'TEMPORAL':
            return self._plan_temporal(facts)
        elif intent == 'COMPARATIVE':
            return self._plan_comparative(facts)
        elif intent == 'SUMMARIZATION':
            return self._plan_summarization(facts)
        else:
            return self._plan_general(facts)

    def _plan_definitional(self, facts: Dict) -> List[Dict]:
        plan = []

        # For definitional queries, always start with SVO action if available
        # (more natural than text_context quotes)
        has_good_action = bool(facts.get('verb') and facts.get('object')
                               and facts['verb'] not in GENERIC_VERB_FORMS
                               and len(facts['object']) > 2)

        if has_good_action:
            plan.append({
                'template_intent': 'DEFINITIONAL',
                'required': {'entity': facts['entity'], 'verb': facts['verb'],
                            'object': facts['object']},
                'discourse_role': 'nucleus',
            })
        elif facts.get('role'):
            plan.append({
                'template_intent': 'DEFINITIONAL',
                'required': {'entity': facts['entity'], 'role': facts['role'],
                            'has_role': True, 'attr': facts.get('attr', ''),
                            'has_attr': facts.get('has_attr', False)},
                'discourse_role': 'nucleus',
            })
        elif facts.get('def_np'):
            plan.append({
                'template_intent': 'DEFINITIONAL',
                'required': {'entity': facts['entity'], 'def_np': facts['def_np'],
                            'has_definition': True},
                'discourse_role': 'nucleus',
            })
        elif facts.get('text_context'):
            plan.append({
                'template_intent': 'DEFINITIONAL',
                'required': {'entity': facts['entity'],
                            'text_context': facts['text_context']},
                'discourse_role': 'nucleus',
            })
        else:
            plan.append({
                'type': 'fallback',
                'text': '{entity} is a character in the story.',
                'discourse_role': 'nucleus',
            })

        # Second sentence: additional action or elaboration
        if facts.get('verb2') and facts.get('obj2') and not has_good_action:
            plan.append({
                'template_intent': 'DEFINITIONAL',
                'required': {'entity': facts['entity'], 'verb': facts['verb2'],
                            'object': facts['obj2']},
                'discourse_role': 'elaboration',
                'connective': 'also',
            })
        elif has_good_action and facts.get('actions') and len(facts['actions']) > 1:
            # Second action
            second = facts['actions'][1] if len(facts['actions']) > 1 else ''
            if second and len(second) > 5:
                parts = second.split()
                v2 = parts[0] if parts else ''
                o2 = ' '.join(parts[1:]) if len(parts) > 1 else ''
                plan.append({
                    'template_intent': 'DEFINITIONAL',
                    'required': {'entity': facts['entity'], 'verb': v2,
                                'object': o2},
                    'discourse_role': 'elaboration',
                    'connective': 'also',
                })

        # Relationships
        if facts.get('related'):
            plan.append({
                'type': 'related_clause',
                'template': 'In the story, {entity} interacts with {related}.',
                'discourse_role': 'satellite',
                'connective': 'furthermore',
            })
        return plan

    def _plan_factual(self, facts: Dict) -> List[Dict]:
        plan = []
        if facts.get('verb'):
            plan.append({
                'template_intent': 'FACTUAL',
                'required': {'entity': facts['entity'], 'verb': facts['verb'],
                            'object': facts['object']},
                'discourse_role': 'nucleus',
            })
        if facts.get('attr'):
            plan.append({
                'type': 'attr_clause',
                'text': f'In the story, {facts["entity"]} is described as someone who is {facts["attr"]}.',
                'discourse_role': 'satellite',
                'connective': 'also',
            })
        return plan

    def _plan_causal(self, facts: Dict) -> List[Dict]:
        plan = []
        if facts.get('verb'):
            plan.append({
                'template_intent': 'CAUSAL',
                'required': {'entity': facts['entity'], 'verb': facts['verb'],
                            'object': facts['object']},
                'discourse_role': 'nucleus',
            })
        return plan

    def _plan_temporal(self, facts: Dict) -> List[Dict]:
        plan = [{'type': 'fallback', 'text': 'The story covers events involving {entity}.',
                'discourse_role': 'nucleus'}]
        return plan

    def _plan_comparative(self, facts: Dict) -> List[Dict]:
        plan = [{'type': 'fallback', 'text': 'Neither entity has enough comparative data.',
                'discourse_role': 'nucleus'}]
        return plan

    def _plan_summarization(self, facts: Dict) -> List[Dict]:
        plan = []
        if facts.get('def_np') or facts.get('role'):
            plan.append({
                'template_intent': 'SUMMARIZATION',
                'required': {'entity': facts['entity'],
                            'role': facts.get('role', facts.get('def_np', '')),
                            'verb': facts.get('verb', 'appears'),
                            'object': facts.get('object', 'in the story'),
                            'related': facts.get('related', 'others')},
                'discourse_role': 'nucleus',
            })
        elif facts.get('text_context'):
            plan.append({
                'type': 'fallback',
                'text': '{text_context}',
                'discourse_role': 'nucleus',
            })
        else:
            plan.append({
                'type': 'fallback',
                'text': '{entity} is a character in the story.',
                'discourse_role': 'nucleus',
            })
        return plan

    def _plan_general(self, facts: Dict) -> List[Dict]:
        plan = []
        if facts.get('verb'):
            plan.append({
                'template_intent': 'GENERAL',
                'required': {'entity': facts['entity'], 'verb': facts['verb'],
                            'object': facts['object']},
                'discourse_role': 'nucleus',
            })
        elif facts.get('text_context'):
            plan.append({
                'type': 'fallback',
                'text': '{text_context}',
                'discourse_role': 'nucleus',
            })
        else:
            plan.append({
                'type': 'fallback',
                'text': 'I don\'t have enough information about {entity}.',
                'discourse_role': 'nucleus',
            })
        return plan


class SentenceConstructor:
    """Convert plan specs into sentences using template engine."""

    def construct(self, spec: Dict, facts: Dict, entity: str,
                  conv: Dict) -> Optional[str]:
        """Build a single sentence from a plan spec."""
        if 'type' in spec and spec['type'] == 'fallback':
            text = spec.get('text', '')
            text = text.replace('{entity}', entity)
            text = text.replace('{entity:pronoun:subject}',
                              verb.pronoun(verb.infer_gender(entity), 'subject'))
            text = text.replace('{related}', facts.get('related', ''))
            text = text.replace('{text_context}', facts.get('text_context', ''))
            return text

        if 'type' in spec and spec['type'] in ('related_clause', 'attr_clause'):
            text = spec.get('template', spec.get('text', ''))
            for key, val in facts.items():
                text = text.replace('{' + key + '}', str(val or ''))
            text = text.replace('{entity}', entity)
            return text

        if 'template_intent' in spec:
            req = spec.get('required', {})
            result = template_engine.render(spec['template_intent'], req, entity, conv)
            if result:
                return result

        return None

    def apply_connective(self, sentence: str, connective: str,
                         is_first: bool = False) -> str:
        """Add discourse connective to a sentence."""
        if not sentence:
            return ''
        if is_first:
            return sentence
        conn_map = {
            'also': 'Also,',
            'furthermore': 'Furthermore,',
            'moreover': 'Moreover,',
            'however': 'However,',
            'therefore': 'Therefore,',
            'consequently': 'Consequently,',
            'in addition': 'In addition,',
            'nevertheless': 'Nevertheless,',
        }
        con = conn_map.get(connective, '')
        if con:
            return f'{con} {sentence[0].lower()}{sentence[1:]}'
        return sentence


class CoherenceAssembler:
    """Assemble sentences into coherent paragraphs."""

    def assemble(self, sentences: List[str], conn_roles: List[str],
                 entity: str) -> str:
        """Combine sentences with discourse connectives for flow."""
        if not sentences:
            return ""

        result_parts = []
        first = True

        for i, sent in enumerate(sentences):
            if not sent:
                continue

            role = conn_roles[i] if i < len(conn_roles) else ''

            # Apply given-new: start with entity reference
            if not first:
                # Use pronoun if entity was the subject of previous sentence
                if role in ('elaboration', 'satellite'):
                    sent = sent  # keep as-is

            # Capitalize first letter
            sent = sent[0].upper() + sent[1:] if sent else sent

            # Ensure ending period
            if sent and not sent.endswith(('.', '!', '?')):
                sent += '.'

            # Remove redundant "In the story, In the story" etc.
            if result_parts and sent.startswith('In the story,'):
                result_parts.append(sent)
            else:
                result_parts.append(sent)

            first = False

        paragraph = ' '.join(result_parts)

        # Clean up double spaces and punctuation
        paragraph = re.sub(r' +', ' ', paragraph)
        paragraph = re.sub(r'\.,', '.', paragraph)
        paragraph = re.sub(r'\s+([.,;:!?])', r'\1', paragraph)

        # Ensure proper sentence boundaries
        paragraph = re.sub(r'\.([A-Z])', r'. \1', paragraph)

        return paragraph.strip()


class QualityAssessor:
    """Check answer quality and decide if retry is needed."""

    CRITICAL_VERBS = {'is', 'are', 'was', 'were', 'has', 'have', 'had',
                      'said', 'told', 'went', 'came', 'made', 'took',
                      'gave', 'found', 'knew', 'thought', 'felt', 'saw',
                      'began', 'became', 'left', 'kept', 'held', 'meant',
                      'met', 'ran', 'spoke', 'sat', 'stood', 'fell',
                      'grew', 'led', 'taught', 'showed', 'brought',
                      'wrote', 'read', 'paid', 'lost', 'sent', 'built',
                      'drew', 'broke', 'spent', 'cut', 'proved'}

    def assess(self, answer: str, entity: str) -> Dict:
        """Assess answer quality and return score + issues."""
        issues = []
        score = 1.0

        if not answer or len(answer) < 10:
            issues.append('too_short')
            score -= 0.3

        if entity.lower() not in answer.lower():
            issues.append('entity_not_mentioned')
            score -= 0.2

        # Check for presence of a real verb (not fragmented SVO)
        words = answer.split()
        has_real_verb = False
        for w in words:
            if w.lower() in self.CRITICAL_VERBS:
                has_real_verb = True
                break

        if not has_real_verb:
            issues.append('no_critical_verb')
            score -= 0.2

        # Check for sentence completeness
        sentence_count = len(re.findall(r'[.!?]', answer))
        if sentence_count < 1:
            issues.append('no_complete_sentence')
            score -= 0.3

        # Check for entity name repetition (too much = bad)
        entity_count = answer.lower().count(entity.lower())
        word_count = len(words)

        if word_count > 0 and entity_count > 0:
            density = entity_count / word_count
            if density > 0.25 and word_count > 10:
                issues.append('entity_overuse')
                score -= 0.1

        # Check if answer is actually descriptive (not just a quote)
        # Descriptive answers contain "is a", "was a", "is the", or character-defining patterns
        DESC_PATTERNS = [' is a ', ' was a ', ' is an ', ' was an ', ' is the ',
                         ' is described ', ' is known ', ' lives in ', ' lives at ']
        has_description = any(p in answer.lower() for p in DESC_PATTERNS)
        # Also consider answers with verb actions as descriptive (e.g., "Elizabeth feels Jane")
        has_actions = bool(re.search(r'\b\w+s \w+', answer))  # simple SVO pattern
        if not has_description and not has_actions and entity.lower() in answer.lower():
            # Answer mentions entity but doesn't describe it
            issues.append('not_descriptive')
            score -= 0.15

        return {
            'score': max(0.0, score),
            'issues': issues,
            'passing': score >= 0.5,
        }


class ReasoningEngine:
    """Main orchestrator: facts -> plan -> sentences -> coherence -> quality."""

    def __init__(self):
        self.fact_extractor = FactExtractor()
        self.answer_planner = AnswerPlanner()
        self.sentence_constructor = SentenceConstructor()
        self.coherence_assembler = CoherenceAssembler()
        self.quality_assessor = QualityAssessor()

    def generate(self, intent: str, entity: str, evidence: Dict,
                 conversation_context: Dict = None) -> str:
        """Generate a natural answer using the full reasoning pipeline."""
        conv = conversation_context or {}

        if not entity:
            return self._fallback_text(evidence)

        # Step 1: Extract structured facts from raw evidence
        facts = self.fact_extractor.extract(evidence, entity)

        # Step 2: Plan the answer structure
        plan = self.answer_planner.plan(intent, facts)

        # Step 3: Construct sentences from plan
        sentences = []
        conn_roles = []
        for i, spec in enumerate(plan):
            sent = self.sentence_constructor.construct(spec, facts, entity, conv)
            if sent:
                sentences.append(sent)
                role = spec.get('discourse_role', '')
                conn_roles.append(role)

        # Step 4: Assemble into coherent paragraph
        answer = self.coherence_assembler.assemble(sentences, conn_roles, entity)

        # Step 5: Add text evidence if answer is too short or low quality
        quality = self.quality_assessor.assess(answer, entity)
        if not quality['passing'] and answer:
            text_context = self.fact_extractor.extract_sentences_context(evidence)
            if text_context:
                answer = text_context

        # Step 6: Synthesis fallback - build answer from available evidence
        # Also trigger if answer is not descriptive (just a quote, no "is a/was a")
        not_descriptive = 'not_descriptive' in quality.get('issues', [])
        if not answer or not quality['passing'] or not_descriptive:
            synthesized = self._synthesize_from_evidence(entity, facts, evidence)
            if synthesized:
                answer = synthesized

        return answer if answer else self._fallback_text(evidence)

    def _synthesize_from_evidence(self, entity: str, facts: Dict,
                                   evidence: Dict) -> str:
        """Build a natural answer by synthesizing available evidence."""
        parts = []

        # 1. Role/definition from evidence
        role = facts.get('role', '')
        defn = facts.get('def_np', '')
        if role:
            parts.append(f"{entity} is {role}.")
        elif defn:
            parts.append(f"{entity} is {defn}.")

        # 2. Attribute/trait (most descriptive available)
        attr = facts.get('attr', '')
        trait = facts.get('trait', '')
        descriptive = attr if attr and len(attr) > 5 and entity.lower() not in attr.lower() else trait
        if descriptive and len(descriptive) > 5:
            parts.append(f"{entity} is described as {descriptive}.")

        # 3. Key actions from SVO (only clean, meaningful ones)
        actions = facts.get('actions', [])
        good_actions = []
        for a in actions[:3]:
            if not a or len(a) < 10:
                continue
            a_lower = a.lower()
            # Skip noise patterns
            if any(g in a_lower for g in ['has the', 'begins them', 'does her', 'was s ', 'might', 'even']):
                continue
            # Skip if verb looks like a fragment (no vowels, too short)
            parts_a = a.split()
            if len(parts_a) < 2:
                continue
            verb_word = parts_a[0] if parts_a else ''
            if len(verb_word) < 3 or not any(c in verb_word for c in 'aeiou'):
                continue
            good_actions.append(a)
        if good_actions:
            action_text = '; '.join(good_actions[:2])
            parts.append(f"{entity} {action_text}.")

        # 4. Relationships
        related = facts.get('related', '')
        if related:
            parts.append(f"In the story, {entity} interacts with {related}.")

        # 5. Text context as last resort (only if nothing else)
        if not parts:
            text_ctx = facts.get('text_context', '')
            if text_ctx and len(text_ctx) > 10:
                parts.append(text_ctx)

        if not parts:
            parts.append(f"{entity} is a character in the story.")

        return ' '.join(parts)

    def _fallback_text(self, evidence: Dict) -> str:
        """Generate simple fallback from text."""
        sentences = evidence.get('text_sentences', [])
        descriptions = evidence.get('descriptions', [])
        for d in descriptions:
            if len(d.split()) >= 5:
                return d
        for s in sentences:
            text = s.get('text', '') if isinstance(s, dict) else str(s)
            if len(text.split()) >= 5:
                return text
        return "I couldn't find relevant information in the book."

    def generate_followups(self, entity: str, evidence: Dict,
                           conversation_context: Dict = None) -> List[str]:
        """Generate relevant follow-up suggestions."""
        suggestions = []
        conv = conversation_context or {}
        discussed = conv.get('entities_discussed', [])

        related = evidence.get('related_entities', [])
        for r in related[:5]:
            if isinstance(r, dict):
                name = r.get('related', '') or r.get('source_id', '') or r.get('target_id', '')
            elif isinstance(r, (list, tuple)):
                name = r[0] if r else ''
            else:
                name = str(r)
            if name and name.lower() not in [d.lower() for d in discussed] and len(name) > 1:
                if name.lower() not in STOPWORDS:
                    suggestions.append(f"Tell me more about {name}.")
                    if len(suggestions) >= 2:
                        break

        svo = evidence.get('svo_triples', [])
        SKIP_VERBS = {'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being',
                       's', 'es', 'ses'}
        for t in svo[:8]:
            if isinstance(t, dict):
                v = t.get('verb', '')
            elif isinstance(t, (list, tuple)):
                v = t[1] if len(t) > 1 else ''
            else:
                continue
            if not v:
                continue
            v_clean = v.strip("'").lower().strip()
            if v_clean in SKIP_VERBS or len(v_clean) <= 2:
                continue
            # Skip verbs that look like fragments (no vowels, too short, etc.)
            if not any(c in v_clean for c in 'aeiou'):
                continue
            if len(v_clean) < 3:
                continue
            # Normalize to base form for "What did X ___?"
            base_v = verb.normalize_verb_form(v_clean)
            if base_v and len(base_v) >= 3 and base_v not in SKIP_VERBS:
                suggestions.append(f"What did {entity} {base_v}?")
                if len(suggestions) >= 3:
                    break

        return suggestions[:3]
