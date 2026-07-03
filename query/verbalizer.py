"""
Verbalizer
Verb conjugation, subject-verb agreement, articles, pronouns, pluralization.
"""

import re
import logging
from typing import Dict, Optional

logger = logging.getLogger('bookbot.query.verbalizer')

IRREGULAR_VERBS = {
    'be': {'pres': {'1s': 'am', '2s': 'are', '3s': 'is', '1p': 'are', '2p': 'are', '3p': 'are'},
           'past': {'1s': 'was', '2s': 'were', '3s': 'was', '1p': 'were', '2p': 'were', '3p': 'were'},
           'ppart': 'been', 'ppres': 'being'},
    'have': {'pres': {'3s': 'has', 'other': 'have'}, 'past': 'had', 'ppart': 'had', 'ppres': 'having'},
    'do': {'pres': {'3s': 'does', 'other': 'do'}, 'past': 'did', 'ppart': 'done', 'ppres': 'doing'},
    'say': {'pres': {'3s': 'says', 'other': 'say'}, 'past': 'said', 'ppart': 'said'},
    'go': {'pres': {'3s': 'goes', 'other': 'go'}, 'past': 'went', 'ppart': 'gone'},
    'get': {'pres': {'3s': 'gets', 'other': 'get'}, 'past': 'got', 'ppart': 'gotten'},
    'make': {'pres': {'3s': 'makes', 'other': 'make'}, 'past': 'made', 'ppart': 'made'},
    'know': {'pres': {'3s': 'knows', 'other': 'know'}, 'past': 'knew', 'ppart': 'known'},
    'think': {'pres': {'3s': 'thinks', 'other': 'think'}, 'past': 'thought', 'ppart': 'thought'},
    'see': {'pres': {'3s': 'sees', 'other': 'see'}, 'past': 'saw', 'ppart': 'seen'},
    'come': {'pres': {'3s': 'comes', 'other': 'come'}, 'past': 'came', 'ppart': 'come'},
    'take': {'pres': {'3s': 'takes', 'other': 'take'}, 'past': 'took', 'ppart': 'taken'},
    'find': {'pres': {'3s': 'finds', 'other': 'find'}, 'past': 'found', 'ppart': 'found'},
    'give': {'pres': {'3s': 'gives', 'other': 'give'}, 'past': 'gave', 'ppart': 'given'},
    'tell': {'pres': {'3s': 'tells', 'other': 'tell'}, 'past': 'told', 'ppart': 'told'},
    'become': {'pres': {'3s': 'becomes', 'other': 'become'}, 'past': 'became', 'ppart': 'become'},
    'leave': {'pres': {'3s': 'leaves', 'other': 'leave'}, 'past': 'left', 'ppart': 'left'},
    'feel': {'pres': {'3s': 'feels', 'other': 'feel'}, 'past': 'felt', 'ppart': 'felt'},
    'put': {'pres': {'3s': 'puts', 'other': 'put'}, 'past': 'put', 'ppart': 'put'},
    'bring': {'pres': {'3s': 'brings', 'other': 'bring'}, 'past': 'brought', 'ppart': 'brought'},
    'begin': {'pres': {'3s': 'begins', 'other': 'begin'}, 'past': 'began', 'ppart': 'begun'},
    'keep': {'pres': {'3s': 'keeps', 'other': 'keep'}, 'past': 'kept', 'ppart': 'kept'},
    'hold': {'pres': {'3s': 'holds', 'other': 'hold'}, 'past': 'held', 'ppart': 'held'},
    'write': {'pres': {'3s': 'writes', 'other': 'write'}, 'past': 'wrote', 'ppart': 'written'},
    'stand': {'pres': {'3s': 'stands', 'other': 'stand'}, 'past': 'stood', 'ppart': 'stood'},
    'hear': {'pres': {'3s': 'hears', 'other': 'hear'}, 'past': 'heard', 'ppart': 'heard'},
    'let': {'pres': {'3s': 'lets', 'other': 'let'}, 'past': 'let', 'ppart': 'let'},
    'mean': {'pres': {'3s': 'means', 'other': 'mean'}, 'past': 'meant', 'ppart': 'meant'},
    'set': {'pres': {'3s': 'sets', 'other': 'set'}, 'past': 'set', 'ppart': 'set'},
    'meet': {'pres': {'3s': 'meets', 'other': 'meet'}, 'past': 'met', 'ppart': 'met'},
    'run': {'pres': {'3s': 'runs', 'other': 'run'}, 'past': 'ran', 'ppart': 'run'},
    'speak': {'pres': {'3s': 'speaks', 'other': 'speak'}, 'past': 'spoke', 'ppart': 'spoken'},
    'sit': {'pres': {'3s': 'sits', 'other': 'sit'}, 'past': 'sat', 'ppart': 'sat'},
    'read': {'pres': {'3s': 'reads', 'other': 'read'}, 'past': 'read', 'ppart': 'read'},
    'pay': {'pres': {'3s': 'pays', 'other': 'pay'}, 'past': 'paid', 'ppart': 'paid'},
    'lose': {'pres': {'3s': 'loses', 'other': 'lose'}, 'past': 'lost', 'ppart': 'lost'},
    'fall': {'pres': {'3s': 'falls', 'other': 'fall'}, 'past': 'fell', 'ppart': 'fallen'},
    'send': {'pres': {'3s': 'sends', 'other': 'send'}, 'past': 'sent', 'ppart': 'sent'},
    'build': {'pres': {'3s': 'builds', 'other': 'build'}, 'past': 'built', 'ppart': 'built'},
    'understand': {'pres': {'3s': 'understands', 'other': 'understand'}, 'past': 'understood', 'ppart': 'understood'},
    'draw': {'pres': {'3s': 'draws', 'other': 'draw'}, 'past': 'drew', 'ppart': 'drawn'},
    'break': {'pres': {'3s': 'breaks', 'other': 'break'}, 'past': 'broke', 'ppart': 'broken'},
    'spend': {'pres': {'3s': 'spends', 'other': 'spend'}, 'past': 'spent', 'ppart': 'spent'},
    'cut': {'pres': {'3s': 'cuts', 'other': 'cut'}, 'past': 'cut', 'ppart': 'cut'},
    'grow': {'pres': {'3s': 'grows', 'other': 'grow'}, 'past': 'grew', 'ppart': 'grown'},
    'lead': {'pres': {'3s': 'leads', 'other': 'lead'}, 'past': 'led', 'ppart': 'led'},
    'teach': {'pres': {'3s': 'teaches', 'other': 'teach'}, 'past': 'taught', 'ppart': 'taught'},
    'prove': {'pres': {'3s': 'proves', 'other': 'prove'}, 'past': 'proved', 'ppart': 'proven'},
    'show': {'pres': {'3s': 'shows', 'other': 'show'}, 'past': 'showed', 'ppart': 'shown'},
}

SUBJECT_PRONOUNS = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}
OBJECT_PRONOUNS = {'me', 'you', 'him', 'her', 'it', 'us', 'them'}
POSSESSIVE_PRONOUNS = {'my', 'your', 'his', 'her', 'its', 'our', 'their'}

PRONOUN_MAP = {
    'subject': {'male': 'he', 'female': 'she', 'unknown': 'they', 'neutral': 'it', 'plural': 'they'},
    'object': {'male': 'him', 'female': 'her', 'unknown': 'them', 'neutral': 'it', 'plural': 'them'},
    'possessive': {'male': 'his', 'female': 'her', 'unknown': 'their', 'neutral': 'its', 'plural': 'their'},
    'possessive_noun': {'male': 'his', 'female': 'hers', 'unknown': 'theirs', 'neutral': 'its', 'plural': 'theirs'},
    'reflexive': {'male': 'himself', 'female': 'herself', 'unknown': 'themself', 'neutral': 'itself', 'plural': 'themselves'},
}

KNOWN_MALE = {'darcy', 'bingley', 'bennet', 'wickham', 'collins', 'gardiner',
              'philips', 'denny', 'goulding', 'jackson',
              'jones', 'king', 'morris', 'nicholls', 'robinson', 'saunders',
              'stone', 'wilson', 'william', 'john', 'james', 'thomas',
              'henry', 'george', 'charles', 'edward', 'richard', 'robert',
              'francis', 'joseph', 'david', 'andrew', 'peter', 'paul',
              'simon', 'stephen', 'samuel', 'michael', 'daniel', 'alexander',
              'benjamin', 'christopher', 'nicholas', 'jonathan', 'timothy',
              'matthew', 'luke', 'mark', 'arthur', 'albert', 'victor',
              'harold', 'walter', 'leonard', 'lawrence', 'vincent'}

KNOWN_FEMALE = {'elizabeth', 'jane', 'lydia', 'kitty', 'mary', 'catherine',
                'charlotte', 'anne', 'lucas', 'long',
                'sarah', 'hannah', 'lucy', 'emma',
                'olivia', 'sophia', 'isabella', 'emily', 'amelia',
                'jessica', 'victoria', 'rebecca', 'rachel', 'margaret',
                'helen', 'eliza', 'maria', 'louisa', 'fanny', 'harriet',
                'caroline', 'georgiana', 'ann', 'diana', 'cecelia',
                'alice', 'wendy', 'dorothy', 'rose', 'lily', 'daisy',
                'poppy', 'jasmine', 'ruby', 'pearl', 'ivy', 'fern',
                'hazel', 'holly', 'heather'}

COMMON_IRREGULAR_PAST = {
    'said': 'say', 'says': 'say',
    'told': 'tell', 'tells': 'tell',
    'thought': 'think', 'thinks': 'think',
    'brought': 'bring', 'brings': 'bring',
    'left': 'leave', 'leaves': 'leave',
    'felt': 'feel', 'feels': 'feel',
    'kept': 'keep', 'keeps': 'keep',
    'held': 'hold', 'holds': 'hold',
    'meant': 'mean', 'means': 'mean',
    'met': 'meet', 'meets': 'meet',
    'ran': 'run', 'runs': 'run',
    'sat': 'sit', 'sits': 'sit',
    'read': 'read', 'reads': 'read',
    'stood': 'stand', 'stands': 'stand',
    'knew': 'know', 'knows': 'know',
    'drew': 'draw', 'draws': 'draw',
    'fell': 'fall', 'falls': 'fall',
    'lost': 'lose', 'loses': 'lose',
    'sent': 'send', 'sends': 'send',
    'built': 'build', 'builds': 'build',
    'taught': 'teach', 'teaches': 'teach',
    'spoke': 'speak', 'speaks': 'speak',
    'broke': 'break', 'breaks': 'break',
    'wrote': 'write', 'writes': 'write',
    'grew': 'grow', 'grows': 'grow',
    'led': 'lead', 'leads': 'lead',
    'spent': 'spend', 'spends': 'spend',
    'cut': 'cut', 'cuts': 'cut',
    'heard': 'hear', 'hears': 'hear',
    'paid': 'pay', 'pays': 'pay',
    'showed': 'show', 'shows': 'show',
    'began': 'begin', 'begins': 'begin',
    'became': 'become', 'becomes': 'become',
    'loved': 'love', 'loves': 'love',
    'walked': 'walk', 'walks': 'walk',
    'looked': 'look', 'looks': 'look',
    'opened': 'open', 'opens': 'open',
    'closed': 'close', 'closes': 'close',
    'started': 'start', 'starts': 'start',
    'stopped': 'stop', 'stops': 'stop',
    'turned': 'turn', 'turns': 'turn',
    'wanted': 'want', 'wants': 'want',
    'asked': 'ask', 'asks': 'ask',
    'answered': 'answer', 'answers': 'answer',
    'called': 'call', 'calls': 'call',
    'continued': 'continue', 'continues': 'continue',
}

COMMON_ROLE_WORDS = {
    'mr': 'male', 'mrs': 'female', 'miss': 'female', 'lady': 'female',
    'lord': 'male', 'sir': 'male', 'king': 'male', 'queen': 'female',
    'duke': 'male', 'duchess': 'female', 'prince': 'male', 'princess': 'female',
    'count': 'male', 'countess': 'female', 'father': 'male', 'mother': 'female',
    'brother': 'male', 'sister': 'female', 'son': 'male', 'daughter': 'female',
    'husband': 'male', 'wife': 'female', 'gentleman': 'male', 'lady': 'female',
    'man': 'male', 'woman': 'female', 'boy': 'male', 'girl': 'female',
}


class Verbalizer:
    """Morphology engine for verb conjugation, agreement, articles, pronouns."""

    @staticmethod
    def conjugate(verb: str, tense: str = 'pres', person: str = '3',
                  number: str = 's') -> str:
        """Conjugate a verb to the specified tense/person/number.

        Args:
            verb: Base form of verb
            tense: 'pres' (present) or 'past'
            person: '1' (I/we), '2' (you), '3' (he/she/it/they)
            number: 's' (singular) or 'p' (plural)

        Returns:
            Conjugated verb form
        """
        verb_lower = verb.lower().strip('.,;:!?')

        if verb_lower in IRREGULAR_VERBS:
            info = IRREGULAR_VERBS[verb_lower]
            if tense == 'pres':
                form_key = f'{person}{number}'
                if form_key in info['pres']:
                    return info['pres'][form_key]
                if 'other' in info['pres']:
                    return info['pres']['other']
                return info['pres'].get('3p', verb_lower)
            elif tense == 'past':
                return info.get('past', verb_lower + 'ed')
            else:
                return verb_lower

        # Regular verb
        if tense == 'past':
            if verb_lower.endswith('e'):
                return verb_lower + 'd'
            if verb_lower.endswith('y') and len(verb_lower) > 2 and verb_lower[-2] not in 'aeiou':
                return verb_lower[:-1] + 'ied'
            return verb_lower + 'ed'

        if tense == 'pres' and person == '3' and number == 's':
            if verb_lower.endswith(('s', 'sh', 'ch', 'x', 'z', 'o')):
                return verb_lower + 'es'
            if verb_lower.endswith('y') and len(verb_lower) > 2 and verb_lower[-2] not in 'aeiou':
                return verb_lower[:-1] + 'ies'
            return verb_lower + 's'

        return verb_lower

    @staticmethod
    def article(noun: str, definite: bool = False) -> str:
        """Return the appropriate article for a noun.

        Args:
            noun: The noun to determine article for
            definite: If True, return 'the'; else choose 'a'/'an'

        Returns:
            Article string (including trailing space) or empty string
        """
        if definite:
            return 'the '

        noun_lower = noun.strip('.,;:!?').lower()
        if not noun_lower:
            return ''

        if noun_lower[0] in 'aeiou':
            return 'an '
        return 'a '

    @staticmethod
    def pronoun(gender: str, case: str = 'subject',
                number: str = 's') -> str:
        """Generate a pronoun based on gender and case.

        Args:
            gender: 'male', 'female', 'unknown', 'neutral'
            case: 'subject', 'object', 'possessive', 'possessive_noun', 'reflexive'
            number: 's' (singular) or 'p' (plural)

        Returns:
            Pronoun string
        """
        if number == 'p':
            gender = 'plural'
        if gender not in PRONOUN_MAP.get(case, {}):
            gender = 'unknown'
        return PRONOUN_MAP.get(case, PRONOUN_MAP['subject']).get(gender, 'they')

    @staticmethod
    def infer_gender(entity_name: str) -> str:
        """Infer gender from entity name or associated words."""
        name_lower = entity_name.lower()

        for word, gender in GENDER_HINTS.items():
            if word in name_lower:
                return gender

        if name_lower.endswith(('a', 'ia', 'ina', 'ella', 'ette')):
            return 'female'

        return 'unknown'

    @staticmethod
    def pluralize(noun: str) -> str:
        """Pluralize a noun."""
        if not noun:
            return ''
        noun_lower = noun.lower()
        if noun_lower.endswith(('s', 'x', 'z', 'ch', 'sh')):
            return noun + 'es'
        if noun_lower.endswith('y') and len(noun_lower) > 2 and noun_lower[-2] not in 'aeiou':
            return noun[:-1] + 'ies'
        if noun_lower.endswith('f'):
            return noun[:-1] + 'ves'
        if noun_lower.endswith('fe'):
            return noun[:-2] + 'ves'
        return noun + 's'

    @staticmethod
    def normalize_verb_form(verb: str) -> str:
        """Get the base form of a verb (lemmatize)."""
        v = verb.lower().strip('.,;:!?')

        BE_FORMS = {'am': 'be', 'is': 'be', 'are': 'be', 'was': 'be',
                    'were': 'be', 'been': 'be', 'being': 'be', 'be': 'be'}
        HAVE_FORMS = {'has': 'have', 'had': 'have', 'having': 'have',
                      'have': 'have'}
        DO_FORMS = {'does': 'do', 'did': 'do', 'done': 'do', 'doing': 'do',
                    'do': 'do'}

        if v in BE_FORMS:
            return BE_FORMS[v]
        if v in HAVE_FORMS:
            return HAVE_FORMS[v]
        if v in DO_FORMS:
            return DO_FORMS[v]
        if v in IRREGULAR_VERBS:
            return v
        if v in COMMON_IRREGULAR_PAST:
            return COMMON_IRREGULAR_PAST[v]

        # Past tense: -ed or -d (verbs ending in e)
        if v.endswith('ed') and len(v) > 3:
            base_ed = v[:-2]  # "restored" -> "restor"
            base_d = v[:-1]   # "loved" -> "love"
            # Prefer strip-d if it ends in 'e' (more likely correct)
            if base_d.endswith('e') and len(base_d) > 2:
                return base_d
            # Try adding 'e': "restor" + "e" = "restore"
            if not base_ed.endswith('e') and len(base_ed) > 2:
                return base_ed + 'e'
            return base_ed
        if v.endswith('d') and len(v) > 3 and v[-2] != 'e':
            base = v[:-1]
            if base in IRREGULAR_VERBS:
                return base
            return base

        # 3rd person: -s / -es
        if v.endswith('s') and not v.endswith('ss') and len(v) > 2:
            # Try stripping just 's' first (loves -> love)
            base_s = v[:-1]
            if base_s in IRREGULAR_VERBS:
                return base_s
            # Handle -ies -> -y
            if v.endswith('ies') and len(v) > 3:
                base = v[:-3] + 'y'
                return base
            # Handle -es: prefer stem without 's' over stripping 'es'
            if v.endswith('es'):
                base_es = v[:-2]
                # If stem+'e' is more likely than just stem
                if base_s.endswith('e'):
                    return base_s
                return base_es
            return base_s

        # Gerund: -ing
        if v.endswith('ing') and len(v) > 3:
            base = v[:-3]
            if base in IRREGULAR_VERBS:
                return base
            # Double consonant: "running" -> "run"
            if len(base) > 1 and base[-1] == base[-2]:
                return base[:-1]
            # Try adding 'e': "provoking" -> "provok" + "e" = "provoke"
            if base + 'e' in IRREGULAR_VERBS or len(base) > 2:
                return base + 'e' if not base.endswith('e') else base
            return base

        return v

    @staticmethod
    def infer_gender(entity_name: str) -> str:
        """Infer gender from entity name or associated words."""
        name_lower = entity_name.lower().strip()

        # Check known names
        if name_lower in KNOWN_MALE:
            return 'male'
        if name_lower in KNOWN_FEMALE:
            return 'female'

        # Check role words embedded in the name
        for word, gender in COMMON_ROLE_WORDS.items():
            if word == name_lower or name_lower.startswith(word + ' ') or name_lower.endswith(' ' + word):
                return gender

        # Feminine name endings
        feminine_endings = ('a', 'ia', 'ina', 'ella', 'ette', 'ine', 'elle',
                           'issa', 'anna', 'inda', 'ina', 'ara', 'ita', 'ena')
        # Masculine name endings
        masculine_endings = ('o', 'us', 'ius', 'er', 'or', 'an', 'en')

        for ending in feminine_endings:
            if name_lower.endswith(ending) and len(name_lower) > 2:
                return 'female'

        for ending in masculine_endings:
            if name_lower.endswith(ending) and len(name_lower) > 2:
                return 'male'

        return 'unknown'
