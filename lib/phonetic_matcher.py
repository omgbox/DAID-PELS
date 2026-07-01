"""
BookBot Phonetic Matcher
High-quality phonetic matching using CMU Pronouncing Dictionary,
Double Metaphone, Jaro-Winkler, and phoneme-based scoring.

Based on: C:\projects\phonetic_matcher
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger('bookbot.lib.phonetic_matcher')

# Lazy-loaded singleton
_instance = None


class Encoder:
    """Phonetic and string encoding algorithms."""

    def __init__(self):
        self._cmu_cache = {}

    def _load_cmu(self):
        if not self._cmu_cache:
            try:
                import pronouncing
                pronouncing.init_cmu()
                for entry in pronouncing.pronunciations:
                    word = entry[0]
                    pron = entry[1]
                    word_clean = re.sub(r'[^\w]', '', word.lower())
                    if word_clean:
                        self._cmu_cache.setdefault(word_clean, []).append(pron)
            except Exception as e:
                logger.warning(f"Failed to load CMU dictionary: {e}")
        return self._cmu_cache

    def cmu_phonemes(self, word: str) -> List[str]:
        cmu = self._load_cmu()
        return cmu.get(word.lower(), [])

    def metaphone(self, word: str) -> str:
        import jellyfish
        return jellyfish.metaphone(word)

    def soundex(self, word: str) -> str:
        import jellyfish
        return jellyfish.soundex(word)

    def nysiis(self, word: str) -> str:
        import jellyfish
        return jellyfish.nysiis(word)

    def jaro_winkler(self, word1: str, word2: str) -> float:
        import jellyfish
        return jellyfish.jaro_winkler_similarity(word1, word2)

    def levenshtein(self, word1: str, word2: str) -> int:
        import jellyfish
        return jellyfish.levenshtein_distance(word1, word2)

    def dice_coefficient(self, word1: str, word2: str) -> float:
        bigrams1 = [word1[i:i+2] for i in range(len(word1)-1)]
        bigrams2 = [word2[i:i+2] for i in range(len(word2)-1)]
        if not bigrams1 or not bigrams2:
            return 0.0
        matches = sum(1 for b in bigrams1 if b in bigrams2)
        return (2.0 * matches) / (len(bigrams1) + len(bigrams2))


class Matcher:
    """Candidate retrieval, scoring, and best-match selection."""

    def __init__(self, weights=None):
        self.encoder = Encoder()
        self._dm = None
        self._cmu_words = None
        self._phoneme_index = None
        self._metaphone_index = None

        self.weights = weights or {
            "phoneme_distance": 0.35,
            "metaphone_match": 0.30,
            "jaro_winkler": 0.25,
            "dice_coefficient": 0.10,
        }

    def _get_dm(self):
        if self._dm is None:
            from abydos.phonetic import DoubleMetaphone
            self._dm = DoubleMetaphone()
        return self._dm

    def _build_indices(self):
        if self._cmu_words is not None:
            return

        self._cmu_words = set()
        self._phoneme_index = {}
        self._metaphone_index = {}

        try:
            import pronouncing
            pronouncing.init_cmu()
            dm = self._get_dm()
            for entry in pronouncing.pronunciations:
                word = entry[0]
                pron = entry[1]
                word_clean = re.sub(r'[^\w]', '', word.lower())
                if not word_clean:
                    continue
                self._cmu_words.add(word_clean)
                self._phoneme_index.setdefault(word_clean, []).append(pron)
                meta = dm.encode(word_clean)
                for m in meta:
                    if m:
                        self._metaphone_index.setdefault(m, set()).add(word_clean)
            logger.info(f"Loaded {len(self._cmu_words):,} CMU words, "
                       f"{len(self._metaphone_index):,} metaphone keys")
        except Exception as e:
            logger.warning(f"Failed to build indices: {e}")

    def get_cmu_words(self):
        self._build_indices()
        return self._cmu_words

    def is_in_dict(self, word: str) -> bool:
        self._build_indices()
        return word.lower() in self._cmu_words

    def get_candidates(self, word: str, max_candidates: int = 10) -> List[str]:
        self._build_indices()
        dm = self._get_dm()
        word_lower = word.lower()
        candidates = set()

        # Tier 1: Double Metaphone lookup
        meta = dm.encode(word_lower)
        for m_val in meta:
            if m_val and m_val in self._metaphone_index:
                for w in self._metaphone_index[m_val]:
                    if w != word_lower:
                        candidates.add(w)

        # Tier 2: Exact phoneme match
        if len(candidates) < max_candidates:
            cmu = self._phoneme_index.get(word_lower, [])
            if cmu:
                target_phone = re.sub(r'\d', '', cmu[0])
                for w, phones in self._phoneme_index.items():
                    if w == word_lower:
                        continue
                    for p in phones:
                        p_clean = re.sub(r'\d', '', p)
                        if p_clean == target_phone:
                            candidates.add(w)
                            break
                    if len(candidates) >= max_candidates * 3:
                        break

        # Tier 3: Levenshtein fallback
        if len(candidates) < max_candidates:
            for w in self._cmu_words:
                if w == word_lower:
                    continue
                dist = self.encoder.levenshtein(word_lower, w)
                if dist <= 2:
                    candidates.add(w)
                if len(candidates) >= max_candidates * 3:
                    break

        if len(candidates) > max_candidates:
            scored = [(c, self.combined_score(word_lower, c)) for c in candidates]
            scored.sort(key=lambda x: x[1], reverse=True)
            return [c for c, s in scored[:max_candidates]]

        return list(candidates)

    def phoneme_score(self, word1: str, word2: str) -> float:
        phones1 = self.encoder.cmu_phonemes(word1)
        phones2 = self.encoder.cmu_phonemes(word2)
        if not phones1 or not phones2:
            return 0.0

        best = 0.0
        for p1 in phones1[:2]:
            for p2 in phones2[:2]:
                s = self.encoder.jaro_winkler(p1, p2)
                if s > best:
                    best = s
        return best

    def metaphone_score(self, word1: str, word2: str) -> float:
        dm = self._get_dm()
        dm1 = dm.encode(word1)
        dm2 = dm.encode(word2)

        if dm1[0] and dm2[0] and dm1[0] == dm2[0]:
            return 1.0
        if dm1[1] and dm2[1] and dm1[1] == dm2[1]:
            return 0.85
        if (dm1[0] and dm2[1] and dm1[0] == dm2[1]) or \
           (dm1[1] and dm2[0] and dm1[1] == dm2[0]):
            return 0.75

        m1 = self.encoder.metaphone(word1)
        m2 = self.encoder.metaphone(word2)
        if m1 and m2 and m1 == m2:
            return 0.9

        return 0.0

    def jelly_metaphone_score(self, word1: str, word2: str) -> float:
        m1 = self.encoder.metaphone(word1)
        m2 = self.encoder.metaphone(word2)
        if m1 and m2 and m1 == m2:
            return 1.0
        return 0.0

    def phonetic_tier(self, word1: str, word2: str) -> int:
        ms = self.metaphone_score(word1, word2)
        jm = self.jelly_metaphone_score(word1, word2)
        if ms > 0 and jm > 0:
            return 3
        if ms > 0:
            return 2
        if jm > 0:
            return 1
        return 0

    def combined_score(self, word1: str, word2: str) -> float:
        ps = self.phoneme_score(word1, word2)
        ms = self.metaphone_score(word1, word2)
        jw = self.encoder.jaro_winkler(word1, word2)

        if ps > 0:
            score = (
                ps * self.weights["phoneme_distance"]
                + ms * self.weights["metaphone_match"]
                + jw * self.weights["jaro_winkler"]
            )
        else:
            total = self.weights["metaphone_match"] + self.weights["jaro_winkler"]
            score = (
                ms * (self.weights["metaphone_match"] / total)
                + jw * (self.weights["jaro_winkler"] / total)
            )
        return round(min(score, 1.0), 4)

    def find_best_match(self, word: str, candidates=None, threshold: float = 0.65) -> Tuple[str, float]:
        if candidates is None:
            candidates = self.get_candidates(word)
        if not candidates:
            return word, 0.0

        scored = []
        for c in candidates:
            tier = self.phonetic_tier(word, c)
            s = self.combined_score(word, c)
            dist = self.encoder.levenshtein(word, c)
            len_diff = len(c) - len(word)
            scored.append((c, tier, s, dist, len_diff))

        scored.sort(key=lambda x: (x[1], x[2], -x[3], x[4]), reverse=True)

        best_word = scored[0][0]
        best_tier = scored[0][1]
        best_score = scored[0][2]

        if best_tier == 0:
            return word, 0.0

        if best_score >= threshold:
            return best_word, best_score
        return word, 0.0


class PhoneticMatcher:
    """Public API for phonetic matching."""

    def __init__(self, threshold: float = 0.65, custom_words=None):
        self.matcher = Matcher()
        self.threshold = threshold
        self._corrections = []

        if custom_words:
            for word in custom_words:
                self.matcher.get_cmu_words().add(word.lower())

    def is_known_word(self, word: str) -> bool:
        return word.lower() in self.matcher.get_cmu_words()

    def find_match(self, word: str) -> Tuple[str, float]:
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if not word_clean:
            return word, 0.0

        if self.matcher.is_in_dict(word_clean):
            return word_clean, 1.0

        best, score = self.matcher.find_best_match(
            word_clean, threshold=self.threshold
        )
        if best != word_clean and score >= self.threshold:
            return best, score
        return word_clean, 0.0

    def find_matches(self, word: str, top_n: int = 5) -> List[Tuple[str, float]]:
        word_clean = re.sub(r'[^\w]', '', word.lower())
        if not word_clean:
            return [(word, 0.0)]

        candidates = self.matcher.get_candidates(word_clean, max_candidates=10)
        scored = [(c, self.matcher.combined_score(word_clean, c)) for c in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]

    def phonetic_similarity(self, word1: str, word2: str) -> float:
        """Compute phonetic similarity between two words."""
        return self.matcher.combined_score(word1.lower(), word2.lower())

    def get_corrections(self) -> List[Tuple[str, str, float]]:
        return list(self._corrections)


def get_phonetic_matcher() -> PhoneticMatcher:
    """Get or create the singleton PhoneticMatcher instance."""
    global _instance
    if _instance is None:
        _instance = PhoneticMatcher()
    return _instance
