# BookBot Generic Libraries
# These are standalone libraries with no BookBot dependencies.
# They can be used by any module or even in other projects.

# FST Engine - Uses Pynini (OpenFst backend) for OCR correction and text normalization
# Falls back to custom implementation if Pynini not available
try:
    from .fst_engine import PyniniFST as FST
    from .fst_engine import PyniniOCRCorrector as OCRCorrector
    from .fst_engine import PyniniTextNormalizer as TextNormalizer
    PYNINI_AVAILABLE = True
except ImportError:
    from .fst_engine import CustomFST as FST
    from .fst_engine import CustomOCRCorrector as OCRCorrector
    from .fst_engine import CustomTextNormalizer as TextNormalizer
    PYNINI_AVAILABLE = False

from .edit_distance import levenshtein, damerau_levenshtein, jaro_winkler
from .phonetic import soundex, metaphone, double_metaphone
from .gematria import gematria_simple, gematria_reduced
from .anagram import anagram_signature, find_anagrams
from .text_utils import normalize_spaces, remove_punctuation, tokenize_words
