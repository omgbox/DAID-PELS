"""
BookBot Configuration
All settings, paths, and constants.

All paths are relative to the bookbot/ directory by default.
Override with environment variables or --book/--dict/--db flags.
"""

import os
from pathlib import Path

# =============================================================================
# PATHS (relative to bookbot/ directory)
# =============================================================================

# Project root = bookbot/ directory
PROJECT_ROOT = Path(__file__).parent

# Default book (Pride and Prejudice - clean Gutenberg text)
# Users: place your book.txt in bookbot/books/ or set BOOKBOT_BOOK_PATH
BOOK_PATH = os.environ.get(
    'BOOKBOT_BOOK_PATH',
    str(PROJECT_ROOT / "books" / "pride_and_prejudice_clean.txt")
)
DICTIONARY_PATH = os.environ.get(
    'BOOKBOT_DICT_PATH',
    str(PROJECT_ROOT / "English_dictionary.csv")
)
DATABASE_PATH = os.environ.get(
    'BOOKBOT_DB_PATH',
    str(PROJECT_ROOT / "bookbot.db")
)
LOG_PATH = os.environ.get(
    'BOOKBOT_LOG_PATH',
    str(PROJECT_ROOT / "bookbot.log")
)
NLTK_DATA_PATH = None  # None = NLTK default (~/nltk_data/)

# Data paths
DATA_DIR = Path(__file__).parent / "data"
IDIOM_LEXICON_PATH = DATA_DIR / "idiom_lexicon.txt"
STOPWORD_LIST_PATH = DATA_DIR / "stopword_list.txt"
GAZETTEER_PATH = DATA_DIR / "gazetteer.txt"
OCR_CORRECTIONS_PATH = DATA_DIR / "ocr_corrections.txt"
POS_GUESS_RULES_PATH = DATA_DIR / "pos_guess_rules.txt"
OCR_FST_CACHE_PATH = DATA_DIR / "ocr_fst.bin"

# =============================================================================
# TRAINING PARAMETERS
# =============================================================================

# Pass 0: OCR Normalization
OCR = {
    'enabled': True,
    'use_fst': False,  # Disabled for now - using simple normalization only
    'fst_cache': False,
    'fst_max_edit_distance': 1,
    'fst_high_freq_threshold': 10,
    'confusion_matrix': 'auto',
    'collapse_spaces': True,
    'remove_metadata': True,
    'metadata_lines_start': 100,
    'metadata_lines_end': 40,
    'remove_page_numbers': True,
    'hyphenation_repair': True,
    'dictionary_validation': False,  # Disabled for performance
    'min_word_length': 2,
    'correction_confidence': 0.9,
}

# Pass 1: Lexical
LEXICAL = {
    'pos_tagger_model': 'averaged_perceptron_tagger',
    'pos_guesser_enabled': True,
    'pos_guesser_default': 'NN',
    'pos_guesser_use_distributional': True,
    'skip_pos_disambiguation': True,
    'min_word_length': 1,
    'single_char_exceptions': {'a', 'I'},
    'max_definition_match_candidates': 5,
    'stopword_source': 'nltk',
}

# Pass 2: Semantic
SEMANTIC = {
    'min_entity_frequency': 2,
    'svo_chunk_grammar': r"""
        NP: {<DT|PP\$>?<JJ>*<NN.*>+}
        VP: {<VB.*><NP|PP|RB>*}
    """,
    'svo_max_sentence_length': 300,
    'gazetteer_auto_build': True,
}

# =============================================================================
# CLAUSE DETECTION PARAMETERS
# =============================================================================

CLAUSE = {
    'coord_conjunctions': {
        'and', 'but', 'or', 'nor', 'yet',
    },
    'subord_conjunctions': {
        'though', 'although', 'because', 'since', 'unless', 'until',
        'while', 'when', 'whenever', 'if', 'as', 'after', 'before',
        'once', 'whereas', 'whether', 'so', 'for',
    },
    'clause_boundary_pos': {'IN', 'CC', 'WRB', 'WDT', 'WP', 'WP$'},
    'clause_punctuation': {';', '--', '—', ':'},
}

# Pass 3: Relational
RELATIONAL = {
    'coref_hobbs_weight': 0.4,
    'coref_heuristic_weight': 0.35,
    'coref_centering_weight': 0.25,
    'coref_max_lookback': 10,
    'metaphor_threshold': 0.6,
    'topic_similarity_threshold': 0.3,
    'topic_max_clusters': 50,
    'topic_max_sentences': 500,  # Auto-sample for O(n²) clustering speed
}

# Pass 4: Discourse
DISCOURSE = {
    'temporal_connectives': [
        "before", "after", "when", "while", "during", "until",
        "since", "as", "meanwhile", "subsequently", "previously",
        "then", "next", "first", "finally", "eventually"
    ],
    'causal_connectives': [
        "because", "therefore", "consequently", "as a result",
        "thus", "hence", "so", "due to", "caused by", "led to",
        "resulted in", "gave rise to"
    ],
    'narrative_arc_thresholds': {
        "exposition": 0.10,
        "rising": 0.40,
        "climax": 0.60,
        "falling": 0.85,
        "resolution": 1.00
    },
}

# =============================================================================
# CONVERGENCE PARAMETERS
# =============================================================================

CONVERGENCE = {
    'method': 'kl_divergence',  # 'simple' or 'kl_divergence'
    'kl_threshold': 0.01,
    'laplace_alpha': 0.01,
    'max_passes': 10,
    'min_new_relationships': 5,
    'min_ratio': 0.01,
    'min_stability': 0.98,
    'check_after_pass': 2,
}

# =============================================================================
# QUERY PARAMETERS
# =============================================================================

QUERY = {
    'bm25_k1': 1.5,
    'bm25_b': 0.75,
    'bm25_top_k': 50,
    'fts5_top_k': 50,
    'merged_top_k': 10,
    'bm25_weight': 0.4,
    'fts5_weight': 0.3,
    'kg_weight': 0.3,
    'max_answer_sentences': 5,
    'max_followup_suggestions': 3,
    'conversation_history_size': 5,
}

# =============================================================================
# CONFIDENCE PARAMETERS
# =============================================================================

CONFIDENCE = {
    'retrieval_weight': 0.30,
    'coverage_weight': 0.25,
    'consensus_weight': 0.20,
    'entity_weight': 0.15,
    'intent_weight': 0.10,
    'min_for_display': 0.10,
}

# =============================================================================
# CONVERSATIONAL PARAMETERS (NEW)
# =============================================================================

CONVERSATIONAL = {
    'enable_general_knowledge': True,
    'enable_personal_statements': True,
    'enable_learning': True,
    'general_knowledge_sources': ['wikipedia', 'local_kb', 'generated'],
    'max_learned_facts': 10000,
    'personalization_strength': 0.3,
    'conversation_style': 'friendly',  # 'friendly', 'formal', 'casual'
}

# Intent routing
INTENT_ROUTING = {
    'GREETING': 'conversational',
    'FAREWELL': 'conversational',
    'HELP': 'conversational',
    'EMOTIONAL': 'conversational',
    'OPINION': 'conversational',
    'PERSONAL_STATEMENT': 'personal',
    'GENERAL_KNOWN': 'knowledge',
    # Book intents stay as 'book'
    'DEFINITIONAL': 'book',
    'FACTUAL': 'book',
    'CAUSAL': 'book',
    'TEMPORAL': 'book',
    'COMPARATIVE': 'book',
    'SUMMARIZATION': 'book',
    'EXPLANATORY': 'book',
    'LISTING': 'book',
}

# =============================================================================
# DATABASE PARAMETERS
# =============================================================================

DATABASE = {
    'cache_size_kb': 64000,
    'mmap_size_bytes': 268435456,
    'wal_autocheckpoint': 1000,
}

# =============================================================================
# LOGGING PARAMETERS
# =============================================================================

LOGGING = {
    'level': os.environ.get('BOOKBOT_LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    'to_file': True,
    'to_console': True,
    'training_progress': True,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(module_name: str) -> dict:
    """Get configuration for a specific module."""
    config_map = {
        'ocr_normalizer': OCR,
        'pos_guesser': LEXICAL,
        'tokenizer': LEXICAL,
        'pos_tagger': LEXICAL,
        'ner_extractor': SEMANTIC,
        'svo_extractor': SEMANTIC,
        'definition_linker': LEXICAL,
        'idf_builder': LEXICAL,
        'idiom_detector': SEMANTIC,
        'metaphor_detector': RELATIONAL,
        'coreference': RELATIONAL,
        'temporal_reasoner': DISCOURSE,
        'entity_graph': SEMANTIC,
        'topic_modeler': RELATIONAL,
        'convergence_tracker': CONVERGENCE,
        'query_classifier': QUERY,
        'bm25_engine': QUERY,
        'answer_engine': QUERY,
        'confidence_scorer': CONFIDENCE,
        'response_formatter': QUERY,
    }
    return config_map.get(module_name, {})
