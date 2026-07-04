-- BookBot Database Schema
-- SQLite with WAL journal mode

-- ============================================================
-- PRAGMA SETTINGS
-- ============================================================

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;
PRAGMA temp_store = MEMORY;
PRAGMA mmap_size = 268435456;
PRAGMA foreign_keys = ON;

-- ============================================================
-- CORE LEXICAL TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS definitions (
    definition_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    word            TEXT NOT NULL,
    pos_canonical   TEXT,
    pos_original    TEXT,
    pos_source      TEXT DEFAULT 'dictionary',
    pos_guess_confidence REAL,
    definition      TEXT NOT NULL,
    word_lower      TEXT NOT NULL,
    word_length     INTEGER,
    def_word_count  INTEGER,
    entry_index     INTEGER DEFAULT 0,
    UNIQUE(word_lower, pos_original, definition, entry_index)
);

CREATE INDEX IF NOT EXISTS idx_def_word_lower ON definitions(word_lower);
CREATE INDEX IF NOT EXISTS idx_def_pos ON definitions(pos_canonical);

CREATE TABLE IF NOT EXISTS vocabulary (
    word            TEXT PRIMARY KEY,
    frequency       INTEGER NOT NULL DEFAULT 0,
    document_freq   INTEGER NOT NULL DEFAULT 0,
    idf             REAL NOT NULL DEFAULT 0.0,
    pos_distribution TEXT,
    is_stopword     INTEGER DEFAULT 0,
    is_content_word INTEGER DEFAULT 1,
    first_occurrence INTEGER,
    last_occurrence  INTEGER,
    definition_id   INTEGER,
    FOREIGN KEY (definition_id) REFERENCES definitions(definition_id)
);

CREATE INDEX IF NOT EXISTS idx_vocab_freq ON vocabulary(frequency DESC);
CREATE INDEX IF NOT EXISTS idx_vocab_idf ON vocabulary(idf DESC);

CREATE TABLE IF NOT EXISTS sentences (
    sentence_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter_id      INTEGER,
    paragraph_id    INTEGER,
    position_in_para INTEGER,
    raw_text        TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    token_count     INTEGER,
    word_count      INTEGER,
    pos_tags        TEXT,
    avg_idf         REAL,
    sentiment_score REAL,
    chapter_position REAL
);

CREATE INDEX IF NOT EXISTS idx_sent_chapter ON sentences(chapter_id, position_in_para);
CREATE INDEX IF NOT EXISTS idx_sent_para ON sentences(paragraph_id);

CREATE TABLE IF NOT EXISTS sentence_tokens (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id     INTEGER NOT NULL,
    position        INTEGER NOT NULL,
    token           TEXT NOT NULL,
    token_lower     TEXT NOT NULL,
    pos_tag         TEXT,
    is_punctuation  INTEGER DEFAULT 0,
    is_stopword     INTEGER DEFAULT 0,
    lemma           TEXT,
    definition_id   INTEGER,
    def_link_confidence REAL,
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id),
    FOREIGN KEY (definition_id) REFERENCES definitions(definition_id),
    UNIQUE(sentence_id, position)
);

CREATE INDEX IF NOT EXISTS idx_stok_sentence ON sentence_tokens(sentence_id);
CREATE INDEX IF NOT EXISTS idx_stok_token ON sentence_tokens(token_lower);
CREATE INDEX IF NOT EXISTS idx_stok_def ON sentence_tokens(definition_id);

CREATE TABLE IF NOT EXISTS word_definition_links (
    link_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id     INTEGER NOT NULL,
    token_position  INTEGER NOT NULL,
    definition_id   INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    match_type      TEXT,
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id),
    FOREIGN KEY (definition_id) REFERENCES definitions(definition_id)
);

CREATE INDEX IF NOT EXISTS idx_wdl_sentence ON word_definition_links(sentence_id);
CREATE INDEX IF NOT EXISTS idx_wdl_def ON word_definition_links(definition_id);

-- ============================================================
-- FTS5 VIRTUAL TABLE
-- ============================================================

CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts USING fts5(
    sentence_id UNINDEXED,
    normalized_text,
    tokenize='porter unicode61 remove_diacritics 2'
);

-- ============================================================
-- OCR CORRECTION TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS ocr_corrections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text   TEXT NOT NULL,
    corrected_text  TEXT NOT NULL,
    rule_applied    TEXT,
    line_number     INTEGER,
    confidence      REAL DEFAULT 1.0
);

CREATE INDEX IF NOT EXISTS idx_ocr_line ON ocr_corrections(line_number);

-- ============================================================
-- ENTITY TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS entities (
    entity_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL,
    entity_type     TEXT,
    frequency       INTEGER DEFAULT 0,
    centrality      REAL DEFAULT 0.0,
    degree_centrality REAL DEFAULT 0.0,
    first_sentence  INTEGER,
    last_sentence   INTEGER,
    description     TEXT,
    gender          TEXT DEFAULT 'unknown',
    number_type     TEXT DEFAULT 'singular',
    animacy         TEXT DEFAULT 'unknown',
    UNIQUE(canonical_name, entity_type)
);

CREATE INDEX IF NOT EXISTS idx_ent_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_ent_centrality ON entities(centrality DESC);
CREATE INDEX IF NOT EXISTS idx_ent_name ON entities(canonical_name);

CREATE TABLE IF NOT EXISTS entity_mentions (
    mention_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id       INTEGER NOT NULL,
    sentence_id     INTEGER NOT NULL,
    token_start     INTEGER,
    token_end       INTEGER,
    mention_text    TEXT,
    is_pronoun      INTEGER DEFAULT 0,
    pronoun_type    TEXT,
    coreference_chain_id INTEGER,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_em_entity ON entity_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_em_sentence ON entity_mentions(sentence_id);
CREATE INDEX IF NOT EXISTS idx_em_coref ON entity_mentions(coreference_chain_id);

-- ============================================================
-- RELATIONSHIP TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS svo_triples (
    triple_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    subject         TEXT NOT NULL,
    subject_entity_id INTEGER,
    verb            TEXT NOT NULL,
    verb_original   TEXT,
    object          TEXT NOT NULL,
    object_entity_id INTEGER,
    sentence_id     INTEGER NOT NULL,
    confidence      REAL DEFAULT 0.5,
    negated         INTEGER DEFAULT 0,
    passive         INTEGER DEFAULT 0,
    clause_type     TEXT,
    FOREIGN KEY (subject_entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (object_entity_id) REFERENCES entities(entity_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_svo_subj ON svo_triples(subject);
CREATE INDEX IF NOT EXISTS idx_svo_verb ON svo_triples(verb);
CREATE INDEX IF NOT EXISTS idx_svo_obj ON svo_triples(object);
CREATE INDEX IF NOT EXISTS idx_svo_sentence ON svo_triples(sentence_id);

CREATE TABLE IF NOT EXISTS knowledge_edges (
    edge_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     TEXT NOT NULL,
    source_id       INTEGER NOT NULL,
    target_type     TEXT NOT NULL,
    target_id       INTEGER NOT NULL,
    edge_type       TEXT NOT NULL,
    weight          REAL DEFAULT 1.0,
    properties      TEXT,
    pass_created    INTEGER,
    UNIQUE(source_type, source_id, target_type, target_id, edge_type)
);

CREATE INDEX IF NOT EXISTS idx_ke_source ON knowledge_edges(source_type, source_id);
CREATE INDEX IF NOT EXISTS idx_ke_target ON knowledge_edges(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_ke_type ON knowledge_edges(edge_type);

-- ============================================================
-- COREFERENCE TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS coreference_chains (
    chain_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    representative  TEXT,
    entity_id       INTEGER,
    mention_count   INTEGER DEFAULT 0,
    start_sentence  INTEGER,
    end_sentence    INTEGER,
    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

CREATE TABLE IF NOT EXISTS coreference_mentions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chain_id        INTEGER NOT NULL,
    mention_id      INTEGER NOT NULL,
    sentence_id     INTEGER NOT NULL,
    token_span      TEXT,
    is_representative INTEGER DEFAULT 0,
    antecedent_id   INTEGER,
    resolution_method TEXT,
    resolution_confidence REAL,
    FOREIGN KEY (chain_id) REFERENCES coreference_chains(chain_id),
    FOREIGN KEY (mention_id) REFERENCES entity_mentions(mention_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_cm_chain ON coreference_mentions(chain_id);
CREATE INDEX IF NOT EXISTS idx_cm_sentence ON coreference_mentions(sentence_id);

-- ============================================================
-- IDIOM TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS idiom_lexicon (
    idiom_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    idiom_text      TEXT NOT NULL UNIQUE,
    meaning         TEXT,
    category        TEXT,
    word_count      INTEGER
);

CREATE INDEX IF NOT EXISTS idx_idiom_text ON idiom_lexicon(idiom_text);

CREATE TABLE IF NOT EXISTS idiom_instances (
    instance_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    idiom_id        INTEGER NOT NULL,
    sentence_id     INTEGER NOT NULL,
    token_start     INTEGER,
    token_end       INTEGER,
    confidence      REAL DEFAULT 1.0,
    FOREIGN KEY (idiom_id) REFERENCES idiom_lexicon(idiom_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_ii_idiom ON idiom_instances(idiom_id);
CREATE INDEX IF NOT EXISTS idx_ii_sentence ON idiom_instances(sentence_id);

-- ============================================================
-- METAPHOR TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS metaphors (
    metaphor_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    type            TEXT,
    expression      TEXT NOT NULL,
    sentence_id     INTEGER NOT NULL,
    literal_field   TEXT,
    figurative_field TEXT,
    source_domain   TEXT,
    target_domain   TEXT,
    confidence      REAL DEFAULT 0.5,
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_met_sentence ON metaphors(sentence_id);
CREATE INDEX IF NOT EXISTS idx_met_type ON metaphors(type);

-- ============================================================
-- TOPIC TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS topics (
    topic_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT,
    top_terms       TEXT,
    sentence_count  INTEGER DEFAULT 0,
    centroid_vector TEXT,
    coherence_score REAL
);

CREATE TABLE IF NOT EXISTS topic_sentences (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic_id        INTEGER NOT NULL,
    sentence_id     INTEGER NOT NULL,
    membership_strength REAL DEFAULT 1.0,
    FOREIGN KEY (topic_id) REFERENCES topics(topic_id),
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id),
    UNIQUE(topic_id, sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_ts_topic ON topic_sentences(topic_id);
CREATE INDEX IF NOT EXISTS idx_ts_sentence ON topic_sentences(sentence_id);

-- ============================================================
-- SEMANTIC FIELD TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS semantic_fields (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    word            TEXT NOT NULL,
    pos             TEXT NOT NULL,
    synset_name     TEXT,
    semantic_field  TEXT NOT NULL,
    definition_gloss TEXT,
    UNIQUE(word, pos, synset_name)
);

CREATE INDEX IF NOT EXISTS idx_sf_word ON semantic_fields(word);
CREATE INDEX IF NOT EXISTS idx_sf_field ON semantic_fields(semantic_field);

-- ============================================================
-- TEMPORAL TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS temporal_events (
    event_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    sentence_id     INTEGER NOT NULL,
    raw_expression  TEXT NOT NULL,
    normalized_time TEXT,
    time_type       TEXT,
    event_description TEXT,
    svo_triple_id   INTEGER,
    FOREIGN KEY (sentence_id) REFERENCES sentences(sentence_id),
    FOREIGN KEY (svo_triple_id) REFERENCES svo_triples(triple_id)
);

CREATE INDEX IF NOT EXISTS idx_te_sentence ON temporal_events(sentence_id);
CREATE INDEX IF NOT EXISTS idx_te_time ON temporal_events(normalized_time);

CREATE TABLE IF NOT EXISTS temporal_order (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_a_id      INTEGER NOT NULL,
    event_b_id      INTEGER NOT NULL,
    relation        TEXT NOT NULL,
    confidence      REAL DEFAULT 0.5,
    source_connective TEXT,
    FOREIGN KEY (event_a_id) REFERENCES temporal_events(event_id),
    FOREIGN KEY (event_b_id) REFERENCES temporal_events(event_id),
    UNIQUE(event_a_id, event_b_id)
);

CREATE INDEX IF NOT EXISTS idx_to_a ON temporal_order(event_a_id);
CREATE INDEX IF NOT EXISTS idx_to_b ON temporal_order(event_b_id);

-- ============================================================
-- CAUSAL TABLES
-- ============================================================

CREATE TABLE IF NOT EXISTS causal_chains (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    cause_sentence_id INTEGER NOT NULL,
    cause_svo_id    INTEGER,
    effect_sentence_id INTEGER NOT NULL,
    effect_svo_id   INTEGER,
    causal_type     TEXT,
    connective      TEXT,
    confidence      REAL DEFAULT 0.5,
    FOREIGN KEY (cause_sentence_id) REFERENCES sentences(sentence_id),
    FOREIGN KEY (effect_sentence_id) REFERENCES sentences(sentence_id)
);

CREATE INDEX IF NOT EXISTS idx_cc_cause ON causal_chains(cause_sentence_id);
CREATE INDEX IF NOT EXISTS idx_cc_effect ON causal_chains(effect_sentence_id);

-- ============================================================
-- NARRATIVE STRUCTURE
-- ============================================================

CREATE TABLE IF NOT EXISTS narrative_structure (
    section_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    section_type    TEXT NOT NULL,
    start_sentence_id INTEGER NOT NULL,
    end_sentence_id INTEGER NOT NULL,
    key_events      TEXT,
    summary         TEXT,
    FOREIGN KEY (start_sentence_id) REFERENCES sentences(sentence_id),
    FOREIGN KEY (end_sentence_id) REFERENCES sentences(sentence_id)
);

-- ============================================================
-- PRAGMATIC RULES
-- ============================================================

CREATE TABLE IF NOT EXISTS pragmatic_rules (
    rule_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    antecedent      TEXT NOT NULL,
    consequent      TEXT NOT NULL,
    rule_type       TEXT,
    source_sentence_ids TEXT,
    confidence      REAL DEFAULT 0.5
);

-- ============================================================
-- DOCUMENT METADATA
-- ============================================================

CREATE TABLE IF NOT EXISTS document_stats (
    stat_key        TEXT PRIMARY KEY,
    stat_value      TEXT,
    description     TEXT
);

-- ============================================================
-- CONVERGENCE LOG
-- ============================================================

CREATE TABLE IF NOT EXISTS convergence_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    pass_number     INTEGER NOT NULL,
    timestamp       TEXT NOT NULL,
    new_relationships INTEGER,
    total_relationships INTEGER,
    kl_divergence   REAL,
    ratio           REAL,
    stability       REAL,
    duration_seconds REAL,
    stopped         INTEGER DEFAULT 0,
    stop_reason     TEXT,
    notes           TEXT
);

-- ============================================================
-- TRAINING METADATA
-- ============================================================

CREATE TABLE IF NOT EXISTS training_metadata (
    key             TEXT PRIMARY KEY,
    value           TEXT,
    updated_at      TEXT
);

-- ============================================================
-- LEARNED KNOWLEDGE (from conversations & Wikipedia)
-- ============================================================

CREATE TABLE IF NOT EXISTS learned_knowledge (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT,
    fact            TEXT,
    source          TEXT,
    confidence      REAL DEFAULT 0.5,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_learned_topic ON learned_knowledge(topic);
CREATE INDEX IF NOT EXISTS idx_learned_source ON learned_knowledge(source);
