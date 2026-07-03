"""
Pipeline - Orchestrates module execution in the correct order.

This class manages the training and query pipelines, ensuring
modules are executed in the correct sequence with proper data flow.
"""

from typing import Any, Dict, List, Optional
import logging
import time

from .pipeline_context import PipelineContext
from .core.base_module import BaseModule


class Pipeline:
    """Orchestrates module execution in the correct order."""

    def __init__(self, config: Dict = None, db_manager=None, logger=None):
        """
        Initialize the pipeline.

        Args:
            config: Configuration dictionary.
            db_manager: Database manager instance.
            logger: Logger instance.
        """
        self.config = config or {}
        self.db = db_manager
        self.logger = logger or logging.getLogger('Pipeline')
        self.modules: Dict[str, BaseModule] = {}

    def register_module(self, name: str, module: BaseModule):
        """
        Register a module with the pipeline.

        Args:
            name: Module name (e.g., 'ocr_normalizer').
            module: Module instance (must inherit BaseModule).
        """
        self.modules[name] = module
        self.logger.debug(f"Registered module: {name}")

    def get_module(self, name: str) -> Optional[BaseModule]:
        """
        Get a registered module by name.

        Args:
            name: Module name.

        Returns:
            Module instance or None if not found.
        """
        return self.modules.get(name)

    def run_training(self, book_path: str, dict_path: str) -> PipelineContext:
        """
        Run the full training pipeline.

        Args:
            book_path: Path to the book text file.
            dict_path: Path to the dictionary CSV file.

        Returns:
            PipelineContext with all training results.
        """
        context = PipelineContext()
        start_time = time.time()

        self.logger.info("BookBot Training Started")
        self.logger.info(f"Book: {book_path}")
        self.logger.info(f"Dictionary: {dict_path}")

        # Load raw data
        with open(book_path, 'r', encoding='utf-8') as f:
            context.raw_text = f.read()

        # Pass 0: OCR Normalization
        if self.config.get('ocr_normalizer', {}).get('enabled', True):
            self.logger.info("=== PASS 0: OCR NORMALIZATION ===")
            ocr = self.modules.get('ocr_normalizer')
            if ocr:
                result = ocr.process(context)
                context.update(result)
                context.pass_number = 0

        # Pass 1: Lexical Foundation
        self.logger.info("=== PASS 1: LEXICAL FOUNDATION ===")
        # Tokenizer
        tokenizer = self.modules.get('tokenizer')
        if tokenizer:
            result = tokenizer.process(context)
            context.update(result)

        # POS Tagger
        pos_tagger = self.modules.get('pos_tagger')
        if pos_tagger:
            result = pos_tagger.process(context)
            context.update(result)

        # POS Guesser (Phase 2: distributional)
        pos_guesser = self.modules.get('pos_guesser')
        if pos_guesser and self.config.get('pos_guesser', {}).get('use_distributional', True):
            pos_guesser.advance_phase()
            result = pos_guesser.process(context)
            context.update(result)

        # Definition Linker
        def_linker = self.modules.get('definition_linker')
        if def_linker:
            result = def_linker.process(context)
            context.update(result)

        # IDF Builder
        idf_builder = self.modules.get('idf_builder')
        if idf_builder:
            result = idf_builder.process(context)
            context.update(result)

        # BM25 Engine
        bm25 = self.modules.get('bm25_engine')
        if bm25:
            result = bm25.process(context)
            context.update(result)

        context.pass_number = 1

        # Iterative passes (Pass 2-4)
        max_passes = self.config.get('convergence', {}).get('max_passes', 10)
        convergence_method = self.config.get('convergence', {}).get('method', 'kl_divergence')

        for cycle in range(max_passes):
            self.logger.info(f"=== ITERATIVE CYCLE {cycle + 1} ===")

            # Pass 2: Semantic Enrichment
            self.logger.info("--- Pass 2: Semantic Enrichment ---")
            ner = self.modules.get('ner_extractor')
            if ner:
                result = ner.process(context)
                context.update(result)

            svo = self.modules.get('svo_extractor')
            if svo:
                result = svo.process(context)
                context.update(result)

            idiom = self.modules.get('idiom_detector')
            if idiom:
                result = idiom.process(context)
                context.update(result)

            entity_graph = self.modules.get('entity_graph')
            if entity_graph:
                result = entity_graph.process(context)
                context.update(result)

            # Pass 3: Relational Deepening
            self.logger.info("--- Pass 3: Relational Deepening ---")
            coref = self.modules.get('coreference')
            if coref:
                result = coref.process(context)
                context.update(result)

            metaphor = self.modules.get('metaphor_detector')
            if metaphor:
                result = metaphor.process(context)
                context.update(result)

            topic = self.modules.get('topic_modeler')
            if topic:
                result = topic.process(context)
                context.update(result)

            # Pass 4: Discourse and Pragmatics
            self.logger.info("--- Pass 4: Discourse and Pragmatics ---")
            temporal = self.modules.get('temporal_reasoner')
            if temporal:
                result = temporal.process(context)
                context.update(result)

            # Convergence Check
            conv_tracker = self.modules.get('convergence_tracker')
            if conv_tracker:
                converged = conv_tracker.process(context)
                if converged.get('converged', False):
                    self.logger.info("CONVERGED")
                    context.convergence_achieved = True
                    break

            context.pass_number = 2 + cycle

        # Training complete
        elapsed = time.time() - start_time
        self.logger.info(f"Training complete in {elapsed:.1f} seconds")
        self.logger.info(f"Total passes: {context.pass_number + 1}")

        return context

    def run_query(self, query: str, context: PipelineContext,
                  conversation_memory=None) -> Dict[str, Any]:
        """
        Run the query pipeline with conversation awareness and structured knowledge.
        """
        self.logger.info(f"Query: {query}")

        # Step 1: Contextual Query Rewriting
        rewriter = self.modules.get('contextual_rewriter')
        rewrite_result = {
            'rewritten_query': query,
            'is_followup': False,
            'followup_type': None,
            'intent_carryover': None,
            'resolutions': [],
            'expanded_terms': [],
        }
        if rewriter and conversation_memory:
            rewriter.set_conversation_memory(conversation_memory)
            rewrite_result = rewriter.process({
                'query': query,
                'intent': 'EXPLANATORY',
            })

        effective_query = rewrite_result['rewritten_query']

        # Step 2: Query Classifier
        query_classifier = self.modules.get('query_classifier')
        intent = rewrite_result.get('intent_carryover') or 'EXPLANATORY'
        if not rewrite_result.get('intent_carryover') and query_classifier:
            intent_result = query_classifier.process({'query': effective_query})
            intent = intent_result.get('intent', 'EXPLANATORY')

        # Step 3: Extract entities from query
        query_entities = self._extract_entities(effective_query, self.db)

        # Step 4: Trilateral BM25 Retrieval
        bm25 = self.modules.get('trilateral_bm25') or self.modules.get('bm25_engine')
        retrieval_results = []
        if bm25:
            full_query = effective_query
            if rewrite_result.get('expanded_terms'):
                full_query = effective_query + ' ' + ' '.join(rewrite_result['expanded_terms'])
            retrieval_results = bm25.process({
                'query': full_query,
                'intent': intent,
                'context': context
            }).get('results', [])

        # Step 5: Structured Knowledge Retrieval
        structured_evidence = {}
        retriever = self.modules.get('structured_retriever')
        if retriever and query_entities:
            for entity in query_entities[:3]:
                actions = retriever.find_entity_actions(entity)
                experience = retriever.find_entity_experience(entity)
                related = retriever.find_related_entities(entity)
                relationships = retriever.find_relationships(entity)
                descriptions = retriever.find_entity_descriptions(entity)
                attributes = retriever.find_entity_attributes(entity)
                definition = retriever.find_entity_definition(entity)

                # Use clean SVO triples from training (better than raw DB actions)
                clean_svo = []
                if context and hasattr(context, 'svo_triples'):
                    entity_lower = entity.lower()
                    for t in context.svo_triples:
                        s = t.get('subject', '') if isinstance(t, dict) else ''
                        if entity_lower in s.lower():
                            clean_svo.append(t)

                structured_evidence[entity] = {
                    'actions': actions,
                    'svo_triples': clean_svo if clean_svo else actions,
                    'experience': experience,
                    'related_entities': related,
                    'relationships': relationships,
                    'descriptions': descriptions,
                    'attributes': attributes,
                    'definition': definition,
                }

        # Step 6: Answer Engine (with structured evidence)
        answer_engine = self.modules.get('answer_engine')
        answer = {}
        if answer_engine:
            answer = answer_engine.process({
                'query': effective_query,
                'intent': intent,
                'retrieval_results': retrieval_results,
                'context': context,
                'structured_evidence': structured_evidence,
                'query_entities': query_entities,
            })

        # Step 6b: Advanced Prose Generation
        if query_entities:
            primary_entity = query_entities[0]
            entity_evidence = structured_evidence.get(primary_entity, {})

            try:
                from .query.advanced_answer import generate_answer

                # Get SVO triples with sentence links
                all_svo = entity_evidence.get('svo_triples', [])
                if not all_svo and context and hasattr(context, 'svo_triples'):
                    entity_lower = primary_entity.lower()
                    all_svo = [t for t in context.svo_triples
                               if entity_lower in t.get('subject', '').lower()]

                # Get related entities
                related = [r.get('related', '') if isinstance(r, dict)
                           else r[0] if isinstance(r, (list, tuple)) and r
                           else str(r) for r in entity_evidence.get('related_entities', [])]

                # Retrieve original sentences
                orig_sentences = []
                if self.db:
                    sids = []
                    for t in all_svo:
                        if isinstance(t, dict):
                            sid = t.get('sentence_id')
                        elif isinstance(t, (list, tuple)) and len(t) >= 5:
                            sid = t[4]  # sentence_id is 5th element
                        else:
                            continue
                        if sid and sid not in sids:
                            sids.append(sid)
                    if sids:
                        placeholders = ','.join('?' * len(sids))
                        rows = self.db.execute(
                            f"SELECT raw_text FROM sentences "
                            f"WHERE sentence_id IN ({placeholders}) "
                            f"AND LENGTH(raw_text) > 25 AND LENGTH(raw_text) < 400",
                            sids
                        )
                        orig_sentences = [r[0] for r in rows]

                if not orig_sentences:
                    # Fallback: get sentences mentioning entity directly
                    if self.db:
                        rows = self.db.execute(
                            "SELECT raw_text FROM sentences "
                            "WHERE raw_text LIKE ? AND LENGTH(raw_text) > 25 "
                            "AND LENGTH(raw_text) < 400 ORDER BY RANDOM() LIMIT 30",
                            (f'%{primary_entity}%',)
                        )
                        orig_sentences = [r[0] for r in rows]

                if orig_sentences:
                    # Get entity info for scoring
                    entity_info = {
                        'frequency': len(all_svo),
                        'n_svo': len(all_svo),
                        'n_related': len(related),
                    }

                    result = generate_answer(
                        primary_entity, orig_sentences,
                        effective_query, intent,
                        related=related, entity_info=entity_info
                    )

                    if result.get('best'):
                        answer = {
                            'text': result['best'],
                            'options': result.get('options', {}),
                        }
            except Exception as e:
                import logging, traceback
                logging.getLogger('bookbot').warning(f"Advanced answer failed: {e}")
                traceback.print_exc()
                # Fallback to prose realizer
                try:
                    from .query.prose_realizer import ProseRealizer
                    prose = ProseRealizer()
                    synth = prose.realize(primary_entity,
                                         [t for t in all_svo if t.get('object') and len(t.get('object', '')) > 1][:10],
                                         related=related)
                    if synth and len(synth) > 20:
                        answer = {'text': synth}
                except Exception:
                    pass

        # Step 7: Confidence Scorer
        conf_scorer = self.modules.get('confidence_scorer')
        confidence = 0.5
        if conf_scorer:
            conf_result = conf_scorer.process({
                'query': effective_query,
                'answer': answer,
                'retrieval_results': retrieval_results
            })
            confidence = conf_result.get('confidence', 0.5)

        # Step 8: Iterative refinement (if answer is poor)
        # Skip refinement if we already have a good synthesized answer
        refiner = self.modules.get('query_refiner')
        current_text = answer.get('text', '') if isinstance(answer, dict) else str(answer)
        has_synthesized = (query_entities and current_text
                          and any(w in current_text.lower() for w in ['interacts with', 'feels', 'reads', 'walks']))
        if refiner and not has_synthesized:
            for attempt in range(2):
                quality = refiner.evaluate_quality(
                    answer.get('text', '') if isinstance(answer, dict) else str(answer),
                    confidence,
                    retrieval_results
                )
                if quality['good']:
                    break
                # Refine query and retry
                refined_query = refiner.refine(effective_query, {
                    'primary_entity': query_entities[0] if query_entities else None,
                    'intent': intent,
                    'poor_reason': quality['reason'],
                }, attempt)
                effective_query = refined_query
                # Re-retrieve with refined query
                if bm25:
                    retrieval_results = bm25.process({
                        'query': refined_query,
                        'intent': intent,
                        'context': context
                    }).get('results', [])
                if answer_engine:
                    answer = answer_engine.process({
                        'query': refined_query,
                        'intent': intent,
                        'retrieval_results': retrieval_results,
                        'context': context,
                        'structured_evidence': structured_evidence,
                        'query_entities': query_entities,
                    })

        # Step 9: Response Formatting
        formatter = self.modules.get('response_formatter')
        response = {}
        if formatter:
            conversation_data = {
                'is_followup': rewrite_result.get('is_followup', False),
                'followup_type': rewrite_result.get('followup_type'),
                'context_entities': conversation_memory.get_context_entities() if conversation_memory else [],
                'entities_discussed': [e for turn in (conversation_memory.history if conversation_memory else [])
                                       for e in turn.get('entities', [])],
            }
            response = formatter.process({
                'query': query,
                'answer': answer,
                'confidence': confidence,
                'intent': intent,
                'conversation': conversation_data,
                'structured_evidence': structured_evidence,
                'query_entities': query_entities,
            })

        # Step 10: Update conversation memory
        if conversation_memory:
            conversation_memory.add_turn(
                user_query=query,
                answer=response.get('answer', ''),
                entities=query_entities,
                intent=intent,
            )

        return response

    def _extract_entities(self, text: str, db=None) -> List[str]:
        """Extract entities from text. Checks DB for known names (case-insensitive)."""
        entities = []
        words = text.split()

        # First pass: capitalized words (standard NER)
        for word in words:
            cleaned = word.strip('.,;:!?()[]"\'')
            if cleaned and cleaned[0].isupper() and len(cleaned) > 2:
                if cleaned.lower() not in {
                    'the', 'what', 'when', 'where', 'who', 'why', 'how',
                    'this', 'that', 'these', 'those', 'tell', 'please',
                }:
                    entities.append(cleaned)

        # Second pass: check DB for known entity names (case-insensitive)
        if not entities and db:
            text_lower = text.lower()
            try:
                clean_words = [w.strip('.,;:!?()[]"\'').lower() for w in words if len(w) > 2]
                if clean_words:
                    placeholders = ','.join('?' * len(clean_words))
                    rows = db.execute(
                        f"SELECT canonical_name FROM entities "
                        f"WHERE LOWER(canonical_name) IN ({placeholders})",
                        clean_words
                    )
                    for r in rows:
                        name = r[0]
                        if name not in entities:
                            entities.append(name)
            except Exception:
                pass

        # Third pass: if still no entities, try matching against known names
        if not entities and db:
            text_lower = text.lower()
            try:
                rows = db.execute("SELECT canonical_name FROM entities")
                for r in rows:
                    name = r[0]
                    if name.lower() in text_lower and len(name) > 2:
                        if name not in entities:
                            entities.append(name)
            except Exception:
                pass

        return entities
