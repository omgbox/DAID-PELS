"""
Pipeline - Orchestrates module execution in the correct order.

This class manages the training and query pipelines, ensuring
modules are executed in the correct sequence with proper data flow.
"""

import re
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
        Run query through the conversational AI pipeline.
        Simple flow: Understand → Retrieve → Generate → Respond
        """
        self.logger.info(f"Query: {query}")

        # Use Conversational AI as primary handler
        conv_ai = self._get_conversational_ai()
        response = conv_ai.chat(query)

        return {
            'answer': response,
            'intent': 'CONVERSATIONAL',
            'route': 'ai',
            'confidence': 0.9,
            'sources': [],
        }

    def _get_conversational_ai(self):
        """Lazy-load the conversational AI."""
        if not hasattr(self, '_conv_ai'):
            from .query.conversational_ai import ConversationalAI
            self._conv_ai = ConversationalAI(self.db)
        return self._conv_ai

        # === Book QA (existing pipeline) ===

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

        # Step 2: Query Classifier (already done above, reuse)
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

                # Retrieve original sentences with quality filtering
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
                            f"AND LENGTH(raw_text) > 40 AND LENGTH(raw_text) < 300 "
                            f"AND raw_text NOT LIKE '%\"%' "  # Skip dialogue
                            f"AND raw_text NOT LIKE '%--%' "  # Skip dashes
                            f"AND raw_text NOT LIKE '%;%' ",  # Skip complex sentences
                            sids
                        )
                        orig_sentences = [r[0] for r in rows]

                if not orig_sentences:
                    # Fallback: get sentences mentioning entity directly
                    if self.db:
                        rows = self.db.execute(
                            "SELECT raw_text FROM sentences "
                            "WHERE raw_text LIKE ? "
                            "AND LENGTH(raw_text) > 40 AND LENGTH(raw_text) < 300 "
                            "AND raw_text NOT LIKE '%\"%' "
                            "AND raw_text NOT LIKE '%--%' "
                            "AND raw_text NOT LIKE '%;%' "
                            "ORDER BY LENGTH(raw_text) DESC LIMIT 30",
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

        # Step 10: Update conversation memory and learn
        if conversation_memory:
            conversation_memory.add_turn(
                user_query=query,
                answer=response.get('answer', ''),
                entities=query_entities,
                intent=intent,
            )

            # Learn from this turn
            user_profile = self.modules.get('user_profile')
            knowledge_retriever = self.modules.get('general_knowledge_retriever')
            conversation_memory.learn_from_turn(
                user_query=query,
                answer=response.get('answer', ''),
                intent=intent,
                user_profile=user_profile,
                knowledge_retriever=knowledge_retriever
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

    def _run_conversational(self, query: str, intent: str,
                           conversation_memory=None) -> Dict[str, Any]:
        """
        Handle conversational intents (greetings, farewells, emotional, etc.).

        Args:
            query: User's query
            intent: Classified intent
            conversation_memory: Conversation memory object

        Returns:
            Response dict
        """
        responder = self.modules.get('conversational_responder')
        if not responder:
            # Import and create if not registered
            from .query.conversational_responder import ConversationalResponder
            responder = ConversationalResponder()

        # Build context from conversation memory and user profile
        context = {}
        if conversation_memory:
            context['user_name'] = conversation_memory.get_user_name() if hasattr(conversation_memory, 'get_user_name') else None

        # Add learned knowledge to context
        knowledge_retriever = self.modules.get('general_knowledge_retriever')
        if knowledge_retriever:
            try:
                # Get recent learned knowledge
                if knowledge_retriever.db:
                    rows = knowledge_retriever.db.execute(
                        "SELECT topic, fact, confidence FROM learned_knowledge "
                        "ORDER BY timestamp DESC LIMIT 10"
                    )
                    context['learned_knowledge'] = [
                        {'topic': r[0], 'fact': r[1], 'confidence': r[2]}
                        for r in rows
                    ]
            except Exception:
                context['learned_knowledge'] = []

        # Add user preferences to context
        user_profile = self.modules.get('user_profile')
        if user_profile:
            try:
                context['user_preferences'] = user_profile.get_preferences()
            except Exception:
                context['user_preferences'] = []

        result = responder.process(intent, query, context)

        # Personalize response
        response_text = result.get('response', '')
        personalizer = self.modules.get('response_personalizer')
        if personalizer:
            user_profile = self.modules.get('user_profile')
            response_text = personalizer.personalize(
                response_text, intent, user_profile, conversation_memory
            )

        return {
            'answer': response_text,
            'intent': intent,
            'route': 'conversational',
            'confidence': 1.0,
            'sources': [],
        }

    def _run_personal(self, query: str, intent: str,
                     conversation_memory=None) -> Dict[str, Any]:
        """
        Handle personal statements ("I like gardening").

        Args:
            query: User's query
            intent: Classified intent
            conversation_memory: Conversation memory object

        Returns:
            Response dict
        """
        handler = self.modules.get('personal_statement_handler')
        if not handler:
            # Import and create if not registered
            from .query.personal_statement_handler import PersonalStatementHandler
            from .query.user_profile import UserProfile

            user_profile = self.modules.get('user_profile')
            if not user_profile:
                user_profile = UserProfile(self.db)
                user_profile.initialize()

            handler = PersonalStatementHandler(user_profile)

        result = handler.process(query)

        # Store in conversation memory if available
        if conversation_memory and result.get('stored'):
            conversation_memory.add_turn(query, result['response'], intent='PERSONAL_STATEMENT')

        # Personalize response
        response_text = result.get('response', '')
        personalizer = self.modules.get('response_personalizer')
        if personalizer:
            user_profile = self.modules.get('user_profile')
            response_text = personalizer.personalize(
                response_text, intent, user_profile, conversation_memory
            )

        return {
            'answer': response_text,
            'intent': intent,
            'route': 'personal',
            'confidence': 1.0,
            'sources': [],
        }

    def _run_knowledge(self, query: str, intent: str, context: PipelineContext,
                      conversation_memory=None) -> Dict[str, Any]:
        """
        Handle general knowledge queries using RAG with CONTEXT.
        """
        retriever = self.modules.get('general_knowledge_retriever')
        if not retriever:
            from .query.general_knowledge_retriever import GeneralKnowledgeRetriever
            retriever = GeneralKnowledgeRetriever(self.db)

        # Step 1: Expand query with conversation context
        expanded_query = self._expand_with_context(query, conversation_memory)

        # Step 2: Retrieve Wikipedia facts
        results = retriever.retrieve(expanded_query, max_results=3)

        if results:
            best = results[0]
            wiki_text = best.get('text', '')
            wiki_title = best.get('title', '')

            # Step 3: Use DistilGPT2 to rewrite as natural answer
            natural_answer = self._rewrite_with_gpt2(query, wiki_text, wiki_title)

            if natural_answer:
                return {
                    'answer': natural_answer,
                    'intent': intent,
                    'route': 'knowledge',
                    'confidence': best.get('confidence', 0.8),
                    'sources': [wiki_title],
                }

            return {
                'answer': wiki_text,
                'intent': intent,
                'route': 'knowledge',
                'confidence': best.get('confidence', 0.8),
                'sources': [wiki_title],
            }

        # No knowledge found
        return {
            'answer': "I don't have specific information about that topic. "
                     "Could you tell me more about what you're interested in?",
            'intent': intent,
            'route': 'knowledge',
            'confidence': 0.1,
            'sources': [],
        }

    def _expand_with_context(self, query: str, conversation_memory=None) -> str:
        """
        Expand query with conversation context.
        E.g., "tell me more about the Demilitarized Zone" 
        becomes "Korean Demilitarized Zone" if we just discussed South Korea.
        """
        import re

        if not conversation_memory:
            return query

        # Get the last answer's entities
        last_turn = conversation_memory.get_last_turn()
        if not last_turn:
            return query

        last_answer = last_turn.get('answer', '')
        last_sources = last_turn.get('sources', [])

        # Check if query contains vague references
        vague_patterns = [
            r'\b(the|this|that|it|its|them|they)\b',
            r'\btell me more\b',
            r'\bwhat about\b',
            r'\band\b',
            r'\balso\b',
        ]

        is_vague = any(re.search(p, query.lower()) for p in vague_patterns)

        if not is_vague:
            return query

        # Extract entities from the last answer
        # Look for capitalized words that might be entities
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', last_answer)

        # Also check sources
        for source in last_sources:
            if source and source not in entities:
                entities.append(source)

        if not entities:
            return query

        # Find the most relevant entity to add to the query
        # Check if any entity words appear in the query
        query_lower = query.lower()
        query_words = set(query_lower.split())

        # Remove common words that might cause false matches
        query_words -= {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'do', 'does', 'did',
                       'have', 'has', 'had', 'will', 'would', 'could', 'should', 'may',
                       'might', 'shall', 'can', 'of', 'in', 'to', 'for', 'with', 'on',
                       'at', 'from', 'by', 'about', 'as', 'into', 'through', 'during',
                       'before', 'after', 'above', 'below', 'between', 'out', 'off',
                       'over', 'under', 'again', 'further', 'then', 'once', 'here',
                       'there', 'when', 'where', 'why', 'how', 'all', 'each', 'every',
                       'both', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                       'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
                       'just', 'and', 'but', 'or', 'if', 'while', 'this', 'that', 'these',
                       'those', 'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'him',
                       'his', 'she', 'her', 'it', 'its', 'they', 'them', 'their', 'tell',
                       'me', 'more', 'about'}

        for entity in entities:
            entity_lower = entity.lower()
            entity_words = set(entity_lower.split())

            # Check if entity words overlap with query words
            overlap = entity_words & query_words

            if overlap:
                # Entity partially matches - return the entity as the new query
                return entity

        # If query is about "the" something, add the main entity
        if query_lower.startswith(('the ', 'this ', 'that ')):
            # Use the main entity from the last answer
            main_entity = entities[0] if entities else ''
            if main_entity:
                return main_entity

        # For "tell me more" queries, add context
        if 'tell me more' in query_lower:
            main_entity = entities[0] if entities else ''
            if main_entity:
                return main_entity

        return query

    def _rewrite_with_gpt2(self, query: str, wiki_text: str, title: str) -> str:
        """
        Use DistilGPT2 to rewrite Wikipedia facts as natural human language.
        This is the RAG (Retrieval-Augmented Generation) approach.
        """
        try:
            from .minigpt import DistilGPT2Generator

            generator = DistilGPT2Generator()
            if not generator.load():
                return None

            # Build a prompt that asks GPT2 to rewrite the Wikipedia text
            # as a natural answer to the user's question
            prompt = f"Question: {query}\nFacts: {wiki_text[:300]}\nAnswer:"

            # Generate natural language response
            response = generator.generate_from_prompt(
                prompt,
                max_tokens=100,
                temperature=0.7
            )

            if response and len(response) > 20:
                # Clean up the response
                # Remove the prompt if it's repeated
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()

                # Take only the first few sentences
                sentences = response.split('. ')
                clean_response = '. '.join(sentences[:3]) + '.'

                # Basic quality check - response should be different from input
                if clean_response.lower() != wiki_text[:len(clean_response)].lower():
                    return clean_response

            return None

        except Exception as e:
            self.logger.debug(f"GPT2 rewrite failed: {e}")
            return None
