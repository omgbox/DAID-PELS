"""
BookBot Pass Manager
Orchestrates Pass 0-4 execution.
"""

import logging
from typing import Dict, List, Optional

from ..core.base_module import BaseModule
from ..pipeline_context import PipelineContext

logger = logging.getLogger('bookbot.training.pass_manager')


class PassManager(BaseModule):
    """Training pass manager module."""

    def __init__(self, config: dict = None, db_manager=None, logger=None):
        super().__init__(config, db_manager, logger)
        self.modules = {}

    def register_module(self, name: str, module: BaseModule):
        """
        Register a module for training.

        Args:
            name: Module name
            module: Module instance
        """
        self.modules[name] = module

    def process(self, input_data) -> dict:
        """
        Process input data through training passes.

        Args:
            input_data: PipelineContext

        Returns:
            Dict with training results
        """
        if not isinstance(input_data, PipelineContext):
            raise ValueError("PassManager requires PipelineContext")

        context = input_data
        results = {}

        # Pass 0: OCR Normalization
        logger.info("=== PASS 0: OCR NORMALIZATION ===")
        ocr = self.modules.get('ocr_normalizer')
        if ocr:
            result = ocr.process(context)
            context.update(result)
            results['pass0'] = result

        # Pass 1: Lexical Foundation
        logger.info("=== PASS 1: LEXICAL FOUNDATION ===")

        # Tokenizer
        tokenizer = self.modules.get('tokenizer')
        if tokenizer:
            result = tokenizer.process(context)
            context.update(result)
            results['tokenizer'] = result

        # POS Tagger
        pos_tagger = self.modules.get('pos_tagger')
        if pos_tagger:
            result = pos_tagger.process(context)
            context.update(result)
            results['pos_tagger'] = result

        # POS Guesser (Phase 2: distributional)
        pos_guesser = self.modules.get('pos_guesser')
        if pos_guesser:
            pos_guesser.advance_phase()
            result = pos_guesser.process(context)
            context.update(result)
            results['pos_guesser'] = result

        # Definition Linker
        def_linker = self.modules.get('definition_linker')
        if def_linker:
            result = def_linker.process(context)
            context.update(result)
            results['definition_linker'] = result

        # IDF Builder
        idf_builder = self.modules.get('idf_builder')
        if idf_builder:
            result = idf_builder.process(context)
            context.update(result)
            results['idf_builder'] = result

        # Iterative passes
        max_passes = self.config.get('convergence', {}).get('max_passes', 10)
        convergence_tracker = self.modules.get('convergence_tracker')

        for cycle in range(max_passes):
            logger.info(f"=== ITERATIVE CYCLE {cycle + 1} ===")

            # Pass 2: Semantic Enrichment
            logger.info("--- Pass 2: Semantic Enrichment ---")
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
            logger.info("--- Pass 3: Relational Deepening ---")
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
            logger.info("--- Pass 4: Discourse and Pragmatics ---")
            temporal = self.modules.get('temporal_reasoner')
            if temporal:
                result = temporal.process(context)
                context.update(result)

            # Convergence Check
            if convergence_tracker:
                result = convergence_tracker.process(context)
                if result.get('converged', False):
                    logger.info("CONVERGED")
                    context.convergence_achieved = True
                    break

        return results
