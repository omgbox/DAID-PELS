# BookBot Core Modules
# These modules inherit from BaseModule and provide BookBot-specific functionality.

from .base_module import BaseModule
from .ocr_normalizer import OCRNormalizer
from .pos_guesser import POSGuesser
from .tokenizer import Tokenizer
from .pos_tagger import POSTagger
from .ner_extractor import NERExtractor
from .svo_extractor import SVOExtractor
from .definition_linker import DefinitionLinker
from .idf_builder import IDFBuilder
from .idiom_detector import IdiomDetector
from .metaphor_detector import MetaphorDetector
from .coreference import Coreference
from .temporal_reasoner import TemporalReasoner
from .entity_graph import EntityGraph
from .topic_modeler import TopicModeler
