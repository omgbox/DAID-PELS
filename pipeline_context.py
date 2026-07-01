"""
PipelineContext - Shared context passed between modules in the pipeline.

This class holds all data that flows between modules during training
and querying. Modules read from and write to this context.
"""

from typing import Any, Dict, List, Optional


class PipelineContext:
    """Shared context passed between modules in the pipeline."""

    def __init__(self):
        # Raw data
        self.raw_text: str = ""
        self.normalized_text: str = ""

        # Tokenized data
        self.sentences: List[Dict] = []
        self.tokens: List[List[Dict]] = []

        # Annotated data
        self.entities: List[Dict] = []
        self.svo_triples: List[Dict] = []
        self.coreferences: List[Dict] = []

        # Knowledge
        self.knowledge_edges: List[Dict] = []
        self.topics: List[Dict] = []
        self.temporal_events: List[Dict] = []
        self.causal_chains: List[Dict] = []
        self.metaphors: List[Dict] = []
        self.idioms: List[Dict] = []

        # Metadata
        self.chapter_boundaries: List[int] = []
        self.paragraph_boundaries: List[int] = []

        # Dictionary data
        self.definitions: Dict[str, List[Dict]] = {}
        self.vocabulary: Dict[str, Dict] = {}

        # Training state
        self.pass_number: int = 0
        self.convergence_achieved: bool = False
        self.training_stats: Dict[str, Any] = {}

    def get(self, key: str, default=None) -> Any:
        """
        Get a value from the context.

        Args:
            key: Attribute name.
            default: Default value if attribute not found.

        Returns:
            Attribute value or default.
        """
        return getattr(self, key, default)

    def set(self, key: str, value: Any):
        """
        Set a value in the context.

        Args:
            key: Attribute name.
            value: Value to set.
        """
        setattr(self, key, value)

    def update(self, data: Dict[str, Any]):
        """
        Update multiple values in the context.

        Args:
            data: Dictionary of key-value pairs to update.
        """
        for key, value in data.items():
            setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert context to dictionary.

        Returns:
            Dictionary representation of the context.
        """
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

    def clear(self):
        """Clear all context data."""
        self.__init__()
