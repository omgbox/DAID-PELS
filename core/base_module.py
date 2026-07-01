"""
BaseModule - Abstract base class for all BookBot modules.

All BookBot modules inherit from this class, which provides:
- Common interface (process, train, validate, get_stats, reset)
- Common dependencies (config, db, logger)
- Convenience methods (get_config)

Usage:
    class MyModule(BaseModule):
        def process(self, input_data):
            # Your implementation here
            return {'result': '...'}
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging


class BaseModule(ABC):
    """Abstract base class for all BookBot modules."""

    def __init__(self, config: Dict = None, db_manager=None, logger=None):
        """
        Initialize the module.

        Args:
            config: Configuration dictionary for this module.
            db_manager: Database manager instance (optional).
            logger: Logger instance (optional, creates default if not provided).
        """
        self.config = config or {}
        self.db = db_manager
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._initialized = False

    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Main processing method. Must be overridden by subclasses.

        Args:
            input_data: Input data to process (type varies by module).

        Returns:
            Dictionary with processing results.
        """
        pass

    def train(self, training_data: Any) -> Dict[str, Any]:
        """
        Training method. Override if module needs training.

        Args:
            training_data: Training data.

        Returns:
            Dictionary with training results.
        """
        return {}

    def validate(self) -> bool:
        """
        Validate module state. Override for custom validation.

        Returns:
            True if module is valid, False otherwise.
        """
        return True

    def get_stats(self) -> Dict[str, Any]:
        """
        Return module statistics. Override for custom stats.

        Returns:
            Dictionary with module statistics.
        """
        return {
            'module': self.__class__.__name__,
            'initialized': self._initialized
        }

    def reset(self):
        """Reset module state. Override for custom reset."""
        self._initialized = False

    def save_state(self, path: str):
        """
        Save module state to disk. Override if needed.

        Args:
            path: Path to save state file.
        """
        pass

    def load_state(self, path: str):
        """
        Load module state from disk. Override if needed.

        Args:
            path: Path to state file.
        """
        pass

    def initialize(self):
        """Initialize module (load models, build indices, etc.)."""
        self._initialized = True

    def get_config(self, key: str, default=None) -> Any:
        """
        Get config value for this module.

        Convenience method that automatically looks up config values
        under the module's class name (lowercased).

        Example:
            If class is OCRNormalizer, looks up config['ocr_normalizer'][key]

        Args:
            key: Configuration key.
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        module_name = self.__class__.__name__.lower()
        return self.config.get(module_name, {}).get(key, default)
