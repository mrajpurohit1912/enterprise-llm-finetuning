"""
src/domain/exceptions.py
Domain-specific exceptions defining business and operational failure states.
"""


class DomainError(Exception):
    """Base exception for all domain layer errors."""


class ConfigurationError(DomainError):
    """Raised when configuration validation or loading fails."""


class DatasetIngestionError(DomainError):
    """Raised when dataset fetching, loading, or split validation fails."""


class PreprocessingError(DomainError):
    """Raised when formatting, tokenization, or batch processing fails."""


class ModelLoadError(DomainError):
    """Raised when foundation model loading or quantization fails."""


class PipelineExecutionError(DomainError):
    """Raised when an end-to-end training pipeline stage fails."""


class TrainerError(DomainError):
    """Raised when trainer initialization or execution fails."""


class ModelEvaluationError(DomainError):
    """Raised when model evaluation, generation, or metric scoring fails."""


class RemoteComputeError(DomainError):
    """Raised when remote compute job submission, polling, or artifact retrieval fails."""


class ModelMergeError(DomainError):
    """Raised when merging models fails."""


class ModelRegistryError(DomainError):
    """Raised when model registration, metadata generation, or artifact upload fails."""

class ModelRegistryError(DomainError):
    """Raised when model registration, metadata generation, or artifact upload fails."""
    pass


