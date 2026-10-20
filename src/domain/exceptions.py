"""
src/domain/exceptions.py
Domain-specific exceptions defining business and operational failure states.
"""


class DomainError(Exception):
    """Base exception for all domain layer errors."""
    pass


class ConfigurationError(DomainError):
    """Raised when configuration validation or loading fails."""
    pass


class DatasetIngestionError(DomainError):
    """Raised when dataset fetching, loading, or split validation fails."""
    pass


class PreprocessingError(DomainError):
    """Raised when formatting, tokenization, or batch processing fails."""
    pass


class ModelLoadError(DomainError):
    """Raised when foundation model loading or quantization fails."""
    pass


class PipelineExecutionError(DomainError):
    """Raised when an end-to-end training pipeline stage fails."""
    pass


class TrainerError(DomainError):
    """Raised when trainer initialization or execution fails."""
    pass


class ModelEvaluationError(DomainError):
    """Raised when model evaluation, generation, or metric scoring fails."""
    pass

