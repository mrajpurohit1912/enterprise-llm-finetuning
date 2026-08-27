"""
src/domain/interfaces/model_loader.py
Port interface for model loading, quantization, and adapter attachment.
"""

from abc import ABC, abstractmethod
from typing import Any


class ModelLoaderBase(ABC):
    """Abstract port for base model loading and PEFT adapter wrapping."""

    @abstractmethod
    def load_model(self, model_id: str, quantization_config: Any, **kwargs: Any) -> Any:
        """
        Load foundation model in specified precision.

        Args:
            model_id: Model repository ID or local directory path.
            quantization_config: QuantizationConfig domain entity.
            kwargs: Additional loader kwargs.

        Returns:
            PreTrainedModel instance.
        """
        raise NotImplementedError("load_model method must be implemented by concrete loader.")

    @abstractmethod
    def apply_peft(self, model: Any, peft_config: Any) -> Any:
        """
        Wrap base model with LoRA/QLoRA adapter layers.

        Args:
            model: Base pretrained model.
            peft_config: PEFTConfig domain entity.

        Returns:
            Wrapped PeftModel instance.
        """
        raise NotImplementedError("apply_peft method must be implemented by concrete loader.")
