"""
src/domain/interfaces/model_loader.py
Port interface for model loading, quantization, and adapter attachment.
"""

from abc import ABC, abstractmethod
from typing import Any
from pathlib import Path
from src.domain.interfaces.config_loader import ConfigLoaderBase


class ModelLoaderBase(ABC):
    """Abstract port for base model loading and PEFT adapter wrapping."""

    def load_model(self, model_id: str, quantization_config: Any = None, device_map: Any = None, **kwargs: Any) -> Any:
        """Alias for load_base_model."""
        return self.load_base_model(model_id, quantization_config=quantization_config, device_map=device_map, **kwargs)

    @abstractmethod
    def load_base_model(self, model_id: str, quantization_config: Any = None, device_map: Any = None, **kwargs: Any) -> Any:
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
    def load_adapters(self,adapter_path:Path):
        """
        Load adapter model in specified precision.

        Args:
            model_path: local directory path.

        Returns:
            PreTrainedModel instance.
        """
        
        raise NotImplementedError("load_adapters method must be implemented by concrete loader.")

    @abstractmethod
    def save_merged_model(self):
        raise NotImplementedError("save_merged_model method must be implemented by concrete loader.")


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
