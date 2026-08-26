# Phase 4: Contract
# Domain Abstract Interfaces (Ports)

**Document Version:** 1.0.0  
**Status:** Approved  

---

## 1. Domain Interface Contracts

The following Python abstract classes define the **exact contracts (Ports)** that all infrastructure adapters and application use cases must adhere to.

```python
"""
src/domain/interfaces/
Core domain abstract interfaces (Ports) in accordance with Clean Architecture.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional
from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase, PreTrainedModel


class DataLoaderBase(ABC):
    """Port for all dataset ingestion adapters (HuggingFace, S3, Local, Kaggle)."""

    @abstractmethod
    def load_data(self, dataset_name: str, split: Optional[str] = None) -> DatasetDict:
        """
        Fetch and load a dataset into a standard DatasetDict.
        
        Args:
            dataset_name: Dataset identifier, URL, or local file path.
            split: Optional specific split to load.
            
        Returns:
            DatasetDict containing loaded data splits.
            
        Raises:
            DatasetLoadError: If loading fails or data source is unreachable.
        """
        raise NotImplementedError


class TokenizerBase(ABC):
    """Port for tokenizer loading and vocabulary management."""

    @abstractmethod
    def get_tokenizer(self, model_name: str) -> PreTrainedTokenizerBase:
        """Load and configure the target model tokenizer."""
        raise NotImplementedError


class PromptFormatterBase(ABC):
    """Port for chat-template and prompt-completion batch transformation strategies."""

    @abstractmethod
    def format_batch(self, batch: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        """
        Format a batch of raw records into model-consumable prompt strings.
        
        Args:
            batch: Dictionary of column lists (e.g. {"question": [...], "answer": [...]}).
            
        Returns:
            Dictionary containing formatted text strings (e.g. {"text": [...]}).
        """
        raise NotImplementedError


class ConfigLoaderBase(ABC):
    """Port for configuration file parsers (YAML, JSON, TOML)."""

    @abstractmethod
    def load_config(self) -> Any:
        """Parse, validate, and return the strongly-typed ExperimentConfig entity."""
        raise NotImplementedError


class ModelLoaderBase(ABC):
    """Port for base model precision loading and PEFT adapter wrapping."""

    @abstractmethod
    def load_model(self, model_id: str, quantization_config: Any) -> PreTrainedModel:
        """Load foundation model in specified precision (4-bit / 16-bit)."""
        raise NotImplementedError

    @abstractmethod
    def apply_peft(self, model: PreTrainedModel, peft_config: Any) -> PreTrainedModel:
        """Wrap base model with LoRA/QLoRA adapter layers."""
        raise NotImplementedError


class ModelRegistryBase(ABC):
    """Port for registering, versioning, and publishing model artifacts."""

    @abstractmethod
    def register(self, artifact_path: Path, model_name: str, metadata: Dict[str, Any]) -> str:
        """
        Publish serialized model to registry (Hugging Face Hub / MLflow / S3).
        
        Returns:
            Registered model URI or version string.
        """
        raise NotImplementedError
```
