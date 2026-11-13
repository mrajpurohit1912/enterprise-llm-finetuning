"""
src/application/usecases/merge_model_usecase.py
Atomic application usecase orchestrating the fusing of low-rank PEFT adapter
weights with a pristine full-precision foundation model.
"""

import logging
from pathlib import Path
from typing import Optional

from src.domain.exceptions import ModelMergeError
from src.domain.interfaces.model_loader import ModelLoaderBase
from src.domain.interfaces.tokenizer import TokenizerBase
from src.domain.schemas.config_schema import ExperimentConfig

logger = logging.getLogger(__name__)


class MergeModelUseCase:
    """
    Orchestrates the fusing of low-rank PEFT adapter weights 
    with a pristine full-precision foundation model on CPU RAM.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        model_loader: ModelLoaderBase,
        tokenizer_loader: Optional[TokenizerBase] = None,
    ):
        self.config = config
        self.model_loader = model_loader
        self.tokenizer_loader = tokenizer_loader

    def merge_and_unload_model(self) -> Path:
        """
        Merge trained LoRA adapter weights into base model and persist standalone artifacts.

        Returns:
            Path: Destination path containing the merged standalone model and tokenizer.

        Raises:
            ModelMergeError: If loading, merging, or serialization encounters an error.
        """
        try:
            # 1. Resolve adapter directory (strip -evaluation suffix if present)
            adapter_dir = Path(self.config.artifact.output_dir)
            if not (adapter_dir / "adapter_model.safetensors").exists():
                fallback = Path(str(adapter_dir).replace("-evaluation", "").replace("_evaluation", ""))
                if (fallback / fallback.name / "adapter_model.safetensors").exists():
                    adapter_dir = fallback / fallback.name
                elif (fallback / "adapter_model.safetensors").exists():
                    adapter_dir = fallback

            logger.info("Resolved adapter checkpoint source directory: '%s'", adapter_dir)

            # 2. Load unquantized 16-bit base model on CPU (avoids GPU VRAM limits)
            logger.info("Loading pristine base model '%s' on CPU...", self.config.llm_model.llm_model_id)
            base_model = self.model_loader.load_base_model(
                model_id=self.config.llm_model.llm_model_id,
                quantization_config=None,
                device_map="cpu",
            )

            # 3. Attach trained LoRA adapter
            logger.info("Attaching trained LoRA adapter weights from '%s'...", adapter_dir)
            peft_model = self.model_loader.load_adapters(base_model, adapter_dir)

            # 4. Fuse weights and save standalone safetensors
            export_dir = Path(self.config.artifact.output_dir) / "merged_model"
            export_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Fusing adapter weights and saving standalone model to '%s'...", export_dir)
            self.model_loader.save_merged_model(peft_model, export_dir)

            # 5. Save tokenizer so the merged directory is a 100% self-contained drop-in
            logger.info("Saving tokenizer artifacts to '%s'...", export_dir)
            if self.tokenizer_loader is not None:
                tokenizer = self.tokenizer_loader.get_tokenizer(str(adapter_dir))
            else:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
            tokenizer.save_pretrained(str(export_dir))

            logger.info("Successfully exported merged standalone model and tokenizer to '%s'.", export_dir)
            print(f"✔ Successfully saved merged standalone model to: {export_dir}")
            return export_dir

        except Exception as exc:
            logger.error("Failed to merge and export model: %s", exc, exc_info=True)
            raise ModelMergeError(f"Failed to merge model: {exc}") from exc
