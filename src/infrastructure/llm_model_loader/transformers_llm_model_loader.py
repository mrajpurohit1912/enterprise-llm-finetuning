from typing import Any, Optional
from transformers import AutoModelForCausalLM, PreTrainedModel
import torch

from src.domain.interfaces.model_loader import ModelLoaderBase
from src.domain.exceptions import ModelLoadError


class TransformerLlmModelLoader(ModelLoaderBase):
    """Hugging Face causal language model loader adapter implementing ModelLoaderBase."""

    def load_model(
        self, 
        model_id: str, 
        quantization_config: Optional[Any] = None, 
        **kwargs: Any,
    ) -> PreTrainedModel:
        """
        Load a foundation LLM in either quantized (4-bit/8-bit) or standard 16-bit precision.

        Args:
            model_id (str): Hugging Face repository ID or local model path.
            quantization_config (Any, optional): BitsAndBytes configuration for k-bit quantization.
            **kwargs (Any): Additional keyword arguments passed to AutoModelForCausalLM.

        Raises:
            ModelLoadError: If loading model weights or device placement fails.

        Returns:
            PreTrainedModel: Loaded and device-mapped model instance.
        """
        if not model_id or not model_id.strip():
            raise ModelLoadError("Model ID cannot be empty or blank.")

        try:
            if quantization_config:
                return AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_id.strip(),
                    quantization_config=quantization_config,
                    device_map="auto",
                    **kwargs,
                )

            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_id.strip(),
                torch_dtype=torch.bfloat16,
                device_map="auto",
                **kwargs,
            )
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load LLM model '{model_id}': {exc}"
            ) from exc

    def apply_peft(self, model: PreTrainedModel, peft_config: Any) -> Any:
        """
        Wrap base model with LoRA/QLoRA adapter layers.

        Args:
            model (PreTrainedModel): Base foundation model.
            peft_config (Any): Domain PEFTConfig or peft.LoraConfig.

        Raises:
            ModelLoadError: If adapter injection fails.

        Returns:
            Any: PeftModel with injected LoRA adapters.
        """
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

            # Enable gradient checkpointing and prepare for k-bit training
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                task_type=getattr(peft_config, "task_type", "CAUSAL_LM"),
                r=getattr(peft_config, "r", 16),
                lora_alpha=getattr(peft_config, "lora_alpha", 32),
                lora_dropout=getattr(peft_config, "lora_dropout", 0.05),
                bias=getattr(peft_config, "bias", "none"),
                target_modules=getattr(peft_config, "target_modules", None),
            )
            peft_model = get_peft_model(model, lora_config)
            peft_model.print_trainable_parameters()
            return peft_model
        except Exception as exc:
            raise ModelLoadError(f"Failed to apply PEFT LoRA adapter: {exc}") from exc

