from typing import Any, Optional
from pathlib import Path
from transformers import AutoModelForCausalLM, PreTrainedModel
from peft import PeftConfig, PeftModel
import torch

from src.domain.interfaces.model_loader import ModelLoaderBase
from src.domain.exceptions import ModelLoadError,ModelMergeError
from src.domain.interfaces.config_loader import ConfigLoaderBase
import os


class TransformerLlmModelLoader(ModelLoaderBase):
    """Hugging Face causal language model loader adapter implementing ModelLoaderBase."""     
        
    def load_base_model(
        self, 
        model_id: str, 
        quantization_config: Optional[Any] = None, 
        device_map: Optional[Any] = None,
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


        if device_map is None:
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", 0))
                device_map = {"": local_rank}
            else:
                device_map = "cpu"

        try:
            bnb_config = self._build_bnb_config(quantization_config)
            if bnb_config is not None:
                if not torch.cuda.is_available():
                    raise ModelLoadError(
                        "4-bit quantization (BitsAndBytes NF4) requires an active NVIDIA CUDA GPU, "
                        "but torch.cuda.is_available() is False. Ensure your NVIDIA driver is running "
                        "or set 'quantization.load_in_4bit: false' in your configuration to run on CPU."
                    )
                compute_dtype = getattr(bnb_config, "bnb_4bit_compute_dtype", torch.float16)
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_id.strip(),
                    quantization_config=bnb_config,
                    dtype=compute_dtype,
                    torch_dtype=compute_dtype,
                    device_map=device_map,
                    **kwargs,
                )
            else:
                compute_dtype = torch.float16 
                model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=model_id.strip(),
                    dtype=compute_dtype,
                    torch_dtype=compute_dtype,
                    device_map=device_map,
                    low_cpu_mem_usage=True,
                    **kwargs,
                )

            # Align model config torch_dtype
            if hasattr(model, "config"):
                model.config.torch_dtype = compute_dtype


            for name, param in model.named_parameters():
                if param is not None and param.dtype == torch.bfloat16:
                    param.data = param.data.to(compute_dtype)

            for name, buf in model.named_buffers():
                if buf is not None and buf.dtype == torch.bfloat16:
                    buf.data = buf.data.to(compute_dtype)

            return model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load LLM model '{model_id}': {exc}"
            ) from exc

    def load_adapters(
            self,
            base_model,
            adapter_path:Path):
        
        try:
            merged_model = PeftModel.from_pretrained(base_model, adapter_path)
            return merged_model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to merge base model and adapter {exc}"
            ) from exc


    def save_merged_model(self, peft_model: Any, output_dir: Path) -> Path:
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
  
            # 1. Fuse weights
            fused_model = peft_model.merge_and_unload()                                                                                                                                
                                                                                                                                                                                       
            # 2. Save standalone safetensors weights                                                                                                                                   
            fused_model.save_pretrained(str(output_dir), safe_serialization=True)                                                                                                      
            return output_dir                                                                                                                                                          
        except Exception as exc:                                                                                                                                                       
            raise ModelMergeError(f"Failed to save merged model: {exc}") from exc    


    def _build_bnb_config(self, quantization_config: Optional[Any]) -> Optional[Any]:
        """Convert domain QuantizationConfig or return existing BitsAndBytesConfig."""
        if quantization_config is None:
            return None

        from transformers import BitsAndBytesConfig

        if isinstance(quantization_config, BitsAndBytesConfig):
            return quantization_config

        if getattr(quantization_config, "load_in_4bit", False):
            compute_dtype_str = getattr(quantization_config, "bnb_4bit_compute_dtype", "bfloat16")
            compute_dtype = getattr(torch, compute_dtype_str, torch.bfloat16)
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=getattr(quantization_config, "bnb_4bit_quant_type", "nf4"),
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=getattr(quantization_config, "bnb_4bit_use_double_quant", True),
            )
        return None

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

            # Enable non-reentrant gradient checkpointing and prepare for k-bit training
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )

            lora_config = LoraConfig(
                task_type=getattr(peft_config, "task_type", "CAUSAL_LM"),
                r=getattr(peft_config, "r", 16),
                lora_alpha=getattr(peft_config, "lora_alpha", 32),
                lora_dropout=getattr(peft_config, "lora_dropout", 0.05),
                bias=getattr(peft_config, "bias", "none"),
                target_modules=getattr(peft_config, "target_modules", None),
            )
            peft_model = get_peft_model(model, lora_config)

            # Defensive alignment: Ensure trainable LoRA parameters and buffers do not retain bfloat16
            for name, param in peft_model.named_parameters():
                if param is not None and param.dtype == torch.bfloat16:
                    param.data = param.data.to(torch.float32)

            for name, buf in peft_model.named_buffers():
                if buf is not None and buf.dtype == torch.bfloat16:
                    buf.data = buf.data.to(torch.float32)

            peft_model.print_trainable_parameters()
            return peft_model
        except Exception as exc:
            raise ModelLoadError(f"Failed to apply PEFT LoRA adapter: {exc}") from exc
