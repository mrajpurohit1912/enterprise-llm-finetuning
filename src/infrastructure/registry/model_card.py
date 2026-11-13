"""
src/infrastructure/registry/model_card.py
Utility for generating enterprise-standard Model Cards with YAML metadata and evaluation lineage.
"""

from typing import Any


class ModelCardGenerator:
    """Generates standardized Markdown Model Cards with Hugging Face Hub YAML frontmatter."""

    @staticmethod
    def generate(
        model_name: str,
        base_model: str | None = None,
        description: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Produce a complete README.md Model Card string.

        Args:
            model_name: Name of the registered model.
            base_model: ID of the base foundation model (e.g. 'meta-llama/Llama-3.2-3B-Instruct').
            description: Freeform description or purpose of the model.
            tags: Category tags (e.g. ['finance', 'lora', 'sft']).
            metadata: Training hyperparameters, dataset information, and evaluation scorecard.

        Returns:
            str: Full Markdown content including YAML frontmatter.
        """
        metadata = metadata or {}
        tags = tags or ["fine-tuned", "lora", "transformers"]
        if "transformers" not in tags:
            tags.append("transformers")

        # 1. Frontmatter
        lines = [
            "---",
            "library_name: transformers",
            "pipeline_tag: text-generation",
        ]
        if base_model:
            lines.append(f"base_model: {base_model}")
        lines.append("tags:")
        for tag in tags:
            lines.append(f"  - {tag}")
        lines.append("---\n")

        # 2. Header and Description
        lines.append(f"# {model_name}\n")
        if description:
            lines.append(f"{description}\n")
        else:
            lines.append(
                f"This is an enterprise fine-tuned causal language model based on `{base_model or 'the foundation model'}`.\n"
            )

        # 3. Model Details
        lines.append("## Model Details\n")
        lines.append(f"- **Model Identifier:** `{model_name}`")
        if base_model:
            lines.append(f"- **Base Foundation Model:** `{base_model}`")
        if "dataset" in metadata:
            lines.append(f"- **Fine-Tuning Dataset:** `{metadata['dataset']}`")
        if "experiment_name" in metadata:
            lines.append(f"- **Experiment Name:** `{metadata['experiment_name']}`")
        lines.append("")

        # 4. Training Hyperparameters (if present)
        hyperparams = metadata.get("hyperparameters") or metadata.get("training_args")
        if hyperparams and isinstance(hyperparams, dict):
            lines.append("## Training Hyperparameters\n")
            lines.append("| Hyperparameter | Value |")
            lines.append("| :--- | :--- |")
            for k, v in sorted(hyperparams.items()):
                if k not in ("output_dir", "logging_dir") and v is not None:
                    lines.append(f"| `{k}` | `{v}` |")
            lines.append("")

        # 5. Evaluation Results (if present)
        evaluation = metadata.get("evaluation")
        if evaluation and isinstance(evaluation, dict):
            lines.append("## Evaluation Scorecard\n")
            task_type = evaluation.get("task_type", "text_generation")
            gate = "PASSED" if evaluation.get("passed_quality_gate") else "FAILED"
            lines.append(f"- **Evaluation Task:** `{task_type}`")
            lines.append(f"- **Quality Gate:** `{gate}`\n")

            metrics = evaluation.get("metrics") or {}
            if metrics:
                lines.append("| Metric | Score |")
                lines.append("| :--- | :--- |")
                for m_name, m_val in metrics.items():
                    if m_val is not None:
                        val_str = (
                            f"{m_val:.4f}" if isinstance(m_val, float) else str(m_val)
                        )
                        lines.append(f"| `{m_name}` | `{val_str}` |")
                lines.append("")

            delta_metrics = evaluation.get("delta_metrics") or {}
            if delta_metrics:
                lines.append("### Baseline Uplift (Delta vs Base Model)\n")
                lines.append("| Metric | Delta Uplift |")
                lines.append("| :--- | :--- |")
                for d_name, d_val in delta_metrics.items():
                    if d_val is not None:
                        lines.append(f"| `{d_name}` | `{d_val:+.4f}` |")
                lines.append("")

        # 6. Usage Example
        lines.append("## Usage Example\n")
        lines.append("```python")
        lines.append("import torch")
        lines.append("from transformers import AutoModelForCausalLM, AutoTokenizer\n")
        lines.append(f'model_id = "{model_name}"')
        lines.append("tokenizer = AutoTokenizer.from_pretrained(model_id)")
        lines.append(
            "model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map='auto')\n"
        )
        lines.append(
            'inputs = tokenizer("Hello! How can I help you today?", return_tensors="pt").to(model.device)'
        )
        lines.append("outputs = model.generate(**inputs, max_new_tokens=64)")
        lines.append("print(tokenizer.decode(outputs[0], skip_special_tokens=True))")
        lines.append("```\n")

        return "\n".join(lines)
