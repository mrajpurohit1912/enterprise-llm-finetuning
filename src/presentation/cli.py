"""
src/presentation/cli.py
Command-line presentation interface for triggering fine-tuning pipelines.
"""

import argparse
import sys
from pathlib import Path

from src.application.usecases.train_pipeline import TrainPipelineUsecase
from src.domain.exceptions import DomainError
from src.infrastructure.config.yaml_loader import YamlConfigLoader
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.infrastructure.huggingface.huggingface_tokenizer import HuggingFaceTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enterprise LLM Fine-Tuning Pipeline CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="src/finetuning_config.yaml",
        help="Path to YAML fine-tuning experiment specification",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    print("=" * 70)
    print("🚀 Enterprise LLM Fine-Tuning Pipeline")
    print(f"📄 Configuration: {config_path.resolve()}")
    print("=" * 70)

    try:
        config_loader = YamlConfigLoader(config_path=config_path)
        dataset_factory = DatasetLoaderFactory()
        tokenizer_loader = HuggingFaceTokenizer()

        pipeline = TrainPipelineUsecase(
            config_loader=config_loader,
            dataset_loader_factory=dataset_factory,
            tokenizer_loader=tokenizer_loader,
        )

        dataset = pipeline.run()

        print("\n" + "=" * 70)
        print("✅ Data Ingestion & Preprocessing Completed Successfully!")
        print("=" * 70)
        print(f"📊 Dataset Structure: {dataset}")

        # Show sample output if available
        if hasattr(dataset, "keys"):
            for split in dataset.keys():
                count = len(dataset[split])
                print(f"   • Split '{split}': {count:,} samples")
                if count > 0 and "text" in dataset[split].column_names:
                    preview = dataset[split][0]["text"]
                    truncated = (preview[:200] + "...") if len(preview) > 200 else preview
                    print(f"     Preview: {truncated!r}")

    except DomainError as err:
        print(f"\n❌ [Domain Error] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"\n💥 [Fatal Error] Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
