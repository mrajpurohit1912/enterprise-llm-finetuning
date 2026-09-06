"""
src/presentation/cli.py
Command-line presentation interface for triggering fine-tuning pipelines.
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv

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
    load_dotenv()   
    args = parse_args()
    config_path = Path(args.config)



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


        print(dataset)
    except DomainError as err:
        print(f"[Domain Error] {err}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f" [Fatal Error] Unexpected failure: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
