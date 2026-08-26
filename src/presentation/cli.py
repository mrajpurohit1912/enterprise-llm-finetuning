from src.infrastructure.config.yaml_loader import YamlConfigLoader
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory
from src.application.usecases.train_pipeline import TrainPipelineUsecase


def main():
    config_loader = YamlConfigLoader(config_path="src/finetuning_config.yaml")
    dataset_loader_factory = DatasetLoaderFactory()

    pipeline = TrainPipelineUsecase(
        config_loader=config_loader,
        dataset_loader_factory=dataset_loader_factory,
    )
    dataset = pipeline.run()
    print(f"[CLI] Successfully completed dataset loading stage: {dataset}")


if __name__ == "__main__":
    main()