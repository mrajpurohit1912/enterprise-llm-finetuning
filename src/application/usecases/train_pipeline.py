from src.domain.schemas.config_schema import ExperimentConfig
from src.domain.interfaces.config_loader import ConfigLoaderBase
from src.application.usecases.load_dataset_usecase import LoadDatasetUseCase
from src.application.usecases.preprocess_dataset_usecase import PreprocessDatasetUseCase
from src.infrastructure.factories.dataset_factory import DatasetLoaderFactory



class TrainPipelineUsecase:
    def __init__(
        self,
        config_loader: ConfigLoaderBase,
        dataset_loader_factory: DatasetLoaderFactory,
    ):
        self.config_loader = config_loader
        self.dataset_loader_factory = dataset_loader_factory

    def run(self):
        # 1. Load Configuration
        self.config = self.config_loader.load_config()
        print(f"[Pipeline] Config loaded for experiment: {self.config.experiment_name}")

        # 2. Dynamic Data Ingestion via Factory
        dataset_loader = self.dataset_loader_factory.get_loader(self.config.dataset.source)
        load_dataset_usecase = LoadDatasetUseCase(loader=dataset_loader)
        dataset = load_dataset_usecase.execute(dataset_config=self.config.dataset)
        print(f"[Pipeline] Successfully loaded dataset: '{self.config.dataset.dataset_name}'")

        # 3. Preprocess Dataset
        preprocess_usecase = PreprocessDatasetUseCase()


        return dataset