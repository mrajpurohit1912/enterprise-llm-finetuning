from src.domain.interfaces.dataset_loader import DataLoaderBase
from src.domain.schemas.config_schema import DatasetConfig

class LoadDatasetUseCase():
    def __init__(self,
                 loader:DataLoaderBase) -> None:
        self._loader = loader

    def execute(
            self,
            dataset_config: DatasetConfig
            ):

        return self._loader.load_data(dataset_config.dataset_name)