from src.domain.schemas.config_schema import DatasetSourceType
from src.domain.interfaces.dataset_loader import DataLoaderBase
from src.infrastructure.huggingface.huggingface_dataset_loader import HuggingFaceDatasetLoader



class DatasetLoaderFactory:

    @staticmethod
    def get_loader(source_type: DatasetSourceType)-> DataLoaderBase:
        if source_type == DatasetSourceType.HUGGINGFACE:
            
            return HuggingFaceDatasetLoader()
        else:
            raise ValueError(f"Unsupported dataset source type: {source_type}")