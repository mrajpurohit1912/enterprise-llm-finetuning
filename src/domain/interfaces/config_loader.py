from abc import ABC, abstractmethod

from src.domain.schemas.config_schema import ExperimentConfig


class ConfigLoaderBase(ABC):
    @abstractmethod
    def load_config(self)->ExperimentConfig:
        raise NotImplementedError("load_config method not implemented")
