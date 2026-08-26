import yaml
from src.domain.schemas.config_schema import ExperimentConfig
from src.domain.interfaces.config_loader import ConfigLoaderBase



class YamlConfigLoader(ConfigLoaderBase):

    def __init__(self,config_path:str):
        self.config_path = config_path

    def load_config(self)->ExperimentConfig:

        with open(self.config_path, 'r') as file:
            config_dict = yaml.safe_load(file)

        return ExperimentConfig(**config_dict)