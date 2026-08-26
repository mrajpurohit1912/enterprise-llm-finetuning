from abc import ABC,abstractmethod



class DataLoaderBase(ABC):

    @abstractmethod
    def load_data(self,dataset_name:str):
        raise NotImplementedError("load_data method not implemented")