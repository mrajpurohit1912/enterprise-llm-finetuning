from abc import ABC,abstractmethod



class TokenizerBase(ABC):

    @abstractmethod
    def get_tokenizer(self,tokenizer_model_name:str):
        raise NotImplementedError("get_tokenizer method not implemented")