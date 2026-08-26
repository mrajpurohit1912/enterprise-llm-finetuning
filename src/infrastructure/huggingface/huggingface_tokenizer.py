from transformers import AutoTokenizer

from domain.interfaces.tokenizer import TokenizerBase


class HuggingFaceTokenizer(TokenizerBase):

    def get_tokenizer(self, model_name: str):

        return AutoTokenizer.from_pretrained(model_name)