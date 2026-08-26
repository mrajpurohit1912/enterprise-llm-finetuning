from transformers import AutoTokenizer

from domain.interfaces.tokenizer import TokenizerBase


class HuggingFaceTokenizer(TokenizerBase):
    def get_tokenizer(self,tokenizer_model_name):
        return AutoTokenizer.from_pretrained(tokenizer_model_name)