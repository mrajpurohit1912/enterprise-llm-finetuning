from datasets import DatasetDict

from domain.interfaces.prompt_formatter import PromptFormatter


class DatasetProcessor:
    def __init__(self,formatter:PromptFormatter) -> None:
        self._formatter = formatter

    def process(self,
                dataset:DatasetDict
                )-> DatasetDict:
        return dataset.map(
            self._formatter.format_batch,
            batched=True)
        