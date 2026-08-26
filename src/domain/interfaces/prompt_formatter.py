from abc import ABC, abstractmethod


class PromptFormatter(ABC):

    @abstractmethod
    def format_batch(self, batch):
        pass