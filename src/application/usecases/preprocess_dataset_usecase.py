"""
src/application/usecases/preprocess_dataset_usecase.py
Use case for orchestrating prompt formatting and preprocessing transformations on datasets.
"""

from typing import Any, Optional
from src.application.services.dataset_processor import DatasetProcessor
from src.domain.exceptions import PreprocessingError


class PreprocessDatasetUseCase:
    """Use case encapsulating the batch mapping of chat templates onto datasets."""

    def __init__(self, processor: DatasetProcessor) -> None:
        if processor is None:
            raise PreprocessingError("DatasetProcessor cannot be None for PreprocessDatasetUseCase.")
        self._processor = processor

    def execute(
        self,
        dataset: Any,
        remove_columns: bool = False,
        num_proc: Optional[int] = None,
    ) -> Any:
        """
        Execute chat-template preprocessing across dataset splits.

        Args:
            dataset: Raw loaded dataset (DatasetDict or Dataset).
            remove_columns: Strip raw non-text columns.
            num_proc: Number of parallel worker processes.

        Returns:
            Preprocessed dataset with formatted 'text' column.

        Raises:
            PreprocessingError: If preprocessing transformation fails.
        """
        return self._processor.process(
            dataset=dataset,
            remove_columns=remove_columns,
            num_proc=num_proc,
        )
