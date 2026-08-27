"""
src/application/services/dataset_processor.py
Application service that maps formatting strategies across dataset splits.
"""

from typing import Optional, Union
from datasets import Dataset, DatasetDict

from src.domain.exceptions import PreprocessingError
from src.domain.interfaces.prompt_formatter import PromptFormatterBase


class DatasetProcessor:
    """Orchestrates batch mapping and transformations across Hugging Face datasets."""

    def __init__(self, formatter: PromptFormatterBase) -> None:
        if formatter is None:
            raise PreprocessingError("PromptFormatter cannot be None.")
        self._formatter = formatter

    def process(
        self,
        dataset: Union[DatasetDict, Dataset],
        remove_columns: bool = False,
        num_proc: Optional[int] = None,
    ) -> Union[DatasetDict, Dataset]:
        """
        Apply prompt formatting across all dataset splits.

        Args:
            dataset: Input DatasetDict or single Dataset split.
            remove_columns: Whether to strip original raw columns, leaving only 'text'.
            num_proc: Number of worker processes for parallel batch mapping.

        Returns:
            Processed dataset with standardized 'text' column.

        Raises:
            PreprocessingError: If dataset is empty, null, or mapping fails.
        """
        if dataset is None:
            raise PreprocessingError("Cannot process null dataset.")

        try:
            if isinstance(dataset, DatasetDict):
                processed_splits = {}
                for split_name, split_data in dataset.items():
                    processed_splits[split_name] = self._process_single_split(
                        split_data, remove_columns=remove_columns, num_proc=num_proc
                    )
                return DatasetDict(processed_splits)
            elif isinstance(dataset, Dataset):
                return self._process_single_split(
                    dataset, remove_columns=remove_columns, num_proc=num_proc
                )
            else:
                raise PreprocessingError(
                    f"Unsupported dataset type: {type(dataset).__name__}. Expected Dataset or DatasetDict."
                )
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(f"Dataset processing failed: {exc}") from exc

    def _process_single_split(
        self,
        split_data: Dataset,
        remove_columns: bool = False,
        num_proc: Optional[int] = None,
    ) -> Dataset:
        """Processes a single dataset partition."""
        if len(split_data) == 0:
            return split_data

        cols_to_remove = (
            [col for col in split_data.column_names if col != "text"]
            if remove_columns
            else None
        )

        processed = split_data.map(
            self._formatter.format_batch,
            batched=True,
            remove_columns=cols_to_remove,
            num_proc=num_proc,
            desc="Formatting chat templates",
        )

        if "text" not in processed.column_names:
            raise PreprocessingError(
                f"Processing completed but 'text' column is missing from output columns: {processed.column_names}"
            )

        return processed
