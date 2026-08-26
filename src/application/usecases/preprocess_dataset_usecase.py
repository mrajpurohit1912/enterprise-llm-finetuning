from application.services.dataset_processor import DatasetProcessor


class PreprocessDatasetUseCase:

    def __init__(

        self,

        processor: DatasetProcessor

    ):

        self._processor = processor


    def execute(self, dataset):

        return self._processor.process(dataset)