from datasets import load_dataset


from domain.interfaces.dataset_loader import DataLoaderBase



class HuggingFaceDatasetLoader(DataLoaderBase):


    def load_data(self,dataset_name):
        return load_dataset(dataset_name)