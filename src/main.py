# from data_loader.base import DataLoaderBase

# from data_loader.interfaces import HuggingFaceDataLoader
# from config import Config


# class MainFacade():
#     def __init__(self,config:Config,data_loader:DataLoaderBase):
#         self.config = config
#         self.data_loader = data_loader

#     def run_full_pipeline(self):
#         data = self.data_loader.load_data(self.config.training_dataset_name)

#         data = self.data_loader.process_data(data,self.config.model_id)
#         print(data['train'][0]['question'])
#         print(f"##########################################")
#         print(data['train'][0]['texts'])


# if __name__ == "__main__":
#     config = Config()
#     data_loader = HuggingFaceDataLoader()
#     main_pipeline = MainFacade(config,data_loader)
#     main_pipeline.run_full_pipeline()

from presentation.cli import main

if __name__ == "__main__":
    main()