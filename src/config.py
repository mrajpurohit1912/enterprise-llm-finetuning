from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv
load_dotenv()

@dataclass
class Config:
    # model_id:str = "Qwen/Qwen2.5-0.5B-Instruct"
    #training_dataset_name:str = "databricks/officeqa"
    output_dir:Path = Path.cwd() / "output" / "lora_finetuning_output"
    hf_token = os.getenv("HF_TOKEN")