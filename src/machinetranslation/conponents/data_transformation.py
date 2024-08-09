
from datasets import Dataset
from transformers import M2M100Tokenizer
import pandas as pd
from machinetranslation.logging import logger
from typing import Tuple
from machinetranslation.entity.config_entity import DataTransformationConfig
import os


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = M2M100Tokenizer.from_pretrained(self.config.tokenizer_path)

    def read_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.config.data_path)
        logger.info(f"Data shape: {df.shape}")
        return df

    def preprocess_function(self, examples):
        inputs = self.tokenizer(examples['en'], max_length=self.config.max_length, truncation=True, padding="max_length")
        targets = self.tokenizer(examples['bn'], max_length=self.config.max_length, truncation=True, padding="max_length")
        inputs["labels"] = targets["input_ids"]
        return inputs

    def transform_data(self) -> Tuple[Dataset, Dataset]:
        df = self.read_data()
        dataset = Dataset.from_pandas(df)
        
        tokenized_dataset = dataset.map(self.preprocess_function, batched=True, remove_columns=dataset.column_names)
        
        train_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(len(tokenized_dataset) * 0.8)))
        eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(len(tokenized_dataset) * 0.8), len(tokenized_dataset)))

        return train_dataset, eval_dataset

    def save_datasets(self, train_dataset: Dataset, eval_dataset: Dataset):
        train_dataset.save_to_disk(os.path.join(self.config.root_dir, "train"))
        eval_dataset.save_to_disk(os.path.join(self.config.root_dir, "eval"))
        logger.info(f"Datasets saved to {self.config.root_dir}")
