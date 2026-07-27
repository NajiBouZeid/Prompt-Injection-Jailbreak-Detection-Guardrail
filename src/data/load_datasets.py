from datasets import load_dataset
import pandas as pd


def load_hf_dataset(dataset_name: str) -> pd.DataFrame:
    """Load any HuggingFace dataset by name and return its first split as a pandas DataFrame."""
    dataset = load_dataset(dataset_name)
    split_name = list(dataset.keys())[0]
    print(f"{dataset_name}: available splits = {list(dataset.keys())}, using '{split_name}'")
    return dataset[split_name].to_pandas()
